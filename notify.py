#!/usr/bin/env python3
"""
notify.py  —  Daily trade digest email
────────────────────────────────────────
Sends an HTML email to NOTIFY_TO with:
  1. 🚨 ACTION REQUIRED  — stop loss hit, hold period expired, profit target, earnings warning
  2. 📊 Portfolio Snapshot — all open trades with live P&L
  3. 📅 Weekly P&L summary (Fridays only)

Run manually:    python3 notify.py          (dry-run, prints email to terminal)
Scheduled run:   python3 notify.py --send   (actually sends the email)

Only --send triggers the real email. The scheduler calls with --send.
"""

import os, sys, csv, smtplib, warnings, logging, contextlib
from pathlib     import Path
from datetime    import datetime, date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from typing      import Optional

import yfinance as yf
from show_tracker import _yf_ticker, fetch_live_price, fetch_company_name  # canonical impls (pre-market aware, GBp-safe)

try:
    from rrg_engine import run_sector_rrg, chart_rrg_scatter, QUAD_EMOJI, QUAD_COLORS
    _HAS_RRG = True
except ImportError:
    _HAS_RRG = False

HERE       = Path(__file__).parent
TRADES_CSV = HERE / "trades.csv"


# ── Bad-price-print detection ────────────────────────────────────────────────
# scan_history.csv occasionally records an implausible price (a bad yfinance
# tick), which then produces a fake ±70% "return". The old defence was a blunt
# `abs(ret) > 15` filter, but that also discarded ~50 REAL large moves and
# systematically flattered the stats: it censored big losses from strategies
# like vcp (+2.39% -> -0.29% once they are counted) and three_weeks_tight
# (+1.76% -> +0.02%).
#
# We therefore judge the PRICE, not the return. Two rules matter:
#
#   1. Only `price_at_scan` is tested. price_d5 / price_d10 are OUTCOMES — an
#      earlier version tested those too, which silently dropped any stock that
#      legitimately doubled and so reintroduced exactly the win-side censoring
#      this replaced.
#   2. Dispersion is measured with the median absolute deviation, not a fixed
#      ±2x band. A bad tick inflates the mean and the standard deviation but
#      barely moves the MAD, so the outlier cannot hide its own detection.
#
# Limitation, stated rather than papered over: a ticker with fewer than
# MIN_OBS scan-time observations has no internal reference to compare against,
# so a bad print on a one-off ticker is not detectable here and will pass
# through. Detecting that needs an external price source.
_SUSPECT_KEYS: Optional[set] = None
_SUSPECT_MIN_OBS  = 3      # scan-time prices needed before a ticker is testable
_SUSPECT_MAD_MULT = 8.0    # flag beyond 8 MADs from the median
_SUSPECT_MIN_RATIO = 1.6   # …and at least 1.6x / 0.625x off, so tight series
                           # (where MAD ≈ 0) don't flag ordinary daily moves


def _load_suspect_price_keys() -> set:
    """Return {(scan_date, ticker, strategy)} for rows with a bad price print."""
    global _SUSPECT_KEYS
    if _SUSPECT_KEYS is not None:
        return _SUSPECT_KEYS
    _SUSPECT_KEYS = set()
    path = HERE / "scan_history.csv"
    if not path.exists():
        return _SUSPECT_KEYS
    try:
        from collections import defaultdict as _dd
        rows_by_ticker = _dd(list)
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = (row.get("ticker") or "").strip()
                if t:
                    rows_by_ticker[t].append(row)

        def _f(v):
            try:
                x = float(v)
                return x if x > 0 else None
            except (TypeError, ValueError):
                return None

        def _median(xs):
            s = sorted(xs); n = len(s)
            return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

        for t, rows in rows_by_ticker.items():
            # Entry-time prices only — never outcomes.
            priced = [(r, _f(r.get("price_at_scan"))) for r in rows]
            obs = [v for _r, v in priced if v is not None]
            if len(obs) < _SUSPECT_MIN_OBS:
                continue
            med = _median(obs)
            if med <= 0:
                continue
            mad = _median([abs(v - med) for v in obs]) or (med * 0.02)
            for r, v in priced:
                if v is None:
                    continue
                ratio = v / med
                if (abs(v - med) > _SUSPECT_MAD_MULT * mad
                        and (ratio > _SUSPECT_MIN_RATIO or ratio < 1.0 / _SUSPECT_MIN_RATIO)):
                    _SUSPECT_KEYS.add((r.get("scan_date", ""), t, r.get("strategy", "")))
    except Exception:
        pass
    return _SUSPECT_KEYS


def _is_suspect_row(row: dict) -> bool:
    """True if this scan_history row has an implausible price print."""
    return (row.get("scan_date", ""), (row.get("ticker") or "").strip(),
            row.get("strategy", "")) in _load_suspect_price_keys()

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import (GMAIL_USER, GMAIL_APP_PASSWORD, NOTIFY_TO,
                        STOP_LOSS_PCT, PROFIT_TARGET, EARNINGS_WARN, HOLD_DAYS, DEFAULT_HOLD_DAYS)
except ImportError:
    print("ERROR: config.py not found. Create it with GMAIL_USER and GMAIL_APP_PASSWORD.")
    sys.exit(1)

TODAY = date.today()
IS_FRIDAY = TODAY.weekday() == 4

# ── Silence yfinance noise ────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

@contextlib.contextmanager
def _quiet():
    devnull = open(os.devnull, "w"); old = sys.stderr; sys.stderr = devnull
    try:    yield
    finally: sys.stderr = old; devnull.close()

# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_trades() -> list[dict]:
    if not TRADES_CSV.exists(): return []
    with open(TRADES_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ── Price fetching ────────────────────────────────────────────────────────────
# _yf_ticker and fetch_live_price imported from show_tracker (canonical, better impl)

_FX_CACHE: dict[str, float] = {}

def fetch_fx_now(currency: str) -> float:
    """Cached FX rate: 1 EUR = X currency. Uses in-memory cache to avoid repeat calls."""
    if currency in ("EUR", ""): return 1.0
    if currency in _FX_CACHE: return _FX_CACHE[currency]
    try:
        with _quiet():
            df = yf.Ticker(f"EUR{currency}=X").history(period="5d", interval="1d", auto_adjust=True)
        if not df.empty:
            rate = float(df["Close"].dropna().iloc[-1])
            _FX_CACHE[currency] = rate
            return rate
    except Exception:
        pass
    return 1.0

def fetch_10d_ema(ticker: str) -> Optional[float]:
    """Return current 10-day EMA for trailing stop check. None on failure."""
    try:
        with _quiet():
            df = yf.Ticker(_yf_ticker(ticker)).history(period="60d", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 10: return None
        close = df["Close"].dropna()
        ema10 = close.ewm(span=10, adjust=False).mean()
        return float(ema10.iloc[-1])
    except Exception:
        return None

def _check_exit_rules(trades: list, prices: dict) -> list[dict]:
    """
    Display-only exit rule checker for OPEN trades.

    Args:
        trades: list of trade dicts (same shape as trades.csv rows)
        prices: dict of ticker → current price in native currency (e.g. USD for US stocks)

    Returns list of alert dicts:
        {"ticker", "company", "trade_type", "type", "message", "priority"}

    Types emitted:
        "PARTIAL_EXIT"  — price reached +1.5R, take half off the table
        "MOVE_STOP_BE"  — after +1.5R, move stop to break-even
        "TRAIL_STOP"    — 10d EMA > stop_loss while price > entry, trail the stop up
    """
    alerts = []
    for t in trades:
        if t.get("status") != "OPEN":
            continue
        ticker = t.get("ticker", "")
        curr_price = prices.get(ticker)
        if curr_price is None:
            continue

        # ── Parse entry / stop fields ──────────────────────────────────────────
        try:
            entry_price = float(t.get("buy_price") or 0)
            if entry_price <= 0:
                continue
        except (ValueError, TypeError):
            continue

        sl_raw = t.get("stop_loss_price") or ""
        try:
            stop_loss = float(sl_raw) if str(sl_raw).strip() else 0.0
        except (ValueError, TypeError):
            stop_loss = 0.0

        if stop_loss <= 0:
            continue  # no actionable stop → skip

        initial_risk = entry_price - stop_loss
        if initial_risk <= 0:
            continue  # stop above entry or zero risk → skip

        ccy = t.get("currency", "USD")
        sym = _ccy_sym(ccy)
        company = t.get("company", "")
        tt = t.get("trade_type", "practice")
        pnl_r = (curr_price - entry_price) / initial_risk

        # ── Rule 1 & 2: >= 1.5R → partial exit + move stop to break-even ──────
        if pnl_r >= 1.5:
            alerts.append({
                "ticker":     ticker,
                "company":    company,
                "trade_type": tt,
                "type":       "PARTIAL_EXIT",
                "message":    (f"Sell half now at ~{sym}{curr_price:.2f} "
                               f"— you're up {pnl_r:.1f}R"),
                "priority":   1,
            })
            alerts.append({
                "ticker":     ticker,
                "company":    company,
                "trade_type": tt,
                "type":       "MOVE_STOP_BE",
                "message":    (f"Move stop loss to your entry price "
                               f"({sym}{entry_price:.2f}) — now risk-free"),
                "priority":   1,
            })

        # ── Rule 3: Trail stop to 10d EMA (only when price > entry) ──────────
        if curr_price > entry_price:
            ema10 = fetch_10d_ema(ticker)
            if ema10 is not None and ema10 > stop_loss:
                alerts.append({
                    "ticker":     ticker,
                    "company":    company,
                    "trade_type": tt,
                    "type":       "TRAIL_STOP",
                    "message":    (f"Trail stop to 10d EMA: {sym}{ema10:.2f} "
                                   f"— locks in gains"),
                    "priority":   2,
                })

    return alerts


def fetch_earnings_date(ticker: str) -> Optional[date]:
    """Return next earnings date if within EARNINGS_WARN days, else None."""
    try:
        with _quiet():
            cal = yf.Ticker(_yf_ticker(ticker)).calendar
        if cal is None: return None
        ed = cal.get("Earnings Date")
        if ed is None: return None
        # calendar returns list or single value
        if hasattr(ed, "__iter__") and not isinstance(ed, str):
            ed = list(ed)[0]
        if hasattr(ed, "date"):
            ed = ed.date()
        elif isinstance(ed, str):
            ed = datetime.strptime(ed[:10], "%Y-%m-%d").date()
        if isinstance(ed, date) and (ed - TODAY).days <= EARNINGS_WARN:
            return ed
    except Exception:
        pass
    return None

# ── P&L computation ───────────────────────────────────────────────────────────

def compute_pnl(trade: dict) -> dict:
    """Returns dict with curr_price_eur, ret_pct, pnl_eur, alerts list."""
    alerts = []
    try:
        buy_price  = float(trade["buy_price"])
        fx_entry   = float(trade["fx_at_entry"] or 1)
        qty        = float(trade["qty"])
        invest_eur = float(trade["investment_eur"] or 1000)
        ccy        = trade.get("currency", "EUR")
        sl_price   = float(trade.get("stop_loss_price") or buy_price * 0.97)
    except (ValueError, TypeError):
        return {"curr_eur": None, "ret_pct": None, "pnl_eur": None, "alerts": []}

    raw_price, fetched_ccy = fetch_live_price(trade["ticker"])
    if raw_price is None:
        return {"curr_eur": None, "ret_pct": None, "pnl_eur": None, "alerts": []}

    # Cross-currency normalisation to stored currency
    if fetched_ccy and fetched_ccy != ccy:
        fx_fetched = fetch_fx_now(fetched_ccy)
        fx_stored  = fetch_fx_now(ccy)
        curr_price = raw_price / fx_fetched * fx_stored if fx_fetched else raw_price
    else:
        curr_price = raw_price

    fx_now     = fetch_fx_now(ccy)
    curr_eur   = round(curr_price / fx_now, 2)
    buy_eur    = round(buy_price / fx_entry, 2)
    ret_pct    = round((curr_price / buy_price - 1) * 100, 2)
    pnl_eur    = round((curr_eur - buy_eur) * qty, 2)
    curr_native = round(curr_price, 2)  # price in stock's native currency

    # ── R-multiple tracking ────────────────────────────────────────────────────
    risk_pct  = (buy_price - sl_price) / buy_price  # e.g. 0.03 for 3% stop
    r_mult    = ret_pct / (risk_pct * 100) if risk_pct > 0 else 0  # current R
    target_1r = buy_price * (1 + risk_pct)       # breakeven move stop target
    target_1p5r = buy_price * (1 + risk_pct * 1.5)  # partial profit price

    # ── Alert checks ──────────────────────────────────────────────────────────
    if curr_price <= sl_price:
        drop_pct = round((curr_price / buy_price - 1) * 100, 2)
        alerts.append(("STOP_LOSS", f"Price {curr_eur:.2f}€ hit stop {round(sl_price/fx_now,2):.2f}€  ({drop_pct:+.1f}%)"))

    if ret_pct >= PROFIT_TARGET * 100:
        alerts.append(("PROFIT_TARGET", f"+{ret_pct:.1f}% — consider taking profits (target {int(PROFIT_TARGET*100)}%)"))

    # Exit rule 1: 1.5R reached → take 1/3 off, move stop to breakeven
    if r_mult >= 1.5 and curr_price > sl_price:
        alerts.append(("PROFIT_TARGET",
            f"✂ +{r_mult:.1f}R reached — take 1/3 off now (${curr_native:.2f}) · move stop to BE (${buy_price:.2f})"))

    # Exit rule 2: stop should be moved to breakeven if >1R
    elif r_mult >= 1.0 and curr_price > sl_price:
        alerts.append(("PROFIT_TARGET",
            f"🔒 +{r_mult:.1f}R — move stop to breakeven (${buy_price:.2f}) to lock in cost-free trade"))

    target_exit = trade.get("target_exit_date", "")
    if target_exit and TODAY >= datetime.strptime(target_exit, "%Y-%m-%d").date():
        days_over = (TODAY - datetime.strptime(target_exit, "%Y-%m-%d").date()).days
        hold_d = trade.get("hold_days", DEFAULT_HOLD_DAYS)
        alerts.append(("HOLD_EXPIRED", f"Hold period ({hold_d}d) expired {days_over} day(s) ago — target exit was {target_exit}"))

    # Exit rule 3: trailing 10d EMA breach — price crossed below
    ema10 = fetch_10d_ema(trade["ticker"])
    if ema10 is not None and curr_price < ema10 and curr_price > sl_price:
        gap_pct = (ema10 - curr_price) / ema10 * 100
        alerts.append(("TRAIL_BREACH",
            f"⚠ Price ${curr_native:.2f} crossed below 10d EMA ${ema10:.2f} ({gap_pct:.1f}% below) — consider tightening stop or exiting"))

    earn_d = fetch_earnings_date(trade["ticker"])
    if earn_d:
        days_to = (earn_d - TODAY).days
        alerts.append(("EARNINGS", f"Earnings in {days_to} day(s) on {earn_d} — consider exiting before"))

    return {"curr_eur": curr_eur, "buy_eur": buy_eur, "ret_pct": ret_pct, "pnl_eur": pnl_eur,
            "curr_native": curr_native, "r_mult": round(r_mult, 2), "alerts": alerts}

# ── Gmail-safe inline style helpers ──────────────────────────────────────────
# Rules: white background, all layout via <table>, all colour via inline style.
# No flexbox, no nth-child, no CSS variables — Gmail strips them all.

_FONT  = "font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;"
_W     = "max-width:860px;margin:0 auto;"
_BG    = "background:#ffffff;"

# ── Design tokens ────────────────────────────────────────────────────────────
# ONE light palette for the whole digest. Previously the portfolio sections were
# light and the scanner sections were near-black, so the email visibly changed
# theme halfway down. Light-only is also the safer choice across clients: Apple
# Mail aggressively auto-inverts dark palettes, Gmail inverts only some of them,
# and classic Outlook on Windows ignores dark mode entirely — a dark design is
# the one thing guaranteed to render three different ways.
_C_POS   = "#1a7f4b"   # green   — gains, pass
_C_NEG   = "#c0392b"   # red     — losses, fail
_C_WARN  = "#b7590a"   # orange  — caution
_C_DIM   = "#64748b"   # slate   — secondary text (was #888888: too low contrast)
_C_BODY  = "#1a1a1a"   # primary text
_C_HEAD  = "#ffffff"   # text on dark header row
_C_THEAD = "#2c3e50"   # navy table header
_C_ROW0  = "#ffffff"
_C_ROW1  = "#f7f8fa"   # zebra stripe
_C_BORD  = "#e0e4ea"   # hairline

# Semantic accents — used for section headers and badges
_A_BLUE   = "#1d4ed8"   # scanner / informational
_A_INDIGO = "#4f46e5"   # regime map, matrix
_A_GREEN  = "#15803d"   # conviction, proven edge
_A_AMBER  = "#b45309"   # watchlist, warnings
_A_RED    = "#b91c1c"   # action required

# Tinted surfaces — 50/100-level backgrounds, all safely light
_S_GREEN  = "#f0fdf4"; _S_GREEN2 = "#dcfce7"
_S_RED    = "#fef2f2"; _S_RED2   = "#fee2e2"
_S_AMBER  = "#fffbeb"; _S_AMBER2 = "#fef3c7"
_S_BLUE   = "#eff6ff"; _S_BLUE2  = "#dbeafe"
_S_INDIGO = "#eef2ff"; _S_SLATE  = "#f1f5f9"

_STRAT_COLORS = {
    "momentum":              ("#1a4a8a", "#dbeafe"),
    "breakout":              ("#4a1a8a", "#ede9fe"),
    "pocket_pivot":          ("#7a4a00", "#fef3c7"),
    "connors_rsi2":          ("#005a6e", "#cffafe"),
    "ema_ribbon":            ("#135e2e", "#dcfce7"),
    "nr7":                   ("#4a4a00", "#fefce8"),
    "bb_squeeze":            ("#003d6e", "#e0f0ff"),
    "high_tight_flag":       ("#6e0a00", "#ffe4e1"),
    "analyst_upgrade":       ("#006e3d", "#d1fae5"),
    "signal_velocity":       ("#6e006e", "#fce7f3"),
    "chokepoint_inflection": ("#005e5e", "#ccfbf1"),
    "stage4_short":          ("#8a0000", "#fee2e2"),
    "defensive_rotation":    ("#1a5a00", "#ecfccb"),
    "cup_handle":            ("#003e8a", "#dbeafe"),
    "power_earnings_gap":    ("#7a2200", "#ffedd5"),
    "darvas_box":            ("#3d2b00", "#fef9c3"),
    "rs_line":               ("#004d2e", "#d1fae5"),
    "vcp":                   ("#2d006e", "#ede9fe"),
    "elder_impulse":         ("#006e1a", "#dcfce7"),
    "holy_grail":            ("#4a2800", "#fff7ed"),
    "connors_3down":         ("#00406e", "#e0f2fe"),
    "williams_pct_r":        ("#006e5a", "#ccfbf1"),
    "bollinger_pctb":        ("#3d004a", "#fae8ff"),
    "connors_r3":            ("#004a1a", "#dcfce7"),
    "connors_tps":           ("#1a3a00", "#f0fce8"),
    "turtle_soup":           ("#004040", "#ccfbf1"),
    "raschke_8020":          ("#3a1a00", "#fff7ed"),
    "wyckoff_spring":        ("#1a3a4a", "#e0f4ff"),
    "weinstein_stage2":      ("#2a004a", "#f3e8ff"),
}

def _c(val, good_if_pos=True):
    if val is None: return _C_DIM
    if val > 0: return _C_POS if good_if_pos else _C_NEG
    if val < 0: return _C_NEG if good_if_pos else _C_POS
    return _C_BODY

def _pct(v): return f"{v:+.1f}%" if v is not None else "─"
def _eur(v): return f"{v:+.0f}€" if v is not None else "─"

def _strat_badge(strat: str) -> str:
    key = strat.lower().replace(" ", "_")
    fg, bg = _STRAT_COLORS.get(key, ("#333", "#eee"))
    return (f'<span style="background:{bg};color:{fg};border-radius:3px;'
            f'padding:2px 6px;font-size:10px;font-weight:700;'
            f'letter-spacing:.04em;white-space:nowrap;">{_strat_short(key)}</span>')

def _trade_badge(tt: str) -> str:
    if tt == "real":
        return ('<span style="background:#d1fae5;color:#065f46;border-radius:3px;'
                'padding:2px 6px;font-size:10px;font-weight:700;">REAL</span>')
    return ('<span style="background:#fef3c7;color:#92400e;border-radius:3px;'
            'padding:2px 6px;font-size:10px;font-weight:600;">PRAC</span>')

def _th(text, align="right", title=""):
    tip = f' title="{title}"' if title else ""
    return (f'<th{tip} style="background:{_C_THEAD};color:{_C_HEAD};'
            f'padding:8px 10px;text-align:{align};font-size:10px;'
            f'font-weight:700;letter-spacing:.06em;text-transform:uppercase;'
            f'white-space:nowrap;border:none;">{text}</th>')

def _td(content, align="right", color=_C_BODY, bold=False, bg=_C_ROW0, extra=""):
    fw = "font-weight:700;" if bold else ""
    return (f'<td style="padding:7px 10px;text-align:{align};color:{color};'
            f'{fw}font-size:12px;border-bottom:1px solid {_C_BORD};'
            f'background:{bg};vertical-align:middle;{extra}">{content}</td>')

def _section_head(icon: str, title: str, subtitle: str = "", color: str = "#2c3e50") -> str:
    sub = (f'<span style="font-size:11px;color:{_C_DIM};margin-left:10px;'
           f'font-weight:normal;">{subtitle}</span>') if subtitle else ""
    return (f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:28px;margin-bottom:8px;">'
            f'<tr><td style="border-left:4px solid {color};padding:4px 10px;">'
            f'<span style="font-size:13px;font-weight:700;color:{color};'
            f'letter-spacing:.04em;text-transform:uppercase;">{icon} {title}</span>'
            f'{sub}</td></tr></table>')

def _kpi_table(items: list[tuple]) -> str:
    """items = [(label, value, color), ...]"""
    cells = ""
    for label, value, color in items:
        cells += (f'<td style="padding:10px 16px;background:#f8f9fc;'
                  f'border:1px solid {_C_BORD};border-radius:6px;'
                  f'text-align:center;white-space:nowrap;">'
                  f'<div style="font-size:10px;color:{_C_DIM};letter-spacing:.05em;'
                  f'text-transform:uppercase;margin-bottom:4px;">{label}</div>'
                  f'<div style="font-size:20px;font-weight:700;color:{color};">{value}</div>'
                  f'</td>'
                  f'<td style="width:8px;"></td>')
    return (f'<table cellpadding="0" cellspacing="0" style="margin:10px 0 14px;">'
            f'<tr>{cells}</tr></table>')

_CCY_SYM = {"USD": "$", "GBP": "£", "EUR": "€", "CAD": "CA$",
            "INR": "₹", "CHF": "CHF ", "SEK": "kr", "DKK": "kr", "NOK": "kr"}

def _ccy_sym(ccy: str) -> str:
    return _CCY_SYM.get(ccy, ccy + " ")


# ── Ticker → currency, for scanner rows (which carry no currency field) ───────
# Scanner results only have a ticker, so prices were being printed with a
# hardcoded "$" — showing Indian stocks as "$1121.00" and London stocks as
# "$6335.00" (which is also pence, not pounds).
_SUFFIX_CCY = {
    ".L": "GBP", ".DE": "EUR", ".PA": "EUR", ".AS": "EUR", ".BR": "EUR",
    ".MI": "EUR", ".MC": "EUR", ".LS": "EUR", ".VI": "EUR", ".HE": "EUR",
    ".IR": "EUR", ".TO": "CAD", ".V": "CAD", ".NS": "INR", ".BO": "INR",
    ".SW": "CHF", ".ST": "SEK", ".CO": "DKK", ".OL": "NOK",
}


def _ticker_ccy(ticker: str) -> str:
    """Infer trading currency from the ticker suffix. Defaults to USD."""
    for sfx, ccy in _SUFFIX_CCY.items():
        if (ticker or "").endswith(sfx):
            return ccy
    return "USD"


def _fmt_price(ticker: str, value) -> str:
    """Format a LIVE SCANNER price with the right currency unit.

    London quotes arrive from yfinance's raw Close column in pence, so .L
    values are suffixed "p" rather than divided by 100 — pence is how the LSE
    actually quotes at that layer, and labelling the unit cannot be wrong the
    way a conversion can. Everything else gets its currency symbol from the
    ticker suffix.

    Use this ONLY for values read straight from the scanner (last_scan.json /
    r["price"]) — the hero pick, conviction cards, scan detail tables. Do NOT
    use it for trades.csv fields (buy_price, stop_loss_price) or a live price
    from show_tracker.fetch_live_price(): both of those are already converted
    to true pounds upstream (scan.py's practice-trade writer divides by 100
    before storing; fetch_live_price's _normalise_ccy divides GBp by 100 on
    read). Reapplying the pence suffix to an already-converted pounds value
    would understate it 100x on screen while the stored number is correct.
    Use _fmt_position_price for those instead.
    """
    if value in (None, "") or value != value:      # None / NaN
        return "─"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "─"
    ccy = _ticker_ccy(ticker)
    dec = 0 if abs(v) >= 1000 else 2               # ₹36,050 not ₹36050.00
    if ccy == "GBP":
        return f"{v:,.{dec}f}p"                    # LSE quotes in pence
    return f"{_ccy_sym(ccy)}{v:,.{dec}f}"


def _fmt_position_price(ticker: str, value) -> str:
    """Format a price that is ALREADY in true native units (pounds, not pence).

    For trades.csv fields (buy_price, stop_loss_price) and any value returned
    by show_tracker.fetch_live_price() — both are pre-converted to real GBP,
    unlike the raw scanner Close used by _fmt_price. See that function's
    docstring for the bug this distinction fixes.
    """
    if value in (None, "") or value != value:
        return "─"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "─"
    ccy = _ticker_ccy(ticker)
    dec = 0 if abs(v) >= 1000 else 2
    return f"{_ccy_sym(ccy)}{v:,.{dec}f}"


# ── Canonical strategy display names ─────────────────────────────────────────
# Previously each render site did s.replace("_"," ").title(), which produced
# "Vcp", "Nr7", "Ema Ribbon", "Ma50 Reclaim", "Bb Squeeze" and "Connors Rsi2".
# One map, used everywhere.
STRAT_LABEL = {
    "pocket_pivot": "Pocket Pivot",     "ema_ribbon": "EMA Ribbon",
    "cup_handle": "Cup Handle",         "connors_rsi2": "Connors RSI2",
    "signal_velocity": "Signal Velocity", "breakout": "Breakout",
    "vcp": "VCP",                       "nr7": "NR7",
    "wyckoff_spring": "Wyckoff Spring", "darvas_box": "Darvas Box",
    "raschke_8020": "Raschke 80/20",    "high_tight_flag": "High Tight Flag",
    "stage4_short": "Stage-4 Short",    "connors_3down": "Connors 3↓",
    "holy_grail": "Holy Grail",         "weinstein_stage2": "Weinstein Stage-2",
    "defensive_rotation": "Def. Rotation", "rs_line": "RS Line",
    "williams_pct_r": "Williams %R",    "bollinger_pctb": "BB %B",
    "turnover_momentum": "Turnover Mom.", "elder_impulse": "Elder Impulse",
    "ma50_reclaim": "MA50 Reclaim",     "momentum_burst": "Momentum Burst",
    "analyst_upgrade": "Analyst Upgrade", "bb_squeeze": "BB Squeeze",
    "connors_r3": "Connors R3",         "momentum": "Momentum",
    "power_earnings_gap": "Power Earnings Gap",
    "three_weeks_tight": "3-Weeks-Tight", "episodic_pivot": "Episodic Pivot",
    "combo_pp_ribbon": "Combo PP+Ribbon",
}

# Compact forms for narrow table cells and chips
STRAT_SHORT = {
    "pocket_pivot": "POCKET PIVOT",    "ema_ribbon": "EMA RIBBON",
    "cup_handle": "CUP & HANDLE",      "connors_rsi2": "CONNORS RSI2",
    "signal_velocity": "SIGNAL VEL.",  "breakout": "BREAKOUT",
    "vcp": "VCP",                      "nr7": "NR7",
    "wyckoff_spring": "WYCKOFF",       "darvas_box": "DARVAS",
    "raschke_8020": "RASCHKE 80/20",   "high_tight_flag": "HTF",
    "stage4_short": "STAGE-4 SHORT",   "connors_3down": "CONNORS 3↓",
    "holy_grail": "HOLY GRAIL",        "weinstein_stage2": "WEINSTEIN S2",
    "defensive_rotation": "DEF. ROT.", "rs_line": "RS LINE",
    "williams_pct_r": "WILLIAMS %R",   "bollinger_pctb": "BB %B",
    "turnover_momentum": "TURNOVER",   "elder_impulse": "ELDER IMPULSE",
    "ma50_reclaim": "MA50 RECLAIM",    "momentum_burst": "MOM. BURST",
    "analyst_upgrade": "ANALYST UPG.", "bb_squeeze": "BB SQUEEZE",
    "connors_r3": "CONNORS R3",        "momentum": "MOMENTUM",
    "power_earnings_gap": "POWER EPS GAP",
    "three_weeks_tight": "3-WEEKS-TIGHT", "episodic_pivot": "EPISODIC PIVOT",
    "combo_pp_ribbon": "COMBO PP+RIB",
}


# Ultra-compact codes for the cross-strategy matrix, which needs one narrow
# column per strategy. Slicing to 6 chars produced unreadable stubs like
# "MA50_R", "THREE_", "EPISOD", "COMBO_", "WYCKOF", "WEINST", "TURNOV".
_MATRIX_ABBR = {
    "three_weeks_tight": "3WT",   "episodic_pivot": "EPIV",
    "combo_pp_ribbon": "COMBO",   "wyckoff_spring": "WYCK",
    "weinstein_stage2": "WEIN",   "ma50_reclaim": "MA50",
    "turnover_momentum": "TURN",  "momentum_burst": "BURST",
    "momentum": "MNTM",           "elder_impulse": "ELDER",
    "connors_rsi2": "RSI2",       "connors_3down": "C3DN",
    "connors_r3": "CR3",          "raschke_8020": "R8020",
    "holy_grail": "HGRL",         "williams_pct_r": "WM%R",
    "bollinger_pctb": "BB%B",     "bb_squeeze": "BBSQ",
    "rs_line": "RSLN",            "darvas_box": "DRVS",
    "vcp": "VCP",                 "nr7": "NR7",
    "ema_ribbon": "RIBBON",       "pocket_pivot": "PP",
    "connors_tps": "CTPS",        "turtle_soup": "TSOUP",
    "signal_velocity": "SVEL",    "high_tight_flag": "HTF",
    "analyst_upgrade": "UPGRD",   "defensive_rotation": "DEFR",
    "power_earnings_gap": "PEG",  "stage4_short": "S4SH",
    "cup_handle": "C&H",          "breakout": "BREAK",
}


def _strat_label(s: str) -> str:
    """Human-readable strategy name — never Title-cases an acronym."""
    return STRAT_LABEL.get(s, (s or "").replace("_", " ").title())


def _strat_short(s: str) -> str:
    """Compact upper-case strategy name for chips and narrow cells."""
    return STRAT_SHORT.get(s, (s or "").replace("_", " ").upper())


def _company_label(r: dict, maxlen: int = 22) -> str:
    """Company name, blank when it is just the ticker echoed back.

    The name cache falls back to the ticker when yfinance has no name, which
    rendered as "AMZN AMZN" / a duplicated ticker under the card headline.
    """
    co = str(r.get("company") or "").strip()
    tk = str(r.get("ticker") or "").strip()
    if not co or co.upper() == tk.upper():
        return ""
    return co[:maxlen]

def _sl_eur_str(t: dict) -> str:
    """Stop-loss price, formatted from the TICKER's currency.

    Deliberately ignores the stored `currency` column: it's wrong on some
    legacy rows (GWW and CVS are booked EUR but are USD stocks — a one-off
    data-fix script corrects the existing rows; see fix_legacy_currency.py).
    Deriving from the ticker suffix is more robust than trusting that column.

    Uses _fmt_position_price, NOT _fmt_price: stop_loss_price in trades.csv is
    stored in true pounds for .L tickers (scan.py converts pence→pounds before
    writing), so it must not get the scanner's pence-suffix treatment.
    """
    if not t.get("stop_loss_price"):
        return "─"
    return _fmt_position_price(t.get("ticker", ""), t.get("stop_loss_price"))

def _portfolio_table(results_slice: list, alert_tickers: set) -> str:
    """Open-positions table.

    Consolidated from 16 columns to 9. At 16 the table could not fit an email
    client's width, so every cell wrapped and company names broke mid-word
    ("Air Products and Chemi" spread over three lines). Related fields are now
    stacked two-per-cell instead of each taking its own column:
        Ticker + company + type   →  Position
        Entry + target dates      →  Dates      (held / remaining underneath)
        Buy + current price       →  Buy → Now
        Qty + invested            →  Size
    """
    thead = ('<table width="100%" cellpadding="0" cellspacing="0" '
             f'style="border-collapse:collapse;font-size:12px;{_FONT}">'
             '<thead><tr>'
             + _th("Position", "left")
             + _th("Strategy", "left")
             + _th("Dates", "left", title="Entry date → target exit date; days held / days remaining")
             + _th("Buy → Now", title="Entry price → current price (native currency)")
             + _th("Size", title="Quantity and amount invested")
             + _th("Ret%")
             + _th("P&L€")
             + _th("R", title="Current R-multiple (1R = initial risk)")
             + _th("SL", title="Stop-loss price (native currency)")
             + '</tr></thead><tbody>')
    rows = ""
    for i, (t, r) in enumerate(results_slice):
        bg = _C_ROW1 if i % 2 else _C_ROW0
        has_alert = t["ticker"] in alert_tickers
        if has_alert: bg = "#fff8f0"
        ret_c = _c(r["ret_pct"])
        pnl_c = _c(r["pnl_eur"])
        try:
            entry_d   = datetime.strptime(t["entry_date"], "%Y-%m-%d").date()
            days_held = f"{(TODAY - entry_d).days}d"
        except Exception:
            days_held = "─"
        target = t.get("target_exit_date", "")
        try:
            days_rem = (datetime.strptime(target, "%Y-%m-%d").date() - TODAY).days
            rem_c    = _C_NEG if days_rem < 0 else (_C_WARN if days_rem <= 1 else _C_DIM)
            days_rem_s = f"{days_rem}d"
        except Exception:
            rem_c = _C_DIM; days_rem_s = "─"

        # ── Position: ticker + company + type badge, stacked ──────────────────
        tick_color = _C_NEG if has_alert else _C_BODY
        _co = _company_label(t, 26)
        _pos_cell = (
            f'<div style="white-space:nowrap;">'
            f'<b style="color:{tick_color};font-size:13px;">{t["ticker"]}</b> '
            f'{_trade_badge(t.get("trade_type","practice"))}</div>'
            + (f'<div style="font-size:10px;color:{_C_DIM};margin-top:1px;">{_co}</div>' if _co else "")
        )
        rows += f"<tr>"
        rows += _td(_pos_cell, "left", bg=bg)
        rows += _td(_strat_badge(t.get("strategy","")), "left", bg=bg, extra="white-space:nowrap;")

        # ── Dates: entry → target on line 1, held / remaining on line 2 ───────
        _dates_cell = (
            f'<div style="white-space:nowrap;font-size:11px;color:{_C_DIM};">'
            f'{t["entry_date"]} → {target or "─"}</div>'
            f'<div style="white-space:nowrap;font-size:10px;color:{_C_DIM};margin-top:1px;">'
            f'held {days_held} &nbsp;·&nbsp; <span style="color:{rem_c};font-weight:700;">'
            f'{days_rem_s} left</span></div>'
        )
        rows += _td(_dates_cell, "left", bg=bg)

        # ── Buy → Now, one cell ───────────────────────────────────────────────
        # Formatted from the ticker suffix, not the stored currency column —
        # see _sl_eur_str for why that column cannot be trusted.
        _tk      = t.get("ticker", "")
        # _fmt_position_price, not _fmt_price: both values here are already in
        # true native units (buy_price post-fix in scan.py; curr_native via
        # show_tracker.fetch_live_price's own pence->pounds conversion) —
        # see _fmt_price's docstring for why reapplying "p" would be wrong.
        _px_cell = (
            f'<div style="white-space:nowrap;">'
            f'<span style="color:{_C_DIM};">{_fmt_position_price(_tk, t.get("buy_price"))}</span>'
            f'<span style="color:{_C_DIM};"> → </span>'
            f'<b style="color:{_C_BODY};">{_fmt_position_price(_tk, r.get("curr_native"))}</b>'
            f'</div>'
        )
        rows += _td(_px_cell, "right", bg=bg)

        # ── Size: qty on line 1, invested on line 2 ───────────────────────────
        # `or 0` guards: a single blank cell here used to take down the whole email
        try:    _qty_v = float(t.get("qty") or 0)
        except (TypeError, ValueError): _qty_v = 0.0
        try:    _inv_v = float(t.get("investment_eur") or 0)
        except (TypeError, ValueError): _inv_v = 0.0
        _size_cell = (
            f'<div style="white-space:nowrap;font-size:11px;color:{_C_BODY};">'
            f'{_qty_v:.4g}</div>'
            f'<div style="white-space:nowrap;font-size:10px;color:{_C_DIM};">'
            f'€{_inv_v:,.0f}</div>'
        )
        rows += _td(_size_cell, "right", bg=bg)
        rows += _td(_pct(r["ret_pct"]), "right", ret_c, bold=True, bg=bg)
        rows += _td(_eur(r["pnl_eur"]), "right", pnl_c, bold=True, bg=bg)
        # R-multiple cell — colour coded: green ≥1R, orange ≥0, red <0
        _rm = r.get("r_mult")
        if _rm is not None:
            _rm_c = "#27ae60" if _rm >= 1.0 else ("#e67e22" if _rm >= 0 else _C_NEG)
            _rm_s = f"{_rm:+.1f}R"
        else:
            _rm_c = _C_DIM; _rm_s = "─"
        rows += _td(_rm_s, "right", _rm_c, bold=(_rm is not None and abs(_rm) >= 1.0), bg=bg)
        rows += _td(_sl_eur_str(t), "right", _C_DIM, bg=bg)
        rows += "</tr>"
    return thead + rows + "</tbody></table>"

def _build_matrix_html() -> str:
    scan_json = HERE / "last_scan.json"
    if not scan_json.exists():
        return ""
    try:
        import json
        data       = json.loads(scan_json.read_text())
        scan_date  = data.get("scan_date", "unknown")
        strategies = data.get("strategies", [])
        rbs        = data.get("results_by_strategy", {})
    except Exception:
        return ""
    if not strategies or not rbs:
        return ""

    all_tickers: dict = {}
    for strat, results in rbs.items():
        for r in results:
            tk = r["ticker"]
            if tk not in all_tickers: all_tickers[tk] = {}
            all_tickers[tk][strat] = r
    if not all_tickers:
        return ""

    col_labels = {
        "momentum":              "MNTM",
        "breakout":              "BREAK",
        "pocket_pivot":          "PP",
        "connors_rsi2":          "RSI2",
        "ema_ribbon":            "RIBBON",
        "nr7":                   "NR7",
        "bb_squeeze":            "BBSQ",
        "high_tight_flag":       "HTF",
        "analyst_upgrade":       "UPGRD",
        "signal_velocity":       "SVEL",
        "chokepoint_inflection": "CHOK",
        "stage4_short":          "S4SH",
        "defensive_rotation":    "DEFR",
        "cup_handle":            "C&H",
        "power_earnings_gap":    "PEG",
    }
    sorted_t   = sorted(all_tickers.items(), key=lambda kv: -len(kv[1]))

    # header
    th_row = (_th("Str", "center") + _th("Ticker", "left") + _th("Company", "left")
              + "".join(_th(col_labels.get(s) or _MATRIX_ABBR.get(s) or _strat_short(s),
                            "center", title=_strat_label(s)) for s in strategies))
    rows = ""
    multi_count = 0
    for i, (ticker, strat_map) in enumerate(sorted_t[:50]):
        company = next((_company_label({**r, "ticker": ticker}, 28)
                        for r in strat_map.values()
                        if _company_label({**r, "ticker": ticker}, 28)), "")
        passes  = len(strat_map)
        multi   = passes > 1
        if multi: multi_count += 1
        bg = "#fff5f5" if multi else (_C_ROW1 if i % 2 else _C_ROW0)
        tc = _C_NEG if multi else _C_BODY
        rows += "<tr>"
        rows += _td(str(passes), "center", _C_DIM if not multi else _C_NEG, bg=bg)
        rows += _td(f'<b style="color:{tc}">{ticker}</b>', "left", bg=bg)
        rows += _td(company or "─", "left", _C_DIM if not multi else _C_NEG, bg=bg)
        for s in strategies:
            if s in strat_map:
                r   = strat_map[s]
                wr  = r.get("wr")
                avg = r.get("avg")
                tip = f"WIN%={wr:.0f}%  AVG={avg:+.1f}%" if wr is not None and avg is not None else "✓"
                cell = f'<span title="{tip}" style="color:{_C_POS}">✓ {int(wr) if wr else ""}%</span>'
                rows += _td(cell, "center", bg=bg)
            else:
                rows += _td("─", "center", _C_DIM, bg=bg)
        rows += "</tr>"

    note = ""
    if multi_count:
        note = (f'<p style="font-size:11px;color:{_C_NEG};margin:6px 0 0;">'
                f'★ {multi_count} ticker(s) highlighted in red passed multiple strategies — highest conviction</p>')

    return (f'<br>{_section_head("🎯","Cross-Strategy Matrix",f"last scan {scan_date}","#6d28d9")}'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:12px;">'
            f'<thead><tr>{th_row}</tr></thead><tbody>{rows}</tbody></table>{note}')



def _load_streak_leaders(min_streak: int = 5) -> list:
    """Return tickers seen in ≥1 scanner on each of the last min_streak consecutive trading days."""
    import csv as _csv
    from collections import defaultdict
    csv_path = HERE / "scan_history.csv"
    if not csv_path.exists(): return []
    ticker_dates: dict = defaultdict(set)
    date_ticker_strats: dict = defaultdict(lambda: defaultdict(list))
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                t = row.get("ticker","").strip()
                sd = row.get("scan_date","").strip()
                st = row.get("strategy","").strip()
                if t and sd:
                    ticker_dates[t].add(sd)
                    date_ticker_strats[sd][t].append(st)
    except Exception:
        return []
    all_dates = sorted({sd for dates in ticker_dates.values() for sd in dates})
    if len(all_dates) < min_streak:
        return []
    recent_dates = all_dates[-min_streak:]
    results = []
    for ticker, dates_seen in ticker_dates.items():
        if all(d in dates_seen for d in recent_dates):
            streak = 0
            for d in reversed(all_dates):
                if d in dates_seen: streak += 1
                else: break
            last_date = max(dates_seen)
            strats = list({s for s in date_ticker_strats[last_date][ticker]})
            results.append({"ticker": ticker, "streak": streak, "strategies": strats, "last_date": last_date})
    return sorted(results, key=lambda x: -x["streak"])


# Regime colour palette — shared across scorecard, scan detail, regime map, conviction cards
REGIME_COLORS = {
    "All-weather": {"icon": "🌤", "color": "#15803d", "bg": "#dcfce7"},
    "Defensive":   {"icon": "🛡",  "color": "#0369a1", "bg": "#e0f2fe"},
    "Momentum":    {"icon": "📈", "color": "#6d28d9", "bg": "#ede9fe"},
    "Momentum+":   {"icon": "📈", "color": "#4f46e5", "bg": "#e0e7ff"},
    "Neutral":     {"icon": "〰", "color": "#475569", "bg": "#f1f5f9"},
    "":            {"icon": "─",  "color": "#64748b", "bg": "#f8fafc"},
}

_regime_cache: dict = {}   # strategy → (icon, label, color, bg)

def _load_regime_characters() -> dict:
    """Compute regime character for each strategy from scan_history.csv. Cached."""
    global _regime_cache
    if _regime_cache:
        return _regime_cache
    import csv as _csv, math as _math
    from collections import defaultdict as _dd
    csv_path = HERE / "scan_history.csv"
    if not csv_path.exists():
        return {}
    strats = _dd(lambda: {"bear": [], "neut": [], "bull": []})
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if not row.get("ret_d5") or not row.get("spy_ret_d5"): continue
                try:
                    ret = float(row["ret_d5"]); spy = float(row["spy_ret_d5"])
                    if _math.isnan(ret) or _is_suspect_row(row): continue
                    s = row["strategy"]
                    if spy <= -1:   strats[s]["bear"].append(ret)
                    elif spy >= 1:  strats[s]["bull"].append(ret)
                    else:           strats[s]["neut"].append(ret)
                except (ValueError, TypeError):
                    pass
    except Exception:
        return {}

    def _wr(lst): return sum(1 for x in lst if x > 0) / len(lst) * 100 if len(lst) >= 3 else None

    def _classify(s, d):
        bull_wr = _wr(d["bull"]); bear_wr = _wr(d["bear"]); neut_wr = _wr(d["neut"])
        n = sum(len(v) for v in d.values())
        if n < 8: return ("─", "", "#94a3b8", "#111827")
        if bull_wr is None or bear_wr is None:
            lbl = "Momentum" if (neut_wr and neut_wr >= 60) else "Neutral"
        else:
            diff = bull_wr - bear_wr
            if bear_wr >= 60:                   lbl = "All-weather"
            elif abs(diff) <= 10 and (bull_wr or 0) >= 50: lbl = "All-weather"
            elif diff >= 20:                    lbl = "Momentum"
            elif diff >= 10:                    lbl = "Momentum+"
            elif diff <= -10:                   lbl = "Defensive"
            else:                               lbl = "Neutral"
        meta = REGIME_COLORS.get(lbl, REGIME_COLORS[""])
        return (meta["icon"], lbl, meta["color"], meta["bg"])

    _regime_cache = {s: _classify(s, d) for s, d in strats.items()}
    return _regime_cache


def _regime_badge(strategy: str) -> str:
    """Inline HTML badge for a strategy's regime character."""
    chars = _load_regime_characters()
    if strategy not in chars: return ""
    icon, lbl, col, bg = chars[strategy]
    if not lbl: return ""
    return (f'<span style="background:{bg};color:{col};font-size:9px;font-weight:700;'
            f'border-radius:3px;padding:1px 5px;margin-left:5px;white-space:nowrap;">'
            f'{icon} {lbl}</span>')


def _regime_for(strategy: str) -> tuple:
    """(icon, label, colour, bg) for a strategy — empty tuple values if unknown."""
    return _load_regime_characters().get(strategy, ("", "", _C_DIM, _S_SLATE))


# Hold period per strategy. Was an inline literal at the conviction-card render
# site, so any strategy missing from it silently reported "Hold 5 days" —
# including three_weeks_tight, episodic_pivot and combo_pp_ribbon.
HOLD_DAYS_MAP = {
    "pocket_pivot": 7,      "ema_ribbon": 7,       "cup_handle": 10,
    "vcp": 10,              "connors_rsi2": 5,     "nr7": 3,
    "breakout": 5,          "wyckoff_spring": 10,  "ma50_reclaim": 7,
    "signal_velocity": 5,   "darvas_box": 7,       "high_tight_flag": 10,
    "momentum_burst": 5,    "elder_impulse": 7,    "rs_line": 10,
    "three_weeks_tight": 7, "episodic_pivot": 10,  "combo_pp_ribbon": 7,
    "bb_squeeze": 5,        "connors_3down": 5,    "connors_r3": 5,
    "raschke_8020": 5,      "holy_grail": 7,       "weinstein_stage2": 10,
    "defensive_rotation": 10, "stage4_short": 5,   "momentum": 5,
    "power_earnings_gap": 10, "turnover_momentum": 5, "analyst_upgrade": 5,
}


# Populated by _build_scanner_results_html(); read by build_email() to render
# the hero block. Reset on every scanner render so a stale pick can't leak
# into a later digest.
_TOP_PICK: dict = {}


def _build_hero_html() -> str:
    """The lead block: today's single best new pick, above everything else.

    Built from _TOP_PICK, which _build_scanner_results_html() fills from the same
    ranked list that produces the "#1 BEST" card, so the two can never disagree.
    Returns "" when there is no HIGH-conviction pick — the digest then opens on
    a plain "no setup today" note rather than an empty frame.
    """
    p = _TOP_PICK
    if not p or not p.get("ticker"):
        return (
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            f'style="margin:0 0 18px;">'
            f'<tr><td style="background:{_S_SLATE};border:1px solid {_C_BORD};'
            f'border-left:4px solid {_C_DIM};border-radius:6px;padding:16px 18px;">'
            f'<div style="font-size:11px;font-weight:700;color:{_C_DIM};'
            f'letter-spacing:.08em;text-transform:uppercase;">Today\'s call</div>'
            f'<div style="font-size:15px;font-weight:600;color:{_C_BODY};margin-top:6px;">'
            f'No high-conviction setup today.</div>'
            f'<div style="font-size:12px;color:{_C_DIM};margin-top:3px;">'
            f'Sitting out is a valid trade. Watchlist below.</div>'
            f'</td></tr></table>'
        )

    tk    = p["ticker"]
    price = _fmt_price(tk, p.get("price"))
    stop  = _fmt_price(tk, p.get("stop"))
    hold  = p.get("hold", 5)
    wr    = p.get("wr")
    avg   = p.get("avg")
    _icon, _rlbl, _rcol, _rbg = p.get("regime") or ("", "", _C_DIM, _S_SLATE)

    # Strategy chips — proven-edge strategies get the green surface
    chips = " ".join(
        f'<span style="background:{_S_GREEN2 if s in _PROVEN_EDGE_SET else _S_BLUE2};'
        f'color:{_A_GREEN if s in _PROVEN_EDGE_SET else _A_BLUE};border-radius:3px;'
        f'padding:2px 7px;font-size:10px;font-weight:700;white-space:nowrap;">'
        f'{_strat_short(s)}</span>'
        for s in (p.get("fired") or [])[:3]
    )

    # Corroborating signals, only shown when actually present
    marks = []
    if p.get("proven"):
        marks.append(f'<span style="color:{_A_GREEN};font-weight:700;">✦ Proven edge</span>')
    if (p.get("vol") or 0) >= 1.5:
        marks.append(f'<span style="color:{_A_AMBER};font-weight:700;">⚡ {p["vol"]:.1f}× volume</span>')
    if (p.get("persist") or 0) >= 2:
        marks.append(f'<span style="color:{_A_BLUE};font-weight:700;">🔁 {p["persist"]}d persistent</span>')
    if _rlbl:
        marks.append(f'<span style="color:{_rcol};font-weight:700;">{_icon} {_rlbl}</span>')
    marks_html = ' &nbsp;·&nbsp; '.join(marks)

    # One metric row: the numbers that decide the trade
    def _metric(label, value, colour=None):
        return (f'<td style="padding:0 18px 0 0;vertical-align:top;">'
                f'<div style="font-size:9px;font-weight:700;color:{_C_DIM};'
                f'letter-spacing:.07em;text-transform:uppercase;white-space:nowrap;">{label}</div>'
                f'<div style="font-size:15px;font-weight:700;color:{colour or _C_BODY};'
                f'margin-top:2px;white-space:nowrap;">{value}</div></td>')

    wr_col = _A_GREEN if (wr or 0) >= 60 else _A_AMBER
    metrics = (
        _metric("Entry", price)
        + _metric("Stop", stop, _C_NEG)
        + _metric("Hold", f"{hold} days")
        + _metric("Win rate", f"{wr:.0f}%" if wr is not None else "─", wr_col)
        + _metric("Avg return", f"{avg:+.2f}%" if avg is not None else "─",
                  _A_GREEN if (avg or 0) > 0 else _C_NEG)
    )

    extra = ""
    if (p.get("n_high") or 0) > 1:
        extra = (f'<div style="font-size:11px;color:{_C_DIM};margin-top:10px;'
                 f'padding-top:9px;border-top:1px solid {_C_BORD};">'
                 f'+{p["n_high"] - 1} more high-conviction pick'
                 f'{"s" if p["n_high"] > 2 else ""} below</div>')

    parts = [
        # shell
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="margin:0 0 18px;">'
        f'<tr><td style="background:{_S_GREEN};border:1px solid #bbf7d0;'
        f'border-left:4px solid {_A_GREEN};border-radius:6px;padding:16px 18px;">',

        # eyebrow
        f'<div style="font-size:11px;font-weight:700;color:{_A_GREEN};'
        f'letter-spacing:.08em;text-transform:uppercase;">★ Today\'s call</div>',

        # ticker + company
        f'<table role="presentation" cellpadding="0" cellspacing="0" style="margin-top:7px;">'
        f'<tr><td style="vertical-align:baseline;white-space:nowrap;">'
        f'<span style="font-size:27px;font-weight:800;color:{_C_BODY};'
        f'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:-.01em;">{tk}</span>'
        f'</td><td style="vertical-align:baseline;padding-left:11px;">'
        f'<span style="font-size:13px;color:{_C_DIM};">{p.get("company","")}</span>'
        f'</td></tr></table>',

        # strategy chips
        f'<div style="margin-top:9px;">{chips}</div>' if chips else '',

        # metric row — entry / stop / hold / WR / avg
        f'<table role="presentation" cellpadding="0" cellspacing="0" style="margin-top:13px;">'
        f'<tr>{metrics}</tr></table>',

        # supporting signals
        f'<div style="font-size:11px;margin-top:11px;">{marks_html}</div>' if marks_html else '',

        # plain-language instruction
        f'<div style="font-size:12px;color:{_C_BODY};margin-top:12px;padding:9px 11px;'
        f'background:#ffffff;border:1px solid #bbf7d0;border-radius:4px;">'
        f'Buy near <b>{price}</b>, stop at <b style="color:{_C_NEG};">{stop}</b>, '
        f'sell in <b>{hold} days</b>.</div>',

        extra,
        '</td></tr></table>',
    ]
    return "".join(parts)


def _build_regime_map_html() -> str:
    """Strategy × market-regime WR grid — shows which strategies are all-weather vs momentum-only."""
    import csv as _csv, math as _math
    from collections import defaultdict as _dd
    csv_path = HERE / "scan_history.csv"
    if not csv_path.exists():
        return ""

    strats = _dd(lambda: {"bear": [], "neut": [], "bull": []})
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if not row.get("ret_d5") or not row.get("spy_ret_d5"):
                    continue
                try:
                    ret = float(row["ret_d5"])
                    spy = float(row["spy_ret_d5"])
                    if _math.isnan(ret) or _is_suspect_row(row): continue
                    s = row["strategy"]
                    if spy <= -1:        strats[s]["bear"].append(ret)
                    elif spy >= 1:       strats[s]["bull"].append(ret)
                    else:                strats[s]["neut"].append(ret)
                except (ValueError, TypeError):
                    pass
    except Exception:
        return ""

    MIN_N = 8

    def _wr(lst):
        if len(lst) < 3: return None
        return sum(1 for x in lst if x > 0) / len(lst) * 100

    def _cell(wr, n):
        if wr is None:
            return (f'<td style="padding:6px 8px;text-align:center;color:#6b7280;font-size:11px;">─</td>')
        bg  = "#dcfce7" if wr >= 65 else ("#ecfccb" if wr >= 55 else ("#ffedd5" if wr >= 45 else "#fee2e2"))
        col = "#15803d" if wr >= 65 else ("#4d7c0f" if wr >= 55 else ("#c2410c" if wr >= 45 else "#b91c1c"))
        return (f'<td style="padding:6px 8px;text-align:center;background:{bg};font-size:12px;'
                f'font-weight:700;color:{col};">{wr:.0f}%<span style="font-size:9px;color:{col};'
                f'opacity:0.7;font-weight:400;"> n={n}</span></td>')

    def _label(bull_wr, bear_wr, neut_wr):
        if bull_wr is None or bear_wr is None:
            if neut_wr and neut_wr >= 60: return ("📈", "Momentum", "#4f46e5")
            return ("─", "", _C_DIM)
        diff = bull_wr - bear_wr
        if bear_wr >= 60:                  return ("🌤", "All-weather", "#15803d")
        if abs(diff) <= 10 and bull_wr>=50: return ("🌤", "All-weather", "#15803d")
        if diff >= 20:                     return ("📈", "Momentum", "#4f46e5")
        if diff >= 10:                     return ("📈", "Momentum+", "#6d28d9")
        if diff <= -10:                    return ("🛡", "Defensive", "#0369a1")
        return ("〰", "Neutral", _C_DIM)

    # Sort: all-weather first, then momentum, then bear-sensitive
    rows_data = []
    for s, d in strats.items():
        n = sum(len(v) for v in d.values())
        if n < MIN_N: continue
        bull_wr = _wr(d["bull"])
        bear_wr = _wr(d["bear"])
        neut_wr = _wr(d["neut"])
        # The point of this grid is regime *comparison*. A row with only one
        # populated cell (e.g. "Elder Impulse ─ 53% ─ ─") says nothing about
        # regime dependence and got no Character label either — it was pure
        # visual noise. Require at least two regimes to compare.
        if sum(x is not None for x in (bull_wr, bear_wr, neut_wr)) < 2:
            continue
        icon, lbl, lcol = _label(bull_wr, bear_wr, neut_wr)
        rows_data.append((s, d, bull_wr, bear_wr, neut_wr, icon, lbl, lcol, n))

    # Sort: all-weather > defensive > momentum > neutral; within each by bull WR
    order = {"All-weather": 0, "Defensive": 1, "Momentum+": 2, "Momentum": 3, "Neutral": 4, "": 5}
    rows_data.sort(key=lambda x: (order.get(x[6], 9), -(x[2] or 0)))

    rows_html = ""
    for i, (s, d, bull_wr, bear_wr, neut_wr, icon, lbl, lcol, n) in enumerate(rows_data):
        bg = _C_ROW1 if i % 2 else _C_ROW0
        name = _strat_label(s)
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:6px 10px;font-size:12px;font-weight:600;color:{_C_BODY};white-space:nowrap;">{name}</td>'
            + _cell(bear_wr, len(d["bear"]))
            + _cell(neut_wr, len(d["neut"]))
            + _cell(bull_wr, len(d["bull"]))
            + f'<td style="padding:6px 10px;font-size:11px;color:{lcol};white-space:nowrap;">{icon} {lbl}</td>'
            f'</tr>'
        )

    if not rows_html:
        return ""

    return (
        f'<br><table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:0;">'
        f'<tr><td style="background:#eef2ff;border-left:4px solid #818cf8;padding:7px 10px;border-radius:3px 3px 0 0;">'
        f'<span style="font-size:11px;font-weight:700;color:#4338ca;letter-spacing:.04em;">'
        f'🗺 STRATEGY REGIME MAP &nbsp;·&nbsp; win rate by market condition (SPY 5d return)</span>'
        f'<span style="font-size:10px;color:#4f46e5;margin-left:12px;">🌤 all-weather &nbsp; 📈 momentum &nbsp; 🛡 defensive</span>'
        f'</td></tr></table>'
        f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
        f'<thead><tr style="background:#eef2ff;">'
        f'<th style="padding:6px 10px;text-align:left;font-size:11px;color:#4f46e5;">Strategy</th>'
        f'<th style="padding:6px 8px;text-align:center;font-size:11px;color:#b91c1c;">🐻 Bear<br><span style="font-weight:400;font-size:9px;">SPY ≤ −1%</span></th>'
        f'<th style="padding:6px 8px;text-align:center;font-size:11px;color:#64748b;">〰 Neutral<br><span style="font-weight:400;font-size:9px;">−1% to +1%</span></th>'
        f'<th style="padding:6px 8px;text-align:center;font-size:11px;color:#15803d;">🐂 Bull<br><span style="font-weight:400;font-size:9px;">SPY ≥ +1%</span></th>'
        f'<th style="padding:6px 10px;text-align:left;font-size:11px;color:#4f46e5;">Character</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>'
    )


def _load_strategy_stats() -> dict:
    """Read scan_history.csv → per-strategy {n, wr, avg, avg_r} from filled ret_d5 rows."""
    csv_path = HERE / "scan_history.csv"
    if not csv_path.exists():
        return {}
    from collections import defaultdict
    import csv as _csv, math as _math
    # Strategies with known tracking issues — exclude from scorecard
    _EXCLUDED_FROM_STATS = {"stage4_short"}  # L020: ret tracking inverted for shorts
    stats = defaultdict(lambda: {"wins":0,"total":0,"sum":0.0,"r_sum":0.0,"r_n":0,"sl":0,"ex_sum":0.0,"ex_n":0})
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                strat = row.get("strategy", "").strip()
                if not strat or strat in _EXCLUDED_FROM_STATS:
                    continue
                try:
                    ret = float(row["ret_d5"])
                    if _math.isnan(ret) or _is_suspect_row(row):
                        continue
                    stats[strat]["total"] += 1
                    stats[strat]["sum"]   += ret
                    if ret > 0:
                        stats[strat]["wins"] += 1
                    if row.get("hit_stop_loss_d5") == "1":
                        stats[strat]["sl"] += 1
                    rm_raw = row.get("r_multiple_d5", "")
                    if rm_raw:
                        rm = float(rm_raw)
                        if not _math.isnan(rm):
                            stats[strat]["r_sum"] += rm
                            stats[strat]["r_n"]   += 1
                    ex_raw = row.get("excess_ret_d5", "")
                    if ex_raw:
                        ex = float(ex_raw)
                        if not _math.isnan(ex):
                            stats[strat]["ex_sum"] += ex
                            stats[strat]["ex_n"]   += 1
                except (ValueError, TypeError, KeyError):
                    pass
    except Exception:
        return {}
    out = {}
    for s, d in stats.items():
        n = d["total"]
        if n < 1:
            continue
        out[s] = {
            "n":      n,
            "wr":     round(100 * d["wins"] / n, 1),
            "avg":    round(d["sum"] / n, 2),
            "avg_r":  round(d["r_sum"] / d["r_n"], 2) if d["r_n"] > 0 else None,
            "sl_pct": round(100 * d["sl"] / n, 1),
            "excess": round(d["ex_sum"] / d["ex_n"], 2) if d["ex_n"] > 0 else None,
        }
    return out

_PROVEN_EDGE_SET = {"pocket_pivot", "ema_ribbon", "cup_handle",
                    "signal_velocity", "connors_rsi2"}
# stage4_short REMOVED: tracking inverted, true WR=11.4% (L020)

def _load_persistence_counts() -> dict:
    """Return {ticker: n_unique_scan_dates} from scan_history.csv."""
    import csv as _csv
    from collections import defaultdict
    p = HERE / "scan_history.csv"
    if not p.exists(): return {}
    ticker_dates = defaultdict(set)
    try:
        with open(p, newline="") as f:
            for row in _csv.DictReader(f):
                t = row.get("ticker","").strip()
                sd = row.get("scan_date","").strip()
                if t and sd:
                    ticker_dates[t].add(sd)
    except Exception:
        return {}
    return {t: len(d) for t, d in ticker_dates.items()}

def _conviction_tier_email(r: dict, multi_tickers: set) -> str:
    """Returns HIGH / MED / LOW conviction string."""
    is_multi   = r.get("ticker") in multi_tickers
    good_score = (r.get("score") or 99) <= 5
    rsi        = r.get("rsi") or 0
    adx        = r.get("adx") or 0
    good_rsi   = 50 <= rsi <= 70
    good_adx   = 16 <= adx <= 35
    if is_multi and good_score and good_rsi and good_adx:
        return "HIGH"
    if is_multi or (good_score and good_rsi and good_adx):
        return "MED"
    return "LOW"

# ── Metal spot snapshot (absorbed from metal_tracker.py) ─────────────────────
_METAL_TICKERS = [
    ("Gold",        "GC=F",   "$/oz"),
    ("Silver",      "SI=F",   "$/oz"),
    ("Copper",      "HG=F",   "$/lb"),
    ("Crude Oil",   "CL=F",   "$/bbl"),
    ("Aluminum",    "ALI=F",  "$/lb"),
    ("Platinum",    "PL=F",   "$/oz"),
    ("Rare Earths", "REMX",   "$ ETF"),
    ("Lithium",     "LIT",    "$ ETF"),
]

def _fetch_metal_snapshot() -> list[dict]:
    """Spot prices for key metals via yfinance. Fail-open returns []."""
    results = []
    for name, ticker, unit in _METAL_TICKERS:
        try:
            with _quiet():
                hist = yf.Ticker(ticker).history(period="14d", interval="1d", auto_adjust=True)
            if hist.empty or len(hist) < 2:
                continue
            c    = hist["Close"].dropna()
            spot = float(c.iloc[-1])
            p1d  = float(c.iloc[-2])
            p7d  = float(c.iloc[max(0, len(c) - 8)])
            results.append({
                "name": name, "ticker": ticker, "unit": unit,
                "spot": round(spot, 2),
                "ch1d": round((spot - p1d) / p1d * 100, 1) if p1d else None,
                "ch7d": round((spot - p7d) / p7d * 100, 1) if p7d else None,
            })
        except Exception:
            pass
    return results


def _fetch_metal_events(n: int = 6) -> list[dict]:
    """Top metal supply shock events via Google News RSS. Fail-open returns []."""
    try:
        import feedparser
    except ImportError:
        return []
    _feeds = [
        "https://news.google.com/rss/search?q=smelter+closure+production+halt+mine+shutdown&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=copper+aluminum+nickel+supply+disruption+shortage&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=rare+earth+lithium+cobalt+supply+deficit&hl=en-US&gl=US&ceid=US:en",
    ]
    seen, events = set(), []
    for url in _feeds:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:4]:
                title = e.get("title", "").strip()
                if not title or title in seen:
                    continue
                seen.add(title)
                parts = title.rsplit(" - ", 1)
                clean = parts[0].strip()
                src   = parts[1].strip() if len(parts) == 2 else ""
                pub   = ""
                try:
                    from datetime import datetime as _dt
                    pub = _dt(*e.published_parsed[:6]).strftime("%b %d")
                except Exception:
                    pass
                events.append({"title": clean, "source": src, "pub": pub,
                                "link": e.get("link", "#")})
        except Exception:
            pass
    return events[:n]


def _build_strategy_performance_html() -> str:
    """Historical WR + avg return by strategy from scan_history.csv."""
    import csv as _csv
    from collections import defaultdict as _dd
    h = HERE / "scan_history.csv"
    if not h.exists():
        return ""
    rows = list(_csv.DictReader(open(h)))
    strat: dict = _dd(list)
    for r in rows:
        if r.get("ret_d5"):
            strat[r["strategy"]].append(float(r["ret_d5"]))

    MIN_N = 5
    data = []
    for s, rets in strat.items():
        if len(rets) < MIN_N:
            continue
        wins = sum(1 for r in rets if r > 0)
        wr   = wins / len(rets) * 100
        avg  = sum(rets) / len(rets)
        data.append((s, len(rets), wr, avg))
    data.sort(key=lambda x: -x[2])  # sort by WR desc

    if not data:
        return ""

    STRAT_ALIAS = {
        "pocket_pivot":"Pocket Pivot","ema_ribbon":"EMA Ribbon","cup_handle":"Cup Handle",
        "connors_rsi2":"Connors RSI2","connors_r3":"Connors R3","connors_tps":"Connors TPS",
        "signal_velocity":"Signal Velocity","breakout":"Breakout","vcp":"VCP",
        "darvas_box":"Darvas Box","rs_line":"RS Line","nr7":"NR7",
        "williams_pct_r":"Williams %R","bollinger_pctb":"BB %B",
        "turtle_soup":"Turtle Soup","raschke_8020":"Raschke 80/20",
        "wyckoff_spring":"Wyckoff Spring","weinstein_stage2":"Weinstein S2",
        "momentum_burst":"Momentum Burst","ma50_reclaim":"MA50 Reclaim",
        "holy_grail":"Holy Grail","analyst_upgrade":"Analyst Upgrade",
        "chokepoint_inflection":"Chokepoint","defensive_rotation":"Defensive Rot.",
        "momentum":"Momentum","elder_impulse":"Elder Impulse","stage4_short":"Stage4 Short",
        "turnover_momentum":"Turnover Mom.","connors_3down":"Connors 3↓",
        "high_tight_flag":"High Tight Flag","power_earnings_gap":"Power EG",
    }

    def _wr_col(wr):
        if wr >= 65: return "#15803d"
        if wr >= 55: return "#b45309"
        return "#b91c1c"

    def _avg_col(avg):
        if avg > 1.0: return "#15803d"
        if avg > 0:   return "#4d7c0f"
        return "#b91c1c"

    rows_html = ""
    for i, (s, n, wr, avg) in enumerate(data):
        bg = "#0d1f12" if i % 2 == 0 else "#111827"
        name = _strat_label(s)
        wr_c  = _wr_col(wr)
        avg_c = _avg_col(avg)
        bar_w = int(wr * 0.6)  # scale to max ~60px for 100%
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:6px 10px;font-size:12px;color:#334155;white-space:nowrap;">{name}</td>'
            f'<td style="padding:6px 8px;font-size:11px;color:#64748b;text-align:center;">{n}</td>'
            f'<td style="padding:6px 10px;">'
            f'  <table cellpadding="0" cellspacing="0"><tr>'
            f'  <td style="width:{bar_w}px;background:{wr_c};height:8px;border-radius:3px;opacity:0.7;"></td>'
            f'  <td style="padding-left:6px;font-size:12px;font-weight:700;color:{wr_c};">{wr:.0f}%</td>'
            f'  </tr></table>'
            f'</td>'
            f'<td style="padding:6px 10px;font-size:12px;font-weight:700;color:{avg_c};text-align:right;">{avg:+.2f}%</td>'
            f'</tr>'
        )

    return (
        f'{_section_head("📊","Strategy Performance","historical win rate · 5-day return","#2563eb")}'
        f'<table width="100%" cellpadding="0" cellspacing="0" style="border-radius:4px;overflow:hidden;margin-bottom:16px;">'
        f'<thead><tr style="background:#e2e8f0;">'
        f'<th style="padding:7px 10px;font-size:11px;color:#64748b;text-align:left;font-weight:600;">STRATEGY</th>'
        f'<th style="padding:7px 8px;font-size:11px;color:#64748b;text-align:center;font-weight:600;">N</th>'
        f'<th style="padding:7px 10px;font-size:11px;color:#64748b;text-align:left;font-weight:600;">WIN RATE</th>'
        f'<th style="padding:7px 10px;font-size:11px;color:#64748b;text-align:right;font-weight:600;">AVG RET</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )


def _build_scanner_results_html(india_mode: bool = False) -> str:
    """Scanner results: leads with conviction cards, then full detail by strategy."""
    # Reset BEFORE any early return. If this sat further down (after the JSON is
    # parsed), a missing last_scan_india.json would leave the previous render's
    # pick in place and the India hero card would show a US ticker.
    global _TOP_PICK
    _TOP_PICK = {}

    scan_json = HERE / ("last_scan_india.json" if india_mode else "last_scan.json")
    if not scan_json.exists():
        return ""
    try:
        import json
        data           = json.loads(scan_json.read_text())
        scan_date      = data.get("scan_date", "unknown")
        strategies     = data.get("strategies", [])
        rbs            = data.get("results_by_strategy", {})
        sector_excess  = data.get("sector_excess", {})
        spy_ret        = data.get("spy_ret", 0.0)
        elder_count    = data.get("elder_impulse_count", 0)
        market_regime  = data.get("market_regime", "NEUTRAL")
    except Exception:
        return ""

    # ── Populate company names from persistent cache (reliable in CI) ────────
    from company_cache import get_names as _get_names
    _all_tickers  = list(dict.fromkeys(r["ticker"] for res in rbs.values() for r in res))
    _name_map     = _get_names(_all_tickers)
    for res in rbs.values():
        for r in res:
            if not r.get("company") or r.get("company") == "MISSING":
                r["company"] = _name_map.get(r["ticker"], r["ticker"])

    # Strategy descriptions pulled from scan.py definitions
    _STRAT_DESC = {
        "momentum":       "Finds stocks that just entered momentum — MACD, RSI(14), EMA9/21 crossovers within last 3 bars. ADX≥22, Minervini≥6. Hold 5d. (O'Neil / IBD)",
        "breakout":       "Catches stocks BEFORE they move — coiling for breakout (COIL) or confirmed today (BREAK). ADX 16-35, RSI 50-62, score≤5. Hold 5d.",
        "pocket_pivot":   "Pocket Pivot — volume surge on up-day exceeding any down-day volume in prior 10 days. ADX 16-35, Minervini≥5. Hold 7d. (Morales & Kacher)",
        "connors_rsi2":   "RSI(2) drops below 10 in an uptrending stock — mean reversion snap-back. ADX 16-35, Minervini≥5. Hold 5d. (Larry Connors)",
        "ema_ribbon":     "EMA 8/13/21/34/55 all expanding upward, price pulls back to ribbon and bounces. ADX 20-35, Minervini≥5. Hold 7d.",
        "nr7":            "Narrowest range of last 7 days — maximum volatility compression before expansion. ADX 16-35, RSI<80, Minervini≥5. Hold 3d. (Toby Crabel)",
        "bb_squeeze":     "Bollinger Bands inside Keltner Channels = squeeze. Entry on expansion breakout. Hold 7d. (John Carter / TTM Squeeze)",
        "cup_handle":     "Rounded cup base (6-65 weeks), handle ≤12% depth in upper half, volume dry-up. ADX 16-35, RSI 50-65, Minervini≥5. Hold 10d. (O'Neil / IBD)",
        "power_earnings_gap": "Gap ≥8% on earnings with 2× volume, gap not filled, stock not extended >20%. Hold 10d. (Gil Morales — Power Earnings Gaps)",
        "analyst_upgrade":    "≥3 analyst upgrades in 5 days including ≥1 tier-1 firm. Hold 7d.",
        "stage4_short":       "Weinstein Stage 4 — confirmed distribution, price below falling SMA30, failed rally. SHORT. Hold 7d.",
        "defensive_rotation": "Sector ETF outperforms SPY >3% with acceleration → stock leaders in that sector. Hold 10d. (Meb Faber)",
        "darvas_box":         "New 52w high → tight box consolidation (≤15% width) → volume breakout above box top. ADX 16-35, Minervini≥5. Hold 5d. (Nicolas Darvas, 1960)",
        "rs_line":            "RS line (stock/SPY) makes new 52w high — leading indicator of institutional accumulation. ADX 16-35, Minervini≥5. Hold 7d. (O'Neil / IBD)",
        "vcp":                "≥3 volatility contractions, each tighter than last, on drying volume. Final contraction ≤10%. ADX 16-35, Minervini≥6. Hold 10d. (Minervini — SEPA)",
        "elder_impulse":      "EMA(13) slope AND MACD histogram both rising = green bar. Signal: 2 consecutive green bars. ADX 16-35, RSI 45-75, Minervini≥5. Hold 5d. (Alexander Elder)",
        "holy_grail":         "ADX peaked >30 then stock pulls back to EMA(20) for ≥2 bars on drying volume, then bounces. ADX floor 16, RSI 40-65, Minervini≥4. Trend-pullback, works in slowing markets. Hold 5d. (Linda Raschke — Street Smarts)",
        "connors_3down":      "3 consecutive lower closes in stock above 200d+50d SMA. RSI(2)<20 (short-term oversold). ADX 16-40, Minervini≥4. Mean-reversion snap-back in any market. Hold 3d. (Larry Connors — Short-Term Trading Strategies)",
        "williams_pct_r":     "Williams %R crosses above -80 from oversold. Stock above 50d+200d SMA. ADX 16-40, RSI 35-65, Minervini≥4. Sideways + mild uptrend specialist. Hold 3d. (Larry Williams — Long-Term Secrets to Short-Term Trading)",
        "bollinger_pctb":     "Bollinger %B<0.20 (near lower band) AND MFI<35 (money outflow) AND %B rising (bounce starting). Above 200d SMA. ADX floor 12 — sideways market specialist. Hold 5d. (John Bollinger — Bollinger on Bollinger Bands)",
        "connors_r3":         "RSI(2) drops 3 consecutive days (first from below 60), final RSI(2)<10. Price above 200d SMA. 90% WR on SPY per Connors Research backtests. Best in sideways/choppy markets. Hold 3d. (Connors — High Probability ETF Trading 2009)",
        "connors_tps":        "Time/Price Scale-In: 3-7 consecutive lower closes, RSI(2) declining each day, RSI(2)<25 on entry, volume orderly (declining). Designed for choppy/sideways markets. Hold 4d. (Connors — High Probability ETF Trading 2009)",
        "turtle_soup":        "New 20-day low (traps shorts) then reverses and closes above prior 20d low the same day. False breakdown reversal. Volume confirms. Works in sideways markets. Hold 3d. (Raschke & Connors — Street Smarts 1996)",
        "raschke_8020":       "Bullish 80-20: Opens in bottom 20% of yesterday's range (weak open), closes above yesterday's midpoint (failed breakdown). Next 1-2 days trend up. Pure price-action, any market condition. Hold 2d. (Linda Raschke — Street Smarts 1996)",
        "three_weeks_tight":  "3 consecutive weekly closes within 1.5% of each other, volume declining week-over-week. Stock digesting a prior move under institutional holding. Entry near the weekly high. Hold 7d. (O'Neil / IBD — classic continuation setup)",
        "episodic_pivot":     "Catalyst gap ≥8% on ≥2.5× volume, gap holds for 3+ days (no fill), stock not extended >25% from gap. Enter on the first pullback. Hold 10d. Historical WR ~70% when gap holds. (Gil Morales & Kacher — 'Trade Like an O'Neil Disciple' 2010)",
        "combo_pp_ribbon":    "PREMIUM SETUP: Pocket Pivot AND EMA Ribbon fire simultaneously. PP = institutional accumulation signal. Ribbon = 8/13/21/34/55 EMAs stacked + expanding + pullback bounce. When both align: institutions buying INTO a strengthening trend. Minervini ≥6. Hold 7d. (Morales & Kacher's preferred entry combination)",
    }

    hist_stats = _load_strategy_stats()

    active = [(s, rbs[s]) for s in strategies if rbs.get(s)]
    if not active:
        return (f'<br>{_section_head("📡","Scanner Results","no signals today","#888888")}'
                f'<p style="color:{_C_DIM};font-size:12px;font-style:italic;padding:8px 0;">'
                f'No strategies fired on {scan_date}.</p>')

    # Sort active strategies by historical WR (best first), then by signal count
    def _sort_key(item):
        s, res = item
        st = hist_stats.get(s, {})
        return (-(st.get("wr", 0) if st.get("n", 0) >= 5 else 0), -len(res))
    active = sorted(active, key=_sort_key)

    total_hits = sum(len(r) for _, r in active)

    # ── Conviction classification ─────────────────────────────────────────────
    all_results_flat = [(s, r) for s, res in rbs.items() for r in res]
    from collections import Counter
    ticker_counts = Counter(r["ticker"] for _, r in all_results_flat)
    multi_tickers = {t for t, n in ticker_counts.items() if n >= 2}

    # Deduplicate: best conviction tier per ticker
    tier_rank = {"HIGH": 0, "MED": 1, "LOW": 2}
    best_by_ticker: dict = {}
    for s, r in all_results_flat:
        t = r["ticker"]
        tier = _conviction_tier_email(r, multi_tickers)
        rank = tier_rank[tier]
        if t not in best_by_ticker or rank < best_by_ticker[t][0]:
            best_by_ticker[t] = (rank, tier, s, r)

    _TREND_STRATS     = {"ema_ribbon", "pocket_pivot", "cup_handle", "breakout",
                         "signal_velocity", "weinstein_stage2", "vcp", "power_earnings_gap"}
    _REVERSION_STRATS = {"connors_rsi2", "nr7", "wyckoff_spring", "raschke_8020",
                         "connors_3down", "bollinger_pctb"}
    _ETF_TO_SECTOR = {
        "XLK":"Technology","XLI":"Industrials","XLV":"Healthcare","XLF":"Financial Services",
        "XLY":"Consumer Cyclical","XLP":"Consumer Defensive","XLB":"Basic Materials",
        "XLE":"Energy","XLC":"Communication Services","XLU":"Utilities","XLRE":"Real Estate",
    }
    # Explicit abbreviations. Slicing the full name to 6 chars collided:
    # "Consumer Cyclical" and "Consumer Defensive" both became "Consum", so the
    # same label appeared in HOT and COLD in the same digest.
    _SECTOR_ABBR = {
        "Technology":"Tech", "Industrials":"Industrial", "Healthcare":"Health",
        "Financial Services":"Financials", "Consumer Cyclical":"Cons Cyc",
        "Consumer Defensive":"Cons Def", "Basic Materials":"Materials",
        "Energy":"Energy", "Communication Services":"Comms",
        "Utilities":"Utilities", "Real Estate":"Real Est",
    }

    def _sector_abbr(name: str) -> str:
        return _SECTOR_ABBR.get(name, (name or "")[:10])

    def _email_rank_score(r: dict, strats_fired: list, persistence: dict) -> float:
        """Mirror of scan.py _rank_score — keep in sync."""
        pts = 0.0
        if any(s in _PROVEN_EDGE_SET for s in strats_fired):
            pts += 3
        # ADX removed: -3.1% WR delta, n=588 — noise not signal
        rsi = r.get("rsi") or 0
        if 50 <= rsi <= 65:   pts += 2
        n = len(strats_fired)
        has_quality = any(hist_stats.get(s, {}).get("wr", 0) >= 50 for s in strats_fired)
        if has_quality:
            if n >= 3:   pts += 3
            elif n == 2: pts += 2
        if (r.get("score") or 99) <= 3: pts += 1
        if 7 <= (r.get("score") or 0) <= 10: pts -= 1  # high score = overextended (33% WR)
        vol = r.get("vol_ratio") or 0
        if vol >= 2.0:       pts += 2   # >2x = 71.1% WR (raw n=135)
        elif 1.5 <= vol < 2: pts += 1   # 1.5-2x = 64.2% WR
        days_seen = persistence.get(r.get("ticker", ""), 0)
        if days_seen >= 3:   pts += 2
        elif days_seen >= 2: pts += 1
        if (r.get("rs_vs_spy") or 0) > 0: pts += 1
        # Regime fit
        is_bull    = elder_count >= 15
        is_neutral = 5 <= elder_count < 15
        has_trend  = any(s in _TREND_STRATS     for s in strats_fired)
        has_revert = any(s in _REVERSION_STRATS for s in strats_fired)
        if (is_bull and has_trend) or (is_neutral and has_revert):
            pts += 1
        # Bottom-3 sector penalty (mirrors scan.py _rank_score)
        if sector_excess:
            ticker_sec = r.get("sector", "")
            sec_etf = next((etf for etf, sn in _ETF_TO_SECTOR.items() if sn == ticker_sec), "")
            if sec_etf and sec_etf in sector_excess:
                ranked_vals = sorted(sector_excess.values())
                if sector_excess[sec_etf] <= ranked_vals[2]:
                    pts -= 1
        return pts

    _persistence = _load_persistence_counts()
    high_picks_raw = [(tier, s, r) for t, (rank, tier, s, r) in best_by_ticker.items() if tier == "HIGH"]
    # Compute rank score for each HIGH pick
    high_picks = sorted(
        high_picks_raw,
        key=lambda x: -_email_rank_score(
            x[2],
            [s for s, res in rbs.items() if any(z["ticker"] == x[2]["ticker"] for z in res)],
            _persistence,
        )
    )
    # Stash the #1 ranked pick so build_email() can lead the digest with it.
    # Taken from the same ranked list the cards use, so the hero can never
    # disagree with "#1 BEST" further down. (_TOP_PICK was already cleared at
    # the top of this function, before the early returns.)
    if high_picks:
        _t_tier, _t_strat, _t_r = high_picks[0]
        _t_fired = sorted(
            [s for s, res in rbs.items() if any(z["ticker"] == _t_r["ticker"] for z in res)],
            key=lambda s: -(hist_stats.get(s, {}).get("wr", 0))
        )
        _t_h = hist_stats.get(_t_strat, {})
        _TOP_PICK = {
            "ticker":  _t_r.get("ticker", ""),
            "company": _company_label(_t_r, 34),
            "price":   _t_r.get("price"),
            "stop":    (_t_r["price"] * 0.97) if _t_r.get("price") else None,
            "strat":   _t_strat,
            "fired":   _t_fired,
            "wr":      _t_h.get("wr"),
            "avg":     _t_h.get("avg"),
            "rsi":     _t_r.get("rsi"),
            "adx":     _t_r.get("adx"),
            "vol":     _t_r.get("vol_ratio"),
            "score":   _t_r.get("score"),
            "hold":    HOLD_DAYS_MAP.get(_t_strat, 5),
            "persist": _persistence.get(_t_r.get("ticker", ""), 0),
            "proven":  any(s in _PROVEN_EDGE_SET for s in _t_fired),
            "regime":  _regime_for(_t_strat),
            "n_high":  len(high_picks),
        }

    med_picks = sorted(
        [(tier, s, r) for t, (rank, tier, s, r) in best_by_ticker.items() if tier == "MED"],
        key=lambda x: (x[2].get("score", 99))
    )

    # Regime colour
    regime_col  = {"BULL": "#16a34a", "NEUTRAL": "#d97706", "BEAR": "#dc2626"}.get(market_regime, "#888")
    regime_icon = {"BULL": "🟢", "NEUTRAL": "🟡", "BEAR": "🔴"}.get(market_regime, "⚪")
    spy_s = f"SPY 10d {'+' if spy_ret >= 0 else ''}{spy_ret:.1f}%"

    from datetime import date as _date
    _scan_weekday = _date.fromisoformat(scan_date).weekday() if scan_date else -1
    _is_friday = (_scan_weekday == 4)

    html = f'<br>{_section_head("📡","Scanner Results",f"scan {scan_date} · {len(active)} strategies fired · {total_hits} signals","#1a5a8a")}'

    # ── India banner ──────────────────────────────────────────────────────────
    if india_mode:
        html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 8px;">'
            f'<tr><td style="background:#f0fdf4;border-left:5px solid #ff9933;padding:7px 14px;border-radius:4px;">'
            f'<span style="font-size:12px;font-weight:700;color:#ff9933;">🇮🇳 INDIA SCAN — Nifty 500 universe · benchmark ^NSEI</span>'
            f'</td></tr></table>'
        )

    # ── Market Regime Bar ─────────────────────────────────────────────────────
    html += (
        f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 8px;">'
        f'<tr><td style="background:{_S_SLATE};padding:7px 14px;border-radius:4px;border-left:4px solid {regime_col};">'
        f'<span style="font-size:12px;font-weight:700;color:{regime_col};">'
        f'{regime_icon} MARKET REGIME: {market_regime}</span>'
        f'<span style="font-size:11px;color:#64748b;margin-left:12px;">'
        f'Market trend pulse: <b style="color:#334155;">{elder_count}/20 stocks in uptrend</b>'
        f'{"  · 🔥 Strong uptrend — good time to enter momentum trades" if elder_count >= 15 else ("  · ⚠ Mixed market — only enter the highest-conviction setups" if elder_count >= 5 else "  · ❄️ Weak/falling market — avoid new longs, wait for recovery")}'
        f'</span>'
        f'</td></tr></table>'
    )

    # ── Friday Warning ────────────────────────────────────────────────────────
    if _is_friday:
        html += (
            '<table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 8px;">'
            '<tr><td style="background:#ffedd5;border-left:5px solid #f97316;padding:8px 14px;border-radius:4px;">'
            '<span style="font-size:12px;font-weight:800;color:#c2410c;">⚠ FRIDAY SCAN — Historical WR=45%, avg -0.31%</span>'
            '<span style="font-size:11px;color:#92400e;margin-left:10px;">Hold entry until Monday. Consider existing positions only.</span>'
            '</td></tr></table>'
        )

    # ── Sector Strength Panel ─────────────────────────────────────────────────
    if sector_excess:
        ranked_sectors = sorted(sector_excess.items(), key=lambda x: -x[1])
        top3  = ranked_sectors[:3]
        bot3  = ranked_sectors[-3:]
        bot3_etfs = {e for e, _ in bot3}

        def _sec_chip(etf, ex):
            sector_name = _ETF_TO_SECTOR.get(etf, etf)
            bg   = "#dcfce7" if ex >= 1.0 else ("#fee2e2" if ex <= -1.0 else "#f1f5f9")
            col  = "#15803d" if ex >= 1.0 else ("#b91c1c" if ex <= -1.0 else "#64748b")
            sign = "+" if ex >= 0 else ""
            return (f'<span style="background:{bg};color:{col};border-radius:3px;'
                    f'padding:2px 7px;font-size:10px;font-weight:700;margin-right:4px;">'
                    f'{_sector_abbr(sector_name)} {sign}{ex:.1f}%</span>')

        top_chips = "".join(_sec_chip(e, x) for e, x in top3)
        bot_chips = "".join(_sec_chip(e, x) for e, x in bot3)

        # Which HIGH picks align with top sectors?
        proven_in_top = []
        all_high = [(s, r) for s, res in rbs.items() for r in res
                    if _conviction_tier_email(r, {t for t, n in __import__('collections').Counter(
                        rx["ticker"] for _, rx in [(s2, r2) for s2, res2 in rbs.items() for r2 in res2]
                    ).items() if n >= 2}) == "HIGH"]
        # simplified: just get ticker→sector from results
        ticker_sector = {}
        for s, res in rbs.items():
            for r in res:
                if r.get("sector"):
                    ticker_sector[r["ticker"]] = r.get("sector", "")

        top_sector_names = {_ETF_TO_SECTOR.get(e, "") for e, _ in top3}
        aligned = [r["ticker"] for _, r in all_high
                   if ticker_sector.get(r["ticker"],"") in top_sector_names]
        aligned = list(dict.fromkeys(aligned))  # dedupe

        html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 10px;">'
            f'<tr><td style="background:#f1f5f9;padding:8px 14px;border-radius:4px;border-left:4px solid #0ea5e9;">'
            f'<div style="font-size:11px;font-weight:700;color:#7dd3fc;margin-bottom:4px;">📊 SECTOR STRENGTH vs SPY (10d) &nbsp;·&nbsp; {spy_s}</div>'
            f'<div style="margin-bottom:3px;"><span style="font-size:10px;color:#64748b;margin-right:6px;">HOT 🔥</span>{top_chips}</div>'
            f'<div style="margin-bottom:4px;"><span style="font-size:10px;color:#64748b;margin-right:6px;">COLD ❄️</span>{bot_chips}</div>'
            + (f'<div style="font-size:10px;color:#15803d;">✓ HIGH conviction picks in HOT sectors: <b>{", ".join(aligned)}</b></div>'
               if aligned else
               f'<div style="font-size:10px;color:#f59e0b;">⚠ No HIGH conviction picks in top sectors today — consider reducing size</div>')
            + f'</td></tr></table>'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A: ACT ON THESE — HIGH conviction cards
    # ══════════════════════════════════════════════════════════════════════════
    _RANK_BADGES = {
        0: ('<span style="background:#16a34a;color:#fff;font-size:11px;font-weight:900;'
            'border-radius:4px;padding:2px 8px;margin-right:8px;">#1 BEST</span>'),
        1: ('<span style="background:#2563eb;color:#fff;font-size:11px;font-weight:900;'
            'border-radius:4px;padding:2px 8px;margin-right:8px;">#2</span>'),
        2: ('<span style="background:#d97706;color:#fff;font-size:11px;font-weight:900;'
            'border-radius:4px;padding:2px 8px;margin-right:8px;">#3</span>'),
    }

    if high_picks:
        html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:12px 0 4px;">'
            f'<tr><td style="background:#dcfce7;border-left:5px solid #16a34a;padding:8px 12px;border-radius:4px 4px 0 0;">'
            f'<span style="font-size:13px;font-weight:800;color:#15803d;letter-spacing:.06em;">'
            f'🎯 ACT ON THESE &nbsp;·&nbsp; {len(high_picks)} stock(s) &nbsp;·&nbsp; ★★★ HIGH CONVICTION</span>'
            f'<br><span style="font-size:10px;color:#166534;">Ranked by: PROVEN edge · multi-strategy (74% WR) · vol 1.5-2x · RSI 50-65 · persistence</span>'
            f'</td></tr></table>'
        )
        for card_idx, (tier, strat, r) in enumerate(high_picks):
            strats_fired = sorted(
                [s for s, res in rbs.items() if any(x["ticker"] == r["ticker"] for x in res)],
                key=lambda s: -(hist_stats.get(s, {}).get("wr", 0))
            )
            rank_badge = _RANK_BADGES.get(card_idx, "")
            proven = any(s in _PROVEN_EDGE_SET for s in strats_fired)
            h = hist_stats.get(strat, {})
            wr_val = h.get("wr")
            avg_val = h.get("avg")
            wr_col = "#16a34a" if (wr_val or 0) >= 60 else "#d97706"
            proven_badge = (
                '<span style="background:#16a34a;color:#fff;font-size:9px;font-weight:700;'
                'border-radius:3px;padding:1px 5px;margin-left:6px;">✦ PROVEN EDGE</span>'
            ) if proven else ""
            strat_chips = " ".join(
                f'<span style="background:{"#dcfce7" if s in _PROVEN_EDGE_SET else "#f1f5f9"};'
                f'color:{"#15803d" if s in _PROVEN_EDGE_SET else "#a5b4fc"};'
                f'border-radius:3px;padding:2px 6px;font-size:10px;font-weight:600;">'
                f'{_strat_short(s)}'
                f'{" " + str(int(hist_stats[s]["wr"])) + "%" if hist_stats.get(s, {}).get("n", 0) >= 5 else ""}'
                f'</span>'
                for s in strats_fired[:4]
            ) + _regime_badge(strat)
            price_s  = _fmt_price(r["ticker"], r.get("price"))
            sl_approx = r["price"] * 0.97 if r.get("price") else None
            sl_s     = _fmt_price(r["ticker"], sl_approx)
            hold_d   = HOLD_DAYS_MAP.get(strat, 5)

            # Sector tag for this card
            ticker_sec = r.get("sector", "")
            sec_etf = next((etf for etf, sname in _ETF_TO_SECTOR.items() if sname == ticker_sec), "")
            sec_ex  = sector_excess.get(sec_etf) if sector_excess else None
            if sec_ex is not None:
                ranked_vals = sorted(sector_excess.values(), reverse=True)
                if sec_ex >= ranked_vals[2]:
                    sec_tag = f'<span style="background:#dcfce7;color:#15803d;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-left:4px;">🔥 {_sector_abbr(ticker_sec)} +{sec_ex:.1f}%</span>'
                elif sec_ex <= ranked_vals[-3]:
                    sec_tag = f'<span style="background:#fee2e2;color:#b91c1c;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-left:4px;">❄️ {_sector_abbr(ticker_sec)} {sec_ex:.1f}%</span>'
                else:
                    sec_tag = ""
            else:
                sec_tag = ""

            # Signal badges: persistence, vol, RS
            _days_seen = _persistence.get(r.get("ticker",""), 0)
            _vol = r.get("vol_ratio") or 0
            _rs  = r.get("rs_vs_spy") or 0
            _sig_badges = ""
            if _days_seen >= 3:
                _sig_badges += '<span style="background:#dbeafe;color:#1d4ed8;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-right:3px;">🔁 PERSIST {_days_seen}d</span>'.replace("{_days_seen}", str(_days_seen))
            elif _days_seen >= 2:
                _sig_badges += '<span style="background:#dbeafe;color:#1d4ed8;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-right:3px;">🔁 2d</span>'
            if _vol >= 2.0:
                _sig_badges += f'<span style="background:#fef3c7;color:#c2410c;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-right:3px;">⚡ VOL {_vol:.1f}x</span>'
            elif _vol >= 1.5:
                _sig_badges += f'<span style="background:#fef3c7;color:#b45309;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-right:3px;">⚡ {_vol:.1f}x</span>'
            if _rs > 0:
                _sig_badges += f'<span style="background:#dcfce7;color:#15803d;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-right:3px;">↑RS +{_rs:.1f}%</span>'

            border_col = "#16a34a" if card_idx == 0 else ("#2563eb" if card_idx == 1 else "#d97706" if card_idx == 2 else "#16a34a")
            html += (
                f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:6px;border-radius:0 0 4px 4px;">'
                f'<tr style="background:#f0fdf4;">'
                f'<td style="padding:10px 14px;border-left:5px solid {border_col};border-bottom:1px solid #bbf7d0;">'
                f'<table width="100%" cellpadding="0" cellspacing="0"><tr>'
                f'<td style="width:140px;vertical-align:top;">'
                + (f'<div style="margin-bottom:4px;">{rank_badge}</div>' if rank_badge else '')
                + f'<div style="font-size:20px;font-weight:800;color:#0f172a;font-family:monospace;">{r["ticker"]}</div>'
                f'<div style="font-size:11px;color:#166534;">{_company_label(r)}{sec_tag}</div>'
                + (f'<div style="margin-top:4px;">{_sig_badges}</div>' if _sig_badges else '')
                + f'</td>'
                f'<td style="vertical-align:top;padding-left:12px;">'
                f'<div style="margin-bottom:5px;">{strat_chips}{proven_badge}</div>'
                f'<div style="font-size:11px;color:#14532d;">'
                f'Price <b>{price_s}</b> &nbsp;·&nbsp; '
                f'Stop Loss <b style="color:#b91c1c;">{sl_s}</b> &nbsp;·&nbsp; '
                f'Hold <b>{hold_d} days</b> &nbsp;·&nbsp; '
                f'Momentum <b>{r.get("rsi", 0):.0f}/100</b> &nbsp;·&nbsp; '
                f'Trend Str. <b>{r.get("adx", 0):.0f}/50</b> &nbsp;·&nbsp; '
                f'Score <b>{r.get("score", 0)}</b>'
                f'</div>'
                f'<div style="font-size:10px;color:#166534;margin-top:3px;font-style:italic;">'
                f'{"💡 Buy near " + price_s + ", set a stop at " + sl_s + " to limit downside, target to sell in " + str(hold_d) + " days."}'
                f'</div>'
                f'</td>'
                f'<td style="width:90px;text-align:right;vertical-align:top;">'
                + (f'<div style="font-size:18px;font-weight:800;color:{wr_col};">{wr_val:.0f}% WR</div>'
                   f'<div style="font-size:10px;color:#166534;">{avg_val:+.2f}% avg</div>'
                   if wr_val else '')
                + f'</td></tr></table></td></tr></table>'
            )
    else:
        html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:12px 0 4px;">'
            f'<tr><td style="background:#fefce8;border-left:5px solid #facc15;padding:8px 12px;border-radius:4px;">'
            f'<span style="font-size:12px;font-weight:700;color:#92400e;">'
            f'⚠ No HIGH conviction signals today — see WATCHLIST below or wait for better setup</span>'
            f'</td></tr></table>'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B: WATCHLIST — MED conviction (compact)
    # ══════════════════════════════════════════════════════════════════════════
    if med_picks:
        html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:14px 0 4px;">'
            f'<tr><td style="background:#fefce8;border-left:5px solid #d97706;padding:6px 12px;border-radius:4px 4px 0 0;">'
            f'<span style="font-size:12px;font-weight:700;color:#b45309;">'
            f'👀 WATCHLIST &nbsp;·&nbsp; {len(med_picks)} stock(s) &nbsp;·&nbsp; ★★ MEDIUM — worth watching, but wait for a stronger signal before buying</span>'
            f'</td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;margin-bottom:8px;">'
            f'<thead><tr style="background:#f1f5f9;">'
            + _th("Ticker","left") + _th("Company","left") + _th("Strategy","left") + _th("Win Rate","right")
            + _th("Score","center") + _th("Momentum","right") + _th("Trend Str.","right") + _th("Price","right")
            + f'</tr></thead><tbody>'
        )
        for i, (tier, strat, r) in enumerate(med_picks[:12]):
            bg = _C_ROW1 if i % 2 else _C_ROW0
            h = hist_stats.get(strat, {})
            proven = strat in _PROVEN_EDGE_SET
            pb = (' <span style="background:#16a34a;color:#fff;font-size:8px;border-radius:2px;padding:0 3px;">✦</span>'
                  if proven else "")
            strats_fired = [s for s, res in rbs.items() if any(x["ticker"] == r["ticker"] for x in res)]
            strat_str = " + ".join(_strat_short(s) for s in strats_fired[:2])
            wr_v = h.get("wr")
            wr_c = "#16a34a" if (wr_v or 0) >= 60 else "#d97706"
            price_s = _fmt_price(r["ticker"], r.get("price"))
            co_s = _company_label(r, 20) or "─"
            html += (
                f'<tr style="background:{bg};">'
                + _td(f'<b>{r["ticker"]}</b>{pb}', "left", bg=bg)
                + _td(f'<span style="font-size:10px;color:{_C_DIM};">{co_s}</span>', "left", bg=bg)
                + _td(strat_str, "left", _C_DIM, bg=bg)
                + _td(f'<span style="color:{wr_c};font-weight:700;">{wr_v:.0f}%</span>' if wr_v else "─", "right", bg=bg)
                + _td(str(r.get("score",0)), "center", bg=bg)
                + _td(f'{r.get("rsi",0):.0f}' if r.get("rsi") else "─", "right", bg=bg)
                + _td(f'{r.get("adx",0):.0f}' if r.get("adx") else "─", "right", bg=bg)
                + _td(price_s, "right", _C_DIM, bg=bg)
                + '</tr>'
            )
        html += '</tbody></table>'

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION C: PERSISTENCE LEADERS — seen ≥5 consecutive trading days
    # ══════════════════════════════════════════════════════════════════════════
    streak_leaders = _load_streak_leaders(min_streak=5)
    if streak_leaders:
        html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:14px 0 4px;">'
            f'<tr><td style="background:#f8fafc;border-left:5px solid #818cf8;padding:6px 12px;border-radius:4px 4px 0 0;">'
            f'<span style="font-size:12px;font-weight:700;color:#4338ca;">'
            f'🔁 PERSISTENCE LEADERS &nbsp;·&nbsp; {len(streak_leaders)} stock(s) &nbsp;·&nbsp; ≥5 consecutive trading days</span>'
            f'</td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;margin-bottom:8px;">'
            f'<thead><tr style="background:#f1f5f9;">'
            + _th("Ticker","left") + _th("Streak","center") + _th("Strategies (latest)","left") + _th("Last Seen","right")
            + f'</tr></thead><tbody>'
        )
        for i, l in enumerate(streak_leaders[:15]):
            bg = _C_ROW1 if i % 2 else _C_ROW0
            strat_str = " · ".join(_strat_short(s) for s in l["strategies"][:3])
            streak_color = "#16a34a" if l["streak"] >= 10 else ("#a5b4fc" if l["streak"] >= 7 else "#d1d5db")
            html += (
                f'<tr style="background:{bg};">'
                f'<td style="padding:4px 8px;color:#334155;font-weight:700;">{l["ticker"]}</td>'
                f'<td style="padding:4px 8px;color:{streak_color};text-align:center;font-weight:700;">{l["streak"]}d</td>'
                f'<td style="padding:4px 8px;color:#64748b;">{strat_str}</td>'
                f'<td style="padding:4px 8px;color:#64748b;text-align:right;">{l["last_date"]}</td>'
                f'</tr>'
            )
        html += '</tbody></table>'

    html += _kpi_table([
        ("HIGH Conviction", str(len(high_picks)), "#16a34a"),
        ("MED Watchlist",   str(len(med_picks)),  "#d97706"),
        ("Persistence",     str(len(streak_leaders)), "#4f46e5"),
        ("Scan Date",       scan_date,             _C_DIM),
    ])

    # ── Strategy Scorecard ────────────────────────────────────────────────────
    if hist_stats:
        scored = sorted(
            [(s, d) for s, d in hist_stats.items() if d["n"] >= 3],
            key=lambda x: (-x[1]["wr"], -x[1]["avg"])
        )
        _rc = _load_regime_characters()
        sc_rows = ""
        for rank, (s, d) in enumerate(scored[:12], 1):
            proven = d["n"] >= 10 and d["wr"] >= 60
            wr_color  = "#16a34a" if d["wr"] >= 60 else ("#d97706" if d["wr"] >= 50 else "#dc2626")
            avg_color = "#16a34a" if d["avg"] > 0 else "#dc2626"
            badge = ('<span style="background:#16a34a;color:#fff;font-size:9px;font-weight:700;'
                     'border-radius:3px;padding:1px 4px;margin-left:6px;">✓</span>') if proven else ""
            s_label  = _strat_label(s)
            avg_r    = d.get("avg_r")
            sl_pct   = d.get("sl_pct")
            excess   = d.get("excess")
            avg_r_col  = "#16a34a" if (avg_r or 0) >= 0.5 else ("#d97706" if (avg_r or 0) >= 0 else "#dc2626")
            sl_col     = "#dc2626" if (sl_pct or 0) >= 20 else ("#d97706" if (sl_pct or 0) >= 12 else "#16a34a")
            excess_col = "#16a34a" if (excess or 0) > 0 else "#dc2626"
            _rchar = _rc.get(s, ("─", "", "#94a3b8", "#111827"))
            _r_icon, _r_lbl, _r_col, _r_bg = _rchar
            _row_border = f'border-left:3px solid {_r_col};'
            _regime_sc_badge = (
                f'<span style="background:{_r_bg};color:{_r_col};font-size:9px;font-weight:700;'
                f'border-radius:3px;padding:1px 4px;margin-left:5px;">{_r_icon} {_r_lbl}</span>'
            ) if _r_lbl else ""
            sc_rows += (
                f'<tr style="background:{_C_ROW1 if rank%2 else _C_ROW0};">'
                f'<td style="padding:5px 8px;font-size:11px;color:{_C_DIM};{_row_border}">{rank}</td>'
                f'<td style="padding:5px 8px;font-size:11px;font-weight:600;color:{_C_BODY};">{s_label}{badge}{_regime_sc_badge}</td>'
                f'<td style="padding:5px 8px;font-size:12px;text-align:right;color:{wr_color};font-weight:700;">{d["wr"]:.0f}%</td>'
                f'<td style="padding:5px 8px;font-size:11px;text-align:right;color:{avg_color};">{d["avg"]:+.2f}%</td>'
                f'<td style="padding:5px 8px;font-size:11px;text-align:right;color:{avg_r_col};">{f"{avg_r:.2f}R" if avg_r is not None else "─"}</td>'
                f'<td style="padding:5px 8px;font-size:11px;text-align:right;color:{sl_col};">{f"{sl_pct:.0f}%" if sl_pct is not None else "─"}</td>'
                f'<td style="padding:5px 8px;font-size:11px;text-align:right;color:{excess_col};">{f"{excess:+.2f}%" if excess is not None else "─"}</td>'
                f'<td style="padding:5px 8px;font-size:11px;text-align:right;color:{_C_DIM};">{d["n"]}</td>'
                '</tr>'
            )
        html += (
            f'<br><table width="100%" cellpadding="0" cellspacing="0" style="margin-top:8px;margin-bottom:0;">'
            f'<tr><td style="background:#f0fdf4;border-left:4px solid #16a34a;padding:6px 10px;border-radius:3px 3px 0 0;">'
            f'<span style="font-size:11px;font-weight:700;color:#16a34a;letter-spacing:.05em;">'
            f'📊 STRATEGY SCORECARD &nbsp;·&nbsp; live from scan_history.csv &nbsp;·&nbsp; ✓ = proven edge (WR≥60%, n≥10)</span></td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;">'
            f'<thead><tr style="background:#f1f5f9;">'
            f'<th style="padding:5px 8px;text-align:left;color:#888;">#</th>'
            f'<th style="padding:5px 8px;text-align:left;color:#888;">Strategy</th>'
            f'<th style="padding:5px 8px;text-align:right;color:#888;" title="5-day win rate">WR%</th>'
            f'<th style="padding:5px 8px;text-align:right;color:#888;" title="Avg 5-day return">Avg Ret</th>'
            f'<th style="padding:5px 8px;text-align:right;color:#888;" title="Avg reward/risk multiple">R:R</th>'
            f'<th style="padding:5px 8px;text-align:right;color:#888;" title="% of signals that hit stop loss">SL Hit</th>'
            f'<th style="padding:5px 8px;text-align:right;color:#888;" title="Avg excess return vs SPY over same period">vs SPY</th>'
            f'<th style="padding:5px 8px;text-align:right;color:#888;" title="Number of signals tracked">N</th>'
            f'</tr></thead><tbody>'
            + sc_rows +
            f'</tbody></table>'
        )

    # ── Regime Map ───────────────────────────────────────────────────────────
    html += _build_regime_map_html()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION C: FULL STRATEGY DETAIL (reference — scroll past if acted on above)
    # ══════════════════════════════════════════════════════════════════════════
    html += (
        f'<br><table width="100%" cellpadding="0" cellspacing="0" style="margin:18px 0 4px;">'
        f'<tr><td style="background:#f1f5f9;border-left:4px solid #4f46e5;padding:6px 12px;border-radius:4px;">'
        f'<span style="font-size:11px;font-weight:600;color:#4338ca;letter-spacing:.04em;">'
        f'📋 FULL SCAN DETAIL — {total_hits} signals across {len(active)} strategies</span>'
        f'</td></tr></table>'
    )

    _rc_detail = _load_regime_characters()
    for strat, results in active:
        is_short = "short" in strat
        fg, bg_badge = _STRAT_COLORS.get(strat, ("#333333", "#eeeeee"))
        strat_label  = strat.upper().replace("_", " ")

        wrs  = [r["wr"]  for r in results if r.get("wr")  is not None]
        avgs = [r["avg"] for r in results if r.get("avg") is not None]
        bt_note = ""
        if wrs:
            bt_note = (f'backtest avg: {sum(wrs)/len(wrs):.0f}% win · '
                       f'{sum(avgs)/len(avgs):+.1f}% return')

        desc_text = _STRAT_DESC.get(strat, "")
        h_stat    = hist_stats.get(strat, {})
        proven    = h_stat.get("n", 0) >= 10 and h_stat.get("wr", 0) >= 60
        live_wr   = h_stat.get("wr")
        live_avg  = h_stat.get("avg")
        live_n    = h_stat.get("n", 0)
        live_note = ""
        if live_wr is not None:
            wr_col  = "#16a34a" if live_wr >= 60 else ("#d97706" if live_wr >= 45 else "#dc2626")
            live_note = (f'<span style="font-size:10px;color:{wr_col};font-weight:700;margin-left:10px;">'
                         f'📈 {live_wr:.0f}% WR · {live_avg:+.2f}% avg (n={live_n})</span>')
        proven_badge = ""
        if proven:
            proven_badge = ('<span style="background:#16a34a;color:#fff;font-size:9px;font-weight:700;'
                            'border-radius:3px;padding:1px 5px;margin-left:8px;">PROVEN EDGE ✓</span>')
        # Regime badge on strategy header
        _drchar = _rc_detail.get(strat, ("─", "", "#94a3b8", "#111827"))
        _d_icon, _d_lbl, _d_col, _d_bg = _drchar
        _detail_regime = (
            f'<span style="background:{_d_bg};color:{_d_col};font-size:9px;font-weight:700;'
            f'border-radius:3px;padding:1px 5px;margin-left:8px;">{_d_icon} {_d_lbl}</span>'
        ) if _d_lbl else ""
        # Use regime colour for border if available, else fall back to strategy colour
        _header_border_col = _d_col if _d_lbl else fg
        html += (f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:16px;margin-bottom:0;">'
                 f'<tr><td style="background:{bg_badge};border-left:4px solid {_header_border_col};'
                 f'padding:6px 10px;border-radius:3px 3px 0 0;">'
                 f'<span style="font-size:11px;font-weight:700;color:{fg};'
                 f'letter-spacing:.05em;text-transform:uppercase;">'
                 f'{strat_label} &nbsp;·&nbsp; {len(results)} signal(s)</span>'
                 + proven_badge
                 + _detail_regime
                 + live_note
                 + (f'<span style="font-size:10px;color:{fg};margin-left:12px;">{bt_note}</span>' if bt_note else "")
                 + (f'<br><span style="font-size:10px;color:{fg};opacity:0.8;font-style:italic;">{desc_text}</span>' if desc_text else "")
                 + f'</td></tr></table>')

        thead = (f'<table width="100%" cellpadding="0" cellspacing="0" '
                 f'style="border-collapse:collapse;font-size:11px;{_FONT}">'
                 '<thead><tr>'
                 + _th("Ticker", "left")
                 + _th("Company", "left")
                 + _th("Mkt",   "center")
                 + _th("Scr",   "center", title="Scanner score")
                 + _th("Price", "right")
                 + _th("RSI",   "right")
                 + _th("ADX",   "right")
                 + _th("Vol×",  "right",  title="Volume ratio vs 20d avg")
                 + _th("Signals/Context", "left")
                 + '</tr></thead><tbody>')

        rows = ""
        sorted_results = sorted(results, key=lambda x: -(x.get("score") or 0))[:15]
        for i, r in enumerate(sorted_results):
            bg      = _C_ROW1 if i % 2 else _C_ROW0
            score_v = r.get("score") or 0
            rsi_v   = r.get("rsi")
            adx_v   = r.get("adx")
            price_v = r.get("price")
            vol_v   = r.get("vol_ratio")
            m_v     = r.get("minervini")
            wr_v    = r.get("wr")
            avg_v   = r.get("avg")
            mkt     = r.get("mkt", "US")
            fresh   = r.get("fresh") or []
            conf    = r.get("conf")  or []

            # Build compact signal string
            sigs = []
            if r.get("phase"):            sigs.append(r["phase"])
            if r.get("pole_return"):      sigs.append(f"pole+{r['pole_return']:.0f}%")
            if r.get("pct_from_piv") is not None: sigs.append(f"piv{r['pct_from_piv']:+.1f}%")
            if r.get("velocity") is not None:     sigs.append(f"vel{r['velocity']:+.1f}")
            if r.get("pct_52w_hi") is not None:   sigs.append(f"52w-{r['pct_52w_hi']:.0f}%")
            if r.get("cup_depth_pct"):            sigs.append(f"cup{r['cup_depth_pct']:.0f}%")
            if r.get("compression"):              sigs.append(f"NR{r['compression']:.0f}%")
            if r.get("squeeze_intensity"):        sigs.append(f"sq{r['squeeze_intensity']:.2f}")
            if r.get("net_score") is not None:    sigs.append(f"net{r['net_score']:+d}")
            for sig in fresh[:3]:
                if sig not in sigs: sigs.append(sig)
            if not sigs:
                sigs = conf[:3]
            sigs_str = " · ".join(str(s) for s in sigs[:5]) or "─"

            score_c = _C_POS if score_v >= 6 else (_C_WARN if score_v >= 4 else _C_DIM)
            rsi_c   = (_C_WARN if rsi_v and rsi_v > 70
                       else (_C_POS if rsi_v and rsi_v < 40 else _C_BODY))
            vol_c   = _C_POS if (vol_v or 0) >= 1.5 else _C_DIM
            tick_c  = _C_NEG if is_short else _C_BODY
            company_s = _company_label(r, 22) or "─"

            rows += "<tr>"
            rows += _td(f'<b style="color:{tick_c}">{r["ticker"]}</b>', "left", bg=bg)
            rows += _td(f'<span style="color:{_C_DIM};font-size:10px;">{company_s}</span>', "left", bg=bg)
            rows += _td(
                f'<span style="background:#f0f4ff;color:#334;border-radius:3px;'
                f'padding:1px 5px;font-size:10px;">{mkt}</span>',
                "center", bg=bg)
            rows += _td(str(score_v), "center", score_c, bold=True, bg=bg)
            rows += _td(_fmt_price(r["ticker"], price_v), "right", _C_DIM, bg=bg)
            rows += _td(f'{rsi_v:.0f}' if rsi_v else "─", "right", rsi_c, bg=bg)
            rows += _td(f'{adx_v:.0f}' if adx_v else "─", "right", _C_DIM, bg=bg)
            rows += _td(f'{vol_v:.1f}×' if vol_v else "─", "right", vol_c, bg=bg)
            rows += _td(sigs_str, "left", _C_DIM, bg=bg,
                        extra="font-size:10px;max-width:200px;overflow:hidden;")
            rows += "</tr>"

        html += thead + rows + "</tbody></table>"

    # ══════════════════════════════════════════════════════════════════════════
    # METALS SNAPSHOT — spot prices + supply events (absorbed from metal_tracker)
    # ══════════════════════════════════════════════════════════════════════════
    metal_prices = _fetch_metal_snapshot()
    if metal_prices:
        def _chg_span(v):
            if v is None: return '<span style="color:#94a3b8;">─</span>'
            col  = "#15803d" if v > 0 else ("#b91c1c" if v < 0 else "#64748b")
            sign = "+" if v > 0 else ""
            return f'<span style="color:{col};font-weight:700;">{sign}{v}%</span>'

        metal_rows = "".join(
            f'<tr style="background:{_C_ROW1 if i % 2 else _C_ROW0};">'
            f'<td style="padding:5px 8px;color:#334155;font-weight:700;font-size:11px;">{m["name"]}</td>'
            f'<td style="padding:5px 8px;color:#64748b;font-size:10px;">{m["ticker"]}</td>'
            f'<td style="padding:5px 8px;color:#334155;text-align:right;font-size:11px;">'
            f'${m["spot"]:,.2f} <span style="color:#94a3b8;font-size:10px;">{m["unit"]}</span></td>'
            f'<td style="padding:5px 8px;text-align:right;">{_chg_span(m["ch1d"])}</td>'
            f'<td style="padding:5px 8px;text-align:right;">{_chg_span(m["ch7d"])}</td>'
            f'</tr>'
            for i, m in enumerate(metal_prices)
        )
        html += (
            f'<br><table width="100%" cellpadding="0" cellspacing="0" style="margin:10px 0 4px;">'
            f'<tr><td style="background:#fefce8;border-left:5px solid #f59e0b;padding:6px 12px;border-radius:4px 4px 0 0;">'
            f'<span style="font-size:12px;font-weight:700;color:#b45309;">⬡ METALS SNAPSHOT &nbsp;·&nbsp; Live Spot Prices</span>'
            f'</td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;margin-bottom:4px;">'
            f'<thead><tr style="background:#f1f5f9;">'
            f'<th style="padding:5px 8px;color:#64748b;text-align:left;">Metal</th>'
            f'<th style="padding:5px 8px;color:#64748b;text-align:left;">Ticker</th>'
            f'<th style="padding:5px 8px;color:#64748b;text-align:right;">Spot</th>'
            f'<th style="padding:5px 8px;color:#64748b;text-align:right;">1-Day Δ</th>'
            f'<th style="padding:5px 8px;color:#64748b;text-align:right;">7-Day Δ</th>'
            f'</tr></thead><tbody>{metal_rows}</tbody></table>'
        )

    metal_events = _fetch_metal_events(n=6)
    if metal_events:
        ev_rows = "".join(
            f'<tr style="background:{_C_ROW1 if i % 2 else _C_ROW0};">'
            f'<td style="padding:5px 8px;color:#64748b;font-size:10px;white-space:nowrap;">{e["pub"]}</td>'
            f'<td style="padding:5px 8px;font-size:11px;">'
            f'<a href="{e["link"]}" style="color:#c9d1d9;text-decoration:none;">{e["title"][:100]}</a>'
            + (f'<br><span style="font-size:10px;color:#94a3b8;">{e["source"]}</span>'
               if e.get("source") else "")
            + f'</td></tr>'
            for i, e in enumerate(metal_events)
        )
        html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;'
            f'font-size:11px;margin-bottom:8px;">'
            f'<thead><tr style="background:#f1f5f9;">'
            f'<th style="padding:5px 8px;color:#64748b;text-align:left;width:55px;">Date</th>'
            f'<th style="padding:5px 8px;color:#64748b;text-align:left;">Supply Shock Events</th>'
            f'</tr></thead><tbody>{ev_rows}</tbody></table>'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # RRG SECTOR ROTATION — Live quadrant analysis + scatter chart (bottom)
    # ══════════════════════════════════════════════════════════════════════════
    if _HAS_RRG:
        try:
            rrg_results = run_sector_rrg(lookback_days=365)
        except Exception:
            rrg_results = []

        if rrg_results:
            _QUAD_BG = {
                "Leading":   "#dcfce7", "Improving": "#dbeafe",
                "Weakening": "#fef3c7", "Lagging":   "#fee2e2",
            }
            _QUAD_COL = {
                "Leading":   "#15803d", "Improving": "#1d4ed8",
                "Weakening": "#b45309", "Lagging":   "#b91c1c",
            }
            rrg_rows = ""
            for i, r in enumerate(rrg_results):
                q         = r["quad"]
                bg        = _C_ROW1 if i % 2 else _C_ROW0
                qbg       = _QUAD_BG.get(q, "#f1f5f9")
                qcol      = _QUAD_COL.get(q, "#64748b")
                emoji     = QUAD_EMOJI.get(q, "⚪")
                ratio_col = "#15803d" if r["rs_ratio"] >= 100 else "#b91c1c"
                mom_col   = "#15803d" if r["rs_mom"]   >= 100 else "#b91c1c"
                rrg_rows += (
                    f'<tr style="background:{bg};">'
                    f'<td style="padding:5px 8px;color:#334155;font-weight:700;font-size:11px;">{r["etf"]}</td>'
                    f'<td style="padding:5px 8px;color:#64748b;font-size:10px;">{r["name"]}</td>'
                    f'<td style="padding:5px 8px;text-align:center;">'
                    f'<span style="background:{qbg};color:{qcol};border-radius:3px;padding:2px 6px;'
                    f'font-size:10px;font-weight:700;">{emoji} {q}</span></td>'
                    f'<td style="padding:5px 8px;color:{ratio_col};text-align:right;font-size:11px;font-weight:600;">{r["rs_ratio"]:.2f}</td>'
                    f'<td style="padding:5px 8px;color:{mom_col};text-align:right;font-size:11px;font-weight:600;">{r["rs_mom"]:.2f}</td>'
                    f'</tr>'
                )
            html += (
                f'<br><table width="100%" cellpadding="0" cellspacing="0" style="margin:10px 0 4px;">'
                f'<tr><td style="background:#f8fafc;border-left:5px solid #818cf8;padding:6px 12px;border-radius:4px 4px 0 0;">'
                f'<span style="font-size:12px;font-weight:700;color:#4338ca;">'
                f'📡 RRG SECTOR ROTATION &nbsp;·&nbsp; Live vs SPY &nbsp;·&nbsp; '
                f'{sum(1 for r in rrg_results if r["quad"]=="Leading")} Leading &nbsp; '
                f'{sum(1 for r in rrg_results if r["quad"]=="Improving")} Improving &nbsp; '
                f'{sum(1 for r in rrg_results if r["quad"]=="Weakening")} Weakening &nbsp; '
                f'{sum(1 for r in rrg_results if r["quad"]=="Lagging")} Lagging'
                f'</span></td></tr></table>'
                f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;margin-bottom:8px;">'
                f'<thead><tr style="background:#f1f5f9;">'
                f'<th style="padding:5px 8px;color:#64748b;text-align:left;font-weight:600;">ETF</th>'
                f'<th style="padding:5px 8px;color:#64748b;text-align:left;font-weight:600;">Sector</th>'
                f'<th style="padding:5px 8px;color:#64748b;text-align:center;font-weight:600;">Status</th>'
                f'<th style="padding:5px 8px;color:#64748b;text-align:right;font-weight:600;" title="vs SPY — above 100 means sector is beating the market">Beating Market?</th>'
                f'<th style="padding:5px 8px;color:#64748b;text-align:right;font-weight:600;" title="above 100 = gap widening, below 100 = gap closing">Gap Widening?</th>'
                f'</tr></thead>'
                f'<tr style="background:#f8fafc;">'
                f'<td colspan="3" style="padding:4px 8px;font-size:10px;color:#94a3b8;font-style:italic;">'
                f'🟢 Leading = best to trade now &nbsp; 🔵 Improving = gaining momentum &nbsp; 🟡 Weakening = slowing &nbsp; 🔴 Lagging = avoid</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#94a3b8;font-style:italic;text-align:right;">&gt;100 = beating SPY</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#94a3b8;font-style:italic;text-align:right;">&gt;100 = accelerating</td>'
                f'</tr>'
                f'<tbody>{rrg_rows}</tbody></table>'
            )
            try:
                rrg_chart_b64 = chart_rrg_scatter(rrg_results)
                if rrg_chart_b64:
                    html += (
                        f'<div style="margin:4px 0 12px;">'
                        f'<img src="data:image/png;base64,{rrg_chart_b64}" '
                        f'style="width:100%;max-width:700px;border-radius:6px;display:block;margin:0 auto;">'
                        f'</div>'
                    )
            except Exception:
                pass

    return html


def build_email(trades: list[dict], india_mode: bool = False) -> str:
    # Practice trades excluded everywhere in the email
    trades        = [t for t in trades if t.get("trade_type") == "real"]
    open_trades   = [t for t in trades if t.get("status") == "OPEN"]
    closed_trades = [t for t in trades if t.get("status") == "CLOSED"]

    ALERT_ORDER = [
        # Exit-rule alerts — shown first (highest priority)
        "PARTIAL_EXIT", "MOVE_STOP_BE", "TRAIL_STOP",
        # Standard alerts
        "STOP_LOSS", "TRAIL_BREACH", "HOLD_EXPIRED", "PROFIT_TARGET", "EARNINGS",
    ]
    ALERT_META  = {
        # Exit rules — amber/orange for partial exit & BE stop; green for trail stop
        "PARTIAL_EXIT":  ("✂",  "Take Partial Profits",    "#f59e0b", "#fffbeb"),
        "MOVE_STOP_BE":  ("🔒", "Move Stop to Break-Even", "#f59e0b", "#fffbeb"),
        "TRAIL_STOP":    ("📈", "Trail Stop to 10d EMA",   "#10b981", "#f0fdf4"),
        # Standard alerts
        "STOP_LOSS":     ("🛑", "Stop Loss Hit",            "#c0392b", "#fff0f0"),
        "TRAIL_BREACH":  ("⚠",  "Trailing EMA Breached",   "#7b3f00", "#fff8ee"),
        "HOLD_EXPIRED":  ("⏰", "Hold Period Expired",      "#b7590a", "#fff8f0"),
        "PROFIT_TARGET": ("🎯", "Profit Target Reached",   "#1a7f4b", "#f0faf4"),
        "EARNINGS":      ("📣", "Earnings Warning",         "#1a5a8a", "#f0f6ff"),
    }
    alerts_by_type: dict = {k: [] for k in ALERT_ORDER}
    results: list[tuple] = []

    print(f"  Fetching prices for {len(open_trades)} open trade(s)...", flush=True)
    for t in open_trades:
        print(f"    {t['ticker']}...", end=" ", flush=True)
        pnl = compute_pnl(t)
        results.append((t, pnl))
        for atype, amsg in pnl["alerts"]:
            if atype in alerts_by_type:
                alerts_by_type[atype].append((t, amsg))
        print("✓")

    # ── Exit-rule alerts (display-only, prepended to action alerts) ──────────
    _prices_native = {t["ticker"]: r.get("curr_native") for t, r in results}
    _trade_by_ticker = {t["ticker"]: t for t in open_trades}
    for _ea in _check_exit_rules(open_trades, _prices_native):
        _atype = _ea["type"]
        if _atype in alerts_by_type:
            alerts_by_type[_atype].append(
                (_trade_by_ticker.get(_ea["ticker"], {}), _ea["message"])
            )

    alert_tickers = {t["ticker"] for items in alerts_by_type.values() for t, _ in items}
    total_alerts  = sum(len(v) for v in alerts_by_type.values())

    # ── Section 1: ACTION REQUIRED ────────────────────────────────────────────
    action_html = ""
    for atype in ALERT_ORDER:
        items = alerts_by_type[atype]
        if not items: continue
        icon, label, fg, bg = ALERT_META[atype]
        # Bucket header
        action_html += (f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:12px;">'
                        f'<tr><td style="background:{bg};border-left:4px solid {fg};'
                        f'padding:8px 12px;border-radius:4px 4px 0 0;">'
                        f'<span style="font-size:12px;font-weight:700;color:{fg};'
                        f'letter-spacing:.05em;text-transform:uppercase;">'
                        f'{icon} {label} &nbsp;({len(items)})</span></td></tr>')
        for j, (t, amsg) in enumerate(items):
            row_bg = "#ffffff" if j % 2 == 0 else "#fafafa"
            tt = t.get("trade_type", "practice")
            action_html += (f'<tr><td style="background:{row_bg};'
                            f'border-left:4px solid {fg};border-bottom:1px solid {_C_BORD};'
                            f'padding:9px 12px;">'
                            f'<table cellpadding="0" cellspacing="0" width="100%"><tr>'
                            f'<td style="width:90px;font-weight:700;font-size:13px;'
                            f'color:{_C_BODY};{_FONT}">{t["ticker"]}</td>'
                            f'<td style="width:160px;font-size:11px;color:{_C_DIM};{_FONT}">'
                            f'{_company_label(t, 28)}</td>'
                            f'<td style="width:50px;">{_trade_badge(tt)}</td>'
                            f'<td style="font-size:12px;color:{_C_BODY};{_FONT}">{amsg}</td>'
                            f'</tr></table></td></tr>')
        action_html += '</table>'

    if not action_html:
        action_html = (f'<p style="color:{_C_DIM};font-style:italic;font-size:13px;'
                       f'padding:10px 0;">✓ No actions required — all positions within parameters.</p>')

    # ── Section 2: Portfolio — real trades only ───────────────────────────────
    snapshot_html = ""
    if results:
        invested  = sum(float(t["investment_eur"] or 0) for t, _ in results)
        pnls      = [r["pnl_eur"] for _, r in results if r["pnl_eur"] is not None]
        total_pnl = sum(pnls) if pnls else None
        wins      = sum(1 for _, r in results if (r["ret_pct"] or 0) > 0)
        kpi_items = [
            ("Invested",   f"€{invested:.0f}", _C_BODY),
            ("Open P&L",   _eur(total_pnl),    _c(total_pnl)),
            ("Positions",  str(len(results)),   _C_BODY),
            ("In Profit",  str(wins),           _C_POS if wins else _C_DIM),
        ]
        snapshot_html += _kpi_table(kpi_items)
        snapshot_html += _portfolio_table(results, alert_tickers)

    if not snapshot_html:
        snapshot_html = f'<p style="color:{_C_DIM};font-style:italic;">No open real positions.</p>'

    # ── Section 3: Weekly P&L (Fridays only) ──────────────────────────────────
    weekly_html = ""
    if IS_FRIDAY and closed_trades:
        week_start  = TODAY - timedelta(days=TODAY.weekday())
        week_closed = [t for t in closed_trades
                       if t.get("actual_sell_date") and
                       datetime.strptime(t["actual_sell_date"], "%Y-%m-%d").date() >= week_start]
        if week_closed:
            thead = ('<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:12px;">'
                     '<thead><tr>'
                     + _th("Ticker","left") + _th("Company","left") + _th("Type","left")
                     + _th("Strategy","left") + _th("Entry") + _th("Sold")
                     + _th("Ret%") + _th("P&L€") + _th("Reason","left")
                     + '</tr></thead><tbody>')
            rows = ""
            for i, t in enumerate(week_closed):
                bg = _C_ROW1 if i % 2 else _C_ROW0
                try:
                    bp  = float(t["buy_price"]); sp = float(t["exit_price"])
                    fx_e = float(t["fx_at_entry"] or 1)
                    fx_x = fetch_fx_now(t.get("currency","EUR"))
                    qty  = float(t["qty"])
                    ret  = round((sp/bp - 1)*100, 2)
                    pnl  = round((sp/fx_x - bp/fx_e)*qty, 2)
                    ret_c = _c(ret); pnl_c = _c(pnl)
                except Exception:
                    ret = None; pnl = None; ret_c = _C_DIM; pnl_c = _C_DIM
                rows += "<tr>"
                rows += _td(f'<b>{t["ticker"]}</b>', "left", bg=bg)
                rows += _td(_company_label(t, 22) or "─", "left", _C_DIM, bg=bg)
                rows += _td(_trade_badge(t.get("trade_type","practice")), "left", bg=bg)
                rows += _td(_strat_badge(t.get("strategy","")), "left", bg=bg)
                rows += _td(t["entry_date"], "right", _C_DIM, bg=bg)
                rows += _td(t.get("actual_sell_date",""), "right", _C_DIM, bg=bg)
                rows += _td(_pct(ret), "right", ret_c, bold=True, bg=bg)
                rows += _td(_eur(pnl), "right", pnl_c, bold=True, bg=bg)
                rows += _td(t.get("exit_reason","─"), "left", _C_DIM, bg=bg)
                rows += "</tr>"
            weekly_html = (f'<br>{_section_head("📅","Week Closed Trades",f"{week_start} → {TODAY}","#2c3e50")}'
                           + thead + rows + '</tbody></table>')

    # ── Section 4: Scanner results ────────────────────────────────────────────
    scanner_html = _build_scanner_results_html(india_mode=india_mode)

    # ── Section 6: Cross-strategy matrix ─────────────────────────────────────
    matrix_html = _build_matrix_html()

    # ── Top-level KPI bar ─────────────────────────────────────────────────────
    all_invested = sum(float(t["investment_eur"] or 0) for t in open_trades)
    all_pnls     = [r["pnl_eur"] for _, r in results if r["pnl_eur"] is not None]
    all_pnl      = sum(all_pnls) if all_pnls else None
    alert_kpi_c  = _C_NEG if total_alerts else _C_DIM
    top_kpi = _kpi_table([
        ("Total Invested",  f"€{all_invested:.0f}", _C_BODY),
        ("Open P&L",        _eur(all_pnl),           _c(all_pnl)),
        ("Open Positions",  str(len(open_trades)),    _C_BODY),
        ("Closed Total",    str(len(closed_trades)),  _C_DIM),
        ("Action Required", str(total_alerts),        alert_kpi_c),
    ])

    # ── Assemble ──────────────────────────────────────────────────────────────
    # Must come after scanner_html was built — that call populates _TOP_PICK.
    hero_html = _build_hero_html()

    alert_banner = ""
    if total_alerts:
        alert_banner = (f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:6px;">'
                        f'<tr><td style="background:#c0392b;padding:10px 16px;border-radius:4px;">'
                        f'<span style="color:#fff;font-weight:700;font-size:13px;">'
                        f'⚠️  {total_alerts} action(s) require your attention today</span></td></tr></table>')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trade Digest {TODAY}</title></head>
<body style="margin:0;padding:16px;{_BG}{_FONT}color:{_C_BODY};">
<div style="{_W}padding:0 4px;">

  <!-- HEADER -->
  <table width="100%" cellpadding="0" cellspacing="0"
         style="border-bottom:2px solid {_C_THEAD};margin-bottom:16px;padding-bottom:12px;">
    <tr>
      <td>
        <div style="font-size:22px;font-weight:700;color:{_C_THEAD};">
          📊 Trade Digest</div>
        <div style="font-size:13px;color:{_C_DIM};margin-top:2px;">
          {TODAY.strftime("%A, %d %b %Y")} &nbsp;·&nbsp; {NOTIFY_TO}
          &nbsp;·&nbsp; SL {int(STOP_LOSS_PCT*100)}% · Target {int(PROFIT_TARGET*100)}%
          · Earnings warn {EARNINGS_WARN}d
        </div>
      </td>
    </tr>
  </table>

  {hero_html}

  {alert_banner}
  {top_kpi}

  {_section_head("🚨","Action Required",f"{total_alerts} alert(s)","#c0392b")}
  {action_html}

  {_section_head("📈","Open Positions (Real)",f"{len(open_trades)} trade(s)","#2c3e50")}
  {snapshot_html}

  {weekly_html}

  {scanner_html}

  {matrix_html}

  <!-- FOOTER -->
  <table width="100%" cellpadding="0" cellspacing="0"
         style="border-top:1px solid {_C_BORD};margin-top:32px;padding-top:10px;">
    <tr><td style="font-size:10px;color:{_C_DIM};">
      Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;·&nbsp; notify.py
    </td></tr>
  </table>

</div></body></html>"""
    return html

# ── Send email ────────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = NOTIFY_TO
    msg.attach(MIMEText(html_body, "html"))

    pw = GMAIL_APP_PASSWORD.replace(" ", "")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, pw)
        server.sendmail(GMAIL_USER, NOTIFY_TO, msg.as_string())

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    do_send    = "--send"  in sys.argv
    india_mode = "--india" in sys.argv

    trades = load_trades()  # empty list is fine — digest still shows scanner signals

    print(f"\n  Building {'🇮🇳 India' if india_mode else '🌍 Intl'} digest for {TODAY}...")
    html = build_email(trades, india_mode=india_mode)

    if do_send:
        open_count = sum(1 for t in trades if t.get('status') == 'OPEN')
        if india_mode:
            subject = f"🇮🇳 India Scan — {TODAY} · {open_count} open"
        else:
            subject = f"🌍 Intl Scan — {TODAY} · {open_count} open"
        print(f"\n  Sending email to {NOTIFY_TO}...", end=" ", flush=True)
        send_email(subject, html)
        print("✅  Sent!")
    else:
        fname = "digest_preview_india.html" if india_mode else "digest_preview.html"
        out = HERE / fname
        out.write_text(html, encoding="utf-8")
        print(f"\n  Dry-run — preview saved to: {out}")
        print("  Run with --send to actually email it.")

if __name__ == "__main__":
    main()
