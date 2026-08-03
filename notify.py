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

# Colour palette (light theme, high contrast)
_C_POS   = "#1a7f4b"   # green
_C_NEG   = "#c0392b"   # red
_C_WARN  = "#b7590a"   # orange
_C_DIM   = "#888888"
_C_BODY  = "#1a1a1a"
_C_HEAD  = "#ffffff"
_C_THEAD = "#2c3e50"   # dark navy header row
_C_ROW0  = "#ffffff"
_C_ROW1  = "#f7f8fa"
_C_BORD  = "#e0e4ea"

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
            f'letter-spacing:.04em;white-space:nowrap;">{strat.upper()}</span>')

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

_CCY_SYM = {"USD": "$", "GBP": "£", "EUR": "€", "CAD": "CA$"}

def _ccy_sym(ccy: str) -> str:
    return _CCY_SYM.get(ccy, ccy + " ")

def _sl_eur_str(t: dict) -> str:
    if not t.get("stop_loss_price"):
        return "─"
    try:
        sl_native = float(t["stop_loss_price"])
        ccy = t.get("currency", "EUR")
        sym = _ccy_sym(ccy)
        return f"{sym}{sl_native:.2f}"
    except Exception:
        return str(t.get("stop_loss_price", "─"))

def _portfolio_table(results_slice: list, alert_tickers: set) -> str:
    thead = ('<table width="100%" cellpadding="0" cellspacing="0" '
             f'style="border-collapse:collapse;font-size:12px;{_FONT}">'
             '<thead><tr>'
             + _th("Ticker", "left") + _th("Company", "left") + _th("Type", "left")
             + _th("Strategy", "left") + _th("Entry") + _th("Exit→", title="Target exit date")
             + _th("Held", title="Calendar days held") + _th("Rem", title="Days to target exit")
             + _th("Buy") + _th("Now") + _th("Qty") + _th("Inv€")
             + _th("Ret%") + _th("P&L€") + _th("R", title="Current R-multiple (1R = initial risk)") + _th("SL", title="Stop-loss price (native currency)")
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

        # Ticker cell — highlight if alerted
        tick_color = _C_NEG if has_alert else _C_BODY
        rows += f"<tr>"
        rows += _td(f'<b style="color:{tick_color}">{t["ticker"]}</b>', "left", bg=bg)
        rows += _td(str(t.get("company",""))[:22], "left", _C_DIM, bg=bg)
        rows += _td(_trade_badge(t.get("trade_type","practice")), "left", bg=bg)
        rows += _td(_strat_badge(t.get("strategy","")), "left", bg=bg)
        rows += _td(t["entry_date"], "right", _C_DIM, bg=bg)
        rows += _td(target or "─", "right", _C_DIM, bg=bg)
        rows += _td(days_held, "right", _C_DIM, bg=bg)
        rows += _td(days_rem_s, "right", rem_c, bg=bg)
        _buy_sym = _ccy_sym(t.get("currency", "EUR"))
        _buy_px  = float(t["buy_price"]) if t.get("buy_price") else None
        rows += _td(f'{_buy_sym}{_buy_px:.2f}' if _buy_px else "─", "right", _C_DIM, bg=bg)
        _now_sym = _ccy_sym(t.get("currency", "EUR"))
        _now_px  = r.get("curr_native")
        rows += _td(f'{_now_sym}{_now_px:.2f}' if _now_px else "─", "right", _C_BODY, bg=bg)
        rows += _td(f'{float(t["qty"]):.0f}', "right", _C_DIM, bg=bg)
        rows += _td(f'€{float(t["investment_eur"]):.0f}', "right", _C_DIM, bg=bg)
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
              + "".join(_th(col_labels.get(s, s[:6].upper()), "center", title=s) for s in strategies))
    rows = ""
    multi_count = 0
    for i, (ticker, strat_map) in enumerate(sorted_t[:50]):
        company = next((r.get("company","") or "" for r in strat_map.values()), "")
        passes  = len(strat_map)
        multi   = passes > 1
        if multi: multi_count += 1
        bg = "#fff5f5" if multi else (_C_ROW1 if i % 2 else _C_ROW0)
        tc = _C_NEG if multi else _C_BODY
        rows += "<tr>"
        rows += _td(str(passes), "center", _C_DIM if not multi else _C_NEG, bg=bg)
        rows += _td(f'<b style="color:{tc}">{ticker}</b>', "left", bg=bg)
        rows += _td(company[:28], "left", _C_DIM if not multi else _C_NEG, bg=bg)
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
    "All-weather": {"icon": "🌤", "color": "#4ade80", "bg": "#052e16"},
    "Defensive":   {"icon": "🛡",  "color": "#38bdf8", "bg": "#0c1a2e"},
    "Momentum":    {"icon": "📈", "color": "#a78bfa", "bg": "#1e1b4b"},
    "Momentum+":   {"icon": "📈", "color": "#818cf8", "bg": "#1e1b4b"},
    "Neutral":     {"icon": "〰", "color": "#94a3b8", "bg": "#1e293b"},
    "":            {"icon": "─",  "color": "#4b5563", "bg": "#111827"},
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
                    if _math.isnan(ret) or abs(ret) > 15: continue
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
        if n < 8: return ("─", "", "#4b5563", "#111827")
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
                    if _math.isnan(ret) or abs(ret) > 15: continue
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
            return (f'<td style="padding:6px 8px;text-align:center;color:#4b5563;font-size:11px;">─</td>')
        bg  = "#052e16" if wr >= 65 else ("#1e3a00" if wr >= 55 else ("#3b1a00" if wr >= 45 else "#3b0000"))
        col = "#4ade80" if wr >= 65 else ("#a3e635" if wr >= 55 else ("#fb923c" if wr >= 45 else "#f87171"))
        return (f'<td style="padding:6px 8px;text-align:center;background:{bg};font-size:12px;'
                f'font-weight:700;color:{col};">{wr:.0f}%<span style="font-size:9px;color:{col};'
                f'opacity:0.7;font-weight:400;"> n={n}</span></td>')

    STRAT_ALIAS = {
        "pocket_pivot":"Pocket Pivot","ema_ribbon":"EMA Ribbon","cup_handle":"Cup Handle",
        "connors_rsi2":"Connors RSI2","signal_velocity":"Signal Velocity","breakout":"Breakout",
        "vcp":"VCP","nr7":"NR7","wyckoff_spring":"Wyckoff Spring","darvas_box":"Darvas Box",
        "raschke_8020":"Raschke 80/20","high_tight_flag":"High Tight Flag","stage4_short":"Stage4 Short",
        "connors_3down":"Connors 3↓","holy_grail":"Holy Grail","weinstein_stage2":"Weinstein S2",
        "defensive_rotation":"Def. Rotation","rs_line":"RS Line","williams_pct_r":"Williams %R",
        "bollinger_pctb":"BB %B","turnover_momentum":"Turnover Mom.","elder_impulse":"Elder Impulse",
        "ma50_reclaim":"MA50 Reclaim","momentum_burst":"Mom. Burst","analyst_upgrade":"Analyst Upg.",
    }

    def _label(bull_wr, bear_wr, neut_wr):
        if bull_wr is None or bear_wr is None:
            if neut_wr and neut_wr >= 60: return ("📈", "Momentum", "#818cf8")
            return ("─", "", _C_DIM)
        diff = bull_wr - bear_wr
        if bear_wr >= 60:                  return ("🌤", "All-weather", "#4ade80")
        if abs(diff) <= 10 and bull_wr>=50: return ("🌤", "All-weather", "#4ade80")
        if diff >= 20:                     return ("📈", "Momentum", "#818cf8")
        if diff >= 10:                     return ("📈", "Momentum+", "#a78bfa")
        if diff <= -10:                    return ("🛡", "Defensive", "#38bdf8")
        return ("〰", "Neutral", _C_DIM)

    # Sort: all-weather first, then momentum, then bear-sensitive
    rows_data = []
    for s, d in strats.items():
        n = sum(len(v) for v in d.values())
        if n < MIN_N: continue
        bull_wr = _wr(d["bull"])
        bear_wr = _wr(d["bear"])
        neut_wr = _wr(d["neut"])
        icon, lbl, lcol = _label(bull_wr, bear_wr, neut_wr)
        rows_data.append((s, d, bull_wr, bear_wr, neut_wr, icon, lbl, lcol, n))

    # Sort: all-weather > defensive > momentum > neutral; within each by bull WR
    order = {"All-weather": 0, "Defensive": 1, "Momentum+": 2, "Momentum": 3, "Neutral": 4, "": 5}
    rows_data.sort(key=lambda x: (order.get(x[6], 9), -(x[2] or 0)))

    rows_html = ""
    for i, (s, d, bull_wr, bear_wr, neut_wr, icon, lbl, lcol, n) in enumerate(rows_data):
        bg = _C_ROW1 if i % 2 else _C_ROW0
        name = STRAT_ALIAS.get(s, s.replace("_", " ").title())
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
        f'<tr><td style="background:#1e1b4b;border-left:4px solid #818cf8;padding:7px 10px;border-radius:3px 3px 0 0;">'
        f'<span style="font-size:11px;font-weight:700;color:#a5b4fc;letter-spacing:.04em;">'
        f'🗺 STRATEGY REGIME MAP &nbsp;·&nbsp; win rate by market condition (SPY 5d return)</span>'
        f'<span style="font-size:10px;color:#6366f1;margin-left:12px;">🌤 all-weather &nbsp; 📈 momentum &nbsp; 🛡 defensive</span>'
        f'</td></tr></table>'
        f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">'
        f'<thead><tr style="background:#1e1b4b;">'
        f'<th style="padding:6px 10px;text-align:left;font-size:11px;color:#818cf8;">Strategy</th>'
        f'<th style="padding:6px 8px;text-align:center;font-size:11px;color:#f87171;">🐻 Bear<br><span style="font-weight:400;font-size:9px;">SPY ≤ −1%</span></th>'
        f'<th style="padding:6px 8px;text-align:center;font-size:11px;color:#94a3b8;">〰 Neutral<br><span style="font-weight:400;font-size:9px;">−1% to +1%</span></th>'
        f'<th style="padding:6px 8px;text-align:center;font-size:11px;color:#4ade80;">🐂 Bull<br><span style="font-weight:400;font-size:9px;">SPY ≥ +1%</span></th>'
        f'<th style="padding:6px 10px;text-align:left;font-size:11px;color:#818cf8;">Character</th>'
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
                    if _math.isnan(ret) or abs(ret) > 15:
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
        if wr >= 65: return "#4ade80"
        if wr >= 55: return "#fbbf24"
        return "#f87171"

    def _avg_col(avg):
        if avg > 1.0: return "#4ade80"
        if avg > 0:   return "#a3e635"
        return "#f87171"

    rows_html = ""
    for i, (s, n, wr, avg) in enumerate(data):
        bg = "#0d1f12" if i % 2 == 0 else "#111827"
        name = STRAT_ALIAS.get(s, s.replace("_"," ").title())
        wr_c  = _wr_col(wr)
        avg_c = _avg_col(avg)
        bar_w = int(wr * 0.6)  # scale to max ~60px for 100%
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:6px 10px;font-size:12px;color:#e2e8f0;white-space:nowrap;">{name}</td>'
            f'<td style="padding:6px 8px;font-size:11px;color:#94a3b8;text-align:center;">{n}</td>'
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
        f'<thead><tr style="background:#1e293b;">'
        f'<th style="padding:7px 10px;font-size:11px;color:#94a3b8;text-align:left;font-weight:600;">STRATEGY</th>'
        f'<th style="padding:7px 8px;font-size:11px;color:#94a3b8;text-align:center;font-weight:600;">N</th>'
        f'<th style="padding:7px 10px;font-size:11px;color:#94a3b8;text-align:left;font-weight:600;">WIN RATE</th>'
        f'<th style="padding:7px 10px;font-size:11px;color:#94a3b8;text-align:right;font-weight:600;">AVG RET</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )


def _build_scanner_results_html() -> str:
    """Scanner results: leads with conviction cards, then full detail by strategy."""
    scan_json = HERE / "last_scan.json"
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

    # ── Batch-fill missing company names ─────────────────────────────────────
    _all_tickers = list({r["ticker"] for res in rbs.values() for r in res})
    _need_fetch  = [t for res in rbs.values() for r in res
                    if not r.get("company") or r.get("company") == "MISSING"
                    for t in [r["ticker"]]]
    _need_fetch  = list(dict.fromkeys(_need_fetch))  # dedupe, preserve order
    if _need_fetch:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        try:
            with _TPE(max_workers=12) as _ex:
                _fetched = dict(zip(_need_fetch, _ex.map(fetch_company_name, _need_fetch)))
        except Exception:
            _fetched = {}
        # Write names back into every result row
        for res in rbs.values():
            for r in res:
                if not r.get("company") or r.get("company") == "MISSING":
                    r["company"] = _fetched.get(r["ticker"], r.get("ticker", ""))
    else:
        _fetched = {}

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

    # ── Market Regime Bar ─────────────────────────────────────────────────────
    html += (
        f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 8px;">'
        f'<tr><td style="background:#0f172a;padding:7px 14px;border-radius:4px;border-left:4px solid {regime_col};">'
        f'<span style="font-size:12px;font-weight:700;color:{regime_col};">'
        f'{regime_icon} MARKET REGIME: {market_regime}</span>'
        f'<span style="font-size:11px;color:#94a3b8;margin-left:12px;">'
        f'Market trend pulse: <b style="color:#e2e8f0;">{elder_count}/20 stocks in uptrend</b>'
        f'{"  · 🔥 Strong uptrend — good time to enter momentum trades" if elder_count >= 15 else ("  · ⚠ Mixed market — only enter the highest-conviction setups" if elder_count >= 5 else "  · ❄️ Weak/falling market — avoid new longs, wait for recovery")}'
        f'</span>'
        f'</td></tr></table>'
    )

    # ── Friday Warning ────────────────────────────────────────────────────────
    if _is_friday:
        html += (
            '<table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 8px;">'
            '<tr><td style="background:#3b1a00;border-left:5px solid #f97316;padding:8px 14px;border-radius:4px;">'
            '<span style="font-size:12px;font-weight:800;color:#fb923c;">⚠ FRIDAY SCAN — Historical WR=45%, avg -0.31%</span>'
            '<span style="font-size:11px;color:#fed7aa;margin-left:10px;">Hold entry until Monday. Consider existing positions only.</span>'
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
            bg   = "#052e16" if ex >= 1.0 else ("#3b0000" if ex <= -1.0 else "#1e293b")
            col  = "#4ade80" if ex >= 1.0 else ("#f87171" if ex <= -1.0 else "#94a3b8")
            sign = "+" if ex >= 0 else ""
            return (f'<span style="background:{bg};color:{col};border-radius:3px;'
                    f'padding:2px 7px;font-size:10px;font-weight:700;margin-right:4px;">'
                    f'{sector_name[:6]} {sign}{ex:.1f}%</span>')

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
            f'<tr><td style="background:#0f1923;padding:8px 14px;border-radius:4px;border-left:4px solid #0ea5e9;">'
            f'<div style="font-size:11px;font-weight:700;color:#7dd3fc;margin-bottom:4px;">📊 SECTOR STRENGTH vs SPY (10d) &nbsp;·&nbsp; {spy_s}</div>'
            f'<div style="margin-bottom:3px;"><span style="font-size:10px;color:#94a3b8;margin-right:6px;">HOT 🔥</span>{top_chips}</div>'
            f'<div style="margin-bottom:4px;"><span style="font-size:10px;color:#94a3b8;margin-right:6px;">COLD ❄️</span>{bot_chips}</div>'
            + (f'<div style="font-size:10px;color:#4ade80;">✓ HIGH conviction picks in HOT sectors: <b>{", ".join(aligned)}</b></div>'
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
            f'<tr><td style="background:#052e16;border-left:5px solid #16a34a;padding:8px 12px;border-radius:4px 4px 0 0;">'
            f'<span style="font-size:13px;font-weight:800;color:#4ade80;letter-spacing:.06em;">'
            f'🎯 ACT ON THESE &nbsp;·&nbsp; {len(high_picks)} stock(s) &nbsp;·&nbsp; ★★★ HIGH CONVICTION</span>'
            f'<br><span style="font-size:10px;color:#86efac;">Ranked by: PROVEN edge · multi-strategy (74% WR) · vol 1.5-2x · RSI 50-65 · persistence</span>'
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
                f'<span style="background:{"#052e16" if s in _PROVEN_EDGE_SET else "#1c1c2e"};'
                f'color:{"#4ade80" if s in _PROVEN_EDGE_SET else "#a5b4fc"};'
                f'border-radius:3px;padding:2px 6px;font-size:10px;font-weight:600;">'
                f'{s.replace("_"," ").upper()}'
                f'{" " + str(int(hist_stats[s]["wr"])) + "%" if hist_stats.get(s, {}).get("n", 0) >= 5 else ""}'
                f'</span>'
                for s in strats_fired[:4]
            ) + _regime_badge(strat)
            price_s  = f'${r["price"]:.2f}' if r.get("price") else "─"
            sl_approx = r["price"] * 0.97 if r.get("price") else None
            sl_s     = f'${sl_approx:.2f}' if sl_approx else "─"
            hold_d   = {"pocket_pivot":7,"ema_ribbon":7,"cup_handle":10,"vcp":10,"connors_rsi2":5,"nr7":3,"breakout":5}.get(strat, 5)

            # Sector tag for this card
            ticker_sec = r.get("sector", "")
            sec_etf = next((etf for etf, sname in _ETF_TO_SECTOR.items() if sname == ticker_sec), "")
            sec_ex  = sector_excess.get(sec_etf) if sector_excess else None
            if sec_ex is not None:
                ranked_vals = sorted(sector_excess.values(), reverse=True)
                if sec_ex >= ranked_vals[2]:
                    sec_tag = f'<span style="background:#052e16;color:#4ade80;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-left:4px;">🔥 {ticker_sec[:8]} +{sec_ex:.1f}%</span>'
                elif sec_ex <= ranked_vals[-3]:
                    sec_tag = f'<span style="background:#3b0000;color:#f87171;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-left:4px;">❄️ {ticker_sec[:8]} {sec_ex:.1f}%</span>'
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
                _sig_badges += '<span style="background:#1e3a5f;color:#93c5fd;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-right:3px;">🔁 PERSIST {_days_seen}d</span>'.replace("{_days_seen}", str(_days_seen))
            elif _days_seen >= 2:
                _sig_badges += '<span style="background:#1e3a5f;color:#93c5fd;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-right:3px;">🔁 2d</span>'
            if _vol >= 2.0:
                _sig_badges += f'<span style="background:#3b2200;color:#fb923c;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-right:3px;">⚡ VOL {_vol:.1f}x</span>'
            elif _vol >= 1.5:
                _sig_badges += f'<span style="background:#3b2200;color:#fbbf24;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-right:3px;">⚡ {_vol:.1f}x</span>'
            if _rs > 0:
                _sig_badges += f'<span style="background:#052e16;color:#4ade80;font-size:9px;font-weight:700;border-radius:3px;padding:1px 5px;margin-right:3px;">↑RS +{_rs:.1f}%</span>'

            border_col = "#16a34a" if card_idx == 0 else ("#2563eb" if card_idx == 1 else "#d97706" if card_idx == 2 else "#16a34a")
            html += (
                f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:6px;border-radius:0 0 4px 4px;">'
                f'<tr style="background:#0f2a1a;">'
                f'<td style="padding:10px 14px;border-left:5px solid {border_col};border-bottom:1px solid #1a4a2a;">'
                f'<table width="100%" cellpadding="0" cellspacing="0"><tr>'
                f'<td style="width:140px;vertical-align:top;">'
                + (f'<div style="margin-bottom:4px;">{rank_badge}</div>' if rank_badge else '')
                + f'<div style="font-size:20px;font-weight:800;color:#f0fdf4;font-family:monospace;">{r["ticker"]}</div>'
                f'<div style="font-size:11px;color:#86efac;">{str(r.get("company","") or "")[:20]}{sec_tag}</div>'
                + (f'<div style="margin-top:4px;">{_sig_badges}</div>' if _sig_badges else '')
                + f'</td>'
                f'<td style="vertical-align:top;padding-left:12px;">'
                f'<div style="margin-bottom:5px;">{strat_chips}{proven_badge}</div>'
                f'<div style="font-size:11px;color:#d1fae5;">'
                f'Price <b>{price_s}</b> &nbsp;·&nbsp; '
                f'Stop Loss <b style="color:#fca5a5;">{sl_s}</b> &nbsp;·&nbsp; '
                f'Hold <b>{hold_d} days</b> &nbsp;·&nbsp; '
                f'Momentum <b>{r.get("rsi", 0):.0f}/100</b> &nbsp;·&nbsp; '
                f'Trend Str. <b>{r.get("adx", 0):.0f}/50</b> &nbsp;·&nbsp; '
                f'Score <b>{r.get("score", 0)}</b>'
                f'</div>'
                f'<div style="font-size:10px;color:#86efac;margin-top:3px;font-style:italic;">'
                f'{"💡 Buy near " + price_s + ", set a stop at " + sl_s + " to limit downside, target to sell in " + str(hold_d) + " days."}'
                f'</div>'
                f'</td>'
                f'<td style="width:90px;text-align:right;vertical-align:top;">'
                + (f'<div style="font-size:18px;font-weight:800;color:{wr_col};">{wr_val:.0f}% WR</div>'
                   f'<div style="font-size:10px;color:#86efac;">{avg_val:+.2f}% avg</div>'
                   if wr_val else '')
                + f'</td></tr></table></td></tr></table>'
            )
    else:
        html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:12px 0 4px;">'
            f'<tr><td style="background:#1a1a00;border-left:5px solid #facc15;padding:8px 12px;border-radius:4px;">'
            f'<span style="font-size:12px;font-weight:700;color:#fde68a;">'
            f'⚠ No HIGH conviction signals today — see WATCHLIST below or wait for better setup</span>'
            f'</td></tr></table>'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B: WATCHLIST — MED conviction (compact)
    # ══════════════════════════════════════════════════════════════════════════
    if med_picks:
        html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:14px 0 4px;">'
            f'<tr><td style="background:#1a1a0a;border-left:5px solid #d97706;padding:6px 12px;border-radius:4px 4px 0 0;">'
            f'<span style="font-size:12px;font-weight:700;color:#fcd34d;">'
            f'👀 WATCHLIST &nbsp;·&nbsp; {len(med_picks)} stock(s) &nbsp;·&nbsp; ★★ MEDIUM — worth watching, but wait for a stronger signal before buying</span>'
            f'</td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;margin-bottom:8px;">'
            f'<thead><tr style="background:#1a1a2e;">'
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
            strat_str = " + ".join(s.replace("_"," ").upper()[:8] for s in strats_fired[:2])
            wr_v = h.get("wr")
            wr_c = "#16a34a" if (wr_v or 0) >= 60 else "#d97706"
            price_s = f'${r["price"]:.2f}' if r.get("price") else "─"
            co_s = str(r.get("company") or "")[:20] or "─"
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
            f'<tr><td style="background:#0a0a1a;border-left:5px solid #818cf8;padding:6px 12px;border-radius:4px 4px 0 0;">'
            f'<span style="font-size:12px;font-weight:700;color:#a5b4fc;">'
            f'🔁 PERSISTENCE LEADERS &nbsp;·&nbsp; {len(streak_leaders)} stock(s) &nbsp;·&nbsp; ≥5 consecutive trading days</span>'
            f'</td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;margin-bottom:8px;">'
            f'<thead><tr style="background:#1a1a2e;">'
            + _th("Ticker","left") + _th("Streak","center") + _th("Strategies (latest)","left") + _th("Last Seen","right")
            + f'</tr></thead><tbody>'
        )
        for i, l in enumerate(streak_leaders[:15]):
            bg = _C_ROW1 if i % 2 else _C_ROW0
            strat_str = " · ".join(s.replace("_"," ").upper() for s in l["strategies"][:3])
            streak_color = "#16a34a" if l["streak"] >= 10 else ("#a5b4fc" if l["streak"] >= 7 else "#d1d5db")
            html += (
                f'<tr style="background:{bg};">'
                f'<td style="padding:4px 8px;color:#e2e8f0;font-weight:700;">{l["ticker"]}</td>'
                f'<td style="padding:4px 8px;color:{streak_color};text-align:center;font-weight:700;">{l["streak"]}d</td>'
                f'<td style="padding:4px 8px;color:#94a3b8;">{strat_str}</td>'
                f'<td style="padding:4px 8px;color:#64748b;text-align:right;">{l["last_date"]}</td>'
                f'</tr>'
            )
        html += '</tbody></table>'

    html += _kpi_table([
        ("HIGH Conviction", str(len(high_picks)), "#16a34a"),
        ("MED Watchlist",   str(len(med_picks)),  "#d97706"),
        ("Persistence",     str(len(streak_leaders)), "#818cf8"),
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
            s_label  = s.replace("_", " ").title()
            avg_r    = d.get("avg_r")
            sl_pct   = d.get("sl_pct")
            excess   = d.get("excess")
            avg_r_col  = "#16a34a" if (avg_r or 0) >= 0.5 else ("#d97706" if (avg_r or 0) >= 0 else "#dc2626")
            sl_col     = "#dc2626" if (sl_pct or 0) >= 20 else ("#d97706" if (sl_pct or 0) >= 12 else "#16a34a")
            excess_col = "#16a34a" if (excess or 0) > 0 else "#dc2626"
            _rchar = _rc.get(s, ("─", "", "#4b5563", "#111827"))
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
            f'<tr><td style="background:#1a2a1a;border-left:4px solid #16a34a;padding:6px 10px;border-radius:3px 3px 0 0;">'
            f'<span style="font-size:11px;font-weight:700;color:#16a34a;letter-spacing:.05em;">'
            f'📊 STRATEGY SCORECARD &nbsp;·&nbsp; live from scan_history.csv &nbsp;·&nbsp; ✓ = proven edge (WR≥60%, n≥10)</span></td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;">'
            f'<thead><tr style="background:#1a1a2e;">'
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
        f'<tr><td style="background:#1a1a2e;border-left:4px solid #4f46e5;padding:6px 12px;border-radius:4px;">'
        f'<span style="font-size:11px;font-weight:600;color:#a5b4fc;letter-spacing:.04em;">'
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
        _drchar = _rc_detail.get(strat, ("─", "", "#4b5563", "#111827"))
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
            company_s = str(r.get("company") or "")[:22] or "─"

            rows += "<tr>"
            rows += _td(f'<b style="color:{tick_c}">{r["ticker"]}</b>', "left", bg=bg)
            rows += _td(f'<span style="color:{_C_DIM};font-size:10px;">{company_s}</span>', "left", bg=bg)
            rows += _td(
                f'<span style="background:#f0f4ff;color:#334;border-radius:3px;'
                f'padding:1px 5px;font-size:10px;">{mkt}</span>',
                "center", bg=bg)
            rows += _td(str(score_v), "center", score_c, bold=True, bg=bg)
            rows += _td(f'${price_v:.2f}' if price_v else "─", "right", _C_DIM, bg=bg)
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
            if v is None: return '<span style="color:#484f58;">─</span>'
            col  = "#4ade80" if v > 0 else ("#f87171" if v < 0 else "#94a3b8")
            sign = "+" if v > 0 else ""
            return f'<span style="color:{col};font-weight:700;">{sign}{v}%</span>'

        metal_rows = "".join(
            f'<tr style="background:{"#1a1a2e" if i % 2 else "#0f0f1a"};">'
            f'<td style="padding:5px 8px;color:#e2e8f0;font-weight:700;font-size:11px;">{m["name"]}</td>'
            f'<td style="padding:5px 8px;color:#94a3b8;font-size:10px;">{m["ticker"]}</td>'
            f'<td style="padding:5px 8px;color:#e2e8f0;text-align:right;font-size:11px;">'
            f'${m["spot"]:,.2f} <span style="color:#484f58;font-size:10px;">{m["unit"]}</span></td>'
            f'<td style="padding:5px 8px;text-align:right;">{_chg_span(m["ch1d"])}</td>'
            f'<td style="padding:5px 8px;text-align:right;">{_chg_span(m["ch7d"])}</td>'
            f'</tr>'
            for i, m in enumerate(metal_prices)
        )
        html += (
            f'<br><table width="100%" cellpadding="0" cellspacing="0" style="margin:10px 0 4px;">'
            f'<tr><td style="background:#1a1200;border-left:5px solid #f59e0b;padding:6px 12px;border-radius:4px 4px 0 0;">'
            f'<span style="font-size:12px;font-weight:700;color:#fcd34d;">⬡ METALS SNAPSHOT &nbsp;·&nbsp; Live Spot Prices</span>'
            f'</td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;margin-bottom:4px;">'
            f'<thead><tr style="background:#1a1a2e;">'
            f'<th style="padding:5px 8px;color:#94a3b8;text-align:left;">Metal</th>'
            f'<th style="padding:5px 8px;color:#94a3b8;text-align:left;">Ticker</th>'
            f'<th style="padding:5px 8px;color:#94a3b8;text-align:right;">Spot</th>'
            f'<th style="padding:5px 8px;color:#94a3b8;text-align:right;">1-Day Δ</th>'
            f'<th style="padding:5px 8px;color:#94a3b8;text-align:right;">7-Day Δ</th>'
            f'</tr></thead><tbody>{metal_rows}</tbody></table>'
        )

    metal_events = _fetch_metal_events(n=6)
    if metal_events:
        ev_rows = "".join(
            f'<tr style="background:{"#1a1a2e" if i % 2 else "#0f0f1a"};">'
            f'<td style="padding:5px 8px;color:#94a3b8;font-size:10px;white-space:nowrap;">{e["pub"]}</td>'
            f'<td style="padding:5px 8px;font-size:11px;">'
            f'<a href="{e["link"]}" style="color:#c9d1d9;text-decoration:none;">{e["title"][:100]}</a>'
            + (f'<br><span style="font-size:10px;color:#484f58;">{e["source"]}</span>'
               if e.get("source") else "")
            + f'</td></tr>'
            for i, e in enumerate(metal_events)
        )
        html += (
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;'
            f'font-size:11px;margin-bottom:8px;">'
            f'<thead><tr style="background:#1a1a2e;">'
            f'<th style="padding:5px 8px;color:#94a3b8;text-align:left;width:55px;">Date</th>'
            f'<th style="padding:5px 8px;color:#94a3b8;text-align:left;">Supply Shock Events</th>'
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
                "Leading":   "#052e16", "Improving": "#0c1a3a",
                "Weakening": "#2d1f00", "Lagging":   "#3b0000",
            }
            _QUAD_COL = {
                "Leading":   "#4ade80", "Improving": "#60a5fa",
                "Weakening": "#fbbf24", "Lagging":   "#f87171",
            }
            rrg_rows = ""
            for i, r in enumerate(rrg_results):
                q         = r["quad"]
                bg        = "#1a1a2e" if i % 2 else "#0f0f1a"
                qbg       = _QUAD_BG.get(q, "#1e293b")
                qcol      = _QUAD_COL.get(q, "#94a3b8")
                emoji     = QUAD_EMOJI.get(q, "⚪")
                ratio_col = "#4ade80" if r["rs_ratio"] >= 100 else "#f87171"
                mom_col   = "#4ade80" if r["rs_mom"]   >= 100 else "#f87171"
                rrg_rows += (
                    f'<tr style="background:{bg};">'
                    f'<td style="padding:5px 8px;color:#e2e8f0;font-weight:700;font-size:11px;">{r["etf"]}</td>'
                    f'<td style="padding:5px 8px;color:#94a3b8;font-size:10px;">{r["name"]}</td>'
                    f'<td style="padding:5px 8px;text-align:center;">'
                    f'<span style="background:{qbg};color:{qcol};border-radius:3px;padding:2px 6px;'
                    f'font-size:10px;font-weight:700;">{emoji} {q}</span></td>'
                    f'<td style="padding:5px 8px;color:{ratio_col};text-align:right;font-size:11px;font-weight:600;">{r["rs_ratio"]:.2f}</td>'
                    f'<td style="padding:5px 8px;color:{mom_col};text-align:right;font-size:11px;font-weight:600;">{r["rs_mom"]:.2f}</td>'
                    f'</tr>'
                )
            html += (
                f'<br><table width="100%" cellpadding="0" cellspacing="0" style="margin:10px 0 4px;">'
                f'<tr><td style="background:#0a0a1a;border-left:5px solid #818cf8;padding:6px 12px;border-radius:4px 4px 0 0;">'
                f'<span style="font-size:12px;font-weight:700;color:#a5b4fc;">'
                f'📡 RRG SECTOR ROTATION &nbsp;·&nbsp; Live vs SPY &nbsp;·&nbsp; '
                f'{sum(1 for r in rrg_results if r["quad"]=="Leading")} Leading &nbsp; '
                f'{sum(1 for r in rrg_results if r["quad"]=="Improving")} Improving &nbsp; '
                f'{sum(1 for r in rrg_results if r["quad"]=="Weakening")} Weakening &nbsp; '
                f'{sum(1 for r in rrg_results if r["quad"]=="Lagging")} Lagging'
                f'</span></td></tr></table>'
                f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;margin-bottom:8px;">'
                f'<thead><tr style="background:#1a1a2e;">'
                f'<th style="padding:5px 8px;color:#94a3b8;text-align:left;font-weight:600;">ETF</th>'
                f'<th style="padding:5px 8px;color:#94a3b8;text-align:left;font-weight:600;">Sector</th>'
                f'<th style="padding:5px 8px;color:#94a3b8;text-align:center;font-weight:600;">Status</th>'
                f'<th style="padding:5px 8px;color:#94a3b8;text-align:right;font-weight:600;" title="vs SPY — above 100 means sector is beating the market">Beating Market?</th>'
                f'<th style="padding:5px 8px;color:#94a3b8;text-align:right;font-weight:600;" title="above 100 = gap widening, below 100 = gap closing">Gap Widening?</th>'
                f'</tr></thead>'
                f'<tr style="background:#0a0a0a;">'
                f'<td colspan="3" style="padding:4px 8px;font-size:10px;color:#484f58;font-style:italic;">'
                f'🟢 Leading = best to trade now &nbsp; 🔵 Improving = gaining momentum &nbsp; 🟡 Weakening = slowing &nbsp; 🔴 Lagging = avoid</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#484f58;font-style:italic;text-align:right;">&gt;100 = beating SPY</td>'
                f'<td style="padding:4px 8px;font-size:10px;color:#484f58;font-style:italic;text-align:right;">&gt;100 = accelerating</td>'
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


def build_email(trades: list[dict]) -> str:
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
                            f'{str(t.get("company",""))[:28]}</td>'
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
            ("REAL",       "REAL",             _C_POS),
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
                rows += _td(t.get("company","")[:22], "left", _C_DIM, bg=bg)
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
    scanner_html = _build_scanner_results_html()

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
    do_send = "--send" in sys.argv

    trades = load_trades()  # empty list is fine — digest still shows scanner signals

    print(f"\n  Building digest for {TODAY}...")
    html = build_email(trades)

    if do_send:
        open_count = sum(1 for t in trades if t.get('status') == 'OPEN')
        subject = f"📊 Trade Digest {TODAY} — {open_count} open"
        print(f"\n  Sending email to {NOTIFY_TO}...", end=" ", flush=True)
        send_email(subject, html)
        print("✅  Sent!")
    else:
        out = HERE / "digest_preview.html"
        out.write_text(html, encoding="utf-8")
        print(f"\n  Dry-run — preview saved to: {out}")
        print("  Run with --send to actually email it.")

if __name__ == "__main__":
    main()
