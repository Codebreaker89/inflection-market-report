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

_PREFIX_MAP = {
    "ETR": (".DE","EUR"), "FRA": (".DE","EUR"), "XETRA": (".DE","EUR"),
    "EPA": (".PA","EUR"), "AMS": (".AS","EUR"), "BIT": (".MI","EUR"),
    "BME": (".MC","EUR"), "LON": (".L","GBP"),
    "TSX": (".TO","CAD"), "CVE": (".TO","CAD"),
    "NYSE": ("","USD"), "NASDAQ": ("","USD"), "NYSEARCA": ("","USD"),
}
_FX_CACHE: dict[str, float] = {}

def _yf_ticker(ticker: str) -> str:
    if ":" in ticker:
        prefix, base = ticker.split(":", 1)
        sfx, _ = _PREFIX_MAP.get(prefix.upper(), ("", "EUR"))
        return base.strip() + sfx
    return ticker

def fetch_fx_now(currency: str) -> float:
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

def fetch_live_price(ticker: str) -> tuple[Optional[float], Optional[str]]:
    yf_t = _yf_ticker(ticker)
    def _norm(p, ccy):
        if ccy in ("GBp","GBX","GBx"): return p/100.0, "GBP"
        return p, (ccy.upper() if ccy else None)
    def _try(sym):
        try:
            with _quiet():
                fi = yf.Ticker(sym).fast_info
            p   = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
            ccy = getattr(fi, "currency", None)
            if p and float(p) > 0:
                return _norm(float(p), ccy)
        except Exception: pass
        try:
            with _quiet():
                df = yf.Ticker(sym).history(period="5d", interval="1d", auto_adjust=True)
            if not df.empty:
                p = float(df["Close"].dropna().iloc[-1])
                ccy = None
                try:
                    with _quiet():
                        ccy = getattr(yf.Ticker(sym).fast_info, "currency", None)
                except Exception: pass
                return _norm(p, ccy)
        except Exception: pass
        return None, None
    return _try(yf_t)

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

    # ── Alert checks ──────────────────────────────────────────────────────────
    if curr_price <= sl_price:
        drop_pct = round((curr_price / buy_price - 1) * 100, 2)
        alerts.append(("STOP_LOSS", f"Price {curr_eur:.2f}€ hit stop {round(sl_price/fx_now,2):.2f}€  ({drop_pct:+.1f}%)"))

    if ret_pct >= PROFIT_TARGET * 100:
        alerts.append(("PROFIT_TARGET", f"+{ret_pct:.1f}% — consider taking profits (target {int(PROFIT_TARGET*100)}%)"))

    target_exit = trade.get("target_exit_date", "")
    if target_exit and TODAY >= datetime.strptime(target_exit, "%Y-%m-%d").date():
        days_over = (TODAY - datetime.strptime(target_exit, "%Y-%m-%d").date()).days
        hold_d = trade.get("hold_days", DEFAULT_HOLD_DAYS)
        alerts.append(("HOLD_EXPIRED", f"Hold period ({hold_d}d) expired {days_over} day(s) ago — target exit was {target_exit}"))

    earn_d = fetch_earnings_date(trade["ticker"])
    if earn_d:
        days_to = (earn_d - TODAY).days
        alerts.append(("EARNINGS", f"Earnings in {days_to} day(s) on {earn_d} — consider exiting before"))

    return {"curr_eur": curr_eur, "buy_eur": buy_eur, "ret_pct": ret_pct, "pnl_eur": pnl_eur,
            "curr_native": curr_native, "alerts": alerts}

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
             + _th("Ret%") + _th("P&L€") + _th("SL", title="Stop-loss price (native currency)")
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



def _load_strategy_stats() -> dict:
    """Read scan_history.csv → per-strategy {n, wr, avg, avg_r} from filled ret_d5 rows."""
    csv_path = HERE / "scan_history.csv"
    if not csv_path.exists():
        return {}
    from collections import defaultdict
    import csv as _csv, math as _math
    # Strategies with known tracking issues — exclude from scorecard
    _EXCLUDED_FROM_STATS = {"stage4_short"}  # L020: ret tracking inverted for shorts
    stats = defaultdict(lambda: {"wins": 0, "total": 0, "sum": 0.0, "r_sum": 0.0, "r_n": 0})
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                strat = row.get("strategy", "").strip()
                if not strat or strat in _EXCLUDED_FROM_STATS:
                    continue
                try:
                    ret = float(row["ret_d5"])
                    if _math.isnan(ret):
                        continue
                    # L021: skip corrupted data (stock splits/yfinance errors)
                    if abs(ret) > 15:
                        continue
                    stats[strat]["total"] += 1
                    stats[strat]["sum"]   += ret
                    if ret > 0:
                        stats[strat]["wins"] += 1
                    rm_raw = row.get("r_multiple_d5", "")
                    if rm_raw:
                        rm = float(rm_raw)
                        if not _math.isnan(rm):
                            stats[strat]["r_sum"] += rm
                            stats[strat]["r_n"]   += 1
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
            "n":     n,
            "wr":    round(100 * d["wins"] / n, 1),
            "avg":   round(d["sum"] / n, 2),
            "avg_r": round(d["r_sum"] / d["r_n"], 2) if d["r_n"] > 0 else None,
        }
    return out

_PROVEN_EDGE_SET = {"pocket_pivot", "ema_ribbon", "cup_handle",
                    "signal_velocity", "connors_rsi2"}
# stage4_short REMOVED: tracking inverted, true WR=11.4% (L020)

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

    def _email_rank_score(r: dict, strats_fired: list) -> float:
        """Same criteria as scan.py _rank_score — higher is better."""
        pts = 0.0
        if any(s in _PROVEN_EDGE_SET for s in strats_fired):
            pts += 3
        adx = r.get("adx") or 0
        if 20 <= adx <= 35:   pts += 2
        elif 16 <= adx < 20 or 35 < adx <= 45: pts += 1
        rsi = r.get("rsi") or 0
        if 50 <= rsi <= 65:   pts += 2
        elif 65 < rsi <= 70:  pts += 1
        n = len(strats_fired)
        if n >= 3:   pts += 2
        elif n == 2: pts += 1
        if (r.get("score") or 99) <= 3: pts += 1
        best_wr = max((hist_stats.get(s, {}).get("wr", 0) for s in strats_fired), default=0)
        if best_wr >= 60: pts += 1
        return pts

    high_picks_raw = [(tier, s, r) for t, (rank, tier, s, r) in best_by_ticker.items() if tier == "HIGH"]
    # Compute rank score for each HIGH pick
    high_picks = sorted(
        high_picks_raw,
        key=lambda x: -_email_rank_score(
            x[2],
            [s for s, res in rbs.items() if any(z["ticker"] == x[2]["ticker"] for z in res)]
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

    html = f'<br>{_section_head("📡","Scanner Results",f"scan {scan_date} · {len(active)} strategies fired · {total_hits} signals","#1a5a8a")}'

    # ── Market Regime Bar ─────────────────────────────────────────────────────
    html += (
        f'<table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 8px;">'
        f'<tr><td style="background:#0f172a;padding:7px 14px;border-radius:4px;border-left:4px solid {regime_col};">'
        f'<span style="font-size:12px;font-weight:700;color:{regime_col};">'
        f'{regime_icon} MARKET REGIME: {market_regime}</span>'
        f'<span style="font-size:11px;color:#94a3b8;margin-left:12px;">'
        f'Elder Impulse: <b style="color:#e2e8f0;">{elder_count} signals</b>'
        f'{"  · 🔥 Uptrend confirmed — favour trend strategies" if elder_count >= 15 else ("  · ⚠ Mixed — use high-conviction only" if elder_count >= 5 else "  · ❄️ Avoid longs — favour mean-reversion")}'
        f'</span>'
        f'</td></tr></table>'
    )

    # ── Sector Strength Panel ─────────────────────────────────────────────────
    _ETF_TO_SECTOR = {
        "XLK":"Technology","XLI":"Industrials","XLV":"Healthcare","XLF":"Financial Services",
        "XLY":"Consumer Cyclical","XLP":"Consumer Defensive","XLB":"Basic Materials",
        "XLE":"Energy","XLC":"Communication Services","XLU":"Utilities","XLRE":"Real Estate",
    }
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
            f'<br><span style="font-size:10px;color:#86efac;">Ranked by: PROVEN edge · RSI 50-65 · ADX 20-35 · multi-strategy · WR≥60%</span>'
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
            )
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
                f'</td>'
                f'<td style="vertical-align:top;padding-left:12px;">'
                f'<div style="margin-bottom:5px;">{strat_chips}{proven_badge}</div>'
                f'<div style="font-size:11px;color:#d1fae5;">'
                f'Price <b>{price_s}</b> &nbsp;·&nbsp; '
                f'SL ~<b style="color:#fca5a5;">{sl_s}</b> &nbsp;·&nbsp; '
                f'Hold <b>{hold_d}d</b> &nbsp;·&nbsp; '
                f'RSI <b>{r.get("rsi", 0):.0f}</b> &nbsp;·&nbsp; '
                f'ADX <b>{r.get("adx", 0):.0f}</b> &nbsp;·&nbsp; '
                f'Score <b>{r.get("score", 0)}</b>'
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
            f'👀 WATCHLIST &nbsp;·&nbsp; {len(med_picks)} stock(s) &nbsp;·&nbsp; ★★ MED — confirm before entering</span>'
            f'</td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;margin-bottom:8px;">'
            f'<thead><tr style="background:#1a1a2e;">'
            + _th("Ticker","left") + _th("Strategy","left") + _th("WR%","right")
            + _th("Score","center") + _th("RSI","right") + _th("ADX","right") + _th("Price","right")
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
            html += (
                f'<tr style="background:{bg};">'
                + _td(f'<b>{r["ticker"]}</b>{pb}', "left", bg=bg)
                + _td(strat_str, "left", _C_DIM, bg=bg)
                + _td(f'<span style="color:{wr_c};font-weight:700;">{wr_v:.0f}%</span>' if wr_v else "─", "right", bg=bg)
                + _td(str(r.get("score",0)), "center", bg=bg)
                + _td(f'{r.get("rsi",0):.0f}' if r.get("rsi") else "─", "right", bg=bg)
                + _td(f'{r.get("adx",0):.0f}' if r.get("adx") else "─", "right", bg=bg)
                + _td(price_s, "right", _C_DIM, bg=bg)
                + '</tr>'
            )
        html += '</tbody></table>'

    html += _kpi_table([
        ("HIGH Conviction", str(len(high_picks)), "#16a34a"),
        ("MED Watchlist",   str(len(med_picks)),  "#d97706"),
        ("Total Signals",   str(total_hits),       _C_DIM),
        ("Scan Date",       scan_date,             _C_DIM),
    ])

    # ── Strategy Scorecard ────────────────────────────────────────────────────
    if hist_stats:
        scored = sorted(
            [(s, d) for s, d in hist_stats.items() if d["n"] >= 3],
            key=lambda x: (-x[1]["wr"], -x[1]["avg"])
        )
        sc_rows = ""
        for rank, (s, d) in enumerate(scored[:10], 1):
            proven = d["n"] >= 10 and d["wr"] >= 60
            wr_color = "#16a34a" if d["wr"] >= 60 else ("#d97706" if d["wr"] >= 45 else "#dc2626")
            avg_color = "#16a34a" if d["avg"] >= 0 else "#dc2626"
            badge = ('<span style="background:#16a34a;color:#fff;font-size:9px;font-weight:700;'
                     'border-radius:3px;padding:1px 4px;margin-left:6px;">PROVEN EDGE ✓</span>') if proven else ""
            s_label = s.replace("_", " ").upper()
            avg_r = d.get("avg_r")
            avg_r_col = "#16a34a" if (avg_r or 0) >= 1.0 else ("#d97706" if (avg_r or 0) >= 0.5 else _C_DIM)
            avg_r_s = f'{avg_r:.2f}R' if avg_r is not None else '─'
            sc_rows += (
                f'<tr style="background:{_C_ROW1 if rank%2 else _C_ROW0};">'
                f'<td style="padding:5px 8px;font-size:11px;color:{_C_BODY};">{rank}</td>'
                f'<td style="padding:5px 8px;font-size:11px;font-weight:600;color:{_C_BODY};">{s_label}{badge}</td>'
                f'<td style="padding:5px 8px;font-size:11px;text-align:right;color:{wr_color};font-weight:700;">{d["wr"]:.0f}%</td>'
                f'<td style="padding:5px 8px;font-size:11px;text-align:right;color:{avg_color};">{d["avg"]:+.2f}%</td>'
                f'<td style="padding:5px 8px;font-size:11px;text-align:right;color:{avg_r_col};font-weight:600;">{avg_r_s}</td>'
                f'<td style="padding:5px 8px;font-size:11px;text-align:right;color:{_C_DIM};">{d["n"]}</td>'
                '</tr>'
            )
        html += (
            f'<br><table width="100%" cellpadding="0" cellspacing="0" style="margin-top:8px;margin-bottom:0;">'
            f'<tr><td style="background:#1a2a1a;border-left:4px solid #16a34a;padding:6px 10px;border-radius:3px 3px 0 0;">'
            f'<span style="font-size:11px;font-weight:700;color:#16a34a;letter-spacing:.05em;">'
            f'📊 STRATEGY SCORECARD &nbsp;·&nbsp; LIVE from scan_history.csv</span></td></tr></table>'
            f'<table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;font-size:11px;">'
            f'<thead><tr>'
            f'<th style="padding:5px 8px;text-align:left;color:#888;background:#1a1a2e;">#</th>'
            f'<th style="padding:5px 8px;text-align:left;color:#888;background:#1a1a2e;">Strategy</th>'
            f'<th style="padding:5px 8px;text-align:right;color:#888;background:#1a1a2e;">WR% d5</th>'
            f'<th style="padding:5px 8px;text-align:right;color:#888;background:#1a1a2e;">Avg% d5</th>'
            f'<th style="padding:5px 8px;text-align:right;color:#888;background:#1a1a2e;font-style:italic;" title="Avg reward:risk multiple — higher = better exits">avgR</th>'
            f'<th style="padding:5px 8px;text-align:right;color:#888;background:#1a1a2e;">n</th>'
            f'</tr></thead><tbody>'
            + sc_rows +
            f'</tbody></table>'
        )

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
        html += (f'<table width="100%" cellpadding="0" cellspacing="0" style="margin-top:16px;margin-bottom:0;">'
                 f'<tr><td style="background:{bg_badge};border-left:4px solid {fg};'
                 f'padding:6px 10px;border-radius:3px 3px 0 0;">'
                 f'<span style="font-size:11px;font-weight:700;color:{fg};'
                 f'letter-spacing:.05em;text-transform:uppercase;">'
                 f'{strat_label} &nbsp;·&nbsp; {len(results)} signal(s)</span>'
                 + proven_badge
                 + live_note
                 + (f'<span style="font-size:10px;color:{fg};margin-left:12px;">{bt_note}</span>' if bt_note else "")
                 + (f'<br><span style="font-size:10px;color:{fg};opacity:0.8;font-style:italic;">{desc_text}</span>' if desc_text else "")
                 + f'</td></tr></table>')

        thead = (f'<table width="100%" cellpadding="0" cellspacing="0" '
                 f'style="border-collapse:collapse;font-size:11px;{_FONT}">'
                 '<thead><tr>'
                 + _th("Ticker", "left")
                 + _th("Mkt",   "center")
                 + _th("Scr",   "center", title="Scanner score")
                 + _th("Price", "right")
                 + _th("RSI",   "right")
                 + _th("ADX",   "right")
                 + _th("M/8",   "right",  title="Minervini score /8")
                 + _th("Vol×",  "right",  title="Volume ratio vs 20d avg")
                 + _th("Signals/Context", "left")
                 + _th("WR%",   "right",  title="Backtest win rate")
                 + _th("Avg%",  "right",  title="Backtest avg return")
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
            m_c     = _C_POS if (m_v or 0) >= 6 else _C_DIM
            tick_c  = _C_NEG if is_short else _C_BODY

            rows += "<tr>"
            rows += _td(f'<b style="color:{tick_c}">{r["ticker"]}</b>', "left", bg=bg)
            rows += _td(
                f'<span style="background:#f0f4ff;color:#334;border-radius:3px;'
                f'padding:1px 5px;font-size:10px;">{mkt}</span>',
                "center", bg=bg)
            rows += _td(str(score_v), "center", score_c, bold=True, bg=bg)
            rows += _td(f'${price_v:.2f}' if price_v else "─", "right", _C_DIM, bg=bg)
            rows += _td(f'{rsi_v:.0f}' if rsi_v else "─", "right", rsi_c, bg=bg)
            rows += _td(f'{adx_v:.0f}' if adx_v else "─", "right", _C_DIM, bg=bg)
            rows += _td(f'{int(m_v)}/8' if m_v is not None else "─", "right", m_c, bg=bg)
            rows += _td(f'{vol_v:.1f}×' if vol_v else "─", "right", vol_c, bg=bg)
            rows += _td(sigs_str, "left", _C_DIM, bg=bg,
                        extra="font-size:10px;max-width:220px;overflow:hidden;")
            rows += _td(f'{wr_v:.0f}%' if wr_v is not None else "─",
                        "right", _C_POS if (wr_v or 0) >= 60 else _C_DIM, bg=bg)
            rows += _td(_pct(avg_v) if avg_v is not None else "─",
                        "right", _c(avg_v) if avg_v is not None else _C_DIM, bg=bg)
            rows += "</tr>"

        html += thead + rows + "</tbody></table>"

    return html


def build_email(trades: list[dict]) -> str:
    open_trades   = [t for t in trades if t.get("status") == "OPEN"]
    closed_trades = [t for t in trades if t.get("status") == "CLOSED"]

    ALERT_ORDER = ["STOP_LOSS", "HOLD_EXPIRED", "PROFIT_TARGET", "EARNINGS"]
    ALERT_META  = {
        "STOP_LOSS":     ("🛑", "Stop Loss Hit",        "#c0392b", "#fff0f0"),
        "HOLD_EXPIRED":  ("⏰", "Hold Period Expired",  "#b7590a", "#fff8f0"),
        "PROFIT_TARGET": ("🎯", "Profit Target Reached","#1a7f4b", "#f0faf4"),
        "EARNINGS":      ("📣", "Earnings Warning",     "#1a5a8a", "#f0f6ff"),
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

    # ── Section 2: Portfolio — split real vs practice ─────────────────────────
    real_res     = [(t, r) for t, r in results if t.get("trade_type") == "real"]
    practice_res = [(t, r) for t, r in results if t.get("trade_type") != "real"]

    snapshot_html = ""

    def _group_kpi(res_slice, label, accent):
        invested  = sum(float(t["investment_eur"] or 0) for t, _ in res_slice)
        pnls      = [r["pnl_eur"] for _, r in res_slice if r["pnl_eur"] is not None]
        total_pnl = sum(pnls) if pnls else None
        wins      = sum(1 for _, r in res_slice if (r["ret_pct"] or 0) > 0)
        pnl_c     = _c(total_pnl)
        items     = [
            (label,       label,              accent),
            ("Invested",  f"€{invested:.0f}", _C_BODY),
            ("Open P&L",  _eur(total_pnl),    pnl_c),
            ("Positions", str(len(res_slice)), _C_BODY),
            ("In Profit", str(wins),           _C_POS if wins else _C_DIM),
        ]
        return _kpi_table(items)

    if real_res:
        snapshot_html += _section_head("💰", "Real Trades", f"{len(real_res)} position(s)", _C_POS)
        snapshot_html += _group_kpi(real_res, "REAL", _C_POS)
        snapshot_html += _portfolio_table(real_res, alert_tickers)

    if practice_res:
        if real_res: snapshot_html += '<br>'
        snapshot_html += _section_head("🧪", "Practice Trades", f"{len(practice_res)} position(s)", _C_WARN)
        snapshot_html += _group_kpi(practice_res, "PRACTICE", _C_WARN)
        snapshot_html += _portfolio_table(practice_res, alert_tickers)

    if not snapshot_html:
        snapshot_html = f'<p style="color:{_C_DIM};font-style:italic;">No open positions.</p>'

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

    # ── Section 4: Scanner results (per-strategy detail) ─────────────────────
    scanner_html = _build_scanner_results_html()

    # ── Section 5: Cross-strategy matrix ─────────────────────────────────────
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

  {_section_head("📈","Open Positions",f"{len(open_trades)} trade(s)","#2c3e50")}
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

    trades = load_trades()
    if not trades:
        print("No trades found in trades.csv"); return

    print(f"\n  Building digest for {TODAY}...")
    html = build_email(trades)

    if do_send:
        subject = f"📊 Trade Digest {TODAY} — {sum(1 for t in trades if t.get('status')=='OPEN')} open"
        open_alerts = sum(1 for t in trades if t.get("status") == "OPEN")
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
