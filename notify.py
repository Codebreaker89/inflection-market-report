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

    fx_now    = fetch_fx_now(ccy)
    curr_eur  = round(curr_price / fx_now, 2)
    buy_eur   = round(buy_price / fx_entry, 2)
    ret_pct   = round((curr_price / buy_price - 1) * 100, 2)
    pnl_eur   = round((curr_eur - buy_eur) * qty, 2)

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

    return {"curr_eur": curr_eur, "buy_eur": buy_eur, "ret_pct": ret_pct, "pnl_eur": pnl_eur, "alerts": alerts}

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
    "momentum":     ("#1a4a8a", "#dbeafe"),
    "breakout":     ("#4a1a8a", "#ede9fe"),
    "pocket_pivot": ("#7a4a00", "#fef3c7"),
    "connors_rsi2": ("#005a6e", "#cffafe"),
    "ema_ribbon":   ("#135e2e", "#dcfce7"),
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

def _sl_eur_str(t: dict) -> str:
    if not t.get("stop_loss_price"):
        return "─"
    try:
        sl_native = float(t["stop_loss_price"])
        ccy = t.get("currency", "EUR")
        fx_now = fetch_fx_now(ccy)
        return f"€{sl_native/fx_now:.2f}"
    except Exception:
        return str(t.get("stop_loss_price", "─"))

def _portfolio_table(results_slice: list, alert_tickers: set) -> str:
    thead = ('<table width="100%" cellpadding="0" cellspacing="0" '
             f'style="border-collapse:collapse;font-size:12px;{_FONT}">'
             '<thead><tr>'
             + _th("Ticker", "left") + _th("Company", "left") + _th("Type", "left")
             + _th("Strategy", "left") + _th("Entry") + _th("Exit→", title="Target exit date")
             + _th("Held", title="Calendar days held") + _th("Rem", title="Days to target exit")
             + _th("Buy€") + _th("Now€") + _th("Qty") + _th("Inv€")
             + _th("Ret%") + _th("P&L€") + _th("SL€", title="Stop-loss price in EUR")
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
        rows += _td(f'€{r["buy_eur"]:.2f}' if r.get("buy_eur") else "─", "right", _C_DIM, bg=bg)
        rows += _td(f'€{r["curr_eur"]:.2f}' if r.get("curr_eur") else "─", "right", _C_BODY, bg=bg)
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

    col_labels = {"momentum":"MNTM","breakout":"BREAK","pocket_pivot":"PP",
                  "connors_rsi2":"RSI2","ema_ribbon":"RIBBON"}
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

    # ── Section 4: Cross-strategy matrix ─────────────────────────────────────
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
          {TODAY.strftime("%A, %d %b %Y")} &nbsp;·&nbsp; dahakehemant@gmail.com
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
