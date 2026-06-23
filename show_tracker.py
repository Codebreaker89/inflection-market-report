#!/usr/bin/env python3
"""
Trade Tracker  |  Momentum & Breakout Strategy Paper Portfolio
──────────────────────────────────────────────────────────────
python3 show_tracker.py                    # show full tracker
python3 show_tracker.py --open             # open trades only
python3 show_tracker.py --closed           # closed trades only
python3 show_tracker.py add                # interactive: add a new trade
python3 show_tracker.py add --ticker AMAT --date 2026-06-10 --price 185.50 \\
        --strategy momentum --signals "MACD RSI50 · VOL ADX↑"
python3 show_tracker.py close              # interactive: close an open trade
python3 show_tracker.py close --id 3 --date 2026-06-22 --price 190.00 --reason 1wk_auto
python3 show_tracker.py risk               # risk module: position limits, P&L tiers, stops, sector breakdown

Data file: trades.csv  (same folder as this script)
Each trade = €1000 invested. Returns shown in both % and EUR.
1-week = entry + 5 trading days. 2-week = entry + 10 trading days.
If the exit date is in the future: shown as "not yet".
If it falls on a holiday: yfinance skips it automatically (trading-day aligned).

Analytics columns auto-populated at entry and when a trade closes:
  rsi_at_entry/exit, adx_at_entry/exit, minervini_at_entry/exit,
  vol_ratio_entry/exit, atr_ratio_entry, market_regime_entry, sector,
  max_dd_1wk, exit_reason

First run / if you see 401 errors:
  pip3 install --upgrade yfinance requests pandas numpy
"""

import os, sys, csv, time, warnings, logging, contextlib, io, webbrowser
from pathlib  import Path
from datetime import datetime, date, timedelta
from typing   import Optional

import requests
import numpy  as np
import pandas as pd
import yfinance as yf

try:
    from config import HOLD_DAYS, DEFAULT_HOLD_DAYS, STOP_LOSS_PCT
except ImportError:
    HOLD_DAYS = {}; DEFAULT_HOLD_DAYS = 7; STOP_LOSS_PCT = 0.03

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

HERE       = Path(__file__).parent
TRADES_CSV = HERE / "trades.csv"
INVEST_EUR = 1000.0

def biz_days_add(start: date, n: int) -> date:
    """Return date that is n business days after start."""
    d, added = start, 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:   # Mon–Fri
            added += 1
    return d

def trade_hold_days(strategy: str) -> int:
    return HOLD_DAYS.get(strategy, DEFAULT_HOLD_DAYS)

def trade_stop_loss(buy_price: float) -> float:
    return round(buy_price * (1 - STOP_LOSS_PCT), 4)

@contextlib.contextmanager
def _quiet():
    devnull = open(os.devnull, "w"); old = sys.stderr; sys.stderr = devnull
    try:    yield
    finally: sys.stderr = old; devnull.close()


# ── CURRENCY HELPERS ──────────────────────────────────────────────────────────

SUFFIX_CCY = {
    ".L":  "GBP",  ".DE": "EUR",  ".PA": "EUR",  ".AS": "EUR",
    ".MC": "EUR",  ".MI": "EUR",  ".BR": "EUR",  ".LS": "EUR",
    ".HE": "EUR",  ".SW": "CHF",  ".ST": "SEK",  ".CO": "DKK",
    ".OL": "NOK",  ".TO": "CAD",
}
FX_PAIR = {          # yfinance symbol: 1 EUR = X local
    "USD": "EURUSD=X",
    "GBP": "EURGBP=X",
    "CAD": "EURCAD=X",
    "CHF": "EURCHF=X",
    "SEK": "EURSEK=X",
    "DKK": "EURDKK=X",
    "NOK": "EURNOK=X",
}

# Exchange prefix (Google Finance / Bloomberg format) → yfinance suffix + currency
_PREFIX_MAP = {
    "ETR":    (".DE", "EUR"),   # XETRA Germany
    "FRA":    (".DE", "EUR"),   # Frankfurt
    "XETRA":  (".DE", "EUR"),
    "EPA":    (".PA", "EUR"),   # Euronext Paris
    "AMS":    (".AS", "EUR"),   # Amsterdam
    "BIT":    (".MI", "EUR"),   # Milan
    "BME":    (".MC", "EUR"),   # Madrid
    "LON":    (".L",  "GBP"),   # London
    "TSX":    (".TO", "CAD"),   # Toronto
    "CVE":    (".TO", "CAD"),
    "NYSE":   ("",    "USD"),
    "NASDAQ": ("",    "USD"),
    "NYSEARCA":("",   "USD"),
}

def _yf_ticker(ticker: str) -> str:
    """Convert exchange:ticker → yfinance format. ETR:ENR → ENR.DE"""
    if ":" in ticker:
        prefix, base = ticker.split(":", 1)
        sfx, _ = _PREFIX_MAP.get(prefix.upper(), ("", "USD"))
        return base.strip() + sfx
    return ticker

def ticker_market(ticker: str) -> str:
    """Return short market label: US, UK, DE, FR, NL, IT, ES, CH, CA."""
    if ":" in ticker:
        prefix = ticker.split(":")[0].upper()
        return {"ETR":"DE","FRA":"DE","XETRA":"DE","EPA":"FR","AMS":"NL",
                "BIT":"IT","BME":"ES","LON":"UK","TSX":"CA","CVE":"CA"}.get(prefix,"US")
    for sfx, mkt in {".L":"UK",".DE":"DE",".PA":"FR",".AS":"NL",
                     ".MI":"IT",".MC":"ES",".SW":"CH",".TO":"CA"}.items():
        if ticker.upper().endswith(sfx): return mkt
    return "US"

def ticker_ccy(ticker: str) -> str:
    """Currency of local price for a given ticker. Defaults to EUR."""
    if ":" in ticker:
        prefix, _ = ticker.split(":", 1)
        _, ccy = _PREFIX_MAP.get(prefix.upper(), ("", "EUR"))
        return ccy
    for sfx, ccy in SUFFIX_CCY.items():
        if ticker.upper().endswith(sfx):
            return ccy
    return "EUR"   # default: user inputs everything in EUR

def fetch_fx_now(ccy: str) -> float:
    if ccy == "EUR": return 1.0
    pair = FX_PAIR.get(ccy)
    if not pair: return 1.0
    try:
        with _quiet():
            df = yf.download(pair, period="5d", interval="1d",
                             progress=False, auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return 1.0

def fetch_fx_on_date(ccy: str, on_date: date) -> float:
    if ccy == "EUR": return 1.0
    pair = FX_PAIR.get(ccy)
    if not pair: return fetch_fx_now(ccy)
    try:
        start = (on_date - timedelta(days=7)).strftime("%Y-%m-%d")
        end   = (on_date + timedelta(days=1)).strftime("%Y-%m-%d")
        with _quiet():
            df = yf.download(pair, start=start, end=end, interval="1d",
                             progress=False, auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df[df.index.date <= on_date]
        if df.empty: return fetch_fx_now(ccy)
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return fetch_fx_now(ccy)


# ── PRICE HELPERS ─────────────────────────────────────────────────────────────

def fetch_price_history(ticker: str) -> Optional[pd.Series]:
    """600-day close history for 1wk/2wk lookups. Normalises exchange:ticker format."""
    yf_t = _yf_ticker(ticker)
    try:
        with _quiet():
            df = yf.download(yf_t, period="600d", interval="1d",
                             progress=False, auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        s = df["Close"].dropna()
        s.index = pd.to_datetime(s.index).date
        return s
    except Exception:
        return None

def price_on_or_before(series: pd.Series, target: date) -> Optional[float]:
    candidates = [d for d in series.index if d <= target]
    if not candidates: return None
    return float(series[max(candidates)])

def price_n_trading_days_after(series: pd.Series, entry: date, n: int):
    trading_days = sorted(d for d in series.index if d > entry)
    if len(trading_days) < n:
        return None, None
    target_date = trading_days[n - 1]
    return float(series[target_date]), target_date

def fetch_live_price(ticker: str) -> tuple[Optional[float], Optional[str]]:
    """
    Returns (price, currency) — freshest available price at any time.

    Priority:
      1. pre_market_price  — if pre-market session is active
      2. last_price        — live during regular session or after-hours
      3. previous_close    — fallback if above are unavailable
      4. history()         — last resort
    """
    yf_t = _yf_ticker(ticker)

    def _normalise_ccy(p, ccy_raw):
        if ccy_raw in ("GBp", "GBX", "GBx"):
            return p / 100.0, "GBP"
        return p, (ccy_raw.upper() if ccy_raw else None)

    def _try(sym):
        tk = yf.Ticker(sym)
        try:
            with _quiet():
                fi = tk.fast_info
            ccy = getattr(fi, "currency", None)
            # 1. Pre-market (most current if available)
            p = getattr(fi, "pre_market_price", None)
            if p and float(p) > 0:
                return _normalise_ccy(float(p), ccy)
            # 2. Last price (live / after-hours)
            p = getattr(fi, "last_price", None)
            if p and float(p) > 0:
                return _normalise_ccy(float(p), ccy)
            # 3. Previous close
            p = getattr(fi, "previous_close", None)
            if p and float(p) > 0:
                return _normalise_ccy(float(p), ccy)
        except Exception:
            ccy = None

        # 4. history() last resort
        try:
            with _quiet():
                df = tk.history(period="5d", interval="1d",
                                auto_adjust=True, prepost=True)
            if not df.empty:
                p = float(df["Close"].dropna().iloc[-1])
                if p > 0:
                    return _normalise_ccy(p, ccy)
        except Exception:
            pass
        return None, None

    price, ccy = _try(yf_t)
    if price is None:
        return None, None
    return price, ccy

def fetch_company_name(ticker: str) -> str:
    try:
        with _quiet():
            info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


# ── CSV HELPERS ───────────────────────────────────────────────────────────────

FIELDNAMES = [
    # Core trade fields
    "id", "entry_date", "ticker", "company", "currency",
    "buy_price", "stop_loss_price", "fx_at_entry", "qty", "investment_eur",
    "trade_type", "strategy", "hold_days", "target_exit_date",
    "signals", "status", "actual_sell_date", "exit_price",
    # Entry analytics (auto-populated when trade is added)
    "rsi_at_entry", "adx_at_entry", "minervini_at_entry",
    "vol_ratio_entry", "atr_ratio_entry",
    "market_regime_entry", "sector",
    # Exit analytics (auto-populated when trade closes)
    "rsi_at_exit", "adx_at_exit", "minervini_at_exit", "vol_ratio_exit",
    # Post-entry analytics
    "max_dd_1wk",
    # Manual close annotation
    "exit_reason",
]

def load_trades() -> list[dict]:
    if not TRADES_CSV.exists(): return []
    with open(TRADES_CSV, newline="") as f:
        reader = csv.DictReader(f, restval="")
        rows = [r for r in reader if r.get("id","").strip()]
    # backfill any missing columns (forward-compat for old CSVs)
    for row in rows:
        for fn in FIELDNAMES:
            if fn not in row:
                row[fn] = ""
    return rows

def save_trades(trades: list[dict]):
    with open(TRADES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(trades)

def next_id(trades: list[dict]) -> str:
    ids = [int(t["id"]) for t in trades if t.get("id","").isdigit()]
    return str(max(ids) + 1) if ids else "1"


# ── ANALYTICS HELPERS ─────────────────────────────────────────────────────────

def _build_indicators_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Inline indicator builder (mirrors momentum_scanner.build_indicators)."""
    c, h, l, v = raw["Close"], raw["High"], raw["Low"], raw["Volume"]
    def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
    def _sma(s, n): return s.rolling(n).mean()
    def _rsi(s, n=14):
        d = s.diff()
        g = d.clip(lower=0).rolling(n).mean()
        lo = (-d.clip(upper=0)).rolling(n).mean()
        return 100 - 100 / (1 + g / lo.replace(0, np.nan))
    def _macd(s):
        m = _ema(s, 12) - _ema(s, 26)
        return m, _ema(m, 9)
    def _adx(high, low, close, n=14):
        tr  = pd.concat([(high - low),
                         (high - close.shift()).abs(),
                         (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/n, adjust=False).mean()
        up  = (high - high.shift()).clip(lower=0)
        dn  = (low.shift() - low).clip(lower=0)
        dmp = up.where(up > dn, 0).ewm(alpha=1/n, adjust=False).mean()
        dmm = dn.where(dn > up, 0).ewm(alpha=1/n, adjust=False).mean()
        dip = 100 * dmp / atr
        dim = 100 * dmm / atr
        dx  = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
        return dx.ewm(alpha=1/n, adjust=False).mean()

    raw = raw.copy()
    raw["sma50"]     = _sma(c, 50)
    raw["sma150"]    = _sma(c, 150)
    raw["sma200"]    = _sma(c, 200)
    raw["ema9"]      = _ema(c, 9)
    raw["ema21"]     = _ema(c, 21)
    raw["rsi"]       = _rsi(c, 14)
    raw["macd"], raw["macd_sig"] = _macd(c)
    raw["macd_hist"] = raw["macd"] - raw["macd_sig"]
    raw["adx"]       = _adx(h, l, c, 14)
    raw["vol_ma20"]  = v.rolling(20).mean()
    raw["52w_high"]  = c.rolling(252).max()
    raw["52w_low"]   = c.rolling(252).min()
    # ATR for atr_ratio
    tr_s = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    raw["atr14"] = tr_s.ewm(alpha=1/14, adjust=False).mean()
    return raw


def _minervini_score(df: pd.DataFrame, idx: int) -> int:
    row = df.iloc[idx]
    return sum([
        row["Close"]  > row["sma150"],
        row["Close"]  > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        row["Close"]  > row["sma50"],
        row["Close"]  >= 1.30 * row["52w_low"],
        row["Close"]  >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[idx - 20]["sma200"],
    ])


def fetch_indicators_at_date(ticker: str, on_date: date) -> dict:
    """Return RSI, ADX, Minervini, vol_ratio, atr_ratio on a specific date."""
    empty = {"rsi": "", "adx": "", "minervini": "", "vol_ratio": "", "atr_ratio": ""}
    try:
        start = (on_date - timedelta(days=420)).strftime("%Y-%m-%d")
        end   = (on_date + timedelta(days=3)).strftime("%Y-%m-%d")
        with _quiet():
            raw = yf.download(ticker, start=start, end=end, interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 220: return empty
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return empty

        df    = _build_indicators_from_raw(raw)
        dates = pd.to_datetime(df.index).date
        cands = [i for i, d in enumerate(dates) if d <= on_date]
        if not cands: return empty
        idx = max(cands)
        if idx < 215: return empty

        row      = df.iloc[idx]
        vol_r    = float(row["Volume"]) / float(row["vol_ma20"]) if row["vol_ma20"] > 0 else 0
        atr_now  = float(row["atr14"])
        atr_prev = float(df.iloc[max(0, idx-20)]["atr14"])
        atr_r    = round(atr_now / atr_prev, 3) if atr_prev > 0 else ""

        return {
            "rsi":       round(float(row["rsi"]),  1),
            "adx":       round(float(row["adx"]),  1),
            "minervini": _minervini_score(df, idx),
            "vol_ratio": round(vol_r, 2),
            "atr_ratio": atr_r,
        }
    except Exception:
        return empty


def fetch_market_regime(on_date: date) -> str:
    """'BULL' if SPY above 50-day SMA on on_date, else 'BEAR'."""
    try:
        start = (on_date - timedelta(days=120)).strftime("%Y-%m-%d")
        end   = (on_date + timedelta(days=3)).strftime("%Y-%m-%d")
        with _quiet():
            df = yf.download("SPY", start=start, end=end, interval="1d",
                             progress=False, auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df["sma50"] = df["Close"].rolling(50).mean()
        dates = pd.to_datetime(df.index).date
        cands = [i for i, d in enumerate(dates) if d <= on_date]
        if not cands: return ""
        idx = max(cands)
        row = df.iloc[idx]
        if pd.isna(row["sma50"]): return ""
        return "BULL" if row["Close"] > row["sma50"] else "BEAR"
    except Exception:
        return ""


def fetch_sector(ticker: str) -> str:
    try:
        with _quiet():
            info = yf.Ticker(ticker).info
        return info.get("sector", "")
    except Exception:
        return ""


def fetch_entry_analytics(ticker: str, entry_date: date, strategy: str) -> dict:
    """
    Single yfinance download → signals string + all entry indicator values.
    Returns: {signals, rsi_at_entry, adx_at_entry, minervini_at_entry,
              vol_ratio_entry, atr_ratio_entry}
    """
    result = {
        "signals": "", "rsi_at_entry": "", "adx_at_entry": "",
        "minervini_at_entry": "", "vol_ratio_entry": "", "atr_ratio_entry": "",
    }
    try:
        start = (entry_date - timedelta(days=420)).strftime("%Y-%m-%d")
        end   = (entry_date + timedelta(days=3)).strftime("%Y-%m-%d")
        with _quiet():
            raw = yf.download(ticker, start=start, end=end, interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 220: return result
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return result

        dates = pd.to_datetime(raw.index).date
        cands = [i for i, d in enumerate(dates) if d <= entry_date]
        if not cands: return result
        idx = max(cands)
        if idx < 215: return result

        sys.path.insert(0, str(HERE))

        # Compute signals via appropriate scanner
        if strategy == "breakout":
            import breakout_scanner as bs
            df_s = bs.build_indicators(raw.copy())
            sig  = bs.score_row(df_s, idx)
            if sig:
                parts = " ".join(sig.get("coil_sigs", []))
                if sig.get("break_sigs"):
                    parts += " ▶ " + " ".join(sig["break_sigs"])
                result["signals"] = parts
        else:
            import momentum_scanner as ms
            df_s = ms.build_indicators(raw.copy())
            sig  = ms.score_row(df_s, idx)
            if sig:
                parts = " ".join(sig.get("fresh", []))
                if sig.get("conf"):
                    parts += " · " + " ".join(sig["conf"])
                result["signals"] = parts

        # Compute indicator values (use our inline builder for atr14)
        df = _build_indicators_from_raw(raw)
        row     = df.iloc[idx]
        vol_r   = float(row["Volume"]) / float(row["vol_ma20"]) if row["vol_ma20"] > 0 else 0
        atr_now  = float(row["atr14"])
        atr_prev = float(df.iloc[max(0, idx-20)]["atr14"])
        atr_r    = round(atr_now / atr_prev, 3) if atr_prev > 0 else ""

        result.update({
            "rsi_at_entry":       round(float(row["rsi"]), 1),
            "adx_at_entry":       round(float(row["adx"]), 1),
            "minervini_at_entry": _minervini_score(df, idx),
            "vol_ratio_entry":    round(vol_r, 2),
            "atr_ratio_entry":    atr_r,
        })
    except Exception:
        pass
    return result


def enrich_trades(trades: list[dict], price_cache: dict) -> bool:
    """
    Backfill missing analytics into existing trades (in-place).
    Returns True if anything changed (caller should save_trades).
    """
    changed = False
    for trade in trades:
        ticker     = trade["ticker"]
        entry_date = datetime.strptime(trade["entry_date"], "%Y-%m-%d").date()

        # ── Entry indicator values ────────────────────────────────────────────
        if not trade.get("rsi_at_entry"):
            strategy = trade.get("strategy", "momentum")
            print(f"  Enriching {ticker} entry analytics...", end=" ", flush=True)
            analytics = fetch_entry_analytics(ticker, entry_date, strategy)
            # only overwrite signals if not already set
            if not trade.get("signals") and analytics.get("signals"):
                trade["signals"] = analytics["signals"]
            trade["rsi_at_entry"]       = analytics["rsi_at_entry"]
            trade["adx_at_entry"]       = analytics["adx_at_entry"]
            trade["minervini_at_entry"] = analytics["minervini_at_entry"]
            trade["vol_ratio_entry"]    = analytics["vol_ratio_entry"]
            trade["atr_ratio_entry"]    = analytics["atr_ratio_entry"]
            print("✓")
            changed = True

        if not trade.get("market_regime_entry"):
            trade["market_regime_entry"] = fetch_market_regime(entry_date)
            changed = True

        if not trade.get("sector"):
            trade["sector"] = fetch_sector(ticker)
            changed = True

        # ── max_dd_1wk: fill once 5 trading days have passed ─────────────────
        if not trade.get("max_dd_1wk"):
            if ticker not in price_cache:
                price_cache[ticker] = fetch_price_history(ticker)
            hist = price_cache[ticker]
            if hist is not None:
                buy_price    = float(trade["buy_price"])
                trading_days = sorted(d for d in hist.index if d > entry_date)
                if len(trading_days) >= 5:
                    prices = [float(hist[d]) for d in trading_days[:5]]
                    min_p  = min(prices)
                    trade["max_dd_1wk"] = round((min_p - buy_price) / buy_price * 100, 2)
                    changed = True

        # ── Exit indicator values (only for CLOSED trades) ────────────────────
        if trade.get("status") == "CLOSED" and trade.get("actual_sell_date"):
            if not trade.get("rsi_at_exit"):
                actual_sell_date = datetime.strptime(trade["actual_sell_date"], "%Y-%m-%d").date()
                print(f"  Enriching {ticker} exit analytics...", end=" ", flush=True)
                ind = fetch_indicators_at_date(ticker, actual_sell_date)
                trade["rsi_at_exit"]       = ind.get("rsi", "")
                trade["adx_at_exit"]       = ind.get("adx", "")
                trade["minervini_at_exit"] = ind.get("minervini", "")
                trade["vol_ratio_exit"]    = ind.get("vol_ratio", "")
                print("✓")
                changed = True

    return changed


# ── ADD TRADE ─────────────────────────────────────────────────────────────────

def add_trade_interactive(args: list[str]):
    def _get(flag, prompt, default=""):
        for i, a in enumerate(args):
            if a == flag and i+1 < len(args): return args[i+1]
        val = input(f"  {prompt} [{default}]: ").strip()
        return val or default

    ALL_STRATEGIES = [
        "momentum", "breakout", "pocket_pivot", "connors_rsi2", "ema_ribbon",
        "nr7", "bb_squeeze", "high_tight_flag", "analyst_upgrade",
        "signal_velocity", "chokepoint_inflection", "stage4_short",
        "defensive_rotation", "cup_handle", "power_earnings_gap",
    ]

    print("\n  ── Add New Trade ──")
    ticker   = _get("--ticker", "Ticker (e.g. AMAT, BP.L)").upper()
    raw_date = _get("--date",   "Entry date (YYYY-MM-DD)", date.today().strftime("%Y-%m-%d"))

    # Strategy: numbered list if not passed via --strategy flag
    strategy = None
    for i, a in enumerate(args):
        if a == "--strategy" and i+1 < len(args):
            strategy = args[i+1]; break
    if not strategy:
        print("\n  Strategy:")
        for idx, s in enumerate(ALL_STRATEGIES, 1):
            print(f"    {idx:>2}. {s}")
        raw_s = input("  Pick number (or type name) [1]: ").strip()
        if raw_s.isdigit() and 1 <= int(raw_s) <= len(ALL_STRATEGIES):
            strategy = ALL_STRATEGIES[int(raw_s) - 1]
        elif raw_s in ALL_STRATEGIES:
            strategy = raw_s
        else:
            strategy = ALL_STRATEGIES[0]

    trade_type = _get("--type", "Trade type  [practice / real]", "practice").lower()
    if trade_type not in ("practice", "real"): trade_type = "practice"

    entry_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
    currency   = ticker_ccy(ticker)

    print(f"\n  Fetching info for {ticker}...", end=" ", flush=True)
    company = fetch_company_name(ticker)
    print(company)

    print(f"  Fetching FX rate (EUR → {currency}) on {entry_date}...", end=" ", flush=True)
    fx = fetch_fx_on_date(currency, entry_date)
    print(f"  1 EUR = {fx:.4f} {currency}")

    print(f"\n  {ticker} ({company}) — entering price in {BOLD(currency)}.")
    if currency != "EUR":
        eur_example = 1000 * fx
        print(f"  Enter the exact price in {currency},")
        print(f"  OR type  EUR:<amount>  if you have the price in euros (e.g. EUR:180 = {180*fx:.2f} {currency})")
    raw_price = _get("--price", f"Buy price (in {currency})")

    if raw_price.upper().startswith("EUR:"):
        eur_amount = float(raw_price.split(":")[1])
        price = round(eur_amount * fx, 4)
        print(f"  → {eur_amount} EUR × {fx:.4f} = {price} {currency}")
    else:
        price = float(raw_price)

    # ── Investment amount ─────────────────────────────────────────────────────
    raw_invest = _get("--invest", "Investment amount in EUR", str(int(INVEST_EUR)))
    invest_eur = float(raw_invest)

    # Allow user to specify qty directly instead
    raw_qty = _get("--qty", "Qty (leave blank to auto-calculate from investment)", "")
    if raw_qty:
        qty        = float(raw_qty)
        invest_eur = round(qty * price / fx, 2)
    else:
        qty = round(invest_eur * fx / price, 4)

    # ── Auto-detect signals AND entry indicators in one fetch ─────────────────
    manual_signals = None
    for i, a in enumerate(args):
        if a == "--signals" and i+1 < len(args):
            manual_signals = args[i+1]

    print(f"\n  Auto-detecting signals + entry indicators on {entry_date}...", end=" ", flush=True)
    analytics = fetch_entry_analytics(ticker, entry_date, strategy)

    if manual_signals is not None:
        signals = manual_signals
    else:
        signals = analytics["signals"]
        if signals:
            print(f"  {signals}")
        else:
            print("  none detected (entry date may be outside signal window)")
            signals = input("  Enter signals manually (or press Enter to skip): ").strip()

    # ── Market regime ─────────────────────────────────────────────────────────
    print(f"  Fetching market regime (SPY vs SMA50)...", end=" ", flush=True)
    regime = fetch_market_regime(entry_date)
    print(regime or "unknown")

    # ── Sector ────────────────────────────────────────────────────────────────
    print(f"  Fetching sector for {ticker}...", end=" ", flush=True)
    sector = fetch_sector(ticker)
    print(sector or "unknown")

    # ── Summary ───────────────────────────────────────────────────────────────
    eur_equiv = price / fx
    rsi_str = str(analytics.get("rsi_at_entry") or "─")
    adx_str = str(analytics.get("adx_at_entry") or "─")
    m_str   = str(analytics.get("minervini_at_entry") or "─")
    vr_str  = str(analytics.get("vol_ratio_entry") or "─")
    ar_str  = str(analytics.get("atr_ratio_entry") or "─")

    hold_d         = trade_hold_days(strategy)
    target_exit    = biz_days_add(entry_date, hold_d)
    stop_loss_px   = trade_stop_loss(price)

    type_label = GRN("REAL") if trade_type == "real" else YLW("PRACTICE")
    print(f"\n  ┌─ Trade Summary ────────────────────────────────────────────")
    print(f"  │  Ticker:      {ticker}  ({company})   [{type_label}]")
    print(f"  │  Entry date:  {entry_date}    Sector: {sector or '─'}    Regime: {regime or '─'}")
    print(f"  │  Buy price:   {price} {currency}  =  €{eur_equiv:.2f} per share")
    print(f"  │  Stop loss:   {stop_loss_px} {currency}  (-{STOP_LOSS_PCT*100:.0f}%)")
    print(f"  │  FX rate:     1 EUR = {fx:.4f} {currency}  (on entry date)")
    print(f"  │  Quantity:    {qty} shares  (€{invest_eur:.2f} invested)")
    print(f"  │  Strategy:    {strategy}  |  Hold: {hold_d} days  |  Target exit: {target_exit}")
    print(f"  │  Signals:     {signals or '—'}")
    print(f"  │  Indicators:  RSI={rsi_str}  ADX={adx_str}  Minervini={m_str}/8  VolRatio={vr_str}  ATRratio={ar_str}")
    print(f"  └────────────────────────────────────────────────────────────")

    confirm = input("\n  Add this trade? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("  Cancelled."); return

    trades = load_trades()
    trades.append({
        "id":                  next_id(trades),
        "entry_date":          entry_date.strftime("%Y-%m-%d"),
        "ticker":              ticker,
        "company":             company,
        "currency":            currency,
        "buy_price":           price,
        "stop_loss_price":     stop_loss_px,
        "fx_at_entry":         round(fx, 6),
        "qty":                 qty,
        "investment_eur":      invest_eur,
        "trade_type":          trade_type,
        "strategy":            strategy,
        "hold_days":           hold_d,
        "target_exit_date":    target_exit.strftime("%Y-%m-%d"),
        "signals":             signals,
        "status":              "OPEN",
        "actual_sell_date":    "",
        "exit_price":          "",
        # entry analytics
        "rsi_at_entry":        analytics.get("rsi_at_entry", ""),
        "adx_at_entry":        analytics.get("adx_at_entry", ""),
        "minervini_at_entry":  analytics.get("minervini_at_entry", ""),
        "vol_ratio_entry":     analytics.get("vol_ratio_entry", ""),
        "atr_ratio_entry":     analytics.get("atr_ratio_entry", ""),
        "market_regime_entry": regime,
        "sector":              sector,
        # exit analytics (empty at entry)
        "rsi_at_exit":         "",
        "adx_at_exit":         "",
        "minervini_at_exit":   "",
        "vol_ratio_exit":      "",
        "max_dd_1wk":          "",
        "exit_reason":         "",
    })
    save_trades(trades)
    print(f"\n  ✅  Trade #{trades[-1]['id']} added to trades.csv")


# ── ANSI COLORS ───────────────────────────────────────────────────────────────

_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _color else text
GRN  = lambda t: _c("32", t);  RED  = lambda t: _c("31", t)
YLW  = lambda t: _c("33", t);  CYN  = lambda t: _c("36", t)
BOLD = lambda t: _c("1",  t);  DIM  = lambda t: _c("2",  t)
MAG  = lambda t: _c("35", t)

def _pct(v):
    if v is None: return DIM("   ─  ")
    return GRN(f"+{v:.1f}%") if v >= 0 else RED(f"{v:.1f}%")

def _eur(v):
    if v is None: return DIM("    ─   ")
    return GRN(f"+€{v:.0f}") if v >= 0 else RED(f"-€{abs(v):.0f}")

def _status(s):
    if s == "OPEN":   return GRN("OPEN  ")
    if s == "CLOSED": return DIM("CLOSED")
    return YLW(s[:6])

def _regime(s):
    if s == "BULL": return GRN("BULL")
    if s == "BEAR": return RED("BEAR")
    return DIM(s or "─")


# ── COMPUTE ONE TRADE ROW ─────────────────────────────────────────────────────

def compute_row(trade: dict, price_cache: dict, fx_cache: dict) -> dict:
    ticker     = trade["ticker"]
    entry_date = datetime.strptime(trade["entry_date"], "%Y-%m-%d").date()
    buy_price  = float(trade["buy_price"])
    fx_entry   = float(trade["fx_at_entry"])
    qty        = float(trade["qty"])
    ccy        = trade["currency"]
    status     = trade["status"]

    if ticker not in price_cache:
        price_cache[ticker] = fetch_price_history(ticker)
    hist = price_cache[ticker]

    if ccy not in fx_cache:
        fx_cache[ccy] = fetch_fx_now(ccy)
    fx_now = fx_cache[ccy]

    def to_eur(local_price, fx): return local_price / fx if fx else None
    def pnl_eur(exit_price_local, fx_exit):
        if exit_price_local is None: return None
        exit_eur  = exit_price_local / fx_exit
        entry_eur = buy_price / fx_entry
        return round((exit_eur - entry_eur) * qty, 2)
    def ret_pct(exit_p):
        if exit_p is None: return None
        return round((exit_p - buy_price) / buy_price * 100, 2)

    result = dict(trade)

    # ── Current price (live quote for OPEN; exit price for CLOSED) ───────────
    if status == "CLOSED" and trade.get("exit_price"):
        curr_price = float(trade["exit_price"])
    else:
        raw_price, fetched_ccy = fetch_live_price(ticker)
        # If yfinance returned the price in a different currency than stored,
        # convert to stored currency using spot FX so ret_pct is apples-to-apples
        if raw_price is not None and fetched_ccy and fetched_ccy != ccy:
            fx_fetched = fetch_fx_now(fetched_ccy)   # 1 EUR = X fetched_ccy
            fx_stored  = fetch_fx_now(ccy)            # 1 EUR = X stored_ccy
            # convert: price_stored = price_fetched / fx_fetched * fx_stored
            curr_price = raw_price / fx_fetched * fx_stored if fx_fetched else raw_price
        else:
            curr_price = raw_price

    result["current_price"] = curr_price
    result["ret_now_pct"]   = ret_pct(curr_price)
    result["pnl_now_eur"]   = pnl_eur(curr_price, fx_now)
    # EUR-denominated prices for display
    result["buy_eur"]     = round(buy_price / fx_entry, 2)
    result["current_eur"] = round(curr_price / fx_now, 2) if curr_price is not None else None

    # ── 1-week exit ───────────────────────────────────────────────────────────
    if status == "CLOSED" and trade.get("actual_sell_date"):
        exit_d = datetime.strptime(trade["actual_sell_date"], "%Y-%m-%d").date()
        # Use stored exit_price (same currency as buy_price) — NOT history,
        # which returns the stock's native currency and causes FX mismatch
        # when buy_price was stored as EUR-converted (e.g. MCHP stored in EUR).
        p1 = float(trade["exit_price"]) if trade.get("exit_price") else (
             price_on_or_before(hist, exit_d) if hist is not None else None)
        d1 = exit_d
    elif hist is not None:
        p1, d1 = price_n_trading_days_after(hist, entry_date, 5)
    else:
        p1, d1 = None, None

    result["price_1wk"]   = p1
    result["date_1wk"]    = d1.strftime("%Y-%m-%d") if d1 else None
    result["ret_1wk_pct"] = ret_pct(p1)
    fx1 = fetch_fx_on_date(ccy, d1) if (p1 is not None and d1 is not None and ccy != "EUR") else fx_now
    result["pnl_1wk_eur"] = pnl_eur(p1, fx1) if p1 else None

    # ── 2-week exit ───────────────────────────────────────────────────────────
    if hist is not None and status != "CLOSED":
        p2, d2 = price_n_trading_days_after(hist, entry_date, 10)
    else:
        p2, d2 = None, None

    result["price_2wk"]   = p2
    result["date_2wk"]    = d2.strftime("%Y-%m-%d") if d2 else None
    result["ret_2wk_pct"] = ret_pct(p2)
    fx2 = fetch_fx_on_date(ccy, d2) if (p2 is not None and d2 is not None and ccy != "EUR") else fx_now
    result["pnl_2wk_eur"] = pnl_eur(p2, fx2) if p2 else None

    # ── max_dd_1wk (live, for display — already persisted by enrich_trades) ───
    if not result.get("max_dd_1wk") and hist is not None:
        trading_days = sorted(d for d in hist.index if d > entry_date)
        if len(trading_days) >= 5:
            prices = [float(hist[d]) for d in trading_days[:5]]
            min_p  = min(prices)
            result["max_dd_1wk"] = round((min_p - buy_price) / buy_price * 100, 2)

    return result


# ── RISK MODULE ───────────────────────────────────────────────────────────────

MAX_POSITIONS   = 5
TRIM_THRESHOLD  = 5.0    # % → trim half, trail rest
CLOSE_THRESHOLD = 8.0    # % → close position
STOP_THRESHOLD  = -3.0   # % → stop breached

def _risk_data(rows: list[dict]) -> dict:
    """Compute all risk metrics from enriched rows."""
    open_rows = [r for r in rows if r.get("status") == "OPEN"]
    total_open = len(open_rows)

    # PnL tiers
    close_now, trim_now, watching, stops_breached = [], [], [], []
    for r in open_rows:
        pct = r.get("ret_now_pct")
        if pct is None:
            continue
        if pct >= CLOSE_THRESHOLD:
            close_now.append(r)
        elif pct >= TRIM_THRESHOLD:
            trim_now.append(r)
        elif pct > 0:
            watching.append(r)
        if pct <= STOP_THRESHOLD:
            stops_breached.append(r)

    # Sector breakdown
    sector_counts: dict[str, int] = {}
    for r in open_rows:
        sec = str(r.get("sector", "")).strip() or "Unknown"
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    # Portfolio weekly drawdown flag
    total_invested = sum(float(r.get("investment_eur", 0)) for r in open_rows)
    pnls = [r["pnl_now_eur"] for r in open_rows if r.get("pnl_now_eur") is not None]
    total_pnl = sum(pnls) if pnls else 0.0
    weekly_dd_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
    weekly_dd_flag = weekly_dd_pct <= -5.0

    return dict(
        open_rows=open_rows, total_open=total_open,
        close_now=close_now, trim_now=trim_now, watching=watching,
        stops_breached=stops_breached, sector_counts=sector_counts,
        total_invested=total_invested, total_pnl=total_pnl,
        weekly_dd_pct=weekly_dd_pct, weekly_dd_flag=weekly_dd_flag,
    )


def print_risk(rows: list[dict]):
    """Terminal risk module: python3 show_tracker.py risk"""
    d = _risk_data(rows)
    now_str = datetime.now().strftime("%Y-%m-%d  %H:%M")
    W2 = 80

    print()
    print("╔" + "═"*(W2-2) + "╗")
    print("║" + BOLD(f"  🛡️  RISK MODULE  ·  {now_str}").ljust(W2+8) + "║")
    print("╠" + "═"*(W2-2) + "╣")

    # Rule 1: Position limit
    pos_line = f"  POSITIONS: {d['total_open']} open"
    if d['total_open'] > MAX_POSITIONS:
        pos_line += f"  ← " + RED(f"OVER LIMIT ({MAX_POSITIONS} max)")
    else:
        pos_line += f"  ← " + GRN(f"OK (max {MAX_POSITIONS})")
    print("║" + pos_line.ljust(W2+8) + "║")

    # Weekly drawdown flag
    if d['weekly_dd_flag']:
        dd_line = f"  WEEKLY DD: {d['weekly_dd_pct']:+.1f}%  ← " + RED("⚠ PAUSE NEW ENTRIES (>5% portfolio loss)")
    else:
        pnl_s = GRN(f"{d['weekly_dd_pct']:+.1f}%") if d['total_pnl'] >= 0 else RED(f"{d['weekly_dd_pct']:+.1f}%")
        dd_line = f"  WEEKLY DD: {d['weekly_dd_pct']:+.1f}%  ← " + GRN("OK")
    print("║" + dd_line.ljust(W2+12) + "║")

    print("╠" + "═"*(W2-2) + "╣")

    # Rule 3: Profit take alerts
    print("║" + BOLD("  TAKE PROFIT ALERTS").ljust(W2+8) + "║")
    if not d['close_now'] and not d['trim_now']:
        print("║" + DIM("  — none —").ljust(W2) + "║")
    for r in sorted(d['close_now'], key=lambda x: x.get('ret_now_pct', 0), reverse=True):
        line = f"  🔴 CLOSE   {r['ticker']:<8}  {r.get('ret_now_pct', 0):+.1f}%  (>{CLOSE_THRESHOLD}% target reached)"
        print("║" + RED(line).ljust(W2+20) + "║")
    for r in sorted(d['trim_now'], key=lambda x: x.get('ret_now_pct', 0), reverse=True):
        line = f"  🟡 TRIM ½  {r['ticker']:<8}  {r.get('ret_now_pct', 0):+.1f}%  (>{TRIM_THRESHOLD}% — trail remaining)"
        print("║" + YLW(line).ljust(W2+20) + "║")

    print("╠" + "═"*(W2-2) + "╣")

    # Rule 4: Stop breaches
    print("║" + BOLD("  STOP LOSS BREACHED  (<-3%)").ljust(W2+8) + "║")
    if not d['stops_breached']:
        print("║" + GRN("  — none —").ljust(W2+8) + "║")
    for r in sorted(d['stops_breached'], key=lambda x: x.get('ret_now_pct', 0)):
        line = f"  🔴 STOP    {r['ticker']:<8}  {r.get('ret_now_pct', 0):+.1f}%  SL={r.get('stop_loss_price', '─')}"
        print("║" + RED(line).ljust(W2+20) + "║")

    print("╠" + "═"*(W2-2) + "╣")

    # Rule 2: Sector breakdown
    print("║" + BOLD("  SECTOR BREAKDOWN").ljust(W2+8) + "║")
    max_count = max(d['sector_counts'].values()) if d['sector_counts'] else 1
    for sec, cnt in sorted(d['sector_counts'].items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / max_count * 20)
        warn = RED("  ⚠ concentrated") if cnt >= 3 else ""
        line = f"  {sec:<20}  {cnt:>2}  {bar}"
        print("║" + (line + warn).ljust(W2+20) + "║")

    print("╠" + "═"*(W2-2) + "╣")

    # All open P&L sorted
    print("║" + BOLD("  ALL OPEN POSITIONS  (sorted by P&L)").ljust(W2+8) + "║")
    for r in sorted(d['open_rows'], key=lambda x: (x.get('ret_now_pct') or 0), reverse=True):
        pct = r.get('ret_now_pct')
        peur = r.get('pnl_now_eur')
        pct_s = _pct(pct)
        eur_s = _eur(peur)
        td = r.get('target_exit_date', '─') or '─'
        print(f"║    {r['ticker']:<8}  {pct_s}  {eur_s}   exit→{td}  [{r.get('sector','─')[:14]}]")

    print("╠" + "═"*(W2-2) + "╣")
    total_s = GRN(f"+€{d['total_pnl']:.0f}") if d['total_pnl'] >= 0 else RED(f"-€{abs(d['total_pnl']):.0f}")
    print(f"║  Total unrealised P&L: {total_s}   Invested: €{d['total_invested']:.0f}")
    print("╚" + "═"*(W2-2) + "╝\n")


def _risk_html_section(rows: list[dict]) -> str:
    """Returns an HTML <section> for the risk module, to embed in dashboard."""
    d = _risk_data(rows)

    pos_cls   = "neg" if d['total_open'] > MAX_POSITIONS else "pos"
    pos_label = f"⚠ {d['total_open']}/{MAX_POSITIONS} — OVER LIMIT" if d['total_open'] > MAX_POSITIONS else f"{d['total_open']}/{MAX_POSITIONS} — OK"

    dd_cls   = "neg" if d['weekly_dd_flag'] else "pos"
    dd_label = f"⚠ {d['weekly_dd_pct']:+.1f}% — PAUSE NEW ENTRIES" if d['weekly_dd_flag'] else f"{d['weekly_dd_pct']:+.1f}% — OK"

    total_cls = "pos" if d['total_pnl'] >= 0 else "neg"
    total_s   = f"+€{d['total_pnl']:.0f}" if d['total_pnl'] >= 0 else f"-€{abs(d['total_pnl']):.0f}"

    # Take profit rows
    tp_rows = ""
    for r in sorted(d['close_now'] + d['trim_now'], key=lambda x: x.get('ret_now_pct', 0), reverse=True):
        pct = r.get('ret_now_pct', 0)
        if pct >= CLOSE_THRESHOLD:
            action = '<span class="risk-close">🔴 CLOSE</span>'
        else:
            action = '<span class="risk-trim">🟡 TRIM ½</span>'
        tp_rows += f"<tr><td class='ticker'>{r['ticker']}</td><td class='pos'>+{pct:.1f}%</td><td>{action}</td><td>{r.get('target_exit_date','─')}</td></tr>"
    if not tp_rows:
        tp_rows = "<tr><td colspan='4' style='color:var(--dim)'>— no take-profit alerts —</td></tr>"

    # Stop breach rows
    sl_rows = ""
    for r in sorted(d['stops_breached'], key=lambda x: x.get('ret_now_pct', 0)):
        pct = r.get('ret_now_pct', 0)
        sl_rows += f"<tr><td class='ticker'>{r['ticker']}</td><td class='neg'>{pct:.1f}%</td><td class='neg'>🔴 STOP BREACHED</td><td>{r.get('stop_loss_price','─')}</td></tr>"
    if not sl_rows:
        sl_rows = "<tr><td colspan='4' style='color:var(--pos)'>— no stop breaches —</td></tr>"

    # Sector breakdown
    max_c = max(d['sector_counts'].values()) if d['sector_counts'] else 1
    sec_rows = ""
    for sec, cnt in sorted(d['sector_counts'].items(), key=lambda x: -x[1]):
        bar_w = int(cnt / max_c * 120)
        warn  = " ⚠" if cnt >= 3 else ""
        cls   = "neg" if cnt >= 3 else "dim"
        sec_rows += f"""<tr>
          <td>{sec}</td>
          <td style="text-align:center">{cnt}</td>
          <td><div style="height:10px;width:{bar_w}px;background:var(--accent);border-radius:3px;display:inline-block"></div></td>
          <td class="{cls}">{warn}</td>
        </tr>"""

    # All positions P&L
    pos_rows = ""
    for r in sorted(d['open_rows'], key=lambda x: (x.get('ret_now_pct') or 0), reverse=True):
        pct  = r.get('ret_now_pct')
        peur = r.get('pnl_now_eur')
        pct_cls = "pos" if (pct or 0) >= 0 else "neg"
        pct_s   = f"+{pct:.1f}%" if pct is not None and pct >= 0 else (f"{pct:.1f}%" if pct is not None else "─")
        eur_s   = f"+€{peur:.0f}" if peur is not None and peur >= 0 else (f"-€{abs(peur):.0f}" if peur is not None else "─")
        sec     = str(r.get('sector','─'))[:16] or '─'
        td      = r.get('target_exit_date','─') or '─'
        pos_rows += f"<tr><td class='ticker'>{r['ticker']}</td><td class='{pct_cls}'>{pct_s}</td><td class='{pct_cls}'>{eur_s}</td><td>{sec}</td><td>{td}</td></tr>"

    return f"""
    <section class="risk-module">
      <h2>🛡️ RISK MODULE</h2>
      <div class="risk-kpi-row">
        <div class="risk-kpi"><div class="kpi-label">Positions</div><div class="kpi-value {pos_cls}">{pos_label}</div></div>
        <div class="risk-kpi"><div class="kpi-label">Portfolio P&L</div><div class="kpi-value {total_cls}">{total_s} / €{d['total_invested']:.0f}</div></div>
        <div class="risk-kpi"><div class="kpi-label">Weekly DD Flag</div><div class="kpi-value {dd_cls}">{dd_label}</div></div>
      </div>
      <div class="risk-grid">
        <div class="risk-panel">
          <div class="risk-panel-title">Take Profit Alerts</div>
          <table><thead><tr><th>Ticker</th><th>Now%</th><th>Action</th><th>Exit Date</th></tr></thead>
          <tbody>{tp_rows}</tbody></table>
        </div>
        <div class="risk-panel">
          <div class="risk-panel-title">Stop Loss Breaches</div>
          <table><thead><tr><th>Ticker</th><th>Now%</th><th>Status</th><th>SL Price</th></tr></thead>
          <tbody>{sl_rows}</tbody></table>
        </div>
        <div class="risk-panel">
          <div class="risk-panel-title">Sector Concentration</div>
          <table><thead><tr><th>Sector</th><th>#</th><th>Bar</th><th></th></tr></thead>
          <tbody>{sec_rows}</tbody></table>
        </div>
        <div class="risk-panel">
          <div class="risk-panel-title">All Open Positions (by P&L)</div>
          <table><thead><tr><th>Ticker</th><th>Now%</th><th>P&L €</th><th>Sector</th><th>Exit Date</th></tr></thead>
          <tbody>{pos_rows}</tbody></table>
        </div>
      </div>
    </section>"""


# ── DISPLAY ───────────────────────────────────────────────────────────────────

W = 112

def print_tracker(rows: list[dict], filter_status: Optional[str] = None):
    if filter_status:
        rows = [r for r in rows if r["status"] == filter_status]

    now      = datetime.now().strftime("%Y-%m-%d  %H:%M")
    open_n   = sum(1 for r in rows if r["status"] == "OPEN")
    closed_n = sum(1 for r in rows if r["status"] == "CLOSED")
    total_inv= sum(float(r["investment_eur"]) for r in rows)

    all_now   = [r["pnl_now_eur"] for r in rows if r.get("pnl_now_eur") is not None]
    total_pnl = sum(all_now) if all_now else None

    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + f"  📊  TRADE TRACKER  ·  {now}  ·  {open_n} open  ·  {closed_n} closed".ljust(W-2) + "║")
    inv_line = f"  Total invested: €{total_inv:.0f}"
    if total_pnl is not None:
        pnl_s = (GRN(f"+€{total_pnl:.0f}") if total_pnl >= 0 else RED(f"-€{abs(total_pnl):.0f}"))
        inv_line += f"  ·  Total P&L now: {pnl_s}"
    print("║" + inv_line.ljust(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")

    if not rows:
        print(DIM("  No trades found. Add one with: python3 show_tracker.py add"))
        return

    strategies = list(dict.fromkeys(r["strategy"] for r in rows))

    for strat in strategies:
        strat_rows = [r for r in rows if r["strategy"] == strat]
        label = {
            "momentum":             "🟢  MOMENTUM  (O'Neil / IBD crossover signals)",
            "breakout":             "🔭  BREAKOUT  (VCP / coil pre-breakout)",
            "pocket_pivot":         "🟠  POCKET PIVOT  (Morales & Kacher)",
            "connors_rsi2":         "🔵  CONNORS RSI(2)  (mean reversion in uptrend)",
            "ema_ribbon":           "🟣  EMA RIBBON  (8/13/21/34/55 expansion pullback)",
            "nr7":                  "⚡  NR7  (Toby Crabel — narrowest range compression)",
            "bb_squeeze":           "🔲  BB SQUEEZE  (TTM Squeeze — Bollinger / John Carter)",
            "high_tight_flag":      "🚀  HIGH TIGHT FLAG  (Minervini / O'Neil — pole + flag)",
            "analyst_upgrade":      "📊  ANALYST UPGRADE  (≥3 firms upgrade in 5 days, tier-1 required)",
            "signal_velocity":      "⚙️   SIGNAL VELOCITY  (TV-style indicator convergence acceleration)",
            "chokepoint_inflection":"🌐  CHOKEPOINT INFLECTION  (macro event → commodity spike → stock lag)",
            "stage4_short":         "🔻  STAGE 4 SHORT  (Weinstein/Minervini — confirmed distribution)",
            "defensive_rotation":   "🛡️   DEFENSIVE ROTATION  (Faber — sector ETF outperforms SPY → stock leaders)",
            "cup_handle":           "☕  CUP & HANDLE  (O'Neil / IBD — rounded base + tight handle at pivot)",
            "power_earnings_gap":   "⚡  POWER EARNINGS GAP  (Gil Morales — 8%+ gap on earnings, 2× volume, gap held)",
        }.get(strat, strat.upper())
        print(f"\n  {BOLD(label)}")
        print()

        hdr = (f"  {'#':>2}  {'TICKER':<8}  {'MKT':<4}  {'COMPANY':<22}  {'ENTRY':<10}  "
               f"{'BUY €':>7}  {'NOW €':>7}  {'QTY':>6}  {'INV€':>6}  "
               f"{'1WK%':>6}  {'1WK€':>7}  "
               f"{'2WK%':>6}  {'2WK€':>7}  "
               f"{'NOW%':>6}  {'NOW€':>7}  {'STATUS':<8}  {'TYPE':<8}  STRAT")
        print(BOLD(hdr))
        print("  " + "─"*(W-2))

        for r in strat_rows:
            buy_e = f"€{r['buy_eur']:.2f}"    if r.get("buy_eur")     else f"{float(r['buy_price']):.2f}"
            now_e = f"€{r['current_eur']:.2f}" if r.get("current_eur") else "─"
            ticker_s = BOLD(f"{r['ticker']:<8}")
            co_s     = f"{str(r['company'])[:22]:<22}"

            tt = r.get("trade_type", "practice")
            type_badge = GRN("REAL    ") if tt == "real" else YLW("PRACTICE")
            strat_val = r.get("strategy", "")
            _strat_badges = {
                "momentum":              CYN("MNTM    "),
                "breakout":              MAG("BRKOUT  "),
                "pocket_pivot":          _c("33","PP      "),
                "connors_rsi2":          _c("36","RSI2    "),
                "ema_ribbon":            _c("35","EMARIBN "),
                "nr7":                   _c("33","NR7     "),
                "bb_squeeze":            _c("36","BBSQZ   "),
                "high_tight_flag":       _c("32","HTF     "),
                "analyst_upgrade":       _c("34","ANUPGRD "),
                "signal_velocity":       _c("35","SIGVEL  "),
                "chokepoint_inflection": _c("31","CHKPNT  "),
                "stage4_short":          _c("31","S4SHORT "),
                "defensive_rotation":    _c("32","DEFROT  "),
                "cup_handle":            _c("33","C&H     "),
                "power_earnings_gap":    _c("33","PEG     "),
            }
            strat_s = _strat_badges.get(strat_val, DIM(f"{strat_val[:8]:<8}"))
            inv_e = f"€{float(r['investment_eur']):.0f}" if r.get("investment_eur") else "─"
            mkt_s = DIM(f"{ticker_market(r['ticker']):<4}")
            row_line = (f"  {r['id']:>2}  {ticker_s}  {mkt_s}  {co_s}  {r['entry_date']:<10}  "
                        f"{buy_e:>7}  {now_e:>7}  {float(r['qty']):>6.2f}  {inv_e:>6}  "
                        f"{_pct(r.get('ret_1wk_pct')):>6}  {_eur(r.get('pnl_1wk_eur')):>7}  "
                        f"{_pct(r.get('ret_2wk_pct')):>6}  {_eur(r.get('pnl_2wk_eur')):>7}  "
                        f"{_pct(r.get('ret_now_pct')):>6}  {_eur(r.get('pnl_now_eur')):>7}  "
                        f"{_status(r['status'])}  {type_badge}  {strat_s}")
            print(row_line)

            # Sub-row 1: signals + exit info + stop/target dates
            sub1_parts = []
            sig_text = str(r.get("signals", "")).strip()
            if sig_text:
                sub1_parts.append(CYN(sig_text[:65]))
            if r.get("stop_loss_price"):
                sub1_parts.append(RED(f"SL {r['stop_loss_price']}"))
            if r.get("target_exit_date"):
                sub1_parts.append(DIM(f"exit→{r['target_exit_date']}"))
            if r["status"] == "CLOSED" and r.get("actual_sell_date"):
                exit_s = f"sold {r['actual_sell_date']} @ {r.get('exit_price','─')}"
                if r.get("exit_reason"):
                    exit_s += f"  [{r['exit_reason']}]"
                sub1_parts.append(DIM(exit_s))
            if sub1_parts:
                print("     " + DIM("  ·  ").join(sub1_parts))

            # Sub-row 2: entry analytics — only show populated fields
            reg = r.get("market_regime_entry", "")
            sec = str(r.get("sector", "")).strip()[:16]
            try:
                dd = float(r.get("max_dd_1wk") or "")
            except (ValueError, TypeError):
                dd = None
            ana = []
            if r.get("rsi_at_entry"):        ana.append(f"RSI {r['rsi_at_entry']}")
            if r.get("adx_at_entry"):        ana.append(f"ADX {r['adx_at_entry']}")
            if r.get("minervini_at_entry"):  ana.append(f"M {r['minervini_at_entry']}/8")
            if r.get("vol_ratio_entry"):     ana.append(f"Vol× {r['vol_ratio_entry']}")
            if reg:                          ana.append(_regime(reg))
            if sec:                          ana.append(f"[{sec}]")
            if dd is not None:
                dd_s = RED(f"MaxDD {dd:.1f}%") if dd < 0 else GRN(f"MaxDD {dd:.1f}%")
                ana.append(dd_s)
            if ana:
                print(DIM("     ") + DIM("  ·  ").join(ana))

            # Sub-row 3: exit analytics (only if closed and populated)
            if r["status"] == "CLOSED" and r.get("rsi_at_exit"):
                ex_ana = []
                if r.get("rsi_at_exit"):      ex_ana.append(f"RSI {r['rsi_at_exit']}")
                if r.get("adx_at_exit"):      ex_ana.append(f"ADX {r['adx_at_exit']}")
                if r.get("minervini_at_exit"): ex_ana.append(f"M {r['minervini_at_exit']}/8")
                if r.get("vol_ratio_exit"):    ex_ana.append(f"Vol× {r['vol_ratio_exit']}")
                if ex_ana:
                    print(DIM("     exit→  ") + DIM("  ·  ").join(ex_ana))
            print()

        # Strategy sub-total
        s_pnls = [r["pnl_now_eur"] for r in strat_rows if r.get("pnl_now_eur") is not None]
        if s_pnls:
            sp = sum(s_pnls)
            print(f"  {DIM(strat + ' total now:')}  {_eur(sp)}")

    print(f"\n  " + "─"*(W-2))
    print(DIM("  Columns: 1WK%/2WK% = 5/10 trading days.  NOW% = unrealised.  "
              "EUR P&L accounts for FX at exit date."))
    print(DIM("  MaxDD = worst intra-week close vs entry price.  "
              "Regime = SPY vs 50 SMA at entry."))
    print(DIM("  To add: python3 show_tracker.py add"))
    print(DIM("  To close a trade: python3 show_tracker.py close"))
    print(DIM("  Risk dashboard:   python3 show_tracker.py risk"))
    print("╚" + "═"*(W-2) + "╝\n")


# ── HTML DASHBOARD ────────────────────────────────────────────────────────────

def _fmt_cell(v, is_pct=False, is_eur=False):
    """Return (display_str, css_class) for a numeric value."""
    if v is None or v == "": return "─", ""
    try:
        f = float(v)
    except (ValueError, TypeError):
        return str(v), ""
    if is_pct or is_eur:
        css = "pos" if f >= 0 else "neg"
        sign = "+" if f >= 0 else ""
        if is_eur:
            return f"{sign}€{f:.0f}", css
        return f"{sign}{f:.1f}%", css
    return str(v), ""

def _kpi_html(label, rows_subset):
    inv  = sum(float(r["investment_eur"]) for r in rows_subset)
    pnls = [r["pnl_now_eur"] for r in rows_subset if r.get("pnl_now_eur") is not None]
    pnl  = sum(pnls) if pnls else None
    pc   = "pos" if (pnl or 0) >= 0 else "neg"
    ps   = (f"+€{pnl:.0f}" if pnl and pnl >= 0 else f"-€{abs(pnl):.0f}" if pnl else "─")
    on   = sum(1 for r in rows_subset if r["status"] == "OPEN")
    cl   = sum(1 for r in rows_subset if r["status"] == "CLOSED")
    return f"""
    <div class="kpi-group">
      <div class="kpi-group-label">{label}</div>
      <div class="kpi-row">
        <div class="kpi-box"><div class="kpi-label">Invested</div><div class="kpi-value">€{inv:.0f}</div></div>
        <div class="kpi-box"><div class="kpi-label">P&L Now</div><div class="kpi-value {pc}">{ps}</div></div>
        <div class="kpi-box"><div class="kpi-label">Open</div><div class="kpi-value">{on}</div></div>
        <div class="kpi-box"><div class="kpi-label">Closed</div><div class="kpi-value">{cl}</div></div>
      </div>
    </div>"""

def generate_dashboard(rows: list[dict], filter_status=None, show_risk=False):
    """Write dashboard.html and open in default browser."""
    if filter_status:
        rows = [r for r in rows if r["status"] == filter_status]

    now_str      = datetime.now().strftime("%Y-%m-%d %H:%M")
    practice_rows = [r for r in rows if r.get("trade_type", "practice") == "practice"]
    real_rows     = [r for r in rows if r.get("trade_type", "practice") == "real"]

    kpi_practice = _kpi_html("🟡 PRACTICE", practice_rows) if practice_rows else ""
    kpi_real     = _kpi_html("🟢 REAL",     real_rows)     if real_rows     else ""
    risk_section = _risk_html_section(rows) if show_risk else ""

    def trade_rows_html(strategy_rows):
        html = ""
        for r in strategy_rows:
            status_cls = "open" if r["status"] == "OPEN" else "closed"

            def p(key, is_pct=False, is_eur=False):
                val = r.get(key)
                s, cls = _fmt_cell(val, is_pct=is_pct, is_eur=is_eur)
                if cls:
                    return f'<span class="{cls}">{s}</span>'
                return s

            # EUR prices
            buy_e   = f"€{r['buy_eur']:.2f}"     if r.get("buy_eur")     else f"€{float(r['buy_price']):.2f}"
            now_e   = f"€{r['current_eur']:.2f}"  if r.get("current_eur") else "─"
            qty_s   = f"{float(r['qty']):.2f}"
            invest_s = f"€{float(r['investment_eur']):.0f}" if r.get("investment_eur") else "─"

            regime    = r.get("market_regime_entry", "")
            regime_cls= "pos" if regime == "BULL" else ("neg" if regime == "BEAR" else "")
            regime_s  = f'<span class="{regime_cls}">{regime or "─"}</span>' if regime_cls else (regime or "─")

            try:
                dd = float(r.get("max_dd_1wk") or "")
                dd_cls = "neg" if dd < 0 else "pos"
                dd_s   = f'<span class="{dd_cls}">{dd:+.1f}%</span>'
            except (ValueError, TypeError):
                dd_s = "─"

            signals   = r.get("signals", "") or "─"
            exit_info = ""
            if r["status"] == "CLOSED" and r.get("actual_sell_date"):
                reason = f" ({r['exit_reason']})" if r.get("exit_reason") else ""
                exit_info = f"Closed {r['actual_sell_date']} @ {r.get('exit_price','─')}{reason}"

            mkt_label = ticker_market(r['ticker'])
            html += f"""
            <tr class="trade-row {status_cls}">
              <td>{r['id']}</td>
              <td class="ticker">{r['ticker']}</td>
              <td><span class="mkt-badge mkt-{mkt_label.lower()}">{mkt_label}</span></td>
              <td>{str(r.get('company',''))[:24]}</td>
              <td>{r['entry_date']}</td>
              <td>{str(r.get('sector',''))[:16] or '─'}</td>
              <td>{regime_s}</td>
              <td>{buy_e}</td>
              <td>{now_e}</td>
              <td>{qty_s}</td>
              <td>{invest_s}</td>
              <td>{p('ret_1wk_pct', is_pct=True)}</td>
              <td>{p('pnl_1wk_eur', is_eur=True)}</td>
              <td>{p('ret_2wk_pct', is_pct=True)}</td>
              <td>{p('pnl_2wk_eur', is_eur=True)}</td>
              <td>{p('ret_now_pct', is_pct=True)}</td>
              <td>{p('pnl_now_eur', is_eur=True)}</td>
              <td>{r.get('rsi_at_entry') or '─'}</td>
              <td>{r.get('adx_at_entry') or '─'}</td>
              <td>{r.get('minervini_at_entry') or '─'}/8</td>
              <td>{r.get('vol_ratio_entry') or '─'}</td>
              <td>{dd_s}</td>
              <td class="signals">{signals}</td>
              <td class="{status_cls}-badge">{r['status']}</td>
              <td class="type-{'real' if r.get('trade_type','practice') == 'real' else 'practice'}-badge">{(r.get('trade_type') or 'practice').upper()}</td>
              <td class="strat-{'momentum' if r.get('strategy')=='momentum' else 'breakout'}-badge">{(r.get('strategy') or '─').upper()}</td>
            </tr>
            <tr class="detail-row">
              <td colspan="26" class="detail-cell">
                1wk→{r.get('date_1wk') or 'not yet'} &nbsp;·&nbsp;
                2wk→{r.get('date_2wk') or 'not yet'}
                {'&nbsp;·&nbsp;' + exit_info if exit_info else ''}
                {'&nbsp;·&nbsp;Exit: RSI=' + str(r.get('rsi_at_exit','─')) +
                 ' ADX=' + str(r.get('adx_at_exit','─')) +
                 ' M=' + str(r.get('minervini_at_exit','─')) + '/8'
                 if r.get('rsi_at_exit') else ''}
              </td>
            </tr>"""
        return html

    strategies   = list(dict.fromkeys(r["strategy"] for r in rows))
    sections_html = ""
    for strat in strategies:
        label = {
            "momentum":              "🟢 MOMENTUM  (O'Neil / IBD)",
            "breakout":              "🔭 BREAKOUT  (VCP / Minervini)",
            "pocket_pivot":          "🟠 POCKET PIVOT  (Morales & Kacher)",
            "connors_rsi2":          "🔵 CONNORS RSI(2)  (mean reversion)",
            "ema_ribbon":            "🟣 EMA RIBBON  (8/13/21/34/55)",
            "nr7":                   "⚡ NR7  (Toby Crabel)",
            "bb_squeeze":            "🔲 BB SQUEEZE  (TTM / John Carter)",
            "high_tight_flag":       "🚀 HIGH TIGHT FLAG  (Minervini / O'Neil)",
            "analyst_upgrade":       "📊 ANALYST UPGRADE  (≥3 firms, tier-1)",
            "signal_velocity":       "⚙️ SIGNAL VELOCITY  (indicator convergence)",
            "chokepoint_inflection": "🌐 CHOKEPOINT INFLECTION  (macro → commodity → stock)",
            "stage4_short":          "🔻 STAGE 4 SHORT  (Weinstein / Minervini)",
            "defensive_rotation":    "🛡️ DEFENSIVE ROTATION  (Faber sector rotation)",
            "cup_handle":            "☕ CUP & HANDLE  (O'Neil / IBD)",
            "power_earnings_gap":    "⚡ POWER EARNINGS GAP  (Gil Morales)",
        }.get(strat, strat.upper())
        s_rows    = [r for r in rows if r["strategy"] == strat]
        s_pnls    = [r["pnl_now_eur"] for r in s_rows if r.get("pnl_now_eur") is not None]
        s_total   = sum(s_pnls) if s_pnls else None
        s_cls     = "pos" if (s_total or 0) >= 0 else "neg"
        s_str     = (f"+€{s_total:.0f}" if s_total and s_total >= 0 else
                     f"-€{abs(s_total):.0f}" if s_total else "")

        sections_html += f"""
        <h2>{label} <span class="strat-pnl {s_cls}">{s_str}</span></h2>
        <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th title="Trade ID">#</th>
              <th title="Exchange ticker symbol">Ticker</th>
              <th title="Market / exchange">Mkt</th>
              <th title="Company name">Company</th>
              <th title="Date position was opened">Entry</th>
              <th title="GICS sector from yfinance">Sector</th>
              <th title="BULL = SPY above 50-day SMA on entry date; BEAR = below">Regime</th>
              <th title="Entry price converted to EUR">Buy €</th>
              <th title="Current live price converted to EUR">Now €</th>
              <th title="Number of shares held">Qty</th>
              <th title="Total EUR invested in this position">Invest€</th>
              <th title="Return % after 5 trading days">1WK%</th>
              <th title="EUR profit/loss after 5 trading days">1WK€</th>
              <th title="Return % after 10 trading days">2WK%</th>
              <th title="EUR profit/loss after 10 trading days">2WK€</th>
              <th title="Current unrealised return %">NOW%</th>
              <th title="Current unrealised EUR profit/loss">NOW€</th>
              <th title="RSI(14) on entry date — 50–70 is healthy momentum zone">RSI</th>
              <th title="ADX(14) on entry — ≥22 required, ≥25 = strong trend">ADX</th>
              <th title="Minervini Trend Template score out of 8 — ≥6 for momentum, ≥5 for breakout">M</th>
              <th title="Volume on entry ÷ 20-day avg — &gt;1.5 = institutional interest">VolX</th>
              <th title="Worst close in first 5 trading days vs entry price (%)">MaxDD</th>
              <th title="Technical signals that fired at entry">Signals</th>
              <th title="OPEN or CLOSED">Status</th>
              <th title="REAL = live money; PRACTICE = paper trade">Type</th>
              <th title="Scanner strategy used: MOMENTUM or BREAKOUT">Strat</th>
            </tr>
          </thead>
          <tbody>
            {trade_rows_html(s_rows)}
          </tbody>
        </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trade Tracker</title>
<style>
  :root {{
    --bg:      #0d0f18;
    --surface: #13151f;
    --card:    #181b28;
    --border:  #232637;
    --text:    #d4d6e3;
    --dim:     #555a72;
    --pos:     #34d399;
    --neg:     #f87171;
    --accent:  #818cf8;
    --warn:    #fbbf24;
    --muted:   #9ca3af;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'SF Mono','Fira Code','Cascadia Code',monospace; font-size: 11px; padding: 20px 24px; line-height: 1.45; }}

  /* ── Header ── */
  header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid var(--border); }}
  header h1 {{ font-size: 15px; font-weight: 700; letter-spacing: .03em; color: var(--text); }}
  header h1 span {{ color: var(--accent); }}
  .meta {{ color: var(--dim); font-size: 10px; margin-top: 3px; }}

  /* ── KPI cards ── */
  .kpi-section {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }}
  .kpi-group {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; }}
  .kpi-group-label {{ font-size: 9px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--dim); margin-bottom: 10px; }}
  .kpi-row {{ display: flex; gap: 10px; }}
  .kpi-box {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 8px 14px; min-width: 80px; }}
  .kpi-label {{ color: var(--dim); font-size: 9px; text-transform: uppercase; letter-spacing: .05em; }}
  .kpi-value {{ font-size: 16px; font-weight: 700; margin-top: 2px; }}

  /* ── Section headers ── */
  h2 {{ font-size: 11px; font-weight: 700; margin: 22px 0 8px; color: var(--muted); letter-spacing: .06em; text-transform: uppercase; display: flex; align-items: center; gap: 8px; }}
  h2::before {{ content: ''; display: block; width: 3px; height: 14px; background: var(--accent); border-radius: 2px; }}
  .strat-pnl {{ font-size: 11px; font-weight: 600; }}

  /* ── Table ── */
  .table-wrap {{ overflow-x: auto; border-radius: 7px; border: 1px solid var(--border); margin-bottom: 4px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead tr {{ background: var(--surface); }}
  th {{ padding: 6px 8px; text-align: right; color: var(--dim); font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; white-space: nowrap; border-bottom: 1px solid var(--border); }}
  th:nth-child(1),th:nth-child(2),th:nth-child(3),th:nth-child(4),th:nth-child(5),th:nth-child(6),th:nth-child(7) {{ text-align: left; }}
  td {{ padding: 5px 8px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; font-size: 11px; }}
  td:nth-child(1),td:nth-child(2),td:nth-child(3),td:nth-child(4),td:nth-child(5),td:nth-child(6),td:nth-child(7) {{ text-align: left; }}
  .trade-row:hover td {{ background: #1a1d2e; }}
  .trade-row.closed {{ opacity: 0.5; }}
  .detail-row td {{ font-size: 10px; color: var(--dim); padding: 1px 8px 7px; }}
  .detail-cell {{ text-align: left !important; }}

  /* ── Cells ── */
  .ticker {{ font-weight: 700; color: var(--accent); font-size: 11px; }}
  .signals {{ color: var(--dim); font-size: 10px; max-width: 160px; overflow: hidden; text-overflow: ellipsis; }}
  .pos {{ color: var(--pos); }}
  .neg {{ color: var(--neg); }}

  /* ── Badges ── */
  .open-badge   {{ color: var(--pos); font-weight: 700; font-size: 10px; }}
  .closed-badge {{ color: var(--dim); font-size: 10px; }}
  .real-badge     {{ background: #14532d; color: var(--pos); font-size: 9px; font-weight: 700; padding: 1px 5px; border-radius: 3px; }}
  .practice-badge {{ background: #451a03; color: var(--warn); font-size: 9px; font-weight: 700; padding: 1px 5px; border-radius: 3px; }}
  .strat-momentum-badge,.strat-breakout-badge {{ font-size: 9px; font-weight: 700; letter-spacing: .4px; color: var(--accent); }}
  .mkt-badge {{ display: inline-block; font-size: 9px; font-weight: 700; padding: 1px 5px; border-radius: 3px; letter-spacing: .04em; }}
  .mkt-us  {{ background:#1e3a5f; color:#60a5fa; }}
  .mkt-uk  {{ background:#3b1f2b; color:#f9a8d4; }}
  .mkt-de  {{ background:#1a2e1a; color:#86efac; }}
  .mkt-fr  {{ background:#1e1b3a; color:#a5b4fc; }}
  .mkt-nl  {{ background:#1e1b3a; color:#a5b4fc; }}
  .mkt-ch  {{ background:#2e1a1a; color:#fca5a5; }}
  .mkt-ca  {{ background:#1e2e20; color:#6ee7b7; }}

  /* ── Footer ── */
  .footer {{ margin-top: 24px; color: var(--dim); font-size: 10px; border-top: 1px solid var(--border); padding-top: 10px; line-height: 1.8; }}

  /* ── Risk Module ── */
  .risk-module {{ margin: 4px 0 20px; }}
  .risk-module h2 {{ font-size: 11px; }}
  .risk-kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
  .risk-kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 7px; padding: 10px 14px; min-width: 220px; }}
  .risk-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .risk-panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 7px; padding: 12px; overflow-x: auto; }}
  .risk-panel-title {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--dim); margin-bottom: 8px; }}
  .risk-panel table {{ width: 100%; border-collapse: collapse; }}
  .risk-panel th {{ font-size: 9px; font-weight: 700; text-transform: uppercase; color: var(--dim); padding: 3px 7px; border-bottom: 1px solid var(--border); text-align: left; }}
  .risk-panel td {{ font-size: 10px; padding: 4px 7px; border-bottom: 1px solid var(--border); }}
  .risk-close {{ color: var(--neg); font-weight: 700; }}
  .risk-trim  {{ color: var(--warn); font-weight: 700; }}
  .dim {{ color: var(--dim); }}
</style>
</head>
<body>
<header>
  <div>
    <h1>📊 Trade <span>Tracker</span></h1>
    <div class="meta">Updated: {now_str} &nbsp;·&nbsp; All prices live via yfinance &nbsp;·&nbsp; EUR P&L uses spot FX</div>
  </div>
</header>
<div class="kpi-section">
  {kpi_practice}
  {kpi_real}
</div>

{risk_section}

{sections_html}

<div class="footer">
  ⚠ Not financial advice.
  &nbsp;·&nbsp; M = Minervini Trend Template /8
  &nbsp;·&nbsp; MaxDD = worst intra-week close vs entry
  &nbsp;·&nbsp; Regime = SPY vs 50 SMA at entry
  &nbsp;·&nbsp; Returns in local currency %; EUR P&amp;L uses FX at exit date
</div>
</body>
</html>"""

    out = HERE / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    webbrowser.open(out.as_uri())
    print(f"  Dashboard → {out}")


# ── CLOSE TRADE ───────────────────────────────────────────────────────────────

def close_trade_interactive(args: list[str]):
    def _get(flag, prompt, default=""):
        for i, a in enumerate(args):
            if a == flag and i+1 < len(args): return args[i+1]
        val = input(f"  {prompt} [{default}]: ").strip()
        return val or default

    trades = load_trades()
    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    if not open_trades:
        print("\n  No open trades to close."); return

    print("\n  ── Close a Trade ──")
    print(f"\n  Open positions:")
    for t in open_trades:
        print(f"    #{t['id']:>3}  {t['ticker']:<10}  entry={t['entry_date']}  "
              f"strategy={t['strategy']}  buy={t['buy_price']} {t['currency']}")

    raw_id = _get("--id", "\n  Trade # to close").strip()
    try:
        trade_id = int(raw_id)
    except ValueError:
        print("  Invalid ID."); return

    trade = next((t for t in trades if str(t.get("id")) == str(trade_id)), None)
    if not trade:
        print(f"  Trade #{trade_id} not found."); return
    if trade.get("status") == "CLOSED":
        print(f"  Trade #{trade_id} is already CLOSED."); return

    ticker    = trade["ticker"]
    currency  = trade.get("currency", "USD")
    buy_price = float(trade.get("buy_price", 0))
    qty       = float(trade.get("qty", 0))
    fx_entry  = float(trade.get("fx_at_entry", 1))
    invest_eur = float(trade.get("investment_eur", 0))

    raw_date = _get("--date", "Close date (YYYY-MM-DD)", date.today().strftime("%Y-%m-%d"))
    sell_date = datetime.strptime(raw_date, "%Y-%m-%d").date()

    raw_price = _get("--price", f"Exit price (in {currency})")
    if raw_price.upper().startswith("EUR:"):
        fx_exit   = fetch_fx_on_date(currency, sell_date)
        eur_amt   = float(raw_price.split(":")[1])
        exit_price = round(eur_amt * fx_exit, 4)
        print(f"  → {eur_amt} EUR × {fx_exit:.4f} = {exit_price} {currency}")
    else:
        exit_price = float(raw_price)

    exit_reasons = ["1wk_auto", "2wk_auto", "manual_stop", "manual_target", "news_exit"]
    print(f"\n  Exit reason options: {', '.join(exit_reasons)}")
    exit_reason = _get("--reason", "Exit reason", "1wk_auto")

    # P&L summary
    pnl_ccy   = round((exit_price - buy_price) * qty, 2)
    pnl_pct   = round((exit_price - buy_price) / buy_price * 100, 2) if buy_price else 0
    fx_exit_r = fetch_fx_on_date(currency, sell_date)
    pnl_eur   = round(pnl_ccy / fx_exit_r, 2) if fx_exit_r else 0

    pnl_str = GRN(f"+{pnl_pct}%  +{pnl_ccy} {currency}  +€{pnl_eur}") if pnl_pct >= 0 \
              else RED(f"{pnl_pct}%  {pnl_ccy} {currency}  €{pnl_eur}")

    print(f"\n  ┌─ Close Summary ────────────────────────────────────────────")
    print(f"  │  Ticker:      {ticker}  (Trade #{trade_id})")
    print(f"  │  Entry:       {trade['entry_date']}  @  {buy_price} {currency}")
    print(f"  │  Exit:        {sell_date}  @  {exit_price} {currency}")
    print(f"  │  P&L:         {pnl_str}")
    print(f"  │  Exit reason: {exit_reason}")
    print(f"  └────────────────────────────────────────────────────────────")

    confirm = input("\n  Close this trade? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("  Cancelled."); return

    # Update trade record
    trade["status"]          = "CLOSED"
    trade["actual_sell_date"] = sell_date.strftime("%Y-%m-%d")
    trade["exit_price"]      = exit_price
    trade["exit_reason"]     = exit_reason

    # Auto-populate exit indicators
    print(f"\n  Fetching exit indicators for {ticker} on {sell_date}...", end=" ", flush=True)
    ind = fetch_indicators_at_date(ticker, sell_date)
    trade["rsi_at_exit"]       = ind.get("rsi", "")
    trade["adx_at_exit"]       = ind.get("adx", "")
    trade["minervini_at_exit"] = ind.get("minervini", "")
    trade["vol_ratio_exit"]    = ind.get("vol_ratio", "")
    print("✓")

    save_trades(trades)
    result = GRN("WIN ✅") if pnl_pct >= 0 else RED("LOSS ❌")
    print(f"\n  Trade #{trade_id} closed — {result}  {pnl_str}")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if args and args[0] == "add":
        add_trade_interactive(args[1:])
        return

    if args and args[0] == "close":
        close_trade_interactive(args[1:])
        return

    if args and args[0] == "risk":
        trades = load_trades()
        if not trades:
            print("\n  No trades yet.\n"); return
        print(f"\n  Loading prices...", flush=True)
        t0 = time.time()
        price_cache, fx_cache = {}, {}
        rows = [compute_row(t, price_cache, fx_cache) for t in trades]
        print(f"  Done in {time.time()-t0:.1f}s")
        print_risk(rows)
        generate_dashboard(rows, show_risk=True)
        return

    filter_status = None
    if "--open"   in args: filter_status = "OPEN"
    if "--closed" in args: filter_status = "CLOSED"

    trades = load_trades()
    if not trades:
        print("\n  No trades yet. Add one with:\n"
              "  python3 show_tracker.py add\n")
        return

    print(f"\n  Loading prices for {len(trades)} trade(s)...", flush=True)
    t0          = time.time()
    price_cache = {}
    fx_cache    = {}

    # Backfill any missing analytics (first-time enrichment for old trades)
    changed = enrich_trades(trades, price_cache)
    if changed:
        save_trades(trades)

    rows = [compute_row(t, price_cache, fx_cache) for t in trades]
    print(f"  Done in {time.time()-t0:.1f}s\n")

    print_risk(rows)
    print_tracker(rows, filter_status)

    if "--no-browser" not in args:
        generate_dashboard(rows, filter_status, show_risk=True)


if __name__ == "__main__":
    main()
