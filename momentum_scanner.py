#!/usr/bin/env python3
"""
Momentum Scanner  |  US + Europe + Canada
──────────────────────────────────────────
Catches stocks that have JUST entered momentum (crossovers within last 3 bars).
For stocks about to break out (pre-move), use breakout_scanner.py instead.

python3 momentum_scanner.py                # full scan + backtest
python3 momentum_scanner.py --no-backtest  # signals only (~5-10 min)
python3 momentum_scanner.py --fast         # 30 US stocks smoke-test
python3 momentum_scanner.py --legend       # column & signal definitions

First run / if you see 401 errors:
  pip3 install --upgrade yfinance

Cron (weekdays 6:30am):
  30 6 * * 1-5 python3 /path/to/momentum_scanner.py --no-backtest >> ~/momentum.log 2>&1
"""

import io, os, sys, time, random, warnings, logging, contextlib
from pathlib import Path
import requests
import numpy  as np
import pandas as pd
import yfinance as yf
from datetime           import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing             import Optional
from scanner_utils import _adx, _ema, _quiet, _rsi, _sma

warnings.filterwarnings("ignore")
# Suppress yfinance's own error/warning spam
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── CONFIG ────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS  = 400
HOLD_DAYS      = 5
MAX_WORKERS    = 25
MIN_PRICE      = 1.0       # lower for EU stocks (some trade < $5 in local currency)
MIN_AVG_VOL    = 100_000   # lower for EU; US stocks typically >> this
FRESH_WINDOW   = 3

# Hard entry filters
MIN_MINERVINI  = 6
MIN_FRESH      = 2
MIN_ADX        = 22
EXCLUDE_SECTORS = {"Utilities", "Real Estate"}

TOP_N   = 50
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; momentum-scanner/1.0)"}

# ── MARKET / BENCHMARK MAP ────────────────────────────────────────────────────
# ticker suffix → (market label, benchmark yfinance symbol)
SUFFIX_MAP = {
    ".L":  ("UK", "^FTSE"),
    ".DE": ("DE", "^GDAXI"),
    ".PA": ("FR", "^FCHI"),
    ".AS": ("NL", "^AEX"),
    ".MC": ("ES", "^IBEX"),
    ".MI": ("IT", "FTSEMIB.MI"),
    ".SW": ("CH", "^SSMI"),
    ".HE": ("FI", "^OMXH25"),
    ".ST": ("SE", "^OMX"),
    ".CO": ("DK", "^OMXC20"),
    ".OL": ("NO", "^OBX"),
    ".TO": ("CA", "^GSPTSE"),
    ".BR": ("BE", "^BFX"),
    ".LS": ("PT", "PSI20.LS"),
}
DEFAULT_MKT   = "US"
DEFAULT_BENCH = "^GSPC"

def get_mkt(ticker: str) -> str:
    for sfx, (mkt, _) in SUFFIX_MAP.items():
        if ticker.endswith(sfx):
            return mkt
    return DEFAULT_MKT

def get_bench(ticker: str) -> str:
    for sfx, (_, bench) in SUFFIX_MAP.items():
        if ticker.endswith(sfx):
            return bench
    return DEFAULT_BENCH


# ── ANSI COLORS ───────────────────────────────────────────────────────────────
_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _color else text
GRN  = lambda t: _c("32", t)
RED  = lambda t: _c("31", t)
YLW  = lambda t: _c("33", t)
CYN  = lambda t: _c("36", t)
BOLD = lambda t: _c("1",  t)
DIM  = lambda t: _c("2",  t)

def ret_fmt(v):
    if v is None: return DIM("   ─  ")
    return GRN(f"+{v:.2f}%") if v >= 0 else RED(f"{v:.2f}%")

def wr_fmt(v):
    if v is None: return DIM("  ─ ")
    return GRN(f"{v:.0f}%") if v >= 60 else (YLW(f"{v:.0f}%") if v >= 45 else RED(f"{v:.0f}%"))


# ── UNIVERSE FETCHERS ─────────────────────────────────────────────────────────

def _fetch_html_tables(url: str):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text)), resp

def _find_ticker_col(tbl: pd.DataFrame) -> Optional[str]:
    for name in ["Ticker", "ticker", "Symbol", "symbol", "EPIC", "Epic", "Code"]:
        if name in tbl.columns:
            return name
    return None

_KNOWN_SFX = {".L",".DE",".PA",".AS",".MC",".MI",".SW",
              ".TO",".HE",".ST",".CO",".OL",".BR",".LS"}

def _clean(tickers, suffix="", max_len=8):
    """
    Clean and normalize ticker symbols.
    - If ticker already has a known exchange suffix → keep as-is (avoids BT.A.L, AIR.PA.DE)
    - Embedded dots in base ticker (share classes like BT.A) → replace with hyphen → BT-A
    - Then append suffix if provided
    """
    out = []
    for t in tickers:
        if not isinstance(t, str): continue
        t = t.strip().upper()
        if not t or t == "-": continue
        # Already has a valid exchange suffix — return as-is
        if any(t.endswith(s) for s in _KNOWN_SFX):
            out.append(t)
            continue
        # Replace embedded dots (e.g. BT.A → BT-A) to avoid malformed tickers
        base = t.replace(".", "-")
        final = base + suffix if suffix else base
        if len(final) <= max_len + 4:   # +4 for longest suffix (.GSPTSE won't apply here)
            out.append(final)
    return out


_UNIVERSE_CACHE = Path(__file__).parent / "universe_cache.json"

def get_sp500_with_sectors():
    try:
        tables, _ = _fetch_html_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tbl = tables[0]
        tbl["Symbol"] = tbl["Symbol"].str.replace(".", "-", regex=False)
        excluded = set(tbl.loc[tbl["GICS Sector"].isin(EXCLUDE_SECTORS), "Symbol"].tolist())
        return tbl["Symbol"].tolist(), excluded
    except Exception as e:
        print(DIM(f"  ⚠ Wikipedia fetch failed ({e.__class__.__name__}) — using cached universe"))
        return [], set()


def _save_universe_cache(universe: dict):
    """Save universe dict to JSON for fallback on next network failure."""
    try:
        import json
        with open(_UNIVERSE_CACHE, "w") as f:
            json.dump(universe, f)
    except Exception:
        pass


def _load_universe_cache() -> dict:
    """Load last known good universe from cache."""
    try:
        import json
        if _UNIVERSE_CACHE.exists():
            data = json.loads(_UNIVERSE_CACHE.read_text())
            print(DIM(f"  ✓ Loaded cached universe: {len(data)} tickers (last known good)"))
            return data
    except Exception:
        pass
    return {}

def get_russell1000():
    try:
        url  = ("https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
                "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund")
        resp = requests.get(url, headers=HEADERS, timeout=20)
        df   = pd.read_csv(io.StringIO(resp.text), skiprows=9)
        return _clean(df["Ticker"].dropna().tolist())
    except Exception:
        return []

def get_ftse100():
    try:
        tables, _ = _fetch_html_tables("https://en.wikipedia.org/wiki/FTSE_100")
        for tbl in tables:
            col = _find_ticker_col(tbl)
            if col:
                return _clean(tbl[col].dropna().tolist(), suffix=".L", max_len=6)
    except Exception:
        pass
    return []

def get_dax40():
    try:
        tables, _ = _fetch_html_tables("https://en.wikipedia.org/wiki/DAX")
        for tbl in tables:
            col = _find_ticker_col(tbl)
            if col and len(tbl) >= 20:
                return _clean(tbl[col].dropna().tolist(), suffix=".DE", max_len=8)
    except Exception:
        pass
    return []

def get_cac40():
    try:
        tables, _ = _fetch_html_tables("https://en.wikipedia.org/wiki/CAC_40")
        for tbl in tables:
            col = _find_ticker_col(tbl)
            if col and len(tbl) >= 15:
                return _clean(tbl[col].dropna().tolist(), suffix=".PA", max_len=8)
    except Exception:
        pass
    return []

def get_stoxx600():
    """iShares STOXX 600 ETF holdings — already includes exchange suffixes."""
    try:
        url  = ("https://www.ishares.com/uk/individual/en/products/251813/"
                "ishares-stoxx-europe-600-ucits-etf/1478372549652.ajax"
                "?fileType=csv&fileName=EXW1_holdings&dataType=fund")
        resp = requests.get(url, headers=HEADERS, timeout=25)
        df   = pd.read_csv(io.StringIO(resp.text), skiprows=9)
        raw  = df["Ticker"].dropna().tolist()
        out  = []
        for t in raw:
            if not isinstance(t, str): continue
            t = t.strip()
            if not t or t == "-" or len(t) > 12: continue
            # iShares uses space separator e.g. "SAP GY" → skip, use only dotted form
            if " " in t: continue
            out.append(t)
        return out
    except Exception:
        return []

def get_tsx_composite():
    """iShares XIC ETF — proxy for TSX Composite."""
    try:
        url  = ("https://www.ishares.com/ca/individual/en/products/239837/"
                "ISHARES-SPTSX-CAPPED-COMPOSITE-INDEX-ETF/1490978163731.ajax"
                "?fileType=csv&fileName=XIC_holdings&dataType=fund")
        resp = requests.get(url, headers=HEADERS, timeout=25)
        df   = pd.read_csv(io.StringIO(resp.text), skiprows=9)
        return _clean(df["Ticker"].dropna().tolist(), suffix=".TO", max_len=8)
    except Exception:
        return []


def get_universe(fast_mode=False) -> dict:
    """Returns {ticker: benchmark_symbol}."""
    print(DIM("  Loading tickers..."), flush=True)

    sp500, excluded = get_sp500_with_sectors()
    r1000           = [] if fast_mode else get_russell1000()
    ftse            = [] if fast_mode else get_ftse100()
    dax             = [] if fast_mode else get_dax40()
    cac             = [] if fast_mode else get_cac40()
    stoxx           = [] if fast_mode else get_stoxx600()
    tsx             = [] if fast_mode else get_tsx_composite()

    # If all fetches failed (network down), fall back to cached universe
    if not sp500 and not ftse and not dax and not r1000:
        cached = _load_universe_cache()
        if cached:
            return cached
        print(DIM("  ⚠ No network and no cache — universe empty. Aborting scan."))
        return {}

    if fast_mode:
        sample = [t for t in random.sample(sp500, 50) if t not in excluded][:30]
        print(DIM(f"  FAST MODE — {len(sample)} US tickers"))
        return {t: DEFAULT_BENCH for t in sample}

    # Build deduped dict {ticker: benchmark}
    universe = {}
    def _add(tickers, excl=set()):
        for t in tickers:
            if t not in excl and t not in universe:
                universe[t] = get_bench(t)

    us_all = list(dict.fromkeys(sp500 + r1000))
    _add([t for t in us_all if t not in excluded])
    _add(ftse); _add(dax); _add(cac); _add(stoxx); _add(tsx)

    counts = {
        "US":    sum(1 for t in universe if get_mkt(t) == "US"),
        "UK":    sum(1 for t in universe if get_mkt(t) == "UK"),
        "DE":    sum(1 for t in universe if get_mkt(t) == "DE"),
        "FR":    sum(1 for t in universe if get_mkt(t) == "FR"),
        "CA":    sum(1 for t in universe if get_mkt(t) == "CA"),
        "Other": sum(1 for t in universe if get_mkt(t) not in {"US","UK","DE","FR","CA"}),
    }
    count_str = "  ".join(f"{k}:{v}" for k, v in counts.items() if v > 0)
    print(DIM(f"  {len(universe)} total  ·  {count_str}"))
    _save_universe_cache(universe)  # persist for fallback on next network failure
    return universe


# ── BENCHMARK RETURNS ─────────────────────────────────────────────────────────

def fetch_benchmark_returns(bench_symbols: set) -> dict:
    """Returns {bench_symbol: 63d_return_pct}."""
    results = {}
    for sym in bench_symbols:
        try:
            with _quiet():
                df = yf.download(sym, period="120d", interval="1d",
                                 progress=False, auto_adjust=True, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            c = df["Close"].dropna()
            if len(c) >= 64:
                results[sym] = float((c.iloc[-1] - c.iloc[-63]) / c.iloc[-63] * 100)
        except Exception:
            pass
    return results


# ── INDICATORS ────────────────────────────────────────────────────────────────

def _macd(s):
    m = _ema(s, 12) - _ema(s, 26)
    return m, _ema(m, 9)

def _stoch_rsi(rsi, n=14):
    lo, hi = rsi.rolling(n).min(), rsi.rolling(n).max()
    return (rsi - lo) / (hi - lo).replace(0, np.nan) * 100

def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]     = _sma(c, 50)
    df["sma150"]    = _sma(c, 150)
    df["sma200"]    = _sma(c, 200)
    df["ema9"]      = _ema(c, 9)
    df["ema21"]     = _ema(c, 21)
    df["rsi"]       = _rsi(c, 14)
    df["stoch_rsi"] = _stoch_rsi(df["rsi"], 14)
    df["macd"], df["macd_sig"] = _macd(c)
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["adx"]       = _adx(h, l, c, 14)
    df["vol_ma20"]  = v.rolling(20).mean()
    df["52w_high"]  = c.rolling(252).max()
    df["52w_low"]   = c.rolling(252).min()
    return df


# ── SCORING ───────────────────────────────────────────────────────────────────

def _cross_up(a, b, w, i):
    for k in range(max(1, i - w + 1), i + 1):
        if a.iloc[k] > b.iloc[k] and a.iloc[k-1] <= b.iloc[k-1]:
            return True
    return False


def score_row(df: pd.DataFrame, idx: int,
              bench_ret63: Optional[float] = None) -> Optional[dict]:
    if idx < 215:
        return None

    row, prev = df.iloc[idx], df.iloc[idx - 1]

    if row["Close"] < MIN_PRICE or row["vol_ma20"] < MIN_AVG_VOL:
        return None
    if row["adx"] < MIN_ADX:
        return None
    if row["adx"] > 35: return None   # ADX cap: overextended trend = chasing

    # Minervini Trend Template — need ≥6/8
    m = sum([
        row["Close"]  > row["sma150"],
        row["Close"]  > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        row["Close"]  > row["sma50"],
        row["Close"]  >= 1.30 * row["52w_low"],
        row["Close"]  >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[idx - 20]["sma200"],
    ])
    if m < MIN_MINERVINI:
        return None

    # Relative strength vs local benchmark
    if bench_ret63 is not None and idx >= 63:
        p63 = float(df.iloc[idx - 63]["Close"])
        if p63 > 0:
            stock_ret = (float(row["Close"]) - p63) / p63 * 100
            if stock_ret < bench_ret63:
                return None

    # Fresh crossover signals
    rsi_50 = pd.Series(50.0, index=df.index)
    fresh = {
        "MACD":    _cross_up(df["macd"],  df["macd_sig"], FRESH_WINDOW, idx),
        "RSI50":   _cross_up(df["rsi"],   rsi_50,         FRESH_WINDOW, idx),
        "EMA921":  _cross_up(df["ema9"],  df["ema21"],    FRESH_WINDOW, idx),
        "P>SMA50": _cross_up(df["Close"], df["sma50"],    FRESH_WINDOW, idx),
    }
    if sum(fresh.values()) < MIN_FRESH:
        return None

    # Confirmation signals
    vol_ratio = float(row["Volume"]) / float(row["vol_ma20"]) if row["vol_ma20"] > 0 else 0
    conf = {
        "VOL":    vol_ratio > 1.5,
        "ADX↑":   row["adx"] > prev["adx"],
        "RSI✓":   50 < row["rsi"] < 70,
        "HIST":   row["macd_hist"] > 0 and prev["macd_hist"] <= 0,
        "StRSI↑": (row["stoch_rsi"] > prev["stoch_rsi"]
                   and row["stoch_rsi"] < 80
                   and prev["stoch_rsi"] < 50),
    }

    score = sum(fresh.values()) * 2 + sum(conf.values()) + 1

    return {
        "score":     score,
        "fresh":     [k for k, v in fresh.items() if v],
        "conf":      [k for k, v in conf.items()  if v],
        "minervini": m,
        "rsi":       round(float(row["rsi"]),       1),
        "adx":       round(float(row["adx"]),       1),
        "price":     round(float(row["Close"]),     2),
        "vol_ratio": round(vol_ratio,               2),
    }


# ── BACKTEST ──────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS:
            continue
        sig = score_row(df, i)   # no RS filter in historical backtest
        if not sig:
            continue
        entry = float(df.iloc[i]["Close"])
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100)
        last = i

    if not rets:
        return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n": len(a),
            "wr":  round(100 * (a > 0).mean(), 1),
            "avg": round(float(a.mean()),       2),
            "med": round(float(np.median(a)),   2)}


# ── PER-TICKER PIPELINE ───────────────────────────────────────────────────────

def analyze_ticker(ticker: str, bench_ret63: Optional[float],
                   with_backtest: bool) -> Optional[dict]:
    try:
        with _quiet():
            raw = yf.download(ticker, period=f"{LOOKBACK_DAYS}d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 220:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns):
            return None

        df  = build_indicators(raw.copy())
        sig = score_row(df, len(df) - 1, bench_ret63)
        if not sig:
            return None

        result = {"ticker": ticker, "mkt": get_mkt(ticker), **sig}
        if with_backtest:
            result.update(run_backtest(df))
        return result
    except Exception:
        return None


# ── DISPLAY ───────────────────────────────────────────────────────────────────

W = 104

def print_results(results: list, with_backtest: bool, bench_returns: dict):
    now  = datetime.now().strftime("%Y-%m-%d  %H:%M")
    hits = len(results)

    # Benchmark summary for header
    key_benches = {"^GSPC": "US", "^FTSE": "UK", "^GDAXI": "DE",
                   "^FCHI": "FR", "^GSPTSE": "CA"}
    bench_parts = []
    for sym, label in key_benches.items():
        if sym in bench_returns:
            bench_parts.append(f"{label}:{ret_fmt(bench_returns[sym])}")
    bench_line = "  ".join(bench_parts)

    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + f"  🚀  MOMENTUM SCANNER  ·  {now}  ·  {hits} stocks matched".ljust(W-2) + "║")
    if bench_line:
        print("║" + f"  Benchmarks (63d):  {bench_line}".ljust(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")

    if not hits:
        print(YLW("  No fresh momentum signals today."))
        return

    print()
    if with_backtest:
        hdr = (f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}  "
               f"{'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'M':>3}  {'SCR':>3}  "
               f"{'#BT':>3}  {'WIN%':>5}  {'AVG':>7}  {'MED':>7}  SIGNALS")
    else:
        hdr = (f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}  "
               f"{'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'M':>3}  {'SCR':>3}  SIGNALS")
    print(BOLD(hdr))
    print("  " + "─" * (W - 2))

    # Group by market for readability
    mkt_order = ["US", "UK", "DE", "FR", "CA",
                 "NL", "ES", "IT", "CH", "SE", "DK", "NO", "FI", "BE", "PT"]
    def mkt_rank(r): return mkt_order.index(r["mkt"]) if r["mkt"] in mkt_order else 99

    sorted_results = sorted(results[:TOP_N], key=lambda r: (-(r.get("wr") or 0), -r["score"], mkt_rank(r)))

    last_mkt = None
    for rank, r in enumerate(sorted_results, 1):
        # Section divider when market changes
        if r["mkt"] != last_mkt:
            mkt_label = {"US":"── United States","UK":"── United Kingdom",
                         "DE":"── Germany","FR":"── France","CA":"── Canada",
                         "NL":"── Netherlands","ES":"── Spain","IT":"── Italy",
                         "CH":"── Switzerland"}.get(r["mkt"], f"── {r['mkt']}")
            print(f"\n  {DIM(mkt_label)}")
            last_mkt = r["mkt"]

        fresh_str = " ".join(r.get("fresh", []))
        conf_str  = ("  · " + " ".join(r.get("conf",[]))) if r.get("conf") else ""
        sig_str   = CYN(fresh_str) + DIM(conf_str)

        ticker_s = BOLD(f"{r['ticker']:<8}")
        mkt_s    = YLW(f"{r['mkt']:<3}") if r["mkt"] != "US" else DIM(f"{r['mkt']:<3}")
        base = (f"  {rank:>3}  {mkt_s}  {ticker_s}  {r['price']:>9.2f}"
                f"  {r['rsi']:>5.1f}  {r['adx']:>5.1f}  {r['vol_ratio']:>5.1f}"
                f"  {r['minervini']:>3}  {r['score']:>3}  ")

        if with_backtest and r.get("n", 0) > 0:
            row = (base + f"{r['n']:>3}  "
                   + f"{wr_fmt(r['wr']):>5}  "
                   + f"{ret_fmt(r['avg']):>7}  "
                   + f"{ret_fmt(r['med']):>7}  "
                   + sig_str)
        elif with_backtest:
            row = base + DIM("  ─     ─      ─      ─   ") + sig_str
        else:
            row = base + sig_str

        print(row)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n  " + "─" * (W - 2))
    if with_backtest:
        rets = [r["avg"] for r in results if r.get("avg") is not None]
        wrs  = [r["wr"]  for r in results if r.get("wr")  is not None]
        if rets:
            good_wr  = sum(1 for w in wrs  if w  >= 60)
            good_ret = sum(1 for v in rets if v >= 1.0)
            print(f"  BACKTEST  ·  {len(rets)} tickers  ·  "
                  f"median win rate {wr_fmt(float(np.median(wrs)))}  ·  "
                  f"median avg return {ret_fmt(float(np.median(rets)))}")
            print(f"  ≥60% win rate: {good_wr}  |  ≥+1% avg return: {good_ret}")

    print(f"\n  {DIM('⚠  Not financial advice. Use stop-losses.')}"
          f"  {DIM('--legend for signal definitions.')}")
    print("╚" + "═"*(W-2) + "╝\n")


def print_legend():
    print()
    print(BOLD("━" * W))
    print(BOLD("  APPENDIX — Column & Signal Reference"))
    print(BOLD("━" * W))
    print("""
  COLUMNS
  ───────
  #       Rank by score
  MKT     Market  US · UK · DE (Germany) · FR (France) · CA (Canada) · etc.
  TICKER  yfinance symbol  (US = no suffix, UK = .L, DE = .DE, FR = .PA, CA = .TO)
  PRICE   Closing price in local currency
  RSI     14-day RSI  (50–70 = momentum zone)
  ADX     Trend strength  (≥22 required, ≥25 strong)
  VOL×    Volume ÷ 20-day avg
  M       Minervini Trend Template score /8
  SCR     fresh×2 + confirmations + 1
  #BT     Historical backtest trades
  WIN%    % profitable after 5 trading days
  AVG     Average 5-day return
  MED     Median 5-day return

  HARD FILTERS
  ────────────
  ADX ≥ 22          Requires an actual trend
  Minervini ≥ 6/8   Stock in quality uptrend base
  Fresh signals ≥ 2  At least 2 crossovers in last 3 bars
  RS > local index  63-day return must beat the stock's own market index
  No Utilities/REIT (US only — sector data only available for S&P 500)

  BENCHMARKS PER MARKET
  ─────────────────────
  US  → ^GSPC (S&P 500)     UK  → ^FTSE         DE → ^GDAXI
  FR  → ^FCHI (CAC 40)      CA  → ^GSPTSE        NL → ^AEX
  ES  → ^IBEX               IT  → FTSEMIB.MI      CH → ^SSMI

  FRESH SIGNALS  (must cross within last 3 bars)
  ───────────────────────────────────────────────
  MACD    MACD line crossed above signal line
  RSI50   RSI crossed above 50
  EMA921  9 EMA crossed above 21 EMA
  P>SMA50 Price reclaimed 50-day SMA

  CONFIRMATION SIGNALS  (after · )
  ─────────────────────────────────
  VOL     Volume > 1.5× 20d avg
  ADX↑    ADX rising
  RSI✓    RSI 50–70
  HIST    MACD histogram just turned positive
  StRSI↑  Stochastic RSI turning up from < 50
""")
    print(BOLD("━" * W))
    print()


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args          = set(sys.argv[1:])
    fast_mode     = "--fast"        in args
    with_backtest = "--no-backtest" not in args
    show_legend   = "--legend"      in args

    if show_legend:
        print_legend()
        return

    universe = get_universe(fast_mode)   # {ticker: benchmark_symbol}

    # Fetch all unique benchmark returns in parallel
    bench_symbols = set(universe.values())
    print(DIM(f"  Fetching {len(bench_symbols)} benchmark indices..."), flush=True)
    bench_returns = fetch_benchmark_returns(bench_symbols)
    fetched = [f"{k.replace('^','')}:{bench_returns[k]:+.1f}%"
               for k in sorted(bench_returns)]
    print(DIM("  " + "  ".join(fetched)))

    bt_label = "backtest ON" if with_backtest else "backtest OFF"
    print(DIM(f"  Scanning {len(universe)} tickers  ·  {bt_label}"
              f"  ·  hold={HOLD_DAYS}d"
              f"  ·  ADX≥{MIN_ADX}  Minervini≥{MIN_MINERVINI}  fresh≥{MIN_FRESH}"))
    print()

    t0, results, done = time.time(), [], 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(analyze_ticker, t, bench_returns.get(bench), with_backtest): t
            for t, bench in universe.items()
        }
        for f in as_completed(futs, timeout=300):
            done += 1
            try:
                r = f.result(timeout=30)
            except Exception:
                continue
            if r:
                results.append(r)
            if done % 150 == 0:
                print(DIM(f"  {done}/{len(universe)}  hits={len(results)}"
                          f"  elapsed={time.time()-t0:.0f}s"), flush=True)

    results.sort(key=lambda x: (-(x.get("wr") or 0), -x["score"]))
    print(DIM(f"  Done in {time.time()-t0:.0f}s — {len(results)} candidates\n"))
    print_results(results, with_backtest, bench_returns)


if __name__ == "__main__":
    main()


# ── PUBLIC API ────────────────────────────────────────────────────────────────
# Aliases for unified scan.py interface
build_universe       = get_universe
compute_bench_returns = fetch_benchmark_returns


def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    """Run the scanning loop silently; return list of result dicts with strategy tag."""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(analyze_ticker, t, bench_returns.get(bench), with_backtest): t
            for t, bench in universe.items()
        }
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
            except Exception:
                continue
            if r:
                r["strategy"] = "momentum"
                results.append(r)
    results.sort(key=lambda x: (-(x.get("wr") or 0), -x["score"]))
    return results
