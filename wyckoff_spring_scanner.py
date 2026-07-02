#!/usr/bin/env python3
"""
Wyckoff Spring Scanner  |  US + Europe + Canada
────────────────────────────────────────────────
A Wyckoff Spring fires when price briefly breaks below a 30-bar trading range's
low on LOW volume (institutional shakeout of weak hands), then closes back above
that support level before a markup phase.

python3 wyckoff_spring_scanner.py                # full scan + backtest
python3 wyckoff_spring_scanner.py --no-backtest  # signals only
python3 wyckoff_spring_scanner.py --fast         # 30 US stocks smoke-test
"""

import io, os, sys, time, random, warnings, logging, contextlib
import requests
import numpy  as np
import pandas as pd
import yfinance as yf
from datetime           import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing             import Optional

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

@contextlib.contextmanager
def _quiet():
    devnull = open(os.devnull, "w")
    old_err = sys.stderr
    sys.stderr = devnull
    try:    yield
    finally:
        sys.stderr = old_err
        devnull.close()


# ── CONFIG ────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS  = 400
HOLD_DAYS      = 5
MAX_WORKERS    = 25
MIN_PRICE      = 1.0
MIN_AVG_VOL    = 300_000
FRESH_WINDOW   = 2   # spring must fire within last N bars
MIN_MINERVINI  = 3
RANGE_BARS     = 30
MIN_RANGE_PCT  = 0.08
EXCLUDE_SECTORS = {"Utilities", "Real Estate"}
TOP_N   = 50
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; wyckoff-spring-scanner/1.0)"}

# ── MARKET / BENCHMARK MAP ────────────────────────────────────────────────────
SUFFIX_MAP = {
    ".L":  ("UK", "^FTSE"),   ".DE": ("DE", "^GDAXI"),
    ".PA": ("FR", "^FCHI"),   ".AS": ("NL", "^AEX"),
    ".MC": ("ES", "^IBEX"),   ".MI": ("IT", "FTSEMIB.MI"),
    ".SW": ("CH", "^SSMI"),   ".HE": ("FI", "^OMXH25"),
    ".ST": ("SE", "^OMX"),    ".CO": ("DK", "^OMXC20"),
    ".OL": ("NO", "^OBX"),    ".TO": ("CA", "^GSPTSE"),
    ".BR": ("BE", "^BFX"),    ".LS": ("PT", "PSI20.LS"),
}
DEFAULT_MKT   = "US"
DEFAULT_BENCH = "^GSPC"

def get_mkt(ticker: str) -> str:
    for sfx, (mkt, _) in SUFFIX_MAP.items():
        if ticker.endswith(sfx): return mkt
    return DEFAULT_MKT

def get_bench(ticker: str) -> str:
    for sfx, (_, bench) in SUFFIX_MAP.items():
        if ticker.endswith(sfx): return bench
    return DEFAULT_BENCH


# ── ANSI COLORS ───────────────────────────────────────────────────────────────
_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _color else text
GRN  = lambda t: _c("32", t);  RED  = lambda t: _c("31", t)
YLW  = lambda t: _c("33", t);  CYN  = lambda t: _c("36", t)
BOLD = lambda t: _c("1",  t);  DIM  = lambda t: _c("2",  t)

def ret_fmt(v):
    if v is None: return DIM("   ─  ")
    return GRN(f"+{v:.2f}%") if v >= 0 else RED(f"{v:.2f}%")

def wr_fmt(v):
    if v is None: return DIM("  ─ ")
    return GRN(f"{v:.0f}%") if v >= 60 else (YLW(f"{v:.0f}%") if v >= 45 else RED(f"{v:.0f}%"))


# ── UNIVERSE FETCHERS ─────────────────────────────────────────────────────────
_KNOWN_SFX = {".L",".DE",".PA",".AS",".MC",".MI",".SW",
              ".TO",".HE",".ST",".CO",".OL",".BR",".LS"}

def _clean(tickers, suffix="", max_len=8):
    out = []
    for t in tickers:
        if not isinstance(t, str): continue
        t = t.strip().upper()
        if not t or t == "-": continue
        if any(t.endswith(s) for s in _KNOWN_SFX):
            out.append(t); continue
        base  = t.replace(".", "-")
        final = base + suffix if suffix else base
        if len(final) <= max_len + 4: out.append(final)
    return out

def _fetch_html(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))

def _ticker_col(tbl):
    for n in ["Ticker","ticker","Symbol","symbol","EPIC","Epic","Code"]:
        if n in tbl.columns: return n
    return None

def get_sp500_with_sectors():
    tbls = _fetch_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    t = tbls[0]; t["Symbol"] = t["Symbol"].str.replace(".", "-", regex=False)
    excl = set(t.loc[t["GICS Sector"].isin(EXCLUDE_SECTORS), "Symbol"])
    return t["Symbol"].tolist(), excl

def get_russell1000():
    try:
        r  = requests.get("https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/"
                          "1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund",
                          headers=HEADERS, timeout=20)
        df = pd.read_csv(io.StringIO(r.text), skiprows=9)
        return _clean(df["Ticker"].dropna().tolist())
    except Exception: return []

def get_ftse100():
    try:
        for t in _fetch_html("https://en.wikipedia.org/wiki/FTSE_100"):
            col = _ticker_col(t)
            if col: return _clean(t[col].dropna().tolist(), ".L", 6)
    except Exception: return []
    return []

def get_dax40():
    try:
        for t in _fetch_html("https://en.wikipedia.org/wiki/DAX"):
            col = _ticker_col(t)
            if col and len(t) >= 20: return _clean(t[col].dropna().tolist(), ".DE")
    except Exception: return []
    return []

def get_cac40():
    try:
        for t in _fetch_html("https://en.wikipedia.org/wiki/CAC_40"):
            col = _ticker_col(t)
            if col and len(t) >= 15: return _clean(t[col].dropna().tolist(), ".PA")
    except Exception: return []
    return []

def get_stoxx600():
    try:
        r  = requests.get("https://www.ishares.com/uk/individual/en/products/251813/"
                          "ishares-stoxx-europe-600-ucits-etf/1478372549652.ajax"
                          "?fileType=csv&fileName=EXW1_holdings&dataType=fund",
                          headers=HEADERS, timeout=25)
        df = pd.read_csv(io.StringIO(r.text), skiprows=9)
        return [t.strip() for t in df["Ticker"].dropna()
                if isinstance(t, str) and t.strip() and " " not in t.strip()
                and t.strip() != "-" and len(t.strip()) <= 12]
    except Exception: return []

def get_tsx():
    try:
        r  = requests.get("https://www.ishares.com/ca/individual/en/products/239837/"
                          "ISHARES-SPTSX-CAPPED-COMPOSITE-INDEX-ETF/1490978163731.ajax"
                          "?fileType=csv&fileName=XIC_holdings&dataType=fund",
                          headers=HEADERS, timeout=25)
        df = pd.read_csv(io.StringIO(r.text), skiprows=9)
        return _clean(df["Ticker"].dropna().tolist(), ".TO")
    except Exception: return []


def get_universe(fast_mode=False) -> dict:
    """Returns {ticker: benchmark_symbol}."""
    print(DIM("  Loading tickers..."), flush=True)
    sp500, excl = get_sp500_with_sectors()
    if fast_mode:
        s = [t for t in random.sample(sp500, 50) if t not in excl][:30]
        print(DIM(f"  FAST MODE — {len(s)} US tickers"))
        return {t: DEFAULT_BENCH for t in s}
    r1000 = get_russell1000(); ftse = get_ftse100(); dax = get_dax40()
    cac   = get_cac40();       stoxx= get_stoxx600(); tsx = get_tsx()
    uni   = {}
    def _add(lst, ex=set()):
        for t in lst:
            if t not in ex and t not in uni: uni[t] = get_bench(t)
    _add([t for t in list(dict.fromkeys(sp500+r1000)) if t not in excl])
    _add(ftse); _add(dax); _add(cac); _add(stoxx); _add(tsx)
    counts = {k: sum(1 for t in uni if get_mkt(t)==k)
              for k in ["US","UK","DE","FR","CA"]}
    other  = sum(1 for t in uni if get_mkt(t) not in counts)
    cs     = "  ".join(f"{k}:{v}" for k,v in counts.items() if v) + f"  Other:{other}"
    print(DIM(f"  {len(uni)} tickers  ·  {cs}"))
    return uni


def fetch_benchmark_returns(syms: set) -> dict:
    results = {}
    for sym in syms:
        try:
            with _quiet():
                df = yf.download(sym, period="120d", interval="1d",
                                 progress=False, auto_adjust=True, threads=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            c = df["Close"].dropna()
            if len(c) >= 64:
                results[sym] = float((c.iloc[-1]-c.iloc[-63])/c.iloc[-63]*100)
        except Exception: pass
    return results


# ── PUBLIC API ALIASES ────────────────────────────────────────────────────────
build_universe        = get_universe
compute_bench_returns = fetch_benchmark_returns


# ── INDICATORS ────────────────────────────────────────────────────────────────
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _adx(high, low, close, n=14):
    tr  = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    up  = (high-high.shift()).clip(lower=0); dn=(low.shift()-low).clip(lower=0)
    dmp = up.where(up>dn,0).ewm(alpha=1/n, adjust=False).mean()
    dmm = dn.where(dn>up,0).ewm(alpha=1/n, adjust=False).mean()
    dip = 100*dmp/atr; dim=100*dmm/atr
    dx  = 100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df


# ── MINERVINI SCORE ───────────────────────────────────────────────────────────

def _compute_minervini(df: pd.DataFrame, idx: int) -> int:
    row = df.iloc[idx]
    if idx < 20:
        return 0
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


# ── WYCKOFF SPRING DETECTION ─────────────────────────────────────────────────

def _find_spring(df: pd.DataFrame, idx: int):
    """
    Check the last 3 bars (up to and including idx) for a spring condition.
    Returns the index of the spring bar if found, else None.
    Spring: bar.low < range_low AND bar.close > range_low AND bar.volume < 20d avg vol.
    The trading range is defined by the 30 bars ending just before the earliest
    candidate spring bar examined (to avoid look-ahead).
    """
    # We check bars idx-2, idx-1, idx
    for k in range(max(0, idx - 2), idx + 1):
        if k < RANGE_BARS + 1:
            continue
        # Range = bars [k-1-RANGE_BARS : k-1] (excludes the spring bar itself)
        range_slice = df.iloc[k - RANGE_BARS: k]
        range_low  = float(range_slice["Low"].min())
        range_high = float(range_slice["High"].max())
        if range_low <= 0:
            continue
        range_width_pct = (range_high - range_low) / range_low
        if range_width_pct < MIN_RANGE_PCT:
            continue
        bar        = df.iloc[k]
        bar_low    = float(bar["Low"])
        bar_close  = float(bar["Close"])
        bar_vol    = float(bar["Volume"])
        vol_ma_val = float(bar["vol_ma20"]) if not pd.isna(bar["vol_ma20"]) else 0.0
        if vol_ma_val <= 0:
            continue
        # Spring: pierced below range_low, closed back above, on sub-average volume
        if bar_low < range_low and bar_close > range_low and bar_vol < vol_ma_val:
            return k, range_low, range_high, range_width_pct
    return None


def score_row(df: pd.DataFrame, idx: int,
              bench_ret63: Optional[float] = None) -> Optional[dict]:
    if idx < 215:
        return None

    row = df.iloc[idx]

    price_val  = float(row["Close"])
    vol_ma_val = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0.0
    if pd.isna(price_val) or price_val < MIN_PRICE:
        return None
    if vol_ma_val < MIN_AVG_VOL:
        return None

    # Minervini filter
    m = _compute_minervini(df, idx)
    if m < MIN_MINERVINI:
        return None

    # ADX filter: must be trending but not overextended
    adx_val = float(row["adx"])
    if pd.isna(adx_val) or adx_val < 10 or adx_val > 40:
        return None

    # Must be above 200-day SMA
    sma200_val = float(row["sma200"])
    if pd.isna(sma200_val) or price_val <= sma200_val:
        return None

    # Must not be too extended (< 95% of 52w high)
    w52_high = float(row["52w_high"])
    if not pd.isna(w52_high) and w52_high > 0 and price_val >= 0.95 * w52_high:
        return None

    # Detect spring in last 3 bars
    result = _find_spring(df, idx)
    if result is None:
        return None
    spring_idx, range_low, range_high, range_width_pct = result

    # Freshness: spring bar must be within FRESH_WINDOW of today
    if idx - spring_idx >= FRESH_WINDOW:
        return None

    spring_bar    = df.iloc[spring_idx]
    spring_vol    = float(spring_bar["Volume"])
    spring_vol_ma = float(spring_bar["vol_ma20"]) if not pd.isna(spring_bar["vol_ma20"]) else 1.0
    vol_ratio     = spring_vol / spring_vol_ma if spring_vol_ma > 0 else 0.0

    rsi_val = float(row["rsi"])
    if pd.isna(rsi_val):
        return None

    range_mid = (range_high + range_low) / 2.0

    # Score (0-5)
    score = 0
    score += 1 if spring_vol < 0.50 * spring_vol_ma else 0          # very clean shakeout
    score += 1 if price_val > range_mid else 0                        # closed above range midpoint
    score += 1 if 35 <= rsi_val <= 60 else 0                          # RSI in healthy zone
    score += 1 if (not pd.isna(w52_high) and w52_high > 0
                   and price_val >= 0.85 * w52_high) else 0           # within 15% of 52w high
    score += 1 if m >= 5 else 0                                        # strong trend template

    return {
        "score":           score,
        "fresh":           ["WS"],
        "conf":            [],
        "minervini":       m,
        "rsi":             round(rsi_val,           1),
        "adx":             round(adx_val,           1),
        "price":           round(price_val,         2),
        "vol_ratio":       round(vol_ratio,         2),
        "spring_low":      round(range_low,         4),
        "range_width_pct": round(range_width_pct * 100, 2),  # stored as %
    }


# ── BACKTEST ──────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS:
            continue
        sig = score_row(df, i)
        if not sig:
            continue
        entry = float(df.iloc[i]["Close"])
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100)
        last = i
    if not rets:
        return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n":   len(a),
            "wr":  round(100*(a>0).mean(), 1),
            "avg": round(float(a.mean()),  2),
            "med": round(float(np.median(a)), 2)}


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
        raw = raw.dropna(subset=["Close","High","Low","Volume"])
        if len(raw) < 220:
            return None
        df      = build_indicators(raw.copy())
        last_idx = len(df) - 1
        sig     = score_row(df, last_idx, bench_ret63)
        if not sig:
            return None
        company = ticker  # no separate company lookup; keep consistent with template
        result  = {
            "ticker":          ticker,
            "company":         company,
            "mkt":             get_mkt(ticker),
            "strategy":        "wyckoff_spring",
            "hold_days":       HOLD_DAYS,
            **sig,
        }
        if with_backtest:
            result.update(run_backtest(df))
        return result
    except Exception:
        return None


# ── DISPLAY ───────────────────────────────────────────────────────────────────

W = 110

def print_results(results: list, with_backtest: bool, bench_returns: dict):
    now  = datetime.now().strftime("%Y-%m-%d  %H:%M")
    hits = len(results)
    key_benches = {"^GSPC":"US","^FTSE":"UK","^GDAXI":"DE","^FCHI":"FR","^GSPTSE":"CA"}
    bench_str   = "  ".join(f"{l}:{ret_fmt(bench_returns.get(s))}"
                            for s, l in key_benches.items() if s in bench_returns)
    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + f"  WYCKOFF SPRING SCANNER  ·  {now}  ·  {hits} stocks matched".ljust(W-2) + "║")
    if bench_str:
        print("║" + f"  Benchmarks (63d):  {bench_str}".ljust(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")
    if not hits:
        print(YLW("  No Wyckoff Spring signals today."))
        return
    print()
    if with_backtest:
        hdr = (f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}  "
               f"{'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'M':>3}  {'SCR':>3}  "
               f"{'RNG%':>6}  {'SPRLO':>8}  "
               f"{'#BT':>3}  {'WIN%':>5}  {'AVG':>7}  {'MED':>7}  CONF")
    else:
        hdr = (f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}  "
               f"{'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'M':>3}  {'SCR':>3}  "
               f"{'RNG%':>6}  {'SPRLO':>8}  CONF")
    print(BOLD(hdr))
    print("  " + "─"*(W-2))
    sorted_r = sorted(results[:TOP_N], key=lambda r: (-(r.get("wr") or 0), -r["score"]))
    for rank, r in enumerate(sorted_r, 1):
        conf_str = " ".join(r.get("conf", []) + r.get("fresh", []))
        mkt_s    = YLW(f"{r['mkt']:<3}") if r["mkt"] != "US" else DIM(f"{r['mkt']:<3}")
        ticker_s = BOLD(f"{r['ticker']:<8}")
        base = (f"  {rank:>3}  {mkt_s}  {ticker_s}  {r['price']:>9.2f}"
                f"  {r['rsi']:>5.1f}  {r['adx']:>5.1f}  {r['vol_ratio']:>5.2f}"
                f"  {r['minervini']:>3}  {r['score']:>3}"
                f"  {r['range_width_pct']:>5.1f}%  {r['spring_low']:>8.2f}  ")
        if with_backtest and r.get("n", 0) > 0:
            row = (base + f"{r['n']:>3}  "
                   + f"{wr_fmt(r['wr']):>5}  "
                   + f"{ret_fmt(r['avg']):>7}  "
                   + f"{ret_fmt(r['med']):>7}  "
                   + CYN(conf_str))
        elif with_backtest:
            row = base + DIM("  ─     ─      ─      ─   ") + CYN(conf_str)
        else:
            row = base + CYN(conf_str)
        print(row)
    print("\n  " + "─"*(W-2))
    print(f"\n  {DIM('Strategy: Wyckoff Spring (accumulation shakeout)  ·  hold=' + str(HOLD_DAYS) + 'd  ·  Minervini≥' + str(MIN_MINERVINI))}")
    print("╚" + "═"*(W-2) + "╝\n")


# ── SCAN (public silent API) ──────────────────────────────────────────────────

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
                results.append(r)
    results.sort(key=lambda x: (-(x.get("wr") or 0), -x["score"]))
    return results


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args          = set(sys.argv[1:])
    fast_mode     = "--fast"        in args
    with_backtest = "--no-backtest" not in args

    universe = get_universe(fast_mode)
    bench_syms = set(universe.values())
    print(DIM(f"  Fetching {len(bench_syms)} benchmark indices..."), flush=True)
    bench_returns = fetch_benchmark_returns(bench_syms)
    fetched = [f"{k.replace('^','')}:{bench_returns[k]:+.1f}%"
               for k in sorted(bench_returns)]
    print(DIM("  " + "  ".join(fetched)))
    print(DIM(f"  Scanning {len(universe)} tickers  ·  "
              f"{'backtest ON' if with_backtest else 'backtest OFF'}"
              f"  ·  hold={HOLD_DAYS}d  ·  Minervini≥{MIN_MINERVINI}  ADX 10-40"
              f"  ·  MinAvgVol={MIN_AVG_VOL:,}"))
    print()

    t0, results, done = time.time(), [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
                for t, b in universe.items()}
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
