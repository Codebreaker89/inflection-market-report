#!/usr/bin/env python3
"""
Weinstein Stage 1→2 Transition Scanner  |  US + Europe + Canada
────────────────────────────────────────────────────────────────
Detects the Stan Weinstein Stage 1→2 transition: when institutional
accumulation ends and markup begins (weekly data, 30-week SMA).

Reference: "Secrets for Profiting in Bull and Bear Markets" (1988)

python3 weinstein_stage2_scanner.py                # full scan
python3 weinstein_stage2_scanner.py --no-backtest  # signals only
python3 weinstein_stage2_scanner.py --fast         # 30 US smoke-test
"""

import io, os, sys, time, random, warnings, logging, contextlib
import requests
import numpy  as np
import pandas as pd
import yfinance as yf
from datetime           import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing             import Optional
from scanner_utils import _adx, _ema, _fetch_html, _quiet, _rsi, _sma

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)


# ── CONFIG ────────────────────────────────────────────────────────────────────
HOLD_DAYS      = 10
MAX_WORKERS    = 20
FRESH_WINDOW   = 3   # crossover must have occurred within last 3 weeks
MIN_PRICE      = 1.0
MIN_AVG_VOL    = 200_000   # 20-day avg volume
EXCLUDE_SECTORS = {"Utilities", "Real Estate"}
TOP_N   = 50
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; weinstein-stage2-scanner/1.0)"}

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


# ── MINERVINI TREND TEMPLATE ──────────────────────────────────────────────────
def _compute_minervini(daily: pd.DataFrame) -> int:
    """Returns Minervini score 0-8 using daily OHLCV data."""
    if len(daily) < 210: return 0
    c = daily["Close"]
    sma50  = c.rolling(50).mean()
    sma150 = c.rolling(150).mean()
    sma200 = c.rolling(200).mean()
    price  = float(c.iloc[-1])
    s50    = float(sma50.iloc[-1])
    s150   = float(sma150.iloc[-1])
    s200   = float(sma200.iloc[-1])
    s200_20ago = float(sma200.iloc[-21]) if len(sma200) >= 21 else float("nan")
    w52_high = float(c.rolling(252).max().iloc[-1])
    w52_low  = float(c.rolling(252).min().iloc[-1])
    score = sum([
        price  > s150,
        price  > s200,
        s150   > s200,
        s50    > s150,
        price  > s50,
        price  >= 1.30 * w52_low,
        price  >= 0.75 * w52_high,
        (not np.isnan(s200_20ago)) and s200 > s200_20ago,
    ])
    return score


# ── WEEKLY WEINSTEIN LOGIC ─────────────────────────────────────────────────────

def _prepare_weekly(wdf: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Flatten MultiIndex columns if present and validate weekly df."""
    if isinstance(wdf.columns, pd.MultiIndex):
        wdf.columns = wdf.columns.droplevel(1)
    needed = {"Close", "Volume"}
    if not needed.issubset(wdf.columns): return None
    wdf = wdf.dropna(subset=["Close"])
    if len(wdf) < 35: return None
    return wdf


def _find_crossover_week(close: pd.Series, sma30: pd.Series,
                          lookback: int = 8) -> Optional[int]:
    """
    Return the index (position in the series) of the most recent week where
    close crossed above sma30 (was below in the prior week, above in this week).
    Only looks within the last `lookback` bars. Returns None if not found.
    """
    n = len(close)
    for i in range(n - 1, max(n - lookback - 1, 1), -1):
        prev_below = float(close.iloc[i - 1]) < float(sma30.iloc[i - 1])
        curr_above = float(close.iloc[i])     > float(sma30.iloc[i])
        if prev_below and curr_above:
            return i
    return None


def _check_ma_slope(sma30: pd.Series) -> tuple:
    """
    Returns (slope_turning_positive: bool, current_slope: float).

    Slope turning positive means:
    - Current slope (last 4 weeks) > 0
    - At least 4 of the 8 weeks before "now" had slope <= 0.01 (flat/down)

    Slope measured as: (sma30[i] - sma30[i-4]) / sma30[i-4]
    """
    n = len(sma30)
    if n < 13: return False, 0.0

    def _slope(idx):
        ref = float(sma30.iloc[idx - 4])
        if ref == 0: return 0.0
        return (float(sma30.iloc[idx]) - ref) / ref

    current_slope = _slope(n - 1)
    if current_slope <= 0:
        return False, current_slope

    # Check that current slope > previous slope (MA accelerating upward)
    prev_slope = _slope(n - 2) if n >= 14 else current_slope
    if current_slope <= prev_slope and current_slope <= 0.005:
        # Rising but barely — require it be clearly positive
        return False, current_slope

    # Count weeks 4-8 bars ago that had slope <= 0.01
    flat_or_down = 0
    for offset in range(4, 12):  # check 8 prior weeks (bars n-5 to n-12)
        idx = n - 1 - offset
        if idx < 4: break
        s = _slope(idx)
        if s <= 0.01:
            flat_or_down += 1
    if flat_or_down < 4:
        return False, current_slope

    # Require at least 2 consecutive rising weeks at the end
    slope_now  = _slope(n - 1)
    slope_prev = _slope(n - 2) if n >= 14 else 0.0
    consecutive_rising = slope_now > 0 and slope_prev > 0
    if not consecutive_rising:
        return False, current_slope

    return True, current_slope


def score_weinstein(wdf: pd.DataFrame, daily: pd.DataFrame,
                    spy_weekly: Optional[pd.Series] = None) -> Optional[dict]:
    """
    Apply Weinstein Stage 1→2 criteria to weekly data.
    Returns a result dict or None if criteria not met.
    """
    close  = wdf["Close"]
    volume = wdf["Volume"]
    n      = len(close)

    if n < 35: return None

    # 30-week SMA
    sma30 = close.rolling(30).mean()
    if pd.isna(sma30.iloc[-1]): return None

    # ── Criterion 2: MA slope turning positive ────────────────────────────────
    slope_ok, current_slope = _check_ma_slope(sma30)
    if not slope_ok: return None

    # ── Criterion 3: Price crossover within last FRESH_WINDOW weeks ───────────
    crossover_idx = _find_crossover_week(close, sma30, lookback=FRESH_WINDOW + 5)
    if crossover_idx is None: return None
    # Must be within last FRESH_WINDOW weeks
    if n - 1 - crossover_idx > FRESH_WINDOW: return None

    # ── Criterion 4: Volume confirmation on crossover week ────────────────────
    vol10_avg = volume.rolling(10).mean()
    if crossover_idx < 10: return None
    avg_vol_at_cross = float(vol10_avg.iloc[crossover_idx - 1])
    if avg_vol_at_cross <= 0: return None
    cross_vol    = float(volume.iloc[crossover_idx])
    vol_ratio    = cross_vol / avg_vol_at_cross
    if vol_ratio < 1.3: return None

    # ── Criterion 5: Not in freefall — at least 10% above 52w low ────────────
    w52_low  = float(close.rolling(52).min().iloc[-1])
    w52_high = float(close.rolling(52).max().iloc[-1])
    price_now = float(close.iloc[-1])
    if w52_low > 0 and price_now < w52_low * 1.10: return None

    # ── Criterion 6: Not extended — below 52w high * 1.1 ─────────────────────
    if w52_high > 0 and price_now > w52_high * 1.1: return None

    # ── Daily data: ADX, RSI, 20-day avg vol ─────────────────────────────────
    if len(daily) < 30: return None
    if isinstance(daily.columns, pd.MultiIndex):
        daily = daily.copy()
        daily.columns = daily.columns.droplevel(1)
    daily = daily.dropna(subset=["Close"])

    dc  = daily["Close"]
    dh  = daily["High"]
    dl  = daily["Low"]
    dv  = daily["Volume"]

    adx_series = _adx(dh, dl, dc, 14)
    adx_val    = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0
    rsi_val    = float(_rsi(dc, 14).iloc[-1])
    daily_price= float(dc.iloc[-1])
    vol_ma20   = float(dv.rolling(20).mean().iloc[-1]) if len(dv) >= 20 else 0.0

    # ── Criterion 7: ADX between 12 and 45 ───────────────────────────────────
    if pd.isna(adx_val) or adx_val < 12 or adx_val > 45: return None

    # ── Criterion 8: 20-day avg volume > 200k ────────────────────────────────
    if vol_ma20 < MIN_AVG_VOL: return None

    # ── Minervini score ───────────────────────────────────────────────────────
    m = _compute_minervini(daily)

    # ── Score (0–5) ───────────────────────────────────────────────────────────
    score = 0

    # +1 if vol > 1.5x avg (strong)
    if vol_ratio > 1.5: score += 1

    # +1 if RS (close/SPY) trending up
    rs_trending = False
    if spy_weekly is not None and len(spy_weekly) >= 5:
        try:
            rs_now  = price_now / float(spy_weekly.iloc[-1])
            rs_4ago = float(close.iloc[-5]) / float(spy_weekly.iloc[-5])
            rs_trending = rs_now > rs_4ago
        except Exception:
            pass
    if rs_trending: score += 1

    # +1 if price within 20% of 52w high (near top of base)
    if w52_high > 0 and price_now >= w52_high * 0.80: score += 1

    # +1 if ADX in sweet spot 16-35
    if 16 <= adx_val <= 35: score += 1

    # +1 if Minervini >= 4
    if m >= 4: score += 1

    # ── Build stage_note ──────────────────────────────────────────────────────
    notes = ["MA turning"]
    notes.append(f"week vol {vol_ratio:.1f}x")
    if rs_trending: notes.append("RS↑")
    stage_note = " · ".join(notes)

    ma30w = round(float(sma30.iloc[-1]), 2)

    conf = []
    if vol_ratio > 1.5:          conf.append("VOL1.5x")
    if rs_trending:              conf.append("RS↑")
    if w52_high > 0 and price_now >= w52_high * 0.80: conf.append("Near52H")
    if 16 <= adx_val <= 35:      conf.append("ADX16-35")
    if m >= 4:                   conf.append(f"M{m}")

    return {
        "score":      score,
        "fresh":      ["W2"],
        "conf":       conf,
        "minervini":  m,
        "rsi":        round(rsi_val, 1),
        "adx":        round(adx_val, 1),
        "price":      round(daily_price, 2),
        "vol_ratio":  round(vol_ratio, 2),
        "ma30w":      ma30w,
        "stage_note": stage_note,
    }


# ── SPY WEEKLY CACHE ──────────────────────────────────────────────────────────
_spy_weekly_cache: Optional[pd.Series] = None

def _get_spy_weekly() -> Optional[pd.Series]:
    global _spy_weekly_cache
    if _spy_weekly_cache is not None:
        return _spy_weekly_cache
    try:
        with _quiet():
            raw = yf.Ticker("SPY").history(period="2y", interval="1wk")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if "Close" in raw.columns and len(raw) >= 10:
            _spy_weekly_cache = raw["Close"].dropna()
            return _spy_weekly_cache
    except Exception:
        pass
    return None


# ── PER-TICKER PIPELINE ───────────────────────────────────────────────────────

def analyze_ticker(ticker: str, bench_ret63: Optional[float] = None,
                   with_backtest: bool = False) -> Optional[dict]:
    try:
        # Weekly data (2 years)
        with _quiet():
            wraw = yf.Ticker(ticker).history(period="2y", interval="1wk")
        wdf = _prepare_weekly(wraw)
        if wdf is None: return None

        # Daily data (1 year for indicators)
        with _quiet():
            draw = yf.download(ticker, period="400d", interval="1d",
                               progress=False, auto_adjust=True, threads=False)
        if draw is None or len(draw) < 30: return None
        if isinstance(draw.columns, pd.MultiIndex): draw.columns = draw.columns.droplevel(1)
        if not {"Close","High","Low","Volume"}.issubset(draw.columns): return None
        draw = draw.dropna(subset=["Close","High","Low","Volume"])
        if len(draw) < 30: return None

        spy_weekly = _get_spy_weekly()

        sig = score_weinstein(wdf, draw, spy_weekly)
        if sig is None: return None

        # Fetch company name
        company = ticker
        try:
            info = yf.Ticker(ticker).info
            company = info.get("shortName") or info.get("longName") or ticker
        except Exception:
            pass

        result = {
            "ticker":     ticker,
            "company":    company,
            "mkt":        get_mkt(ticker),
            "strategy":   "weinstein_stage2",
            "hold_days":  HOLD_DAYS,
            **sig,
        }
        return result
    except Exception:
        return None


# ── DISPLAY ───────────────────────────────────────────────────────────────────
W = 110

def print_results(results: list):
    now  = datetime.now().strftime("%Y-%m-%d  %H:%M")
    hits = len(results)
    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + f"  WEINSTEIN STAGE 1→2 SCANNER  ·  {now}  ·  {hits} stocks matched".ljust(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")
    if not hits:
        print(YLW("  No Weinstein Stage 2 signals today.")); return
    print()
    hdr = (f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}  "
           f"{'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'MA30W':>8}  {'M':>3}  {'SCR':>3}  "
           f"STAGE NOTE / CONF")
    print(BOLD(hdr))
    print("  " + "─"*(W-2))
    sorted_r = sorted(results[:TOP_N], key=lambda r: -r["score"])
    for rank, r in enumerate(sorted_r, 1):
        conf_str = " ".join(r.get("conf", []))
        note_str = r.get("stage_note", "")
        display  = f"{note_str}  {CYN(conf_str)}" if conf_str else note_str
        mkt_s    = YLW(f"{r['mkt']:<3}") if r["mkt"] != "US" else DIM(f"{r['mkt']:<3}")
        ticker_s = BOLD(f"{r['ticker']:<8}")
        row = (f"  {rank:>3}  {mkt_s}  {ticker_s}  {r['price']:>9.2f}"
               f"  {r['rsi']:>5.1f}  {r['adx']:>5.1f}  {r['vol_ratio']:>5.2f}"
               f"  {r['ma30w']:>8.2f}  {r['minervini']:>3}  {r['score']:>3}  "
               f"{display}")
        print(row)
    print("\n  " + "─"*(W-2))
    print(f"\n  {DIM('Strategy: Weinstein Stage 1→2  ·  hold=' + str(HOLD_DAYS) + 'd  ·  weekly 30SMA crossover')}")
    print("╚" + "═"*(W-2) + "╝\n")


# ── SCAN (public silent API) ──────────────────────────────────────────────────

def scan(universe: dict, bench_returns: dict, with_backtest: bool = False) -> list:
    """Run the scanning loop silently; return list of result dicts with strategy tag."""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(analyze_ticker, t, bench_returns.get(bench), with_backtest): t
            for t, bench in universe.items()
        }
        for f in as_completed(futs, timeout=300):
            try:
                r = f.result(timeout=60)
            except Exception:
                continue
            if r:
                results.append(r)
    results.sort(key=lambda x: -x["score"])
    return results


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args      = set(sys.argv[1:])
    fast_mode = "--fast" in args

    universe = get_universe(fast_mode)
    bench_syms = set(universe.values())
    print(DIM(f"  Fetching {len(bench_syms)} benchmark indices..."), flush=True)
    bench_returns = fetch_benchmark_returns(bench_syms)
    fetched = [f"{k.replace('^','')}:{bench_returns[k]:+.1f}%"
               for k in sorted(bench_returns)]
    if fetched:
        print(DIM("  " + "  ".join(fetched)))
    print(DIM(f"  Scanning {len(universe)} tickers  ·  "
              f"Weinstein Stage 1→2  ·  hold={HOLD_DAYS}d  ·  FRESH_WINDOW={FRESH_WINDOW}wk"))
    print()

    t0, results, done = time.time(), [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(analyze_ticker, t, bench_returns.get(b)): t
                for t, b in universe.items()}
        for f in as_completed(futs, timeout=600):
            done += 1
            try:
                r = f.result(timeout=60)
            except Exception:
                continue
            if r: results.append(r)
            if done % 150 == 0:
                print(DIM(f"  {done}/{len(universe)}  hits={len(results)}"
                          f"  elapsed={time.time()-t0:.0f}s"), flush=True)

    results.sort(key=lambda x: -x["score"])
    print(DIM(f"  Done in {time.time()-t0:.0f}s — {len(results)} candidates\n"))
    print_results(results)


if __name__ == "__main__":
    main()
