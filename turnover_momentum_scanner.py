#!/usr/bin/env python3
"""
Turnover Momentum Scanner  |  Medhat & Schmeling (RFS 2022)
─────────────────────────────────────────────────────────────────────────────
Classic 12-1 month momentum filtered by LOW share turnover.

Low turnover = uncrowded position → stronger continuation, fewer crashes.
High turnover momentum = overcrowded, prone to momentum crashes.

Reference: Medhat & Schmeling, "Short-term Momentum", RFS 2022.

python3 turnover_momentum_scanner.py                # full scan + backtest
python3 turnover_momentum_scanner.py --no-backtest  # signals only
python3 turnover_momentum_scanner.py --fast         # 30 US stocks smoke-test
"""

import os, sys, time, warnings, logging
import numpy  as np
import pandas as pd
import yfinance as yf
from datetime           import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing             import Optional
from scanner_utils      import _adx, _quiet, _rsi, _sma, _ema

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)


# ── CONFIG ────────────────────────────────────────────────────────────────────
HOLD_DAYS      = 5
STRATEGY_NAME  = "turnover_momentum"
LOOKBACK_DAYS  = 420       # ~420 days ≈ 1.5yr → enough for 252d return + buffer
MAX_WORKERS    = 20
MIN_PRICE      = 5.0
MIN_ADX        = 16
MIN_MINERVINI  = 5
MIN_RSI        = 50
MAX_RSI        = 75
TOP_N          = 20
MOM_PCTILE     = 67        # top 33% = above 67th percentile


# ── SUFFIX / MARKET MAP ───────────────────────────────────────────────────────
SUFFIX_MAP = {
    ".L":  "UK",  ".DE": "DE",  ".PA": "FR",  ".AS": "NL",
    ".MC": "ES",  ".MI": "IT",  ".SW": "CH",  ".HE": "FI",
    ".ST": "SE",  ".CO": "DK",  ".OL": "NO",  ".TO": "CA",
    ".BR": "BE",  ".LS": "PT",
}

def _get_market(ticker: str) -> str:
    for sfx, mkt in SUFFIX_MAP.items():
        if ticker.endswith(sfx):
            return mkt
    return "US"


# ── ANSI COLORS ───────────────────────────────────────────────────────────────
_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _color else text
GRN  = lambda t: _c("32", t);  RED  = lambda t: _c("31", t)
YLW  = lambda t: _c("33", t);  CYN  = lambda t: _c("36", t)
BOLD = lambda t: _c("1",  t);  DIM  = lambda t: _c("2",  t)

def _ret_fmt(v):
    if v is None: return DIM("   ─  ")
    return GRN(f"+{v:.2f}%") if v >= 0 else RED(f"{v:.2f}%")

def _wr_fmt(v):
    if v is None: return DIM("  ─ ")
    return GRN(f"{v:.0f}%") if v >= 60 else (YLW(f"{v:.0f}%") if v >= 45 else RED(f"{v:.0f}%"))


# ── DATA FETCH ────────────────────────────────────────────────────────────────

def _fetch_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    """Download ~1.5yr of daily OHLCV. Returns None on failure or insufficient data."""
    try:
        with _quiet():
            raw = yf.download(
                ticker, period=f"{LOOKBACK_DAYS}d", interval="1d",
                progress=False, auto_adjust=True, threads=False,
            )
        if raw is None or len(raw) < 260:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if not {"Close", "High", "Low", "Volume"}.issubset(raw.columns):
            return None
        raw = raw.dropna(subset=["Close", "High", "Low", "Volume"])
        if len(raw) < 260:
            return None
        return raw
    except Exception:
        return None


def _get_shares_outstanding(ticker: str) -> float:
    """Fetch sharesOutstanding from yfinance .info. Fail-open to 0."""
    try:
        info = yf.Ticker(ticker).info
        val  = info.get("sharesOutstanding") or 0
        return float(val)
    except Exception:
        return 0.0


def _get_ticker_meta(ticker: str) -> dict:
    """Fetch company name and sector. Fail-open to empty strings."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "company": info.get("shortName") or info.get("longName") or "",
            "sector":  info.get("sector") or "",
        }
    except Exception:
        return {"company": "", "sector": ""}


# ── INDICATORS ────────────────────────────────────────────────────────────────

def _build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["Close"], df["High"], df["Low"]
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df


def _minervini_score(df: pd.DataFrame, idx: int) -> int:
    """Compute Minervini Trend Template score (0-8)."""
    if idx < 20:
        return 0
    row = df.iloc[idx]
    return sum([
        bool(row["Close"]  > row["sma150"]),
        bool(row["Close"]  > row["sma200"]),
        bool(row["sma150"] > row["sma200"]),
        bool(row["sma50"]  > row["sma150"]),
        bool(row["Close"]  > row["sma50"]),
        bool(row["Close"]  >= 1.30 * row["52w_low"]),
        bool(row["Close"]  >= 0.75 * row["52w_high"]),
        bool(row["sma200"] > df.iloc[idx - 20]["sma200"]),
    ])


# ── PER-TICKER METRICS ────────────────────────────────────────────────────────

def _compute_ticker_metrics(ticker: str) -> Optional[dict]:
    """
    Download data and compute all metrics needed for cross-sectional ranking.
    Returns None on failure or if the stock fails quality gates (price, RSI, ADX, Minervini).
    Returns a dict with ret_12_1m and weekly_turnover as raw floats (not yet ranked).
    """
    df = _fetch_ohlcv(ticker)
    if df is None:
        return None

    df  = _build_indicators(df)
    idx = len(df) - 1

    # Need at least 252 trading days for the 12-1m return window
    if idx < 252:
        return None

    price = float(df["Close"].iloc[idx])
    if price < MIN_PRICE:
        return None

    rsi_val = float(df["rsi"].iloc[idx])
    adx_val = float(df["adx"].iloc[idx])
    if pd.isna(rsi_val) or pd.isna(adx_val):
        return None
    if not (MIN_RSI <= rsi_val <= MAX_RSI):
        return None
    if adx_val < MIN_ADX:
        return None

    m = _minervini_score(df, idx)
    if m < MIN_MINERVINI:
        return None

    # 12-1 month return: (close[-21] / close[-252]) - 1
    # Uses close 21 trading days ago (skip last month) vs 252 days ago
    c_skip  = float(df["Close"].iloc[-21])
    c_start = float(df["Close"].iloc[-252])
    if c_start <= 0:
        return None
    ret_12_1m = c_skip / c_start - 1.0

    # Weekly turnover = mean(weekly_volume / shares_outstanding) across last 52 weeks
    shares = _get_shares_outstanding(ticker)
    weekly_vol = df["Volume"].resample("W").sum()
    last_52w   = weekly_vol.iloc[-52:] if len(weekly_vol) >= 52 else weekly_vol
    if shares > 0 and len(last_52w) > 0:
        weekly_turnover = float((last_52w / shares).mean())
    else:
        weekly_turnover = 0.0   # fail-open: treated as minimum turnover (still qualifies)

    vol_ma    = float(df["vol_ma20"].iloc[idx])
    vol_ratio = float(df["Volume"].iloc[idx]) / vol_ma if vol_ma > 0 else 0.0

    return {
        "ticker":          ticker,
        "market":          _get_market(ticker),
        "price":           round(price, 2),
        "rsi":             round(rsi_val, 1),
        "adx":             round(adx_val, 1),
        "minervini":       m,
        "vol_ratio":       round(vol_ratio, 2),
        "ret_12_1m":       ret_12_1m,       # raw; rounded to % only in final result dict
        "weekly_turnover": weekly_turnover,  # raw; used only for cross-sectional ranking
    }


# ── BACKTEST ──────────────────────────────────────────────────────────────────

def _run_backtest(ticker: str) -> dict:
    """
    Simplified per-ticker backtest: enters when 12-1m momentum is positive,
    holds HOLD_DAYS, records return.
    """
    try:
        df = _fetch_ohlcv(ticker)
        if df is None or len(df) < 252 + HOLD_DAYS + 25:
            return {"n": 0, "wr": None, "avg": None, "med": None}
        df   = _build_indicators(df)
        rets = []
        last = -HOLD_DAYS
        for i in range(252, len(df) - HOLD_DAYS - 1):
            if i - last < HOLD_DAYS:
                continue
            # Momentum signal: price 21 days ago > price 252 days ago
            if i < 252 or (i - 21) < 0:
                continue
            c_skip  = float(df["Close"].iloc[i - 21])
            c_start = float(df["Close"].iloc[i - 252])
            if c_start <= 0 or c_skip <= c_start:
                continue
            entry = float(df["Close"].iloc[i])
            exit_ = float(df["Close"].iloc[i + HOLD_DAYS])
            rets.append((exit_ - entry) / entry * 100)
            last = i
        if not rets:
            return {"n": 0, "wr": None, "avg": None, "med": None}
        a = np.array(rets)
        return {
            "n":   len(a),
            "wr":  round(100 * (a > 0).mean(), 1),
            "avg": round(float(a.mean()),       2),
            "med": round(float(np.median(a)),   2),
        }
    except Exception:
        return {"n": 0, "wr": None, "avg": None, "med": None}


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def run(tickers_with_info: list, with_backtest: bool = True) -> list:
    """
    Cross-sectional Medhat & Schmeling turnover-filtered momentum scanner.

    Algorithm:
      1. Download 1yr OHLCV for all tickers in parallel; compute 12-1m return
         and weekly turnover.
      2. Keep only the top 33% by 12-1m return (strong momentum).
      3. Among those, keep only stocks with turnover BELOW the median of the
         FULL scanned universe (low turnover = uncrowded).
      4. Apply quality gates (price > $5, RSI 50-75, Minervini ≥ 5, ADX > 16)
         — these are enforced in step 1 as pre-filters.
      5. Return top 20 sorted by Minervini score then momentum.

    Args:
        tickers_with_info: list of dicts with at least {"ticker": str, "market": str}
        with_backtest:     if True, run a simple per-ticker momentum backtest

    Returns:
        list of result dicts (up to TOP_N=20) with keys:
        ticker, strategy, market, price, rsi, adx, score, vol_ratio,
        minervini, company, sector, ret_12_1m, turnover_rank
        (plus n, wr, avg, med if with_backtest=True)
    """
    tickers = [d["ticker"] for d in tickers_with_info]
    if not tickers:
        return []

    print(DIM(f"  [turnover_momentum] computing metrics for {len(tickers)} tickers..."),
          flush=True)

    # ── Step 1: Per-ticker metrics (parallel) ──────────────────────────────
    raw_metrics: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_compute_ticker_metrics, t): t for t in tickers}
        for f in as_completed(futs, timeout=600):
            t = futs[f]
            try:
                r = f.result(timeout=60)
                if r:
                    raw_metrics[t] = r
            except Exception:
                pass

    if not raw_metrics:
        return []

    # ── Step 2: Cross-sectional momentum filter — top 33% ─────────────────
    all_rets      = [m["ret_12_1m"] for m in raw_metrics.values()]
    mom_threshold = float(np.percentile(all_rets, MOM_PCTILE))
    mom_pass      = {t: m for t, m in raw_metrics.items()
                     if m["ret_12_1m"] >= mom_threshold}

    if not mom_pass:
        return []

    # ── Step 3: Turnover filter — below median of the FULL universe ────────
    all_turnovers    = np.array([m["weekly_turnover"] for m in raw_metrics.values()])
    turnover_median  = float(np.median(all_turnovers))
    low_turn_pass    = {t: m for t, m in mom_pass.items()
                        if m["weekly_turnover"] <= turnover_median}

    if not low_turn_pass:
        return []

    # ── Step 4: Build result dicts + optional backtest ─────────────────────
    results = []
    for t, m in low_turn_pass.items():
        # Percentile rank in turnover among ALL scanned stocks (0 = least crowded)
        turnover_pct = float((all_turnovers <= m["weekly_turnover"]).mean())

        meta = _get_ticker_meta(t)

        sig: dict = {
            "ticker":        t,
            "strategy":      STRATEGY_NAME,
            "market":        m["market"],
            "price":         m["price"],
            "rsi":           m["rsi"],
            "adx":           m["adx"],
            "score":         m["minervini"],   # Minervini score used as scan score
            "vol_ratio":     m["vol_ratio"],
            "minervini":     m["minervini"],
            "company":       meta["company"],
            "sector":        meta["sector"],
            "ret_12_1m":     round(m["ret_12_1m"] * 100, 1),
            "turnover_rank": round(turnover_pct, 2),
        }
        if with_backtest:
            sig.update(_run_backtest(t))
        results.append(sig)

    # Sort: Minervini desc, then 12-1m return desc
    results.sort(key=lambda r: (-r["minervini"], -r["ret_12_1m"]))
    return results[:TOP_N]


# ── SCAN (adapter for scan.py SCANNER_MAP) ────────────────────────────────────

def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    """
    Adapter so this scanner can be wired into scan.py's SCANNER_MAP.
    universe: {ticker: benchmark_symbol}
    bench_returns: not used (cross-sectional scanner, no benchmark needed)
    """
    tickers_with_info = [{"ticker": t, "market": _get_market(t)} for t in universe]
    return run(tickers_with_info, with_backtest=with_backtest)


# ── DISPLAY ───────────────────────────────────────────────────────────────────

_W = 114

def _print_results(results: list, with_backtest: bool) -> None:
    now  = datetime.now().strftime("%Y-%m-%d  %H:%M")
    hits = len(results)
    print()
    print("╔" + "═" * (_W - 2) + "╗")
    title = f"  TURNOVER MOMENTUM SCANNER  ·  {now}  ·  {hits} stocks matched"
    print("║" + title.ljust(_W - 2) + "║")
    sub = ("  Medhat & Schmeling (RFS 2022)  ·  top-33% 12-1m momentum  ·  "
           "below-median turnover  ·  hold=" + str(HOLD_DAYS) + "d")
    print("║" + sub.ljust(_W - 2) + "║")
    print("╚" + "═" * (_W - 2) + "╝")
    if not hits:
        print(YLW("  No signals today."))
        return
    print()

    if with_backtest:
        hdr = (
            f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}"
            f"  {'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'M':>3}"
            f"  {'12-1M%':>7}  {'TURN':>5}"
            f"  {'#BT':>3}  {'WIN%':>5}  {'AVG':>7}  {'MED':>7}  COMPANY"
        )
    else:
        hdr = (
            f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}"
            f"  {'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'M':>3}"
            f"  {'12-1M%':>7}  {'TURN':>5}  COMPANY"
        )
    print(BOLD(hdr))
    print("  " + "─" * (_W - 2))

    for rank, r in enumerate(results, 1):
        mkt_s    = YLW(f"{r['market']:<3}") if r["market"] != "US" else DIM(f"{r['market']:<3}")
        tkr_s    = BOLD(f"{r['ticker']:<8}")
        ret_s    = f"{r['ret_12_1m']:>+.1f}%"
        turn_s   = f"{r['turnover_rank']:.2f}"
        meta_s   = DIM(r.get("company", "") or "")
        base = (
            f"  {rank:>3}  {mkt_s}  {tkr_s}  {r['price']:>9.2f}"
            f"  {r['rsi']:>5.1f}  {r['adx']:>5.1f}  {r['vol_ratio']:>5.1f}  {r['minervini']:>3}"
            f"  {ret_s:>7}  {turn_s:>5}  "
        )
        if with_backtest and r.get("n", 0) > 0:
            row = (base
                   + f"{r['n']:>3}  "
                   + f"{_wr_fmt(r.get('wr')):>5}  "
                   + f"{_ret_fmt(r.get('avg')):>7}  "
                   + f"{_ret_fmt(r.get('med')):>7}  "
                   + meta_s)
        elif with_backtest:
            row = base + DIM("  ─     ─      ─      ─   ") + meta_s
        else:
            row = base + meta_s
        print(row)

    print("\n  " + "─" * (_W - 2))
    print(f"\n  {DIM('Low turnover_rank = less crowded.  TURN=0.00 → most uncrowded in universe.')}")
    print("╚" + "═" * (_W - 2) + "╝\n")


# ── STANDALONE MAIN ───────────────────────────────────────────────────────────

def main() -> None:
    """Run the full scan independently (same universe as pocket_pivot_scanner)."""
    from pocket_pivot_scanner import get_universe

    args          = set(sys.argv[1:])
    fast_mode     = "--fast"        in args
    with_backtest = "--no-backtest" not in args

    print(DIM("Loading universe..."), flush=True)
    universe          = get_universe(fast_mode)
    tickers_with_info = [{"ticker": t, "market": _get_market(t)} for t in universe]

    print(DIM(
        f"  {len(tickers_with_info)} tickers  ·  "
        f"{'backtest ON' if with_backtest else 'backtest OFF'}"
        f"  ·  hold={HOLD_DAYS}d  ·  {STRATEGY_NAME}"
    ))
    print()

    t0      = time.time()
    results = run(tickers_with_info, with_backtest=with_backtest)
    elapsed = time.time() - t0
    print(DIM(f"  Done in {elapsed:.0f}s — {len(results)} signals\n"))
    _print_results(results, with_backtest)


if __name__ == "__main__":
    main()
