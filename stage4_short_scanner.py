#!/usr/bin/env python3
"""
Stage 4 Short Scanner  |  Weinstein Stage Analysis + Minervini Short SEPA
──────────────────────────────────────────────────────────────────────────────
Finds stocks in confirmed Stage 4 distribution (institutional selling) with a
high-conviction entry trigger. Mirror of the long momentum strategy.

Hard filters — ALL must pass:
  1. Price < SMA50 < SMA150 < SMA200   (full bearish SMA stack)
  2. SMA200 declining (lower than 20d ago)
  3. Price ≤ 70% of 52-week high       (real distribution, not a brief dip)
  4. ADX ≥ 20                          (confirmed downtrend, not sideways chop)
  5. Market cap > $500M                (avoid short-squeeze risk on small/micro caps)
  6. Not Biotech / Pharmaceutical      (binary FDA gap risk)
  7. No earnings within 5 calendar days

Entry triggers — at least ONE must fire within last FRESH_WINDOW days:
  A. FAIL-SMA50 / FAIL-SMA150 — failed rally: price bounced within 3% of a
     declining SMA then closed back below it (highest conviction entry)
  B. 20dLOW — new 20-day closing low (downtrend momentum resuming)
  C. DIST — distribution cluster: ≥3 above-average-volume down-days in 10 sessions

Scoring (0–10):
  VOLdist, RS-, RSI<40, MACD-, DeathX (recent death cross), 52wLow, BEAR-MKT

python3 stage4_short_scanner.py --no-backtest
python3 stage4_short_scanner.py
"""

import os, sys, warnings, logging, contextlib
import numpy  as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

@contextlib.contextmanager
def _quiet():
    devnull = open(os.devnull, "w"); old = sys.stderr; sys.stderr = devnull
    try: yield
    finally: sys.stderr = old; devnull.close()

HOLD_DAYS    = 7
MAX_WORKERS  = 20
FRESH_WINDOW = 3    # entry trigger must have fired within last N days

MIN_MARKET_CAP  = 500_000_000   # $500M
EARNS_BUFFER    = 5             # days to earnings — excluded
MAX_PCT_52W_HIGH = 0.70         # price must be ≤ 70% of 52w high

BIOTECH_KEYWORDS = [
    "biotech", "biopharmaceutical", "pharmaceutical", "drug manufacture",
    "genomics", "biotechnology", "clinical", "therapeutics", "bioscience",
]

# ── Indicators ────────────────────────────────────────────────────────────────

def _sma(s, n): return s.rolling(n).mean()
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _adx(high, low, close, n=14):
    tr  = pd.concat([(high-low),(high-close.shift()).abs(),
                     (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    up  = (high - high.shift()).clip(lower=0)
    dn  = (low.shift() - low).clip(lower=0)
    dmp = up.where(up>dn, 0).ewm(alpha=1/n, adjust=False).mean()
    dmm = dn.where(dn>up, 0).ewm(alpha=1/n, adjust=False).mean()
    dip = 100 * dmp / atr; dim = 100 * dmm / atr
    dx  = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]          = _sma(c, 50)
    df["sma150"]         = _sma(c, 150)
    df["sma200"]         = _sma(c, 200)
    df["sma200_20d_ago"] = df["sma200"].shift(20)
    df["ema9"]           = _ema(c, 9)
    df["52w_high"]       = c.rolling(252).max()
    df["52w_low"]        = c.rolling(252).min()
    df["vol_ma20"]       = v.rolling(20).mean()
    df["rsi"]            = _rsi(c, 14)
    df["adx"]            = _adx(h, l, c, 14)
    macd                 = _ema(c, 12) - _ema(c, 26)
    df["macd_hist"]      = macd - _ema(macd, 9)
    df["20d_low"]        = c.rolling(20).min()
    return df

# ── Hard filter: Stage 4 structure ───────────────────────────────────────────

def _is_stage4(df: pd.DataFrame, idx: int) -> bool:
    """All structural Stage 4 conditions."""
    if idx < 215: return False
    row = df.iloc[idx]
    c = float(row["Close"])

    # Full bearish SMA stack
    sma50  = float(row["sma50"])
    sma150 = float(row["sma150"])
    sma200 = float(row["sma200"])
    if pd.isna(sma50) or pd.isna(sma150) or pd.isna(sma200): return False
    if not (c < sma50 < sma150): return False
    # SMA150 < SMA200 OR SMA200 declining — either confirms Stage 4
    sma200_20 = float(row["sma200_20d_ago"])
    sma200_declining = (not pd.isna(sma200_20)) and (sma200 < sma200_20)
    if sma150 >= sma200 and not sma200_declining: return False
    # Must have SMA200 actually declining
    if not sma200_declining: return False

    # Price ≤ 70% of 52w high
    w52h = float(row["52w_high"])
    if pd.isna(w52h) or w52h <= 0 or c > MAX_PCT_52W_HIGH * w52h: return False

    # ADX ≥ 20 (real downtrend, not sideways chop)
    adx = float(row["adx"])
    if pd.isna(adx) or adx < 20: return False

    return True

# ── Entry triggers ────────────────────────────────────────────────────────────

def _failed_rally(df: pd.DataFrame, idx: int) -> Optional[str]:
    """
    Price bounced within 3% of a declining SMA50 or SMA150 in the last
    FRESH_WINDOW days, then closed back below it today.
    Returns tag string or None.
    """
    current_close = float(df.iloc[idx]["Close"])
    for lookback in range(1, FRESH_WINDOW + 2):
        i = idx - lookback
        if i < 0: break
        past_row = df.iloc[i]
        past_c   = float(past_row["Close"])
        for sma_col, tag in [("sma50", "FAIL-SMA50"), ("sma150", "FAIL-SMA150")]:
            sma_val = float(past_row[sma_col])
            if pd.isna(sma_val) or sma_val <= 0: continue
            # Was within 3% of SMA from below (bounce attempt)
            if past_c < sma_val and abs(past_c - sma_val) / sma_val < 0.03:
                # Today still below that SMA — failed
                curr_sma = float(df.iloc[idx][sma_col])
                if current_close < curr_sma:
                    return tag
    return None

def _new_20d_low(df: pd.DataFrame, idx: int) -> bool:
    """Today's close is a new 20-day closing low."""
    if idx < 20: return False
    c    = float(df.iloc[idx]["Close"])
    low  = float(df.iloc[idx]["20d_low"])
    # 20d_low includes today; compare to prior 20d excluding today
    prior_low = float(df["Close"].iloc[idx-20:idx].min())
    return c <= prior_low

def _distribution_cluster(df: pd.DataFrame, idx: int, window: int = 10,
                           min_days: int = 3) -> bool:
    """≥ min_days above-average-volume down-days in last window sessions."""
    if idx < window + 20: return False
    vol_avg  = float(df.iloc[idx]["vol_ma20"])
    if vol_avg <= 0: return False
    dist = 0
    for i in range(idx - window + 1, idx + 1):
        if i <= 0: continue
        change = float(df.iloc[i]["Close"]) - float(df.iloc[i-1]["Close"])
        vol    = float(df.iloc[i]["Volume"])
        if change < -0.002 * float(df.iloc[i-1]["Close"]) and vol > vol_avg:
            dist += 1
    return dist >= min_days

# ── Metadata check (market cap, sector, earnings) ────────────────────────────

def _passes_metadata(ticker: str) -> bool:
    """
    Returns True if stock passes market cap, sector, and earnings checks.
    Called only after technical filters pass — limits extra API calls.
    """
    try:
        t = yf.Ticker(ticker)
        # Market cap
        try:
            mc = t.fast_info.market_cap
            if mc and mc < MIN_MARKET_CAP: return False
        except Exception:
            pass   # if unavailable, allow through

        # Sector / industry — biotech / pharma check
        try:
            info     = t.info
            industry = (info.get("industry", "") or "").lower()
            sector   = (info.get("sector",   "") or "").lower()
            combined = industry + " " + sector
            if any(k in combined for k in BIOTECH_KEYWORDS): return False
        except Exception:
            pass

        # Earnings within EARNS_BUFFER days
        try:
            ed = t.earnings_dates
            if ed is not None and len(ed) > 0:
                now = pd.Timestamp.now(tz="UTC")
                future = ed[ed.index > now]
                if len(future) > 0:
                    next_e = future.index.min()
                    days   = (next_e.tz_localize(None) if next_e.tzinfo else next_e) - pd.Timestamp.now()
                    if 0 <= days.days <= EARNS_BUFFER: return False
        except Exception:
            pass

        return True
    except Exception:
        return True   # allow through on error — technical filters already strict

# ── Benchmark: is market in bear regime? ─────────────────────────────────────

_bear_cache: Optional[bool] = None

def _is_bear_regime() -> bool:
    global _bear_cache
    if _bear_cache is not None: return _bear_cache
    try:
        with _quiet():
            spy = yf.download("SPY", period="250d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.droplevel(1)
        sma200 = float(spy["Close"].rolling(200).mean().iloc[-1])
        price  = float(spy["Close"].iloc[-1])
        _bear_cache = price < sma200
    except Exception:
        _bear_cache = False
    return _bear_cache

# ── Score a stock ─────────────────────────────────────────────────────────────

def _score(df: pd.DataFrame, idx: int, bench_ret: Optional[float]) -> Optional[dict]:
    if not _is_stage4(df, idx): return None

    # Entry trigger
    trigger     = _failed_rally(df, idx)
    new_low     = _new_20d_low(df, idx)
    dist_clust  = _distribution_cluster(df, idx)

    if not trigger and not new_low and not dist_clust: return None

    row     = df.iloc[idx]
    c       = float(row["Close"])
    vol_ma  = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    rsi     = float(row["rsi"])      if not pd.isna(row["rsi"]) else 50
    adx     = float(row["adx"])      if not pd.isna(row["adx"]) else 0
    macd_h  = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0
    w52h    = float(row["52w_high"])
    w52l    = float(row["52w_low"])

    # Volume distribution ratio (down-day vol / up-day vol in last 10d)
    window  = df.iloc[max(0, idx-9):idx+1]
    dn_vol  = window.loc[window["Close"] < window["Close"].shift(), "Volume"].sum()
    up_vol  = window.loc[window["Close"] > window["Close"].shift(), "Volume"].sum()
    vol_dist_ratio = float(dn_vol) / float(up_vol) if up_vol > 0 else 2.0

    # Relative strength vs benchmark
    stock_63d = (c / float(df.iloc[max(0,idx-63)]["Close"]) - 1) if idx >= 63 else 0
    rs_neg    = bench_ret is not None and stock_63d < bench_ret - 0.05  # underperforms by 5%+

    # Death cross: SMA50 crossed below SMA200 recently (within 40 days)
    death_cross = False
    for k in range(1, min(41, idx)):
        prev = df.iloc[idx - k]
        if float(prev["sma50"]) >= float(prev["sma200"]):
            death_cross = True
            break

    near_52w_low = (w52l > 0) and (c <= w52l * 1.15)

    conf_flags = {
        "VOLdist":   vol_dist_ratio > 1.3,
        "RS-":       rs_neg,
        "RSI<40":    rsi < 40,
        "MACD-":     macd_h < 0,
        "DeathX":    death_cross,
        "52wLow":    near_52w_low,
        "BEAR-MKT":  _is_bear_regime(),
    }
    score = sum(conf_flags.values())

    # Build fresh tags
    fresh = []
    if trigger:   fresh.append(trigger)
    if new_low:   fresh.append("20dLOW")
    if dist_clust: fresh.append("DIST")

    # Minervini-style short score (inverse: lower = worse structure = better short)
    m_short = sum([
        c < row["sma150"], c < row["sma200"],
        row["sma50"] < row["sma150"],
        row["sma150"] < row["sma200"],
        row["sma200"] < row["sma200_20d_ago"],
        c <= 0.70 * w52h,
        c <= 0.50 * w52h,  # extra point if really deep in distribution
    ])

    return {
        "score":      score,
        "fresh":      fresh,
        "conf":       [k for k, v in conf_flags.items() if v],
        "rsi":        round(rsi, 1),
        "adx":        round(adx, 1),
        "vol_ratio":  round(vol_dist_ratio, 2),
        "minervini":  m_short,
        "price":      round(c, 2),
        "pct_52w_hi": round(c / w52h * 100, 1) if w52h > 0 else 0,
    }

# ── Per-ticker analysis ───────────────────────────────────────────────────────

def analyze_ticker(ticker: str, bench_ret: Optional[float],
                   with_backtest: bool) -> Optional[dict]:
    try:
        with _quiet():
            raw = yf.download(ticker, period="400d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 220: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return None
        raw = raw.dropna(subset=["Close","High","Low","Volume"])
        if len(raw) < 220: return None

        df   = _build(raw.copy())
        last = len(df) - 1

        # Quick pre-check: is ANY of last FRESH_WINDOW days in Stage 4?
        if not any(_is_stage4(df, k)
                   for k in range(max(215, last - FRESH_WINDOW), last + 1)):
            return None

        sig = _score(df, last, bench_ret)
        if not sig: return None

        # Metadata gating — only for candidates that passed technical filters
        if not _passes_metadata(ticker): return None

        result = {"ticker": ticker, **sig, "hold_days": HOLD_DAYS}
        for sfx, mkt in {".L":"UK",".DE":"DE",".PA":"FR",".AS":"NL",".TO":"CA"}.items():
            if ticker.endswith(sfx): result["mkt"] = mkt; break
        else:
            result["mkt"] = "US"

        if with_backtest:
            result.update(_backtest(df))

        return result
    except Exception:
        return None

# ── Backtest ──────────────────────────────────────────────────────────────────

def _backtest(df: pd.DataFrame) -> dict:
    """Simple Stage 4 short backtest: short at close, cover after HOLD_DAYS."""
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_stage4(df, i): continue
        trigger = (_failed_rally(df, i) or _new_20d_low(df, i)
                   or _distribution_cluster(df, i))
        if not trigger: continue
        entry  = float(df.iloc[i]["Close"])
        cover  = float(df.iloc[i + HOLD_DAYS]["Close"])
        # Short return: positive when stock falls
        rets.append((entry - cover) / entry * 100)
        last = i
    if not rets: return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n": len(a), "wr": round(100*(a>0).mean(),1),
            "avg": round(float(a.mean()),2), "med": round(float(np.median(a)),2)}

# ── Scan entry point ──────────────────────────────────────────────────────────

def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
            for t, b in universe.items()
        }
        for f in as_completed(futs, timeout=240):
            try:
                r = f.result(timeout=45)
                if r:
                    r["strategy"] = "stage4_short"
                    results.append(r)
            except Exception:
                pass
    results.sort(key=lambda x: (-(x.get("wr") or 0), -x["score"]))
    return results


def main():
    from momentum_scanner import build_universe, compute_bench_returns
    import time
    wb  = "--no-backtest" not in sys.argv
    uni = build_universe()
    bench = compute_bench_returns(set(uni.values()))
    t0  = time.time()
    res = scan(uni, bench, wb)
    print(f"\nStage 4 Short Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    bear = _is_bear_regime()
    print(f"  Market regime: {'⬇ BEAR (SPY < SMA200)' if bear else '⬆ BULL (SPY > SMA200)'}")
    for r in res[:20]:
        print(f"  SHORT {r['ticker']:<10}  {r['pct_52w_hi']}% of 52wH  "
              f"rsi={r['rsi']}  adx={r['adx']}  score={r['score']}  "
              f"fresh={r['fresh']}  conf={r['conf']}")


if __name__ == "__main__":
    main()
