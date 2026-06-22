#!/usr/bin/env python3
"""
Cup and Handle Scanner  |  William O'Neil / IBD
──────────────────────────────────────────────────────────────────────────────
Detects the classic Cup and Handle base pattern before a breakout.

Cup criteria:
  - Depth 12–35% from left lip to cup low  (not too shallow, not too deep)
  - Duration 30–200 trading days           (6 weeks to ~9 months)
  - Rounded bottom: cup low in middle 60% of cup duration (not a V-bottom)
  - Right lip recovery within 5% of left lip high

Handle criteria:
  - Duration 5–25 trading days
  - Depth ≤ 12% from handle high to handle low  (tight consolidation)
  - Handle sits in upper half of the cup (doesn't undercut cup midpoint)
  - Volume drying up in handle (institutional patience, no distribution)

Entry trigger:
  - Current price within 3% BELOW the pivot (top of handle) — ready to break

Hard filters:
  - Price > SMA50 and SMA200
  - Minervini ≥ 5
  - Volume > 50k average (liquidity)
  - ADX ≥ 15 (some trend present before the base)

Hold: 10 days

python3 cup_handle_scanner.py --no-backtest
python3 cup_handle_scanner.py
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

HOLD_DAYS    = 10
MAX_WORKERS  = 25
FRESH_WINDOW = 3   # must be in pivot zone within last N days

CUP_DEPTH_MIN     = 0.12
CUP_DEPTH_MAX     = 0.35
CUP_DUR_MIN       = 30    # bars
CUP_DUR_MAX       = 200   # bars
RIGHT_LIP_TOL     = 0.05  # right lip within 5% of left lip
HANDLE_DUR_MIN    = 5
HANDLE_DUR_MAX    = 25
HANDLE_DEPTH_MAX  = 0.12
PIVOT_ZONE        = 0.03  # entry within 3% below pivot
ROUNDED_LOW_RANGE = (0.15, 0.85)   # cup low must be in middle 70% of cup

# ── Helpers ───────────────────────────────────────────────────────────────────

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
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["sma200_20d_ago"] = df["sma200"].shift(20)
    df["vol_ma20"] = v.rolling(20).mean()
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    macd           = _ema(c, 12) - _ema(c, 26)
    df["macd_hist"] = macd - _ema(macd, 9)
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df

# ── Core pattern detection ────────────────────────────────────────────────────

def _detect_cup_handle(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """
    Scan for the best (tightest) cup-and-handle formation ending at `idx`.
    Tries handle lengths from HANDLE_DUR_MIN to HANDLE_DUR_MAX.
    Returns best match or None.
    """
    close = df["Close"]
    vol   = df["Volume"]
    vol_ma = float(df["vol_ma20"].iloc[idx])

    c_now = float(close.iloc[idx])
    if c_now <= 0: return None

    best = None

    for h_len in range(HANDLE_DUR_MIN, HANDLE_DUR_MAX + 1):
        # Need enough bars before the handle for a cup
        if idx - h_len < CUP_DUR_MIN: continue

        # ── Handle region ─────────────────────────────────────────────────────
        h_start = idx - h_len + 1
        h_slice = close.iloc[h_start : idx + 1]
        h_high  = float(h_slice.max())
        h_low   = float(h_slice.min())
        if h_high <= 0: continue

        h_depth = (h_high - h_low) / h_high
        if h_depth > HANDLE_DEPTH_MAX: continue   # handle too deep

        # Current price in pivot zone (within 3% below handle high)
        if c_now < h_high * (1 - PIVOT_ZONE): continue
        if c_now > h_high * 1.02: continue        # already broken out — skip

        # ── Cup region ────────────────────────────────────────────────────────
        cup_end   = idx - h_len         # bar immediately before handle
        cup_start = max(0, idx - CUP_DUR_MAX)
        if cup_end - cup_start < CUP_DUR_MIN: continue

        cup_slice  = close.iloc[cup_start : cup_end + 1]
        cup_len    = len(cup_slice)
        left_lip   = float(cup_slice.max())
        cup_low_v  = float(cup_slice.min())
        if left_lip <= 0: continue

        cup_depth  = (left_lip - cup_low_v) / left_lip
        if not (CUP_DEPTH_MIN <= cup_depth <= CUP_DEPTH_MAX): continue

        # Right lip (handle high) must recover to within 5% of left lip
        if h_high < left_lip * (1 - RIGHT_LIP_TOL): continue
        if h_high > left_lip * 1.08: continue    # right lip far above left → suspicious

        # Rounded bottom: cup low must NOT be in first/last 15% of cup duration
        cup_low_pos = int(cup_slice.values.argmin())
        rel_pos     = cup_low_pos / cup_len
        if not (ROUNDED_LOW_RANGE[0] <= rel_pos <= ROUNDED_LOW_RANGE[1]): continue

        # Handle in upper half of cup
        cup_midpoint = cup_low_v + (left_lip - cup_low_v) / 2
        if h_low < cup_midpoint: continue

        # Volume drying in handle vs 20d average before handle
        h_vol_avg    = float(vol.iloc[h_start : idx + 1].mean())
        vol_drying   = h_vol_avg < vol_ma * 0.85

        # Score: tighter handle + bigger cup recovery + volume drying = better
        score_val = (1 - h_depth / HANDLE_DEPTH_MAX) * 3 + (cup_depth - CUP_DEPTH_MIN) + (1 if vol_drying else 0)

        if best is None or score_val > best["_score"]:
            best = {
                "cup_depth_pct":   round(cup_depth * 100, 1),
                "cup_duration":    cup_end - cup_start,
                "handle_depth_pct": round(h_depth * 100, 1),
                "handle_duration": h_len,
                "pivot":           round(h_high, 2),
                "left_lip":        round(left_lip, 2),
                "vol_drying":      vol_drying,
                "_score":          score_val,
            }

    return best

# ── Per-bar scoring ───────────────────────────────────────────────────────────

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row    = df.iloc[idx]
    c      = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None

    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 50_000: return None

    # Hard filters
    sma50  = float(row["sma50"])  if not pd.isna(row["sma50"])  else 0
    sma200 = float(row["sma200"]) if not pd.isna(row["sma200"]) else 0
    if c <= sma50 or c <= sma200: return None

    adx = float(row["adx"]) if not pd.isna(row["adx"]) else 0
    if adx < 15: return None

    # Minervini
    m = sum([
        c > row["sma150"] if not pd.isna(row["sma150"]) else False,
        c > sma200,
        float(row["sma150"]) > sma200 if not pd.isna(row["sma150"]) else False,
        sma50 > float(row["sma150"]) if not pd.isna(row["sma150"]) else False,
        c > sma50,
        c >= 1.30 * float(row["52w_low"]) if not pd.isna(row["52w_low"]) else False,
        c >= 0.75 * float(row["52w_high"]) if not pd.isna(row["52w_high"]) else False,
        sma200 > float(row["sma200_20d_ago"]) if not pd.isna(row["sma200_20d_ago"]) else False,
    ])
    if m < 5: return None

    pattern = _detect_cup_handle(df, idx)
    if not pattern: return None

    rsi      = float(row["rsi"])      if not pd.isna(row["rsi"])      else 50
    macd_h   = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0
    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0

    conf_flags = {
        f"CUP{pattern['cup_depth_pct']}%":     True,   # always present — cup confirmed
        f"HDL{pattern['handle_depth_pct']}%":  True,   # handle depth shown
        "VOLdry":   pattern["vol_drying"],
        "MACD+":    macd_h > 0,
        "RSI50-70": 50 <= rsi <= 70,
        "M≥6":      m >= 6,
        "VOL1.5x":  vol_ratio > 1.5,
    }
    score = sum(conf_flags.values())

    return {
        "score":            score,
        "fresh":            [f"C&H", f"PIV@{pattern['pivot']}"],
        "conf":             [k for k, v in conf_flags.items() if v],
        "rsi":              round(rsi, 1),
        "adx":              round(adx, 1),
        "vol_ratio":        round(vol_ratio, 2),
        "minervini":        m,
        "price":            round(c, 2),
        "cup_depth_pct":    pattern["cup_depth_pct"],
        "cup_duration":     pattern["cup_duration"],
        "handle_depth_pct": pattern["handle_depth_pct"],
        "handle_duration":  pattern["handle_duration"],
        "pivot":            pattern["pivot"],
        "vol_drying":       pattern["vol_drying"],
    }

# ── Backtest ──────────────────────────────────────────────────────────────────

def _backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        sig = _score(df, i)
        if not sig: continue
        entry = float(df.iloc[i]["Close"])
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100)
        last = i
    if not rets: return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n": len(a), "wr": round(100*(a>0).mean(),1),
            "avg": round(float(a.mean()),2), "med": round(float(np.median(a)),2)}

# ── Per-ticker analysis ───────────────────────────────────────────────────────

def analyze_ticker(ticker: str, bench_ret: Optional[float],
                   with_backtest: bool) -> Optional[dict]:
    try:
        with _quiet():
            raw = yf.download(ticker, period="500d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 220: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return None
        raw = raw.dropna(subset=["Close","High","Low","Volume"])
        if len(raw) < 220: return None

        df   = _build(raw.copy())
        last = len(df) - 1

        # Check if pattern exists in last FRESH_WINDOW bars
        found = any(_score(df, k) is not None
                    for k in range(max(215, last - FRESH_WINDOW + 1), last + 1))
        if not found: return None

        sig = _score(df, last)
        if not sig: return None

        result = {"ticker": ticker, **sig, "hold_days": HOLD_DAYS}
        for sfx, mkt in {".L":"UK",".DE":"DE",".PA":"FR",".AS":"NL",".TO":"CA"}.items():
            if ticker.endswith(sfx): result["mkt"] = mkt; break
        else:
            result["mkt"] = "US"

        if with_backtest: result.update(_backtest(df))
        return result
    except Exception:
        return None

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
                    r["strategy"] = "cup_handle"
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
    print(f"\nCup & Handle Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10}  cup={r['cup_depth_pct']}%/{r['cup_duration']}d  "
              f"hdl={r['handle_depth_pct']}%/{r['handle_duration']}d  "
              f"pivot={r['pivot']}  m={r['minervini']}  score={r['score']}")


if __name__ == "__main__":
    main()
