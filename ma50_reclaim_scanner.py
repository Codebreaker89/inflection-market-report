#!/usr/bin/env python3
"""
50 SMA Reclaim Scanner  |  Minervini / IBD
───────────────────────────────────────────
Detects stocks reclaiming their 50-day SMA after a pullback — the exact
point where institutions add to winning positions. Tight stop (below 50 SMA),
clean R:R, high WR in confirmed uptrends.

Signal criteria:
  • Stock was BELOW 50 SMA within the last 1-7 trading days (recent pullback)
  • Stock close is NOW above 50 SMA (reclaimed)
  • Volume on reclaim day ≥ 1.2x 20-day average (institutional buying)
  • 200 SMA trending up (current > 30 days ago)
  • 50 SMA above 200 SMA (Stage 2 structure)
  • RSI between 35-65 (not overbought, not broken)
  • Price > $5, avg volume > 200k

python3 ma50_reclaim_scanner.py --no-backtest
"""

import os, sys, warnings, logging
import numpy  as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from scanner_utils import _adx, _rsi, _sma, _quiet

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

HOLD_DAYS     = 7
MAX_WORKERS   = 25
PULLBACK_DAYS = 7   # stock must have been below 50 SMA within this window
MIN_VOL_X     = 1.2
FRESH_DAYS    = 2


def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma20"]  = _sma(c, 20)
    df["sma50"]  = _sma(c, 50)
    df["sma150"] = _sma(c, 150)
    df["sma200"] = _sma(c, 200)
    df["rsi"]    = _rsi(c, 14)
    df["adx"]    = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df


def _is_reclaim(df: pd.DataFrame, idx: int) -> bool:
    """True if today reclaims 50 SMA after recent pullback below it."""
    if idx < 230: return False
    row = df.iloc[idx]
    c      = float(row["Close"])
    sma50  = float(row["sma50"])
    sma200 = float(row["sma200"]) if not pd.isna(row["sma200"]) else None

    if pd.isna(sma50) or c < 5.0: return False

    # 1. Today is ABOVE 50 SMA
    if c <= sma50: return False

    # 2. Was BELOW 50 SMA within the last PULLBACK_DAYS bars (the pullback)
    was_below = any(
        float(df.iloc[idx - k]["Close"]) < float(df.iloc[idx - k]["sma50"])
        for k in range(1, PULLBACK_DAYS + 1)
        if not pd.isna(df.iloc[idx - k]["sma50"])
    )
    if not was_below: return False

    # 3. 50 SMA above 200 SMA (Stage 2)
    sma150 = float(row["sma150"]) if not pd.isna(row["sma150"]) else None
    if sma200 is not None and not pd.isna(sma50) and sma50 < sma200: return False

    # 4. 200 SMA trending up (not in Stage 3/4)
    if sma200 is not None and idx >= 230:
        sma200_30ago = float(df.iloc[idx - 30]["sma200"]) if not pd.isna(df.iloc[idx - 30]["sma200"]) else None
        if sma200_30ago is not None and sma200 < sma200_30ago: return False

    # 5. Volume confirmation
    vol_ma = float(row["vol_ma20"])
    if vol_ma < 200_000: return False
    vol_ratio = float(row["Volume"]) / vol_ma
    if vol_ratio < MIN_VOL_X: return False

    return True


def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if not _is_reclaim(df, idx): return None
    row = df.iloc[idx]
    c = float(row["Close"])

    rsi = float(row["rsi"]); adx = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx): return None
    if rsi > 70: return None    # overbought — not a clean reclaim
    if rsi < 30: return None    # too oversold — may be a falling knife

    vol_ma    = float(row["vol_ma20"])
    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    sma50     = float(row["sma50"])
    dist_pct  = (c - sma50) / sma50 * 100  # how far above 50 SMA

    # Minervini template
    m = sum([
        c > float(row["sma150"]) if not pd.isna(row["sma150"]) else False,
        c > float(row["sma200"]) if not pd.isna(row["sma200"]) else False,
        float(row["sma150"]) > float(row["sma200"]) if not pd.isna(row["sma150"]) and not pd.isna(row["sma200"]) else False,
        float(row["sma50"]) > float(row["sma150"]) if not pd.isna(row["sma150"]) else False,
        c > float(row["sma50"]),
        c >= 1.30 * float(row["52w_low"]) if not pd.isna(row["52w_low"]) else False,
        c >= 0.75 * float(row["52w_high"]) if not pd.isna(row["52w_high"]) else False,
        float(row["sma200"]) > float(df.iloc[idx - 20]["sma200"]) if idx >= 220 and not pd.isna(row["sma200"]) else False,
    ])
    if m < 5: return None

    conf = {
        "50SMA_RECLAIM": True,
        "VOL≥1.2x":      vol_ratio >= 1.2,
        "RSI35-65":      35 <= rsi <= 65,
        "ADX>15":        adx > 15,
        "M≥6":           m >= 6,
        "TIGHT<3%":      dist_pct <= 3.0,  # reclaimed but not far above — tight stop possible
    }
    score = sum(conf.values())

    return {
        "score":      score,
        "fresh":      ["50SMA_RECLAIM"],
        "conf":       [k for k, v in conf.items() if v],
        "minervini":  m,
        "rsi":        round(rsi, 1),
        "adx":        round(adx, 1),
        "price":      round(c, 2),
        "vol_ratio":  round(vol_ratio, 2),
        "dist_50sma": round(dist_pct, 2),
    }


def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(230, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_reclaim(df, i): continue
        row = df.iloc[i]
        rsi = float(row["rsi"])
        if pd.isna(rsi) or rsi > 70 or rsi < 30: continue
        entry = float(df.iloc[i]["Close"])
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100)
        last = i
    if not rets: return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n": len(a), "wr": round(100*(a>0).mean(),1),
            "avg": round(float(a.mean()),2), "med": round(float(np.median(a)),2)}


def analyze_ticker(ticker: str, bench_ret: Optional[float], with_backtest: bool) -> Optional[dict]:
    try:
        with _quiet():
            raw = yf.download(ticker, period="400d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 230: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return None
        raw = raw.dropna(subset=["Close","High","Low","Volume"])
        if len(raw) < 230: return None
        df   = _build(raw.copy())
        last = len(df) - 1
        found = any(_is_reclaim(df, k)
                    for k in range(max(230, last - FRESH_DAYS + 1), last + 1))
        if not found: return None
        sig = _score(df, last)
        if not sig: return None
        result = {"ticker": ticker, **sig, "hold_days": HOLD_DAYS}
        for sfx, mkt in {".L":"UK",".DE":"DE",".PA":"FR",".AS":"NL",".TO":"CA"}.items():
            if ticker.endswith(sfx): result["mkt"] = mkt; break
        else:
            result["mkt"] = "US"
        if with_backtest: result.update(run_backtest(df))
        return result
    except Exception:
        return None


def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
                for t, b in universe.items()}
        for f in as_completed(futs):
            r = f.result()
            if r: results.append(r)
    return sorted(results, key=lambda x: (-x.get("score",0), -x.get("vol_ratio",0)))


if __name__ == "__main__":
    import argparse
    from momentum_scanner import build_universe, compute_bench_returns
    p = argparse.ArgumentParser(); p.add_argument("--no-backtest", action="store_true")
    args = p.parse_args()
    u = build_universe(); br = compute_bench_returns(set(u.values()))
    res = scan(u, br, not args.no_backtest)
    for r in res[:10]:
        print(f"  {r['ticker']:<8} dist50={r.get('dist_50sma',0):+.1f}%  vol={r.get('vol_ratio',0):.1f}x  "
              f"rsi={r.get('rsi',0):.0f}  score={r.get('score',0)}  M={r.get('minervini',0)}")
