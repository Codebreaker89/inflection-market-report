#!/usr/bin/env python3
"""
Momentum Burst Scanner  |  Stockbee / Pradeep Bonde
─────────────────────────────────────────────────────
Detects the FIRST explosive day of a new momentum move after a period of
volatility compression. Stocks tend to move 8-40% in 3-5 days after this trigger.

Signal criteria:
  • Price up ≥4% from prior close (range expansion breakout)
  • Volume ≥ 1.5x 20-day average (institutional participation)
  • Prior consolidation: NR3-5 days (or low ATR) in the 3-7 days before signal
  • NOT already up 3 consecutive days before today (fresh move, not extended)
  • Trend Intensity (TI): 7d avg close / 65d avg close ≥ 1.03
    (short trend above medium trend — in a rising structure)
  • Price > $5, avg volume > 300k (liquid)
  • Minervini score ≥ 5 (Stage 2 uptrend context)

python3 momentum_burst_scanner.py --no-backtest
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

HOLD_DAYS   = 5
MAX_WORKERS = 25
MIN_BURST   = 0.04    # 4% up from prior close
MIN_VOL_X   = 1.5     # 1.5x avg volume
MIN_TI      = 1.03    # trend intensity: 7d avg / 65d avg
FRESH_DAYS  = 2       # signal within last N bars


def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["ret1"]      = c.pct_change()                    # daily return
    df["range"]     = h - l
    df["sma7"]      = _sma(c, 7)
    df["sma20"]     = _sma(c, 20)
    df["sma50"]     = _sma(c, 50)
    df["sma65"]     = _sma(c, 65)
    df["sma150"]    = _sma(c, 150)
    df["sma200"]    = _sma(c, 200)
    df["rsi"]       = _rsi(c, 14)
    df["adx"]       = _adx(h, l, c, 14)
    df["vol_ma20"]  = v.rolling(20).mean()
    df["atr14"]     = _sma(df["range"], 14)
    df["52w_high"]  = c.rolling(252).max()
    df["52w_low"]   = c.rolling(252).min()
    # Prior 3-day range (NR3-like consolidation): min range of 3 days before signal
    df["range_3d_min"] = df["range"].shift(1).rolling(3).min()
    df["range_10d_avg"]= df["range"].shift(1).rolling(10).mean()
    return df


def _is_burst(df: pd.DataFrame, idx: int) -> bool:
    """Return True if idx is a valid momentum burst day."""
    if idx < 230: return False
    row  = df.iloc[idx]
    prev = df.iloc[idx - 1]

    c       = float(row["Close"])
    c_prev  = float(prev["Close"])
    if c_prev <= 0: return False

    # 1. Price up ≥4%
    ret = (c - c_prev) / c_prev
    if ret < MIN_BURST: return False

    # 2. Volume surge
    vol_ma = float(row["vol_ma20"])
    if vol_ma < 300_000: return False
    vol_ratio = float(row["Volume"]) / vol_ma
    if vol_ratio < MIN_VOL_X: return False

    # 3. Price > $5
    if c < 5.0: return False

    # 4. Not already up 3 consecutive days before today (fresh, not extended)
    prior_rets = [float(df.iloc[idx - k]["ret1"]) for k in range(1, 4)]
    if all(r > 0 for r in prior_rets): return False   # already running 3 days

    # 5. Prior consolidation: today's range expansion vs prior 3-day min range
    range_today   = float(row["range"])
    range_3d_min  = float(row["range_3d_min"])
    if range_3d_min > 0 and range_today < range_3d_min * 1.5:
        return False   # no real expansion vs prior compression

    # 6. Trend intensity: 7d avg / 65d avg ≥ MIN_TI
    sma7  = float(row["sma7"])
    sma65 = float(row["sma65"])
    if sma65 <= 0 or sma7 / sma65 < MIN_TI: return False

    # 7. Price above 50 SMA (trend context)
    if c < float(row["sma50"]): return False

    return True


def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if not _is_burst(df, idx): return None
    row  = df.iloc[idx]
    prev = df.iloc[idx - 1]
    c    = float(row["Close"])

    rsi = float(row["rsi"]); adx = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx): return None

    vol_ma    = float(row["vol_ma20"])
    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    ret1      = (c - float(prev["Close"])) / float(prev["Close"]) * 100

    # Minervini template score
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
        "BURST≥4%":  ret1 >= 4.0,
        "VOL≥1.5x":  vol_ratio >= 1.5,
        "TI≥1.03":   float(row["sma7"]) / float(row["sma65"]) >= MIN_TI if float(row["sma65"]) > 0 else False,
        "M≥6":       m >= 6,
        "RSI50-70":  50 <= rsi <= 70,
    }
    score = sum(conf.values())

    return {
        "score":      score,
        "fresh":      ["BURST"],
        "conf":       [k for k, v in conf.items() if v],
        "minervini":  m,
        "rsi":        round(rsi, 1),
        "adx":        round(adx, 1),
        "price":      round(c, 2),
        "vol_ratio":  round(vol_ratio, 2),
        "burst_pct":  round(ret1, 2),
    }


def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(230, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_burst(df, i): continue
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
        # Freshness: burst must have fired within last FRESH_DAYS bars
        found = any(_is_burst(df, k)
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
    return sorted(results, key=lambda x: (-x.get("vol_ratio",0), x.get("score",0)))


if __name__ == "__main__":
    import argparse
    from momentum_scanner import build_universe, compute_bench_returns
    p = argparse.ArgumentParser(); p.add_argument("--no-backtest", action="store_true")
    args = p.parse_args()
    u = build_universe(); br = compute_bench_returns(set(u.values()))
    res = scan(u, br, not args.no_backtest)
    for r in res[:10]:
        print(f"  {r['ticker']:<8} burst={r.get('burst_pct',0):+.1f}%  vol={r.get('vol_ratio',0):.1f}x  "
              f"rsi={r.get('rsi',0):.0f}  score={r.get('score',0)}  M={r.get('minervini',0)}")
