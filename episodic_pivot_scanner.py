#!/usr/bin/env python3
"""
Episodic Pivot Scanner  |  Gil Morales & Chris Kacher
──────────────────────────────────────────────────────
A catalyst-driven permanent institutional repricing: gap ≥8% on 2.5× volume,
held above gap_day_open, not overextended (≤25% above gap close).
Minervini ≥4, ADX 16-45.

python3 episodic_pivot_scanner.py                # full scan + backtest
python3 episodic_pivot_scanner.py --no-backtest  # signals only
"""

import os, sys, warnings, logging
import numpy  as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from scanner_utils import _adx, _ema, _quiet, _rsi, _sma

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOLD_DAYS      = 10
MAX_WORKERS    = 25
MIN_PRICE      = 1.0
MIN_AVG_VOL    = 100_000
MIN_MINERVINI  = 4
MIN_ADX        = 16
MAX_ADX        = 45
GAP_WINDOW     = 10    # look back N trading days for the gap event
MIN_GAP_PCT    = 0.08  # gap must be ≥8%
MIN_GAP_VOL_X  = 2.5   # gap-day volume must be ≥2.5× 20d avg
MAX_EXTENSION  = 1.25  # current close must not exceed 1.25× gap_day_close


def _get_mkt(ticker: str) -> str:
    for sfx, mkt in {".L": "UK", ".DE": "DE", ".PA": "FR",
                      ".AS": "NL", ".TO": "CA"}.items():
        if ticker.endswith(sfx):
            return mkt
    return "US"


def _build(df: pd.DataFrame) -> pd.DataFrame:
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


def _minervini(df: pd.DataFrame, idx: int) -> int:
    row = df.iloc[idx]
    c   = float(row["Close"])
    return sum([
        c > row["sma150"],
        c > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        c > row["sma50"],
        c >= 1.30 * row["52w_low"],
        c >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[max(0, idx - 20)]["sma200"],
    ])


def _find_ep_gap(df: pd.DataFrame, end_idx: int) -> Optional[dict]:
    """
    Scan the last GAP_WINDOW bars ending at end_idx for an Episodic Pivot.
    Returns dict with gap info or None.
    """
    # Search window: end_idx-GAP_WINDOW .. end_idx (inclusive)
    start = max(1, end_idx - GAP_WINDOW + 1)
    for gap_idx in range(start, end_idx + 1):
        row      = df.iloc[gap_idx]
        prev_row = df.iloc[gap_idx - 1]

        gap_open  = float(row["Open"])
        prev_close = float(prev_row["Close"])
        gap_close  = float(row["Close"])

        if prev_close <= 0:
            continue

        gap_pct = (gap_open - prev_close) / prev_close
        if gap_pct < MIN_GAP_PCT:
            continue

        # Volume check on gap day
        vm20 = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0.0
        if vm20 < MIN_AVG_VOL:
            continue
        gap_vol_ratio = float(row["Volume"]) / vm20 if vm20 > 0 else 0.0
        if gap_vol_ratio < MIN_GAP_VOL_X:
            continue

        # Check gap has not been filled in subsequent bars
        filled = False
        for j in range(gap_idx + 1, end_idx + 1):
            if float(df.iloc[j]["Close"]) < gap_open:
                filled = True
                break
        if filled:
            continue

        # Not overextended: current close ≤ 1.25× gap_day_close
        current_close = float(df.iloc[end_idx]["Close"])
        if gap_close > 0 and current_close > MAX_EXTENSION * gap_close:
            continue

        return {
            "gap_idx":       gap_idx,
            "gap_pct":       gap_pct,
            "gap_open":      gap_open,
            "gap_close":     gap_close,
            "gap_vol_ratio": gap_vol_ratio,
        }

    return None


def _score_ticker(df: pd.DataFrame) -> Optional[dict]:
    if len(df) < 220:
        return None

    last_idx = len(df) - 1
    row = df.iloc[last_idx]

    price = float(row["Close"])
    if price < MIN_PRICE:
        return None

    vol_ma20 = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0.0
    if vol_ma20 < MIN_AVG_VOL:
        return None

    adx_val = float(row["adx"])
    if pd.isna(adx_val) or adx_val < MIN_ADX or adx_val > MAX_ADX:
        return None

    m = _minervini(df, last_idx)
    if m < MIN_MINERVINI:
        return None

    ep = _find_ep_gap(df, last_idx)
    if ep is None:
        return None

    rsi_val = float(row["rsi"])
    if pd.isna(rsi_val):
        return None

    prev_row = df.iloc[last_idx - 1]
    prev_adx = float(prev_row["adx"]) if not pd.isna(prev_row["adx"]) else adx_val

    gap_pct       = ep["gap_pct"]
    gap_vol_ratio = ep["gap_vol_ratio"]
    gap_close     = ep["gap_close"]
    vol_ratio     = float(row["Volume"]) / vol_ma20 if vol_ma20 > 0 else 0.0

    conf = {
        "GAP≥10%":    gap_pct >= 0.10,
        "VOL3x":      gap_vol_ratio >= 3.0,
        "GAP_HELD":   price >= ep["gap_open"],
        "NOT_OVEREXT": gap_close > 0 and price <= 1.15 * gap_close,
        "ADX↑":       adx_val > prev_adx,
    }

    # Score: gap_pct // 5 (percentage points / 5) + vol_ratio_on_gap_day // 1, cap at 8
    raw_score = int(gap_pct * 100 // 5) + int(gap_vol_ratio // 1)
    score = min(raw_score, 8)

    return {
        "score":     score,
        "fresh":     ["EP_GAP"],
        "conf":      [k for k, v in conf.items() if v],
        "minervini": m,
        "rsi":       round(rsi_val, 1),
        "adx":       round(adx_val, 1),
        "price":     round(price, 2),
        "vol_ratio": round(vol_ratio, 2),
        "gap_pct":   round(gap_pct * 100, 2),
        "gap_vol_x": round(gap_vol_ratio, 2),
    }


def _run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -HOLD_DAYS
    for i in range(220, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS:
            continue
        ep = _find_ep_gap(df, i)
        if ep is None:
            continue
        row = df.iloc[i]
        vm20 = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0.0
        if vm20 < MIN_AVG_VOL:
            continue
        adx_v = float(row["adx"]) if not pd.isna(row["adx"]) else 0.0
        if adx_v < MIN_ADX or adx_v > MAX_ADX:
            continue
        m = _minervini(df, i)
        if m < MIN_MINERVINI:
            continue
        entry = float(df.iloc[i]["Close"])
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100)
        last = i
    if not rets:
        return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n":   len(a),
            "wr":  round(100 * (a > 0).mean(), 1),
            "avg": round(float(a.mean()), 2),
            "med": round(float(np.median(a)), 2)}


def _analyze(ticker: str, with_backtest: bool) -> Optional[dict]:
    try:
        with _quiet():
            raw = yf.download(ticker, period="400d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 220:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
            return None
        raw = raw.dropna(subset=["Close", "High", "Low", "Volume"])
        if len(raw) < 220:
            return None

        df  = _build(raw.copy())
        sig = _score_ticker(df)
        if not sig:
            return None

        result = {"ticker": ticker, "mkt": _get_mkt(ticker),
                  **sig, "hold_days": HOLD_DAYS}
        if with_backtest:
            result.update(_run_backtest(df))
        return result
    except Exception:
        return None


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    """Run episodic pivot scan across universe; return list of result dicts."""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_analyze, t, with_backtest): t for t in universe}
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
            except Exception:
                continue
            if r:
                r["strategy"] = "episodic_pivot"
                results.append(r)
    results.sort(key=lambda x: (-(x.get("wr") or 0), -x["score"]))
    return results


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    from momentum_scanner import build_universe, compute_bench_returns
    import time
    wb  = "--no-backtest" not in sys.argv
    uni = build_universe()
    bench = compute_bench_returns(set(uni.values()))
    t0  = time.time()
    res = scan(uni, bench, wb)
    print(f"\nEpisodic Pivot Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  gap={r.get('gap_pct','?')}%  "
              f"vol_x={r.get('gap_vol_x','?')}  fresh={r['fresh']}  conf={r['conf']}")


if __name__ == "__main__":
    main()
