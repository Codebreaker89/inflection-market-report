#!/usr/bin/env python3
"""
Three-Weeks-Tight Scanner  |  O'Neil / IBD
───────────────────────────────────────────
Three consecutive weekly closing prices within 1.5% of each other, on drying
volume — a coiling setup before a breakout. Minervini ≥5/8, ADX 16-35, RSI 45-70.

python3 three_weeks_tight_scanner.py                # full scan + backtest
python3 three_weeks_tight_scanner.py --no-backtest  # signals only
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
HOLD_DAYS      = 7
MAX_WORKERS    = 25
MIN_PRICE      = 1.0
MIN_AVG_VOL    = 100_000
MIN_MINERVINI  = 5
MIN_ADX        = 16
MAX_ADX        = 35
TIGHT_THRESH   = 0.015   # max spread across 3 weekly closes


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


def _score_ticker(df: pd.DataFrame) -> Optional[dict]:
    """Run 3-weeks-tight detection on the last bar of df (daily OHLCV)."""
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

    rsi_val = float(row["rsi"])
    if pd.isna(rsi_val) or rsi_val < 45 or rsi_val > 70:
        return None

    m = _minervini(df, last_idx)
    if m < MIN_MINERVINI:
        return None

    # ── Build weekly bars from daily OHLCV ───────────────────────────────────
    weekly = (
        df[["Open", "High", "Low", "Close", "Volume"]]
        .resample("W")
        .agg({"Open": "first", "High": "max", "Low": "min",
              "Close": "last", "Volume": "sum"})
        .dropna(subset=["Close"])
    )
    if len(weekly) < 4:
        return None

    # Use last 4 weeks: weeks[-4], [-3], [-2], [-1]
    # Check closes of last 3 completed weeks: indices -3, -2, -1
    w3  = weekly.iloc[-3]
    w2  = weekly.iloc[-2]
    w1  = weekly.iloc[-1]

    closes = [float(w3["Close"]), float(w2["Close"]), float(w1["Close"])]
    max_c  = max(closes)
    min_c  = min(closes)

    if min_c <= 0:
        return None

    spread = (max_c - min_c) / min_c
    if spread > TIGHT_THRESH:
        return None

    # Volume: declining week over week OR all below 20-day average
    vol3, vol2, vol1 = float(w3["Volume"]), float(w2["Volume"]), float(w1["Volume"])
    vol_declining = (vol1 < vol2 < vol3)
    vol_below_avg = (vol_ma20 > 0 and vol1 < vol_ma20 and vol2 < vol_ma20 and vol3 < vol_ma20)
    if not (vol_declining or vol_below_avg):
        return None

    # Confirmation signals
    w52_high = float(row["52w_high"]) if not pd.isna(row["52w_high"]) else 0.0
    vol_ratio = float(row["Volume"]) / vol_ma20 if vol_ma20 > 0 else 0.0

    conf = {
        "TIGHT<1%":      spread <= 0.01,
        "VOL_DECLINING": vol_declining,
        "NEAR_52H":      w52_high > 0 and price >= 0.95 * w52_high,
        "RSI50-65":      50 <= rsi_val <= 65,
    }
    score = sum(conf.values())

    return {
        "score":     score,
        "fresh":     ["3WT"],
        "conf":      [k for k, v in conf.items() if v],
        "minervini": m,
        "rsi":       round(rsi_val, 1),
        "adx":       round(adx_val, 1),
        "price":     round(price, 2),
        "vol_ratio": round(vol_ratio, 2),
        "spread_pct": round(spread * 100, 2),
    }


def _run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS:
            continue
        # Build weekly bars up to index i
        sub = df.iloc[:i + 1]
        weekly = (
            sub[["Open", "High", "Low", "Close", "Volume"]]
            .resample("W")
            .agg({"Open": "first", "High": "max", "Low": "min",
                  "Close": "last", "Volume": "sum"})
            .dropna(subset=["Close"])
        )
        if len(weekly) < 4:
            continue
        w3 = weekly.iloc[-3]; w2 = weekly.iloc[-2]; w1 = weekly.iloc[-1]
        closes = [float(w3["Close"]), float(w2["Close"]), float(w1["Close"])]
        max_c, min_c = max(closes), min(closes)
        if min_c <= 0 or (max_c - min_c) / min_c > TIGHT_THRESH:
            continue
        vol3, vol2, vol1 = float(w3["Volume"]), float(w2["Volume"]), float(w1["Volume"])
        row = df.iloc[i]
        vm20 = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0.0
        vol_declining = vol1 < vol2 < vol3
        vol_below_avg = (vm20 > 0 and vol1 < vm20 and vol2 < vm20 and vol3 < vm20)
        if not (vol_declining or vol_below_avg):
            continue
        m = _minervini(df, i)
        if m < MIN_MINERVINI:
            continue
        adx_v = float(row["adx"]) if not pd.isna(row["adx"]) else 0.0
        if adx_v < MIN_ADX or adx_v > MAX_ADX:
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
    """Run 3-weeks-tight scan across universe; return list of result dicts."""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_analyze, t, with_backtest): t for t in universe}
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
            except Exception:
                continue
            if r:
                r["strategy"] = "three_weeks_tight"
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
    print(f"\n3-Weeks-Tight Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  spread={r.get('spread_pct','?')}%  "
              f"fresh={r['fresh']}  conf={r['conf']}")


if __name__ == "__main__":
    main()
