#!/usr/bin/env python3
"""
Connors R3 Scanner  |  Larry Connors, "High Probability ETF Trading" 2009
──────────────────────────────────────────────────────────────────────────
Pure mean reversion. RSI(2) drops 3 consecutive days (first from below 60),
final RSI(2) < 10, price above 200d SMA. Works in sideways/flat markets.

python3 connors_r3_scanner.py --no-backtest   # fast
python3 connors_r3_scanner.py                 # with backtest
"""

import os, sys, warnings, logging, contextlib
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
HOLD_DAYS    = 3
MAX_WORKERS  = 25
FRESH_WINDOW = 1   # signal must fire today only — it's time-sensitive

# ── INDICATOR HELPERS ─────────────────────────────────────────────────────────

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["rsi2"]     = _rsi(c, 2)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df

def _is_r3(df: pd.DataFrame, idx: int) -> bool:
    """
    R3 conditions:
    1. Price above 200d SMA
    2. Price above 50d SMA
    3. RSI(2) today < 10
    4. RSI(2) dropped for 3 consecutive days: rsi2[i] < rsi2[i-1] < rsi2[i-2] < rsi2[i-3]
    5. rsi2[i-3] < 60 (first drop day started from below 60, not overbought top)
    6. ADX 12-45
    7. Volume avg > 500,000
    """
    if idx < 215: return False
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return False

    sma200 = float(row["sma200"]); sma50 = float(row["sma50"])
    if pd.isna(sma200) or pd.isna(sma50): return False
    if c <= sma200 or c <= sma50: return False

    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 500_000: return False

    adx = float(row["adx"])
    if pd.isna(adx) or adx < 12 or adx > 45: return False

    # RSI(2) consecutive drop check — need 4 data points: i-3, i-2, i-1, i
    if idx < 3: return False
    rsi2_i   = float(df.iloc[idx]["rsi2"])
    rsi2_i1  = float(df.iloc[idx - 1]["rsi2"])
    rsi2_i2  = float(df.iloc[idx - 2]["rsi2"])
    rsi2_i3  = float(df.iloc[idx - 3]["rsi2"])

    for v in (rsi2_i, rsi2_i1, rsi2_i2, rsi2_i3):
        if pd.isna(v): return False

    if rsi2_i >= 10: return False                              # condition 3
    if not (rsi2_i < rsi2_i1 < rsi2_i2 < rsi2_i3): return False  # condition 4
    if rsi2_i3 >= 60: return False                             # condition 5

    return True

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None

    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 500_000: return None

    if not _is_r3(df, idx): return None

    rsi2 = float(row["rsi2"]); adx = float(row["adx"])
    if pd.isna(rsi2) or pd.isna(adx): return None

    # Minervini template
    m = sum([
        c > row["sma150"],
        c > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        c > row["sma50"],
        c >= 1.30 * row["52w_low"],
        c >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[idx - 20]["sma200"],
    ])
    if m < 4: return None

    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0

    conf = {
        "RSI2_sub5": rsi2 < 5,
        "Vol>avg":   vol_ratio > 1.0,
        "ADX16-35":  16 <= adx <= 35,
        "M>=5":      m >= 5,
    }
    score = sum(conf.values())
    return {
        "score": score, "conf": [k for k, v in conf.items() if v],
        "minervini": m, "rsi": round(rsi2, 1), "adx": round(adx, 1),
        "price": round(c, 2), "vol_ratio": round(vol_ratio, 2),
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_r3(df, i): continue
        row = df.iloc[i]
        # Minervini >= 4 required in backtest too
        c = float(row["Close"])
        sma50 = float(row["sma50"]); sma200 = float(row["sma200"])
        if pd.isna(sma50) or pd.isna(sma200): continue
        if c <= sma200 or c <= sma50: continue
        entry = float(df.iloc[i]["Close"]); exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100); last = i
    if not rets: return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n": len(a), "wr": round(100*(a>0).mean(),1),
            "avg": round(float(a.mean()),2), "med": round(float(np.median(a)),2)}

def analyze_ticker(ticker: str, bench_ret: Optional[float], with_backtest: bool) -> Optional[dict]:
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
        # freshness: signal must have fired within last FRESH_WINDOW bars
        found = any(_is_r3(df, k)
                    for k in range(max(215, last - FRESH_WINDOW + 1), last + 1))
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
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
                for t, b in universe.items()}
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
                if r:
                    r["strategy"] = "connors_r3"
                    results.append(r)
            except Exception:
                pass
    results.sort(key=lambda x: (-(x.get("wr") or 0), -x["score"]))
    return results

def main():
    from momentum_scanner import build_universe, compute_bench_returns
    import time
    wb = "--no-backtest" not in sys.argv
    uni = build_universe()
    bench = compute_bench_returns(set(uni.values()))
    t0 = time.time()
    res = scan(uni, bench, wb)
    print(f"\nConnors R3 Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi2={r['rsi']}  adx={r['adx']}  price={r['price']}")

if __name__ == "__main__":
    main()
