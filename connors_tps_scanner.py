#!/usr/bin/env python3
"""
Connors TPS Scanner  |  Connors Research 2009
──────────────────────────────────────────────
Time/Price Scale-In: 3-5 consecutive lower closes, price above 200d SMA,
RSI(2) declining each down day, RSI(2) < 25 on entry, volume declining
(orderly pullback). Designed for choppy/sideways markets.

python3 connors_tps_scanner.py --no-backtest   # fast
python3 connors_tps_scanner.py                 # with backtest
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
HOLD_DAYS       = 4
MAX_WORKERS     = 25
FRESH_WINDOW    = 1   # signal must fire today only
MIN_CONSECUTIVE = 3   # minimum consecutive lower closes required
MAX_CONSECUTIVE = 7   # cap — too many = falling knife

# ── INDICATOR HELPERS ─────────────────────────────────────────────────────────

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma10"]    = _sma(c, 10)
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["rsi2"]     = _rsi(c, 2)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df

def _count_consecutive_lower_closes(df: pd.DataFrame, idx: int) -> int:
    """Count how many consecutive lower closes end at idx."""
    count = 0
    for k in range(1, MAX_CONSECUTIVE + 2):
        if idx - k < 0: break
        if float(df.iloc[idx - k + 1]["Close"]) < float(df.iloc[idx - k]["Close"]):
            count += 1
        else:
            break
    return count

def _is_tps(df: pd.DataFrame, idx: int) -> bool:
    """
    TPS conditions:
    1. Price above 200d SMA
    2. Price above 50d SMA
    3. 3-7 consecutive lower closes ending today
    4. RSI(2) declining each of those consecutive days
    5. RSI(2) today < 25
    6. Volume orderly: declining in last 3 days or avg(last3) < avg(prior10)
    7. ADX 12-40
    8. Not in free-fall: today's close not more than 8% below 10-day SMA
    """
    if idx < 215: return False
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return False

    sma200 = float(row["sma200"]); sma50 = float(row["sma50"])
    if pd.isna(sma200) or pd.isna(sma50): return False
    if c <= sma200 or c <= sma50: return False

    adx = float(row["adx"])
    if pd.isna(adx) or adx < 12 or adx > 40: return False

    # Not in free-fall: close not more than 8% below 10-day SMA
    sma10 = float(row["sma10"])
    if not pd.isna(sma10) and c < sma10 * 0.92: return False

    # Condition 3: consecutive lower closes
    n_lower = _count_consecutive_lower_closes(df, idx)
    if n_lower < MIN_CONSECUTIVE or n_lower > MAX_CONSECUTIVE: return False

    # Condition 5: RSI(2) today < 25
    rsi2_today = float(row["rsi2"])
    if pd.isna(rsi2_today) or rsi2_today >= 25: return False

    # Condition 4: RSI(2) declining each of the consecutive down days
    # We need rsi2[i] < rsi2[i-1] < ... < rsi2[i - n_lower + 1]
    rsi2_vals = []
    for k in range(n_lower):
        v = float(df.iloc[idx - k]["rsi2"])
        if pd.isna(v): return False
        rsi2_vals.append(v)
    # rsi2_vals[0] = today, rsi2_vals[1] = yesterday, etc.
    # Must be strictly decreasing: vals[0] < vals[1] < ... < vals[n-1]
    for k in range(len(rsi2_vals) - 1):
        if rsi2_vals[k] >= rsi2_vals[k + 1]: return False

    # Condition 6: volume orderly (declining in last 3 days or avg last3 < avg prior10)
    if idx >= 12:
        vols = [float(df.iloc[idx - k]["Volume"]) for k in range(13)]
        vol_last3  = np.mean(vols[:3])
        vol_prior10 = np.mean(vols[3:13])
        vol_declining_days = vols[0] < vols[1] or vols[1] < vols[2]
        vol_orderly = vol_declining_days or (vol_last3 < vol_prior10)
        if not vol_orderly: return False

    return True

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None

    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0

    if not _is_tps(df, idx): return None

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

    # Volume orderly check (reuse logic for score component)
    vol_decline = False
    if idx >= 12:
        vols = [float(df.iloc[idx - k]["Volume"]) for k in range(13)]
        vol_last3   = np.mean(vols[:3])
        vol_prior10 = np.mean(vols[3:13])
        vol_decline = (vols[0] < vols[1] or vols[1] < vols[2]) or (vol_last3 < vol_prior10)

    conf = {
        "RSI2_sub15": rsi2 < 15,
        "VolDecline":  vol_decline,
        "ADX16-35":   16 <= adx <= 35,
        "M>=5":        m >= 5,
    }
    score = sum(conf.values())
    n_lower = _count_consecutive_lower_closes(df, idx)
    return {
        "score": score, "conf": [k for k, v in conf.items() if v],
        "minervini": m, "rsi": round(rsi2, 1), "adx": round(adx, 1),
        "price": round(c, 2), "vol_ratio": round(vol_ratio, 2),
        "consecutive_lower": n_lower,
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_tps(df, i): continue
        row = df.iloc[i]
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
        found = any(_is_tps(df, k)
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
                    r["strategy"] = "connors_tps"
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
    print(f"\nConnors TPS Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi2={r['rsi']}  adx={r['adx']}  n_lower={r.get('consecutive_lower','?')}")

if __name__ == "__main__":
    main()
