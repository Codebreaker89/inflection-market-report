#!/usr/bin/env python3
"""
High Tight Flag Scanner  |  Minervini / O'Neil
────────────────────────────────────────────────
Stage 1: stock surges ≥90% in ≤8 weeks (40 trading days) — the "pole".
Stage 2: stock consolidates ≤25% from the 8-week high — the "flag".
Entry:   current price within 15% of the 8-week high (flag forming or breaking out).
Rare signal but extremely high conviction.

python3 high_tight_flag_scanner.py --no-backtest
python3 high_tight_flag_scanner.py
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

HOLD_DAYS    = 10
MAX_WORKERS  = 25
FRESH_WINDOW = 5          # flag can persist a few days; wider window
POLE_DAYS    = 40         # 8 trading weeks
POLE_MIN_RET = 0.90       # ≥90% gain in pole
FLAG_MAX_DD  = 0.25       # flag must not pull back more than 25%
ENTRY_ZONE   = 0.15       # current price within 15% of 8-week high

def _atr(high, low, close, n=14):
    tr = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    df["atr"]      = _atr(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    # rolling 8-week (40-day) min for the pole base
    df["low_40d"]  = c.rolling(POLE_DAYS).min()
    # rolling 8-week max = pole high
    df["high_40d"] = c.rolling(POLE_DAYS).max()
    return df

def _is_htf(df: pd.DataFrame, idx: int) -> tuple:
    """Returns (is_htf, pole_return, flag_drawdown) or (False, 0, 0)."""
    if idx < POLE_DAYS + 20: return False, 0.0, 0.0
    row = df.iloc[idx]
    c = float(row["Close"])
    low_pole  = float(row["low_40d"])
    high_pole = float(row["high_40d"])
    if low_pole <= 0: return False, 0.0, 0.0
    pole_ret = (high_pole - low_pole) / low_pole
    if pole_ret < POLE_MIN_RET: return False, 0.0, 0.0
    # flag: current price vs pole high
    drawdown_from_peak = (high_pole - c) / high_pole
    if drawdown_from_peak > FLAG_MAX_DD: return False, 0.0, 0.0
    # must still be close to the 8-week high (in the entry zone)
    if drawdown_from_peak > ENTRY_ZONE: return False, 0.0, 0.0
    return True, round(pole_ret * 100, 1), round(drawdown_from_peak * 100, 1)

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 220: return None
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None
    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 100_000: return None

    is_htf, pole_ret, flag_dd = _is_htf(df, idx)
    if not is_htf: return None

    rsi = float(row["rsi"]); adx = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx): return None

    # ATR ratio cap: flag must have CONTRACTING volatility (not still parabolic).
    # MRVL had atr_ratio=2.457 (vol expanding — entered mid-parabola) → blocked.
    # SNDK had atr_ratio=1.423 → passes (flag genuinely compressing).
    atr_ratio_val = None
    if idx >= 25 and "atr" in df.columns:
        atr_recent = float(df["atr"].iloc[idx - 5 : idx].mean())
        atr_prior  = float(df["atr"].iloc[idx - 20 : idx - 5].mean())
        if atr_prior > 0:
            atr_ratio_val = round(atr_recent / atr_prior, 3)
            if atr_ratio_val > 1.8: return None   # vol still expanding — not a real flag

    m = sum([
        c > row["sma150"], c > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        c > row["sma50"],
        c >= 1.30 * row["52w_low"],
        c >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[idx - 20]["sma200"],
    ])
    if m < 5: return None  # needs some trend structure

    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    tightness = 1.0 - (flag_dd / 25.0)   # tighter flag = higher tightness

    conf = {
        f"POLE+{int(pole_ret)}%": True,    # always shown — core signal
        f"FLAG-{int(flag_dd)}%": True,      # drawdown shown
        "VOL1.5x":  vol_ratio > 1.5,
        "RSI>50":   rsi > 50,
        "M≥6":      m >= 6,
    }
    score = sum(conf.values())

    return {
        "score": score, "fresh": ["HTF"], "conf": [k for k, v in conf.items() if v],
        "minervini": m, "rsi": round(rsi, 1), "adx": round(adx, 1),
        "price": round(c, 2), "vol_ratio": round(vol_ratio, 2),
        "pole_return": pole_ret, "flag_drawdown": flag_dd,
        "atr_ratio": atr_ratio_val,
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(220, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        is_htf, _, _ = _is_htf(df, i)
        if not is_htf: continue
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
        if raw is None or len(raw) < 225: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return None
        raw = raw.dropna(subset=["Close","High","Low","Volume"])
        if len(raw) < 225: return None
        df   = _build(raw.copy())
        last = len(df) - 1
        found = any(_is_htf(df, k)[0]
                    for k in range(max(220, last - FRESH_WINDOW + 1), last + 1))
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
                    r["strategy"] = "high_tight_flag"
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
    print(f"\nHigh Tight Flag Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"pole={r.get('pole_return','?')}%  flag_dd={r.get('flag_drawdown','?')}%")

if __name__ == "__main__":
    main()
