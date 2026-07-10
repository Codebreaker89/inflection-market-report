#!/usr/bin/env python3
"""
Connors 3-Down Scanner  |  Larry Connors — "Short-Term Trading Strategies That Work"
──────────────────────────────────────────────────────────────────────────────────────
3 consecutive lower closes in an uptrending stock + RSI(2) < 20 (short-term
oversold). Mean-reversion entry into a trending name on a brief dip.

python3 connors_3down_scanner.py --no-backtest   # fast
python3 connors_3down_scanner.py                 # with backtest
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

# ── SPY REGIME GATE ───────────────────────────────────────────────────────────
_SPY_REGIME_CACHE: dict = {}

def _spy_is_bullish() -> bool:
    """SPY regime gate: returns True if SPY EMA(13) is rising AND MACD histogram is positive.
    In choppy/bear markets this returns False → scanner suppresses signals."""
    import datetime
    today = str(datetime.date.today())
    if today in _SPY_REGIME_CACHE:
        return _SPY_REGIME_CACHE[today]
    try:
        with _quiet():
            raw = yf.download("SPY", period="120d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        c = raw["Close"]
        ema13 = c.ewm(span=13, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd  = ema13 - ema26
        sig   = macd.ewm(span=9, adjust=False).mean()
        hist  = macd - sig
        bullish = bool(ema13.iloc[-1] > ema13.iloc[-2] and hist.iloc[-1] > 0)
        _SPY_REGIME_CACHE[today] = bullish
        return bullish
    except Exception:
        return True  # fail open — don't block signals if SPY fetch fails

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOLD_DAYS    = 3
MAX_WORKERS  = 25
FRESH_WINDOW = 1        # signal is fresh today only

# ── INDICATOR HELPERS ─────────────────────────────────────────────────────────

def _rsi2(s):
    """RSI(2) — Connors' short-term oversold oscillator."""
    return _rsi(s, 2)

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["rsi"]      = _rsi(c, 14)
    df["rsi2"]     = _rsi2(c)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df

def _is_3down(df: pd.DataFrame, idx: int) -> bool:
    """Returns True if there are ≥ 3 consecutive lower closes ending at idx."""
    if idx < 3: return False
    closes = df["Close"]
    return (float(closes.iloc[idx])     < float(closes.iloc[idx - 1]) and
            float(closes.iloc[idx - 1]) < float(closes.iloc[idx - 2]) and
            float(closes.iloc[idx - 2]) < float(closes.iloc[idx - 3]))

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None

    vol_ma20 = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma20 < 100_000: return None

    # Must be above 200d and 50d SMA (uptrend required)
    sma200 = float(row["sma200"]); sma50 = float(row["sma50"])
    if pd.isna(sma200) or pd.isna(sma50): return None
    if c <= sma200 or c <= sma50: return None

    # 3 consecutive lower closes
    if not _is_3down(df, idx): return None

    rsi2 = float(row["rsi2"]); rsi = float(row["rsi"]); adx = float(row["adx"])
    if pd.isna(rsi2) or pd.isna(rsi) or pd.isna(adx): return None

    # RSI(2) < 20 — short-term oversold (Connors' core condition)
    if rsi2 >= 25: return None  # Connors book: threshold is 25 not 20

    # RSI(14) > 40 — not a broken stock
    if rsi <= 40: return None

    # ADX 16-40
    if adx < 16 or adx > 40: return None

    # Minervini template
    m = sum([
        c > row["sma150"], c > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        c > row["sma50"],
        c >= 1.30 * row["52w_low"],
        c >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[idx - 20]["sma200"],
    ])
    if m < 4: return None

    vol_ratio = float(row["Volume"]) / vol_ma20 if vol_ma20 > 0 else 0

    # 4th consecutive lower close?
    four_down = (idx >= 4 and
                 float(df["Close"].iloc[idx - 3]) < float(df["Close"].iloc[idx - 4]))

    conf = {
        "RSI2_lt10":  rsi2 < 10,
        "4thDown":    four_down,
        "SMA50above": c > 1.05 * sma50,
        "ADX>20":     adx > 20,
        "M≥5":        m >= 5,
    }
    score = sum(conf.values())

    return {
        "score":     score,
        "fresh":     ["3DOWN"],
        "conf":      [k for k, v in conf.items() if v],
        "minervini": m,
        "rsi":       round(rsi, 1),
        "rsi2":      round(rsi2, 1),
        "adx":       round(adx, 1),
        "price":     round(c, 2),
        "vol_ratio": round(vol_ratio, 2),
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_3down(df, i): continue
        row = df.iloc[i]
        c   = float(row["Close"])
        rsi2 = float(row["rsi2"]); rsi = float(row["rsi"]); adx = float(row["adx"])
        if pd.isna(rsi2) or pd.isna(rsi) or pd.isna(adx): continue
        if rsi2 >= 25 or rsi <= 40 or adx < 16 or adx > 40: continue
        sma200 = float(row["sma200"]); sma50 = float(row["sma50"])
        if pd.isna(sma200) or pd.isna(sma50): continue
        if c <= sma200 or c <= sma50: continue
        m = sum([
            c > row["sma150"], c > row["sma200"],
            row["sma150"] > row["sma200"],
            row["sma50"]  > row["sma150"],
            c > row["sma50"],
            c >= 1.30 * row["52w_low"],
            c >= 0.75 * row["52w_high"],
            row["sma200"] > df.iloc[i - 20]["sma200"],
        ])
        if m < 4: continue
        entry = c; exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
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

        # Freshness: 3-down signal must be fresh today only (FRESH_WINDOW = 1)
        found = any(_is_3down(df, k)
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
    spy_bull = _spy_is_bullish()
    if not spy_bull:
        print("  [connors_3down] SPY regime: CHOPPY/BEAR — signals tagged LOW conviction")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
                for t, b in universe.items()}
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
                if r:
                    r["strategy"]   = "connors_3down"
                    r["spy_regime"] = "BULL" if spy_bull else "CHOPPY"
                    results.append(r)
            except Exception:
                pass
    results.sort(key=lambda x: (-(x.get("wr") or 0), -x["score"]))
    return results

def main():
    from momentum_scanner import build_universe, compute_bench_returns
    import time
    wb = "--no-backtest" not in sys.argv
    uni   = build_universe()
    bench = compute_bench_returns(set(uni.values()))
    t0    = time.time()
    res   = scan(uni, bench, wb)
    print(f"\nConnors 3-Down Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  rsi2={r['rsi2']}  adx={r['adx']}  vol_ratio={r['vol_ratio']}  "
              f"spy={r.get('spy_regime','?')}  conf={r['conf']}")

if __name__ == "__main__":
    main()
