#!/usr/bin/env python3
"""
NR7 Breakout Scanner  |  Toby Crabel
──────────────────────────────────────
Today's high-low range is the NARROWEST of the last 7 days — maximum
volatility compression before an expansion move. Requires price above
50d SMA (trend context) and Minervini ≥5.

python3 nr7_scanner.py --no-backtest   # fast
python3 nr7_scanner.py                 # with backtest
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

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOLD_DAYS    = 3
MAX_WORKERS  = 25
NR_WINDOW    = 7        # range must be narrowest of last N days
FRESH_WINDOW = 2        # signal must have fired within last N bars

# ── INDICATOR HELPERS ─────────────────────────────────────────────────────────
def _sma(s, n): return s.rolling(n).mean()
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _adx(high, low, close, n=14):
    tr  = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    up  = (high-high.shift()).clip(lower=0); dn = (low.shift()-low).clip(lower=0)
    dmp = up.where(up>dn,0).ewm(alpha=1/n, adjust=False).mean()
    dmm = dn.where(dn>up,0).ewm(alpha=1/n, adjust=False).mean()
    dip = 100*dmp/atr; dim = 100*dmm/atr
    dx  = 100*(dip-dim).abs()/(dip+dim).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["range"]   = h - l
    df["sma50"]   = _sma(c, 50)
    df["sma150"]  = _sma(c, 150)
    df["sma200"]  = _sma(c, 200)
    df["sma20"]   = _sma(c, 20)
    df["rsi"]     = _rsi(c, 14)
    df["adx"]     = _adx(h, l, c, 14)
    df["vol_ma20"]= v.rolling(20).mean()
    df["52w_high"]= c.rolling(252).max()
    df["52w_low"] = c.rolling(252).min()
    # NR7: today's range < all prior 6 days
    df["min_range_6"]  = df["range"].shift(1).rolling(NR_WINDOW-1).min()
    return df

def _is_nr7(df: pd.DataFrame, idx: int) -> bool:
    if idx < NR_WINDOW + 5: return False
    row = df.iloc[idx]
    return float(row["range"]) < float(row["min_range_6"])

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None
    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 100_000: return None
    if float(row["sma50"]) > c: return None         # price must be above 50d SMA

    if not _is_nr7(df, idx): return None

    rsi = float(row["rsi"]); adx = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx): return None
    if adx < 16: return None    # ADX floor: ≤15 = no trend (WR 41% in backtest)
    if adx > 35: return None    # ADX cap: overextended trend = not a healthy base
    if rsi > 80: return None    # NR7 + RSI>80 = narrow range at overbought top, not a flag

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
    if m < 5: return None

    vol_ratio   = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    range_ratio = float(row["range"]) / float(row["min_range_6"]) if float(row["min_range_6"]) > 0 else 1.0
    compression = max(0.0, 1.0 - range_ratio)  # closer to 1 = more compressed

    conf = {
        "VOL>avg":  vol_ratio > 1.0,
        "RSI50-70": 50 <= rsi <= 70,
        "ADX>20":   adx > 20,
        "M≥6":      m >= 6,
    }
    score = sum(conf.values())
    return {
        "score": score, "fresh": ["NR7"], "conf": [k for k, v in conf.items() if v],
        "minervini": m, "rsi": round(rsi, 1), "adx": round(adx, 1),
        "price": round(c, 2), "vol_ratio": round(vol_ratio, 2),
        "compression": round(compression, 3),
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_nr7(df, i): continue
        row = df.iloc[i]
        if float(row["sma50"]) > float(row["Close"]): continue
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
        # freshness: NR7 must have fired within last FRESH_WINDOW bars
        found = any(_is_nr7(df, k)
                    for k in range(max(215, last - FRESH_WINDOW + 1), last + 1))
        if not found: return None
        sig = _score(df, last)
        if not sig: return None
        result = {"ticker": ticker, **sig, "hold_days": HOLD_DAYS}
        # derive mkt from ticker suffix (reuse ema_ribbon pattern)
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
                    r["strategy"] = "nr7"
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
    print(f"\nNR7 Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  compress={r.get('compression','?')}")

if __name__ == "__main__":
    main()
