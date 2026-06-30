#!/usr/bin/env python3
"""
Turtle Soup Scanner  |  Raschke & Connors, "Street Smarts" 1996
────────────────────────────────────────────────────────────────
Stock makes a NEW 20-day low (today's low < all lows in prior 20 days),
then REVERSES and closes ABOVE that prior 20-day low.  False breakdown
trap — price sucks in shorts, then snaps back.  Works best in sideways
markets where false breakouts are common.

python3 turtle_soup_scanner.py --no-backtest   # fast
python3 turtle_soup_scanner.py                 # with backtest
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
TS_WINDOW    = 20       # 20-day low window
FRESH_WINDOW = 1        # signal fires today only

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
    df["sma50"]    = _sma(c, 50)
    df["sma200"]   = _sma(c, 200)
    df["sma150"]   = _sma(c, 150)
    df["sma20"]    = _sma(c, 20)
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    # Prior 20-day low: min of the 20 bars BEFORE today (shift(1) then rolling 20)
    df["prior_20d_low"] = l.shift(1).rolling(TS_WINDOW).min()
    # Age of the prior 20-day low: how many bars ago was that low set?
    # We'll compute it on-the-fly in _is_turtle_soup; store shifted low series for age check.
    return df

def _age_of_prior_low(df: pd.DataFrame, idx: int) -> int:
    """Return how many bars ago the prior 20-day low was set (among bars [idx-20 .. idx-1])."""
    if idx < TS_WINDOW + 1:
        return 0
    window_lows = df["Low"].iloc[idx - TS_WINDOW: idx]
    min_val     = window_lows.min()
    # Find the MOST RECENT bar in the window that matches the minimum
    positions   = window_lows[window_lows == min_val].index
    if len(positions) == 0:
        return 0
    last_pos = df.index.get_loc(positions[-1])
    return idx - 1 - last_pos  # how many bars before today

def _is_turtle_soup(df: pd.DataFrame, idx: int) -> bool:
    """Return True if bar at idx is a valid Turtle Soup signal."""
    if idx < 215: return False
    row  = df.iloc[idx]
    low  = float(row["Low"])
    cls  = float(row["Close"])
    p20l = float(row["prior_20d_low"])
    if pd.isna(p20l): return False
    # 1. Today's low broke below the prior 20-day low
    if low >= p20l: return False
    # 2. Today's close is ABOVE the prior 20-day low (reversal)
    if cls <= p20l: return False
    # 9. The 20-day low being broken must be >= 5 days old
    age = _age_of_prior_low(df, idx)
    if age < 5: return False
    return True

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row  = df.iloc[idx]
    c    = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None

    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 100_000: return None

    # 3 & 4. Price above 200d and 50d SMA
    if pd.isna(row["sma200"]) or c < float(row["sma200"]): return None
    if pd.isna(row["sma50"])  or c < float(row["sma50"]):  return None

    if not _is_turtle_soup(df, idx): return None

    # 5. Volume > 1.2x 20-day avg
    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    if vol_ratio < 1.2: return None

    rsi = float(row["rsi"]); adx = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx): return None

    # 6. ADX 12-40
    if adx < 12 or adx > 40: return None
    # 7. RSI 30-70
    if rsi < 30 or rsi > 70: return None

    # 8. Minervini >= 4
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

    prior_20d_low = float(row["prior_20d_low"])

    conf = {
        "CloseStrong": c > prior_20d_low * 1.005,
        "VolSurge":    vol_ratio > 1.5,
        "ADX16-35":    16 <= adx <= 35,
        "M>=5":        m >= 5,
    }
    score = sum(conf.values())
    return {
        "score": score, "conf": [k for k, v in conf.items() if v],
        "minervini": m, "rsi": round(rsi, 1), "adx": round(adx, 1),
        "price": round(c, 2), "vol_ratio": round(vol_ratio, 2),
        "prior_20d_low": round(prior_20d_low, 2),
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_turtle_soup(df, i): continue
        row = df.iloc[i]
        vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
        if vol_ma < 100_000: continue
        vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
        if vol_ratio < 1.2: continue
        c = float(row["Close"])
        if pd.isna(row["sma200"]) or c < float(row["sma200"]): continue
        if pd.isna(row["sma50"])  or c < float(row["sma50"]):  continue
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
        # Freshness: signal must have fired within last FRESH_WINDOW bars
        found = any(_is_turtle_soup(df, k)
                    for k in range(max(215, last - FRESH_WINDOW + 1), last + 1))
        if not found: return None
        sig = _score(df, last)
        if not sig: return None
        result = {"ticker": ticker, "strategy": "turtle_soup", **sig, "hold_days": HOLD_DAYS}
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
                if r: results.append(r)
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
    print(f"\nTurtle Soup Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  p20l={r.get('prior_20d_low','?')}  "
              f"price={r['price']}")

if __name__ == "__main__":
    main()
