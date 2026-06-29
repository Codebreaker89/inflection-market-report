#!/usr/bin/env python3
"""
Williams %R Reversal Scanner  |  Larry Williams
─────────────────────────────────────────────────
%R drops into oversold (<-80) then crosses back above — a mean-reversion
signal within an uptrend (price above both 50d and 200d SMA).
Classic from "Long-Term Secrets to Short-Term Trading".

python3 williams_pct_r_scanner.py --no-backtest   # fast
python3 williams_pct_r_scanner.py                 # with backtest
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
WR_PERIOD    = 14       # Williams %R lookback
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

def _williams_r(high, low, close, n=14):
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["sma20"]    = _sma(c, 20)
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    df["wr"]       = _williams_r(h, l, c, WR_PERIOD)
    return df

def _is_wr_cross(df: pd.DataFrame, idx: int) -> bool:
    """True if %R crosses above -80 today (from below -80 in prior 5 bars)."""
    if idx < WR_PERIOD + 10: return False
    wr_today = float(df["wr"].iloc[idx])
    wr_prev  = float(df["wr"].iloc[idx - 1])
    if pd.isna(wr_today) or pd.isna(wr_prev): return False
    # today must be above -80 (exited oversold), yesterday below -80
    if not (wr_today > -80 and wr_prev <= -80): return False
    # must have dipped below -80 within last 5 bars (already satisfied by prev<=80,
    # but confirm it actually entered oversold zone)
    recent = df["wr"].iloc[max(0, idx - 5): idx]
    return bool((recent < -80).any())

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None
    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 100_000: return None

    # Trend filter: above both 50d and 200d SMA
    sma50  = float(row["sma50"])
    sma200 = float(row["sma200"])
    if pd.isna(sma50) or pd.isna(sma200): return None
    if c <= sma50 or c <= sma200: return None

    # Williams %R cross above -80
    if not _is_wr_cross(df, idx): return None

    wr_val = float(row["wr"])
    rsi    = float(row["rsi"])
    adx    = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx) or pd.isna(wr_val): return None

    # ADX: moderate trend (floor 16, cap 40)
    if adx < 16 or adx > 40: return None

    # RSI: not structurally broken, not overbought
    if not (35 <= rsi <= 65): return None

    # Volume: some participation on reversal day
    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    if vol_ratio < 0.8: return None

    # Minervini template
    sma150 = float(row["sma150"])
    m = sum([
        c > sma150, c > sma200,
        sma150 > sma200,
        sma50  > sma150,
        c > sma50,
        c >= 1.30 * float(row["52w_low"]),
        c >= 0.75 * float(row["52w_high"]),
        sma200 > float(df.iloc[idx - 20]["sma200"]),
    ])
    if m < 4: return None

    # Deep oversold: lowest %R in last 5 bars
    recent_wr = df["wr"].iloc[max(0, idx - 5): idx]
    deep_oversold = bool((recent_wr < -90).any())

    conf = {
        "RSI40-55":   40 <= rsi <= 55,
        "DeepOversold": deep_oversold,
        "ADX>20":     adx > 20,
        "VOL>avg":    vol_ratio >= 1.0,
        "M≥5":        m >= 5,
        "NearSMA50":  (c - sma50) / sma50 <= 0.05,
    }
    score = sum(conf.values())
    return {
        "score":      score,
        "fresh":      ["WR-CROSS"],
        "conf":       [k for k, v in conf.items() if v],
        "minervini":  m,
        "rsi":        round(rsi, 1),
        "adx":        round(adx, 1),
        "price":      round(c, 2),
        "vol_ratio":  round(vol_ratio, 2),
        "williams_r": round(wr_val, 1),
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_wr_cross(df, i): continue
        row = df.iloc[i]
        c = float(row["Close"])
        sma50  = float(row["sma50"])
        sma200 = float(row["sma200"])
        if pd.isna(sma50) or pd.isna(sma200): continue
        if c <= sma50 or c <= sma200: continue
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
        # freshness: %R cross above -80 within last FRESH_WINDOW bars
        found = any(_is_wr_cross(df, k)
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
                    r["strategy"] = "williams_pct_r"
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
    print(f"\nWilliams %R Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  %R={r.get('williams_r','?')}")

if __name__ == "__main__":
    main()
