#!/usr/bin/env python3
"""
Bollinger Band Squeeze Scanner  |  Bollinger / John Carter TTM Squeeze
───────────────────────────────────────────────────────────────────────
Squeeze = Bollinger Bands collapse INSIDE Keltner Channels.
Fire = squeeze just released + MACD histogram positive (momentum direction).
Great for low-volatility sideways markets where other strategies fail.

python3 bb_squeeze_scanner.py --no-backtest
python3 bb_squeeze_scanner.py
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

HOLD_DAYS   = 7
MAX_WORKERS = 25
FRESH_WINDOW = 3

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

def _atr(high, low, close, n=14):
    tr = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    # Bollinger Bands (20, 2)
    bb_mid          = _sma(c, 20)
    bb_std          = c.rolling(20).std()
    df["bb_upper"]  = bb_mid + 2 * bb_std
    df["bb_lower"]  = bb_mid - 2 * bb_std
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    # Keltner Channels (20, 1.5×ATR14) — Carter parameters
    kc_atr          = _atr(h, l, c, 14)
    df["kc_upper"]  = bb_mid + 1.5 * kc_atr
    df["kc_lower"]  = bb_mid - 1.5 * kc_atr
    # TTM Squeeze: BB is INSIDE KC
    df["squeeze"]   = (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])
    # 6-month (126-day) rolling min of bb_width for squeeze intensity context
    df["bb_width_6m_min"] = df["bb_width"].rolling(126).min()
    # MACD momentum (12/26/9)
    macd            = _ema(c, 12) - _ema(c, 26)
    sig             = _ema(macd, 9)
    df["macd_hist"] = macd - sig
    # Structural indicators
    df["sma50"]     = _sma(c, 50)
    df["sma150"]    = _sma(c, 150)
    df["sma200"]    = _sma(c, 200)
    df["rsi"]       = _rsi(c, 14)
    df["adx"]       = _adx(h, l, c, 14)
    df["vol_ma20"]  = v.rolling(20).mean()
    df["52w_high"]  = c.rolling(252).max()
    df["52w_low"]   = c.rolling(252).min()
    return df

def _squeeze_fired(df: pd.DataFrame, idx: int) -> bool:
    """True if squeeze just released — was ON yesterday, OFF today."""
    if idx < 130: return False
    row = df.iloc[idx]; prev = df.iloc[idx - 1]
    was_squeezed = bool(prev["squeeze"])
    now_free     = not bool(row["squeeze"])
    macd_pos     = float(row["macd_hist"]) > 0
    return was_squeezed and now_free and macd_pos

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None
    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 100_000: return None

    if not _squeeze_fired(df, idx): return None

    rsi = float(row["rsi"]); adx = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx): return None

    m = sum([
        c > row["sma150"], c > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        c > row["sma50"],
        c >= 1.30 * row["52w_low"],
        c >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[idx - 20]["sma200"],
    ])

    vol_ratio  = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    bb_width   = float(row["bb_width"])
    bw_min_6m  = float(row["bb_width_6m_min"])
    # squeeze intensity: how close to 6-month low is the width?
    intensity  = 1.0 - (bb_width / bw_min_6m) if bw_min_6m > 0 else 0.0

    conf = {
        "VOL1.5x":  vol_ratio > 1.5,
        "RSI>50":   rsi > 50,
        "ADX>15":   adx > 15,
        "M≥5":      m >= 5,
        "MACD+":    float(row["macd_hist"]) > 0,
    }
    score = sum(conf.values())

    return {
        "score": score, "fresh": ["BB-SQZ"], "conf": [k for k, v in conf.items() if v],
        "minervini": m, "rsi": round(rsi, 1), "adx": round(adx, 1),
        "price": round(c, 2), "vol_ratio": round(vol_ratio, 2),
        "squeeze_intensity": round(intensity, 3),
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _squeeze_fired(df, i): continue
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
        found = any(_squeeze_fired(df, k)
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
                    r["strategy"] = "bb_squeeze"
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
    print(f"\nBB Squeeze Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  sqz_intensity={r.get('squeeze_intensity','?')}")

if __name__ == "__main__":
    main()
