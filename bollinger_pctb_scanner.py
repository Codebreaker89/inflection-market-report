#!/usr/bin/env python3
"""
Bollinger %B + MFI Scanner  |  John Bollinger
───────────────────────────────────────────────
Price touches the lower Bollinger Band (%B < 0.20) with Money Flow Index
confirming institutional selling, then bounces back — a mean-reversion
signal that works even in low-trend, sideways markets.
Classic from "Bollinger on Bollinger Bands".

python3 bollinger_pctb_scanner.py --no-backtest   # fast
python3 bollinger_pctb_scanner.py                 # with backtest
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
HOLD_DAYS    = 5
MAX_WORKERS  = 25
BB_PERIOD    = 20       # Bollinger Band period
BB_STD       = 2        # standard deviations
MFI_PERIOD   = 14       # Money Flow Index period
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

def _bollinger_pctb(close, n=20, std_mult=2):
    mid   = close.rolling(n).mean()
    sigma = close.rolling(n).std(ddof=0)
    upper = mid + std_mult * sigma
    lower = mid - std_mult * sigma
    pctb  = (close - lower) / (upper - lower).replace(0, np.nan)
    return pctb, upper, lower

def _mfi(high, low, close, volume, n=14):
    tp  = (high + low + close) / 3
    rmf = tp * volume
    pos = rmf.where(tp > tp.shift(), 0.0)
    neg = rmf.where(tp < tp.shift(), 0.0)
    pos_mf = pos.rolling(n).sum()
    neg_mf = neg.rolling(n).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - 100 / (1 + mfr)

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
    pctb, upper, lower = _bollinger_pctb(c, BB_PERIOD, BB_STD)
    df["pct_b"]    = pctb
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["mfi"]      = _mfi(h, l, c, v, MFI_PERIOD)
    return df

def _is_bb_oversold(df: pd.DataFrame, idx: int) -> bool:
    """True if %B < 0.20 today, rising vs yesterday, and was even lower in last 5 bars."""
    if idx < BB_PERIOD + MFI_PERIOD + 10: return False
    pctb_today = float(df["pct_b"].iloc[idx])
    pctb_prev  = float(df["pct_b"].iloc[idx - 1])
    if pd.isna(pctb_today) or pd.isna(pctb_prev): return False
    # today below 0.20 AND rising (bouncing off lower band)
    if not (pctb_today < 0.20 and pctb_today > pctb_prev): return False
    # confirm it was even lower in last 5 bars (actual band touch)
    recent = df["pct_b"].iloc[max(0, idx - 5): idx]
    return bool((recent < pctb_today).any())

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None
    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 100_000: return None

    # Trend filter: above 200d SMA
    sma200 = float(row["sma200"])
    sma50  = float(row["sma50"])
    if pd.isna(sma200) or pd.isna(sma50): return None
    if c <= sma200: return None

    # %B oversold with bounce
    if not _is_bb_oversold(df, idx): return None

    pctb_val = float(row["pct_b"])
    mfi_val  = float(row["mfi"])
    rsi      = float(row["rsi"])
    adx      = float(row["adx"])
    if pd.isna(pctb_val) or pd.isna(mfi_val) or pd.isna(rsi) or pd.isna(adx): return None

    # MFI must show money flowing out (setup condition)
    if mfi_val >= 35: return None

    # RSI: in healthy pullback range
    if not (30 <= rsi <= 60): return None

    # ADX floor: 12 (designed for low-trend sideways markets)
    if adx < 12: return None

    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0

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

    conf = {
        "MFI_lt25":  mfi_val < 25,
        "pctB_lt10": pctb_val < 0.10,
        "RSI35-50":  35 <= rsi <= 50,
        "SMAabove":  c > sma50,
        "ADX_low":   adx < 20,
        "M≥5":       m >= 5,
    }
    score = sum(conf.values())
    return {
        "score":     score,
        "fresh":     ["BB-OVERSOLD"],
        "conf":      [k for k, v in conf.items() if v],
        "minervini": m,
        "rsi":       round(rsi, 1),
        "adx":       round(adx, 1),
        "price":     round(c, 2),
        "vol_ratio": round(vol_ratio, 2),
        "pct_b":     round(pctb_val, 3),
        "mfi":       round(mfi_val, 1),
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_bb_oversold(df, i): continue
        row = df.iloc[i]
        c      = float(row["Close"])
        sma200 = float(row["sma200"])
        mfi    = float(row["mfi"])
        if pd.isna(sma200) or pd.isna(mfi): continue
        if c <= sma200: continue
        if mfi >= 35: continue
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
        # freshness: %B < 0.20 AND rising within last FRESH_WINDOW bars
        found = any(_is_bb_oversold(df, k)
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
                    r["strategy"] = "bollinger_pctb"
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
    print(f"\nBollinger %B Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  %B={r.get('pct_b','?')}  mfi={r.get('mfi','?')}")

if __name__ == "__main__":
    main()
