#!/usr/bin/env python3
"""
Elder Impulse System Scanner  |  Alexander Elder — "Come Into My Trading Room"
───────────────────────────────────────────────────────────────────────────────
GREEN bar = EMA(13) rising AND MACD histogram rising.
Signal fires on 2 consecutive GREEN bars (confirmed impulse).

python3 elder_impulse_scanner.py --no-backtest   # fast
python3 elder_impulse_scanner.py                 # with backtest
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
HOLD_DAYS    = 5
MAX_WORKERS  = 25
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

    # EMAs
    df["ema13"]  = _ema(c, 13)
    df["ema26"]  = _ema(c, 26)
    df["ema52"]  = _ema(c, 52)

    # SMAs
    df["sma50"]  = _sma(c, 50)
    df["sma150"] = _sma(c, 150)
    df["sma200"] = _sma(c, 200)

    # RSI / ADX
    df["rsi"]    = _rsi(c, 14)
    df["adx"]    = _adx(h, l, c, 14)

    # Volume
    df["vol_ma20"] = v.rolling(20).mean()

    # 52-week high/low
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()

    # MACD: EMA(12) - EMA(26); Signal = EMA(9) of MACD; Hist = MACD - Signal
    df["macd"]        = _ema(c, 12) - _ema(c, 26)
    df["macd_signal"] = _ema(df["macd"], 9)
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Impulse components
    df["ema13_rising"] = df["ema13"] > df["ema13"].shift(1)
    df["macd_rising"]  = df["macd_hist"] > df["macd_hist"].shift(1)

    # GREEN bar = both rising
    df["green_bar"] = df["ema13_rising"] & df["macd_rising"]

    return df

def _is_green(df: pd.DataFrame, idx: int) -> bool:
    """Return True if green_bar is True at position idx."""
    if idx < 0 or idx >= len(df): return False
    return bool(df.iloc[idx]["green_bar"])

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None

    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 100_000: return None

    # Price above EMA(13)
    if c <= float(row["ema13"]): return None

    # ADX filter
    adx = float(row["adx"])
    if pd.isna(adx) or adx < 16 or adx > 35: return None

    # RSI filter
    rsi = float(row["rsi"])
    if pd.isna(rsi) or rsi < 45 or rsi > 75: return None

    # Must have 2 consecutive green bars ending at idx
    if not (_is_green(df, idx) and _is_green(df, idx - 1)): return None

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

    # Confidence signals
    macd_hist = df["macd_hist"]
    macd_accel = (
        idx >= 2 and
        float(macd_hist.iloc[idx])   > float(macd_hist.iloc[idx - 1]) and
        float(macd_hist.iloc[idx-1]) > float(macd_hist.iloc[idx - 2])
    )

    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0

    conf = {
        "MACD_accel": macd_accel,
        "EMA_stack":  float(row["ema13"]) > float(row["ema26"]) > float(row["ema52"]),
        "RSI50-70":   50 <= rsi <= 70,
        "ADX>20":     adx > 20,
        "VOL1.2x":    vol_ratio >= 1.2,
        "M≥6":        m >= 6,
    }
    score = sum(conf.values())

    return {
        "score":     score,
        "fresh":     ["IMPULSE-GREEN"],
        "conf":      [k for k, v in conf.items() if v],
        "minervini": m,
        "rsi":       round(rsi, 1),
        "adx":       round(adx, 1),
        "price":     round(c, 2),
        "vol_ratio": round(vol_ratio, 2),
    }

def run_backtest(df: pd.DataFrame) -> dict:
    """Signal = 2 consecutive green bars. Hold HOLD_DAYS bars."""
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not (_is_green(df, i) and _is_green(df, i - 1)): continue
        row = df.iloc[i]
        c = float(row["Close"])
        if c <= float(row["ema13"]): continue
        entry = c
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100)
        last = i
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

        # Freshness: at least one green bar in the last FRESH_WINDOW bars
        found = any(_is_green(df, k)
                    for k in range(max(215, last - FRESH_WINDOW + 1), last + 1))
        if not found: return None

        sig = _score(df, last)
        if not sig: return None

        # Market suffix
        mkt = "US"
        for sfx, m in {".L":"UK",".DE":"DE",".PA":"FR",".AS":"NL",".TO":"CA"}.items():
            if ticker.endswith(sfx): mkt = m; break

        result = {
            "ticker":    ticker,
            "mkt":       mkt,
            "hold_days": HOLD_DAYS,
            **sig,
        }
        if with_backtest: result.update(run_backtest(df))
        return result
    except Exception:
        return None

def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    spy_bull = _spy_is_bullish()
    if not spy_bull:
        print("  [elder_impulse] SPY regime: CHOPPY/BEAR — signals tagged LOW conviction")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
                for t, b in universe.items()}
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
                if r:
                    r["strategy"] = "elder_impulse"
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
    print(f"\nElder Impulse Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  vol_ratio={r['vol_ratio']}  "
              f"conf={r['conf']}")

if __name__ == "__main__":
    main()
