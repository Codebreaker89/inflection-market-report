#!/usr/bin/env python3
"""
Raschke Holy Grail Scanner  |  Linda Bradford Raschke & Laurence Connors — "Street Smarts"
────────────────────────────────────────────────────────────────────────────────────────────
ADX(14) > 30 recently (strong trend) → price pulls back to EMA(20) → enter on
the first bounce back above EMA(20). Pullback must be ≥2 bars, RSI 40-65,
volume drying up during the pullback.

python3 raschke_holy_grail_scanner.py --no-backtest   # fast
python3 raschke_holy_grail_scanner.py                 # with backtest
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
FRESH_WINDOW = 2        # pullback-to-EMA20 must have fired within last N bars
ADX_LOOKBACK = 10       # bars to look back for ADX > 30

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
    df["ema20"]    = _ema(c, 20)
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df

def _is_holy_grail(df: pd.DataFrame, idx: int) -> bool:
    """
    Returns True if the Holy Grail setup is present at idx:
    - ADX > 30 in any of the last ADX_LOOKBACK bars
    - Price pulled back to within 2% of EMA20 from above (lasted ≥ 2 bars)
    - Today's close > EMA20 (still above / bouncing)
    - RSI 40-65 during pullback
    - Volume during pullback ≤ 1.0× 20d avg (drying up)
    """
    if idx < ADX_LOOKBACK + 5: return False

    # ADX was > 30 recently
    adx_recent = df["adx"].iloc[idx - ADX_LOOKBACK: idx + 1]
    if adx_recent.max() <= 30: return False

    c     = float(df["Close"].iloc[idx])
    ema20 = float(df["ema20"].iloc[idx])
    if pd.isna(ema20) or ema20 <= 0: return False

    # Today: close > EMA20 (bouncing) and within 2% from above
    pct_from_ema = (c - ema20) / ema20
    if pct_from_ema < 0 or pct_from_ema > 0.02: return False

    # Look back ≥ 2 bars for the pullback (close was near/below EMA20)
    pullback_bars = 0
    for k in range(idx - 1, max(idx - 10, 0), -1):
        pk    = float(df["Close"].iloc[k])
        ema_k = float(df["ema20"].iloc[k])
        if pd.isna(ema_k) or ema_k <= 0: break
        dist  = (pk - ema_k) / ema_k
        if -0.03 <= dist <= 0.02:   # within 3% of EMA20 (pulling back toward it)
            pullback_bars += 1
        else:
            break

    if pullback_bars < 2: return False

    # RSI in 40-65 at idx
    rsi = float(df["rsi"].iloc[idx])
    if pd.isna(rsi) or not (40 <= rsi <= 65): return False

    # Volume during pullback ≤ 1.0× 20d avg
    vol_ma20 = float(df["vol_ma20"].iloc[idx])
    if vol_ma20 > 0:
        # average volume over the pullback period
        pb_start = max(0, idx - pullback_bars)
        pb_vol   = df["Volume"].iloc[pb_start: idx + 1].mean()
        if pb_vol > vol_ma20 * 1.0: return False

    return True

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row = df.iloc[idx]
    c = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None

    vol_ma20 = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma20 < 100_000: return None

    if not _is_holy_grail(df, idx): return None

    rsi = float(row["rsi"]); adx = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx): return None
    if adx < 16: return None   # some trend still present (no cap — strategy works with declining ADX)

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

    ema20    = float(row["ema20"])
    vol_ratio = float(row["Volume"]) / vol_ma20 if vol_ma20 > 0 else 0
    pct_from_ema = (c - ema20) / ema20 if ema20 > 0 else 1.0

    # ADX peak in lookback window
    adx_peak = float(df["adx"].iloc[max(0, idx - ADX_LOOKBACK): idx + 1].max())

    conf = {
        "RSI40-55":  40 <= rsi <= 55,
        "ADXwas30":  adx_peak > 35,
        "VolDry":    vol_ratio < 0.8,
        "M≥5":       m >= 5,
        "NearEMA":   pct_from_ema <= 0.01,
    }
    score = sum(conf.values())

    return {
        "score":     score,
        "fresh":     ["HG-PULLBACK"],
        "conf":      [k for k, v in conf.items() if v],
        "minervini": m,
        "rsi":       round(rsi, 1),
        "adx":       round(adx, 1),
        "price":     round(c, 2),
        "vol_ratio": round(vol_ratio, 2),
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_holy_grail(df, i): continue
        row = df.iloc[i]
        c   = float(row["Close"])
        adx = float(row["adx"])
        if pd.isna(adx) or adx < 16: continue
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

        # Freshness: Holy Grail setup fired within last FRESH_WINDOW bars
        found = any(_is_holy_grail(df, k)
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
        print("  [holy_grail] SPY regime: CHOPPY/BEAR — signals tagged LOW conviction")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
                for t, b in universe.items()}
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
                if r:
                    r["strategy"]   = "holy_grail"
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
    print(f"\nRaschke Holy Grail Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  vol_ratio={r['vol_ratio']}  "
              f"spy={r.get('spy_regime','?')}  conf={r['conf']}")

if __name__ == "__main__":
    main()
