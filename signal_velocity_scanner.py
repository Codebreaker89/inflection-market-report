#!/usr/bin/env python3
"""
Signal Velocity Scanner  |  TradingView-equivalent indicator convergence
─────────────────────────────────────────────────────────────────────────
Computes a 15-indicator buy/sell score (TV-style) each day, then fires when
that score is ACCELERATING upward — multiple indicators flipping bullish in
quick succession. Catches inflection points BEFORE a clean trend is established.

Score: each indicator gives +1 (buy), -1 (sell), 0 (neutral).
       Net score = sum across all 15 indicators. Range: -15 to +15.
Signal: net_score_today >= 3  AND  net_score_today - net_score_3d_ago >= 6
        AND 3 consecutive days of increase.

python3 signal_velocity_scanner.py --no-backtest
python3 signal_velocity_scanner.py
"""

import os, sys, warnings, logging, contextlib
import numpy  as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from scanner_utils import _ema, _quiet, _rsi, _sma

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

HOLD_DAYS       = 5
MAX_WORKERS     = 25
FRESH_WINDOW    = 2
MIN_SCORE       = 3      # net score must be at least mildly bullish
MIN_VELOCITY    = 6      # must gain ≥6 net points over 3 days
MAX_SCORE       = 12     # reject if already maxed out (too late to the party)

def _stoch(high, low, close, k=14, d=3):
    lowest  = low.rolling(k).min()
    highest = high.rolling(k).max()
    pct_k   = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    pct_d   = pct_k.rolling(d).mean()
    return pct_k, pct_d

def _cci(high, low, close, n=20):
    tp  = (high + low + close) / 3
    ma  = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))

def _williams_r(high, low, close, n=14):
    hh = high.rolling(n).max(); ll = low.rolling(n).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)

def _adx_val(high, low, close, n=14):
    tr  = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    up  = (high-high.shift()).clip(lower=0); dn = (low.shift()-low).clip(lower=0)
    dmp = up.where(up>dn,0).ewm(alpha=1/n, adjust=False).mean()
    dmm = dn.where(dn>up,0).ewm(alpha=1/n, adjust=False).mean()
    dip = 100*dmp/atr; dim = 100*dmm/atr
    dx  = 100*(dip-dim).abs()/(dip+dim).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), dip, dim

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # ── Moving averages (7 signals: close vs MA) ──────────────────────────────
    df["sma20"]  = _sma(c, 20);  df["sma50"]  = _sma(c, 50)
    df["sma100"] = _sma(c, 100); df["sma200"] = _sma(c, 200)
    df["ema20"]  = _ema(c, 20);  df["ema50"]  = _ema(c, 50)
    df["ema200"] = _ema(c, 200)
    df["sma150"] = _sma(c, 150)

    # ── Oscillators ───────────────────────────────────────────────────────────
    df["rsi14"]       = _rsi(c, 14)
    df["stoch_k"], df["stoch_d"] = _stoch(h, l, c, 14, 3)
    df["cci20"]       = _cci(h, l, c, 20)
    df["wpr14"]       = _williams_r(h, l, c, 14)
    macd              = _ema(c, 12) - _ema(c, 26)
    df["macd_hist"]   = macd - _ema(macd, 9)
    df["adx"], df["dip"], df["dim"] = _adx_val(h, l, c, 14)
    # Bull Bear Power = close - EMA13
    df["bbp"]         = c - _ema(c, 13)
    # Ultimate Oscillator (simplified)
    bp   = c - pd.concat([l, c.shift()], axis=1).min(axis=1)
    tr   = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum().replace(0, np.nan)
    avg14= bp.rolling(14).sum() / tr.rolling(14).sum().replace(0, np.nan)
    avg28= bp.rolling(28).sum() / tr.rolling(28).sum().replace(0, np.nan)
    df["uo"]          = 100 * (4*avg7 + 2*avg14 + avg28) / 7

    df["vol_ma20"]    = v.rolling(20).mean()
    df["52w_high"]    = c.rolling(252).max()
    df["52w_low"]     = c.rolling(252).min()
    return df


def _compute_signal_score(df: pd.DataFrame, idx: int) -> Optional[int]:
    """
    Returns net buy/sell score for the row at idx.
    +1 = buy signal, -1 = sell signal, 0 = neutral.
    """
    if idx < 210: return None
    row = df.iloc[idx]
    c   = float(row["Close"])
    if pd.isna(c): return None

    score = 0

    # ── 7 Moving average signals ──────────────────────────────────────────────
    for ma_col in ["sma20","sma50","sma100","sma200","ema20","ema50","ema200"]:
        v = row.get(ma_col)
        if v is None or pd.isna(v): continue
        score += 1 if c > float(v) else -1

    # ── 8 Oscillator signals ──────────────────────────────────────────────────
    # RSI: <40 = buy, >60 = sell
    rsi = row["rsi14"]
    if not pd.isna(rsi):
        score += 1 if float(rsi) < 40 else (-1 if float(rsi) > 60 else 0)

    # Stochastic: %K < 20 = buy, >80 = sell; also %K above %D = buy direction
    sk, sd = row["stoch_k"], row["stoch_d"]
    if not pd.isna(sk) and not pd.isna(sd):
        if float(sk) < 20:   score += 1
        elif float(sk) > 80: score -= 1
        score += 1 if float(sk) > float(sd) else -1

    # CCI: < -100 = buy, > +100 = sell
    cci = row["cci20"]
    if not pd.isna(cci):
        score += 1 if float(cci) < -100 else (-1 if float(cci) > 100 else 0)

    # Williams %R: < -80 = buy, > -20 = sell
    wpr = row["wpr14"]
    if not pd.isna(wpr):
        score += 1 if float(wpr) < -80 else (-1 if float(wpr) > -20 else 0)

    # MACD histogram: positive = buy
    mh = row["macd_hist"]
    if not pd.isna(mh):
        score += 1 if float(mh) > 0 else -1

    # Bull Bear Power: >0 = buy
    bbp = row["bbp"]
    if not pd.isna(bbp):
        score += 1 if float(bbp) > 0 else -1

    # Ultimate Oscillator: >55 = buy, <45 = sell
    uo = row["uo"]
    if not pd.isna(uo):
        score += 1 if float(uo) > 55 else (-1 if float(uo) < 45 else 0)

    return score


def _velocity_signal(df: pd.DataFrame, idx: int) -> tuple:
    """Returns (fires, score_today, velocity) or (False, 0, 0)."""
    if idx < 215: return False, 0, 0
    s0 = _compute_signal_score(df, idx)
    s1 = _compute_signal_score(df, idx - 1)
    s2 = _compute_signal_score(df, idx - 2)
    s3 = _compute_signal_score(df, idx - 3)
    if any(x is None for x in [s0, s1, s2, s3]): return False, 0, 0
    velocity = s0 - s3   # gain over 3 days
    # 3 consecutive days of increase
    consecutive = (s0 > s1) and (s1 > s2)
    if s0 < MIN_SCORE: return False, 0, 0
    if s0 > MAX_SCORE: return False, 0, 0   # already fully priced in
    if velocity < MIN_VELOCITY: return False, 0, 0
    if not consecutive: return False, 0, 0
    return True, s0, velocity


def _score_result(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    fires, net_score, velocity = _velocity_signal(df, idx)
    if not fires: return None
    row = df.iloc[idx]
    c   = float(row["Close"])
    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 100_000: return None

    rsi_val = float(row["rsi14"]) if not pd.isna(row["rsi14"]) else 0.0
    adx_val = float(row["adx"])   if not pd.isna(row["adx"])   else 0.0

    m = sum([
        c > row["sma150"], c > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        c > row["sma50"],
        c >= 1.30 * row["52w_low"],
        c >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[idx - 20]["sma200"],
    ])

    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    score = min(velocity, 10) + (2 if net_score >= 8 else 0) + (1 if vol_ratio > 1.5 else 0)

    conf = []
    if net_score >= 8:  conf.append(f"SCORE:{net_score}/15")
    if velocity >= 8:   conf.append(f"VEL+{velocity}")
    if vol_ratio > 1.5: conf.append("VOL1.5x")
    if adx_val > 20:    conf.append("ADX>20")

    return {
        "score": score, "fresh": [f"SV+{velocity}"], "conf": conf,
        "minervini": m, "rsi": round(rsi_val, 1), "adx": round(adx_val, 1),
        "price": round(c, 2), "vol_ratio": round(vol_ratio, 2),
        "net_score": net_score, "velocity": velocity,
    }


def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        fires, _, _ = _velocity_signal(df, i)
        if not fires: continue
        entry = float(df.iloc[i]["Close"]); exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100); last = i
    if not rets: return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n": len(a), "wr": round(100*(a>0).mean(),1),
            "avg": round(float(a.mean()),2), "med": round(float(np.median(a)),2)}


def analyze_ticker(ticker: str, bench_ret: Optional[float],
                   with_backtest: bool) -> Optional[dict]:
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
        found = any(_velocity_signal(df, k)[0]
                    for k in range(max(215, last - FRESH_WINDOW + 1), last + 1))
        if not found: return None
        sig = _score_result(df, last)
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
                    r["strategy"] = "signal_velocity"
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
    print(f"\nSignal Velocity Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  net={r['net_score']}/15  "
              f"vel=+{r['velocity']}  m={r['minervini']}  rsi={r['rsi']}")

if __name__ == "__main__":
    main()
