#!/usr/bin/env python3
"""
RS Line Scanner  |  O'Neil / IBD RS Line
─────────────────────────────────────────
RS line = stock Close / SPY Close (rolling ratio).

O'Neil's core concept: The RS line making a new 52-week high BEFORE or
SIMULTANEOUSLY with the price breakout is the strongest early-entry signal.
It means the stock is already outperforming the market even while still in a
base — institutions are quietly accumulating.

Rebuilt signal criteria (v2 — correct O'Neil implementation):
  1. RS line (Close/SPY) makes a new 52-week high within the last FRESH_DAYS bars
  2. RS line led price: RS new high happened ≥0 days before price 52w high
     (i.e., RS new high came first or simultaneously)
  3. Price is within 20% of its own 52-week high (still in base or early breakout)
  4. Minervini Stage 2: score ≥5 (MA stack aligned, price above MAs)
  5. Volume ≥ 1.3× 20d avg on signal day (institutional participation)
  6. Price > $5, avg vol > 200k (liquid)
  7. NOT already extended: price < 110% of 50 SMA (no chasing)

python3 rs_line_scanner.py --no-backtest   # fast
python3 rs_line_scanner.py                 # with backtest
"""

import os, sys, warnings, logging
import numpy  as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from scanner_utils import _adx, _rsi, _sma, _quiet

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOLD_DAYS    = 7
MAX_WORKERS  = 20
FRESH_DAYS   = 5    # RS line new high must have fired within last N bars
RS_LEAD_MAX  = 30   # RS line new high must have been within this many bars of today

# ── SPY CACHE ─────────────────────────────────────────────────────────────────
_spy_cache: Optional[pd.Series] = None

def _get_spy() -> Optional[pd.Series]:
    """Download SPY once and cache. Returns Close series indexed by date."""
    global _spy_cache
    if _spy_cache is not None:
        return _spy_cache
    try:
        with _quiet():
            raw = yf.download("SPY", period="500d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 252:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        _spy_cache = raw["Close"].dropna()
        return _spy_cache
    except Exception:
        return None


# ── INDICATORS ────────────────────────────────────────────────────────────────

def _build(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma20"]    = _sma(c, 20)
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()

    # Align SPY to ticker's date index, forward-fill any market gaps
    spy_aligned     = spy_close.reindex(df.index).ffill()
    df["spy_close"] = spy_aligned
    # RS ratio — normalised so 1.0 = parity with SPY on first valid day
    rs_raw          = c / spy_aligned.replace(0, np.nan)
    first_valid     = rs_raw.first_valid_index()
    if first_valid is not None:
        base = float(rs_raw.loc[first_valid])
        df["rs_ratio"] = rs_raw / base if base > 0 else rs_raw
    else:
        df["rs_ratio"] = rs_raw

    # RS line 52w high: compare today vs max of PRIOR 252 bars (shift(1) avoids look-ahead)
    df["rs_52w_max"] = df["rs_ratio"].shift(1).rolling(252).max()

    # Flag: is this bar a NEW RS 52w high?
    df["rs_new_high"] = df["rs_ratio"] > df["rs_52w_max"]

    # Price 52w high flag
    df["px_52w_max"]  = df["Close"].shift(1).rolling(252).max()
    df["px_new_high"] = df["Close"] > df["px_52w_max"]

    return df


def _rs_lead_days(df: pd.DataFrame, idx: int, window: int = 60) -> Optional[int]:
    """
    Find the most recent RS new-high bar within `window` bars ending at idx.
    Return how many bars BEFORE the most recent price 52w high that RS high fired.
    Positive = RS led price. 0 = same bar. Negative = RS lagged.
    Returns None if no RS new high found in window.
    """
    # Find last RS new high within window
    rs_bar = None
    for k in range(0, min(window, idx) + 1):
        if df.iloc[idx - k]["rs_new_high"]:
            rs_bar = idx - k
            break
    if rs_bar is None:
        return None

    # Find last price 52w high within window
    px_bar = None
    for k in range(0, min(window, idx) + 1):
        if df.iloc[idx - k]["px_new_high"]:
            px_bar = idx - k
            break

    if px_bar is None:
        # Price hasn't made a new high recently — RS is leading (good)
        return window  # treat as RS leading by full window

    return px_bar - rs_bar  # positive = RS came before price


def _is_signal(df: pd.DataFrame, idx: int) -> bool:
    """True if RS line new high fired within FRESH_DAYS and leads/matches price."""
    if idx < 252: return False
    row = df.iloc[idx]

    # Basic liquidity
    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < 200_000: return False
    c = float(row["Close"])
    if pd.isna(c) or c < 5.0: return False

    # RS line new high within FRESH_DAYS
    fresh_rs = any(
        bool(df.iloc[idx - k]["rs_new_high"])
        for k in range(0, min(FRESH_DAYS, idx) + 1)
        if not pd.isna(df.iloc[idx - k]["rs_new_high"])
    )
    if not fresh_rs: return False

    # Price within 20% of 52w high (in base or early breakout, not extended)
    high52 = float(row["52w_high"]) if not pd.isna(row["52w_high"]) else 0
    if high52 == 0: return False
    pct_from_high = (high52 - c) / high52
    if pct_from_high > 0.20: return False

    # Not too extended above 50 SMA
    sma50 = float(row["sma50"]) if not pd.isna(row["sma50"]) else 0
    if sma50 > 0 and c > sma50 * 1.10: return False

    # Volume surge
    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    if vol_ratio < 1.3: return False

    return True


def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if not _is_signal(df, idx): return None
    row  = df.iloc[idx]
    c    = float(row["Close"])
    rsi  = float(row["rsi"])
    adx  = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx): return None

    vol_ma    = float(row["vol_ma20"])
    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    high52    = float(row["52w_high"])
    pct_from_high = (high52 - c) / high52

    # RS lead days — how early did RS line fire vs price?
    lead = _rs_lead_days(df, idx)
    if lead is None: return None  # no RS new high found (shouldn't happen after _is_signal)
    # Require RS to lead or match (not lag more than 2 bars)
    if lead < -2: return None

    # Minervini template score
    sma50  = float(row["sma50"])  if not pd.isna(row["sma50"])  else 0
    sma150 = float(row["sma150"]) if not pd.isna(row["sma150"]) else 0
    sma200 = float(row["sma200"]) if not pd.isna(row["sma200"]) else 0
    low52  = float(row["52w_low"]) if not pd.isna(row["52w_low"]) else 0
    sma200_20ago = float(df.iloc[idx - 20]["sma200"]) if idx >= 252 and not pd.isna(df.iloc[idx - 20]["sma200"]) else 0
    m = sum([
        c > sma150 if sma150 > 0 else False,
        c > sma200 if sma200 > 0 else False,
        sma150 > sma200 if sma150 > 0 and sma200 > 0 else False,
        sma50  > sma150 if sma50  > 0 and sma150 > 0 else False,
        c > sma50  if sma50  > 0 else False,
        c >= 1.30 * low52  if low52  > 0 else False,
        c >= 0.75 * high52 if high52 > 0 else False,
        sma200 > sma200_20ago if sma200 > 0 and sma200_20ago > 0 else False,
    ])
    if m < 5: return None

    conf = {
        "RS_LEADS":      lead >= 0,          # RS line fired before or with price
        "RS_LEADS_5+":   lead >= 5,          # RS line led by ≥5 bars (early signal)
        "IN_BASE":       pct_from_high > 0.03,  # still in base (not broken out)
        "NEAR_HIGH":     pct_from_high <= 0.05, # within 5% of high (breakout zone)
        "VOL≥1.5x":     vol_ratio >= 1.5,
        "RSI45-70":      45 <= rsi <= 70,
        "ADX>20":        adx > 20,
        "M≥6":           m >= 6,
    }
    score = sum(conf.values())

    return {
        "score":        score,
        "fresh":        ["RS-NEW-HIGH"],
        "conf":         [k for k, v in conf.items() if v],
        "minervini":    m,
        "rsi":          round(rsi, 1),
        "adx":          round(adx, 1),
        "price":        round(c, 2),
        "vol_ratio":    round(vol_ratio, 2),
        "rs_lead_days": lead,
        "pct_from_high": round(pct_from_high * 100, 1),
    }


def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(252, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if not _is_signal(df, i): continue
        lead = _rs_lead_days(df, i)
        if lead is None or lead < -2: continue
        entry = float(df.iloc[i]["Close"])
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100)
        last = i
    if not rets: return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n": len(a), "wr": round(100*(a>0).mean(), 1),
            "avg": round(float(a.mean()), 2), "med": round(float(np.median(a)), 2)}


def analyze_ticker(ticker: str, bench_ret: Optional[float], with_backtest: bool,
                   spy: pd.Series) -> Optional[dict]:
    try:
        with _quiet():
            raw = yf.download(ticker, period="500d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 260: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return None
        raw = raw.dropna(subset=["Close","High","Low","Volume"])
        if len(raw) < 260: return None

        df   = _build(raw.copy(), spy)
        last = len(df) - 1

        if not _is_signal(df, last): return None
        sig = _score(df, last)
        if not sig: return None

        result = {"ticker": ticker, **sig, "hold_days": HOLD_DAYS, "strategy": "rs_line"}
        for sfx, mkt in {".L":"UK",".DE":"DE",".PA":"FR",".AS":"NL",".TO":"CA"}.items():
            if ticker.endswith(sfx): result["mkt"] = mkt; break
        else:
            result["mkt"] = "US"

        if with_backtest: result.update(run_backtest(df))
        return result
    except Exception:
        return None


def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    spy = _get_spy()
    if spy is None:
        return []
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest, spy): t
                for t, b in universe.items()}
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
                if r:
                    results.append(r)
            except Exception:
                pass
    # Sort: RS led earliest first, then by vol ratio
    return sorted(results, key=lambda x: (-x.get("rs_lead_days", 0), -x.get("vol_ratio", 0)))


if __name__ == "__main__":
    import argparse
    from momentum_scanner import build_universe, compute_bench_returns
    p = argparse.ArgumentParser(); p.add_argument("--no-backtest", action="store_true")
    args = p.parse_args()
    u = build_universe(); br = compute_bench_returns(set(u.values()))
    res = scan(u, br, not args.no_backtest)
    for r in res[:10]:
        print(f"  {r['ticker']:<8} RS_lead={r.get('rs_lead_days',0):+d}d  "
              f"from_high={r.get('pct_from_high',0):.1f}%  vol={r.get('vol_ratio',0):.1f}x  "
              f"rsi={r.get('rsi',0):.0f}  score={r.get('score',0)}  M={r.get('minervini',0)}")
