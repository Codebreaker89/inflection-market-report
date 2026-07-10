#!/usr/bin/env python3
"""
VCP Scanner  |  Minervini Volatility Contraction Pattern
─────────────────────────────────────────────────────────
Series of price contractions, each tighter than the last, on drying volume.
Requires ≥3 contractions with decreasing amplitude, final contraction ≤10%,
volume drying, price near base high, and Minervini template ≥6.

python3 vcp_scanner.py --no-backtest   # fast
python3 vcp_scanner.py                 # with backtest
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
HOLD_DAYS    = 10
MAX_WORKERS  = 25
LOOKBACK     = 60       # bars to look back for VCP detection
SWING_WINDOW = 2        # pivot half-width (swing at i uses i-2:i+3)
FRESH_WINDOW = 2        # VCP must be valid within last N bars

# ── INDICATOR HELPERS ─────────────────────────────────────────────────────────

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["sma20"]    = _sma(c, 20)
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ma40"] = v.rolling(40).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df

# ── VCP DETECTION ─────────────────────────────────────────────────────────────
def _find_swing_highs(highs: pd.Series) -> list:
    """Return list of (index_position, price) for all swing highs in the series."""
    result = []
    arr = highs.values
    n   = len(arr)
    for i in range(SWING_WINDOW, n - SWING_WINDOW):
        window = arr[i - SWING_WINDOW: i + SWING_WINDOW + 1]
        if arr[i] == window.max():
            result.append((i, arr[i]))
    return result

def _find_swing_lows(lows: pd.Series) -> list:
    """Return list of (index_position, price) for all swing lows in the series."""
    result = []
    arr = lows.values
    n   = len(arr)
    for i in range(SWING_WINDOW, n - SWING_WINDOW):
        window = arr[i - SWING_WINDOW: i + SWING_WINDOW + 1]
        if arr[i] == window.min():
            result.append((i, arr[i]))
    return result

def _detect_vcp(df: pd.DataFrame, end_idx: int) -> Optional[dict]:
    """
    Analyse the LOOKBACK bars ending at end_idx.
    Returns dict with n_contractions, final_contraction_pct, base_high
    or None if no valid VCP found.
    """
    start = max(0, end_idx - LOOKBACK + 1)
    sub_h = df["High"].iloc[start: end_idx + 1]
    sub_l = df["Low"].iloc[start: end_idx + 1]

    swing_highs = _find_swing_highs(sub_h)
    swing_lows  = _find_swing_lows(sub_l)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None

    # Pair consecutive swing high → nearest following swing low to form contractions
    contractions = []
    sh_list = swing_highs  # sorted by position ascending (they come from a left-to-right scan)
    sl_list = swing_lows

    for sh_pos, sh_price in sh_list:
        # Find a swing low that comes after this swing high
        following_lows = [(sl_pos, sl_price) for sl_pos, sl_price in sl_list if sl_pos > sh_pos]
        if not following_lows:
            continue
        sl_pos, sl_price = following_lows[0]  # nearest low after the high
        if sh_price <= 0:
            continue
        contraction_pct = (sh_price - sl_price) / sh_price
        contractions.append({
            "sh_pos": sh_pos, "sh_price": sh_price,
            "sl_pos": sl_pos, "sl_price": sl_price,
            "pct":    contraction_pct,
        })

    if len(contractions) < 3:
        return None

    # Check decreasing amplitude: each contraction must be smaller than the previous
    pcts = [c["pct"] for c in contractions]
    is_decreasing = all(pcts[i] < pcts[i-1] for i in range(1, len(pcts)))
    if not is_decreasing:
        # Try to find the longest decreasing subsequence from most recent bars
        # Walk backwards and find the longest valid tail
        valid_tail = [contractions[-1]]
        for c in reversed(contractions[:-1]):
            if c["pct"] > valid_tail[0]["pct"]:
                valid_tail.insert(0, c)
            else:
                break
        if len(valid_tail) < 3:
            return None
        contractions = valid_tail
        pcts = [c["pct"] for c in contractions]

    n_contractions = len(contractions)
    final_pct      = pcts[-1]
    base_high      = contractions[0]["sh_price"]

    # Final contraction must be ≤ 10%
    if final_pct > 0.10:
        return None

    return {
        "n_contractions":    n_contractions,
        "final_contraction": round(final_pct * 100, 2),  # as percent
        "base_high":         base_high,
        "contractions":      contractions,
    }

def _vcp_valid(df: pd.DataFrame, idx: int) -> Optional[dict]:
    """Run VCP detection at idx and return VCP info dict or None."""
    if idx < 215: return None
    return _detect_vcp(df, idx)

def _score(df: pd.DataFrame, idx: int) -> Optional[dict]:
    if idx < 215: return None
    row = df.iloc[idx]
    c   = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None

    vol_ma20 = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    vol_ma40 = float(row["vol_ma40"]) if not pd.isna(row["vol_ma40"]) else 0
    if vol_ma20 < 100_000: return None

    rsi = float(row["rsi"]); adx = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx): return None
    if adx < 16 or adx > 35: return None

    # Volume drying: recent 20d avg < 40d avg × 0.90
    vol_drying = (vol_ma40 > 0) and (vol_ma20 < vol_ma40 * 0.90)
    if not vol_drying: return None

    # VCP detection
    vcp = _vcp_valid(df, idx)
    if vcp is None: return None

    n_c        = vcp["n_contractions"]
    final_pct  = vcp["final_contraction"]
    base_high  = vcp["base_high"]

    # Price within 10% of base high (near pivot)
    if base_high <= 0: return None
    pct_from_base = (base_high - c) / base_high
    if pct_from_base > 0.10: return None

    # Minervini template
    m = sum([
        c > row["sma150"],
        c > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        c > row["sma50"],
        c >= 1.30 * row["52w_low"],
        c >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[idx - 20]["sma200"],
    ])
    if m < 6: return None

    vol_ratio = float(row["Volume"]) / vol_ma20 if vol_ma20 > 0 else 0

    conf = {
        "Contractions≥4": n_c >= 4,
        "TightFinal":     final_pct <= 6.0,
        "VolDry":         vol_drying,
        "NearPivot":      pct_from_base <= 0.05,
        "RSI50-65":       50 <= rsi <= 65,
        "M≥7":            m >= 7,
    }
    score = sum(conf.values())

    return {
        "score":          score,
        "fresh":          [f"VCP-{n_c}C"],
        "conf":           [k for k, v in conf.items() if v],
        "minervini":      m,
        "rsi":            round(rsi, 1),
        "adx":            round(adx, 1),
        "price":          round(c, 2),
        "vol_ratio":      round(vol_ratio, 2),
        "n_contractions": n_c,
        "final_pct":      final_pct,
        "base_high":      round(base_high, 2),
    }

def run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -10
    for i in range(215, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        vcp = _vcp_valid(df, i)
        if vcp is None: continue
        row = df.iloc[i]
        c   = float(row["Close"])
        m   = sum([
            c > row["sma150"], c > row["sma200"],
            row["sma150"] > row["sma200"],
            row["sma50"] > row["sma150"],
            c > row["sma50"],
            c >= 1.30 * row["52w_low"],
            c >= 0.75 * row["52w_high"],
            row["sma200"] > df.iloc[i - 20]["sma200"],
        ])
        if m < 6: continue
        base_high = vcp["base_high"]
        if base_high <= 0 or (base_high - c) / base_high > 0.10: continue
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

        # Freshness: VCP valid within last FRESH_WINDOW bars
        found = any(_vcp_valid(df, k) is not None
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
        print("  [vcp] SPY regime: CHOPPY/BEAR — signals tagged LOW conviction")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
                for t, b in universe.items()}
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
                if r:
                    r["strategy"] = "vcp"
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
    uni = build_universe()
    bench = compute_bench_returns(set(uni.values()))
    t0 = time.time()
    res = scan(uni, bench, wb)
    print(f"\nVCP Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  fresh={r['fresh']}  "
              f"final_pct={r.get('final_pct','?')}%  contractions={r.get('n_contractions','?')}")

if __name__ == "__main__":
    main()
