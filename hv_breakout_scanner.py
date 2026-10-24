#!/usr/bin/env python3
"""
HV-Ratio Volatility Breakout  |  Crabel / Raschke / Connors (Street Smarts)
──────────────────────────────────────────────────────────────────────────
Universe: 100-day historical volatility in the top quartile of today's list
          (structurally high-vol names only).
Coil:     HV(6) / HV(100) ≤ 0.50 on YESTERDAY — usually wild, quiet now.
Trigger:  TODAY closes above yesterday's high → LONG
          TODAY closes below yesterday's low  → SHORT
          (two-sided; first filled side wins — at EOD only one can fire)
Skip:     yesterday's range > 2 × ATR(14) (stop too wide)

Hold 3 trading days. Direction-agnostic by design.

python3 hv_breakout_scanner.py --no-backtest
python3 hv_breakout_scanner.py
"""

import os, sys, warnings, logging
import numpy  as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from scanner_utils import _adx, _quiet, _rsi, _sma

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOLD_DAYS       = 3
MAX_WORKERS     = 25
HV_SHORT        = 6
HV_LONG         = 100
HV_RATIO_MAX    = 0.50      # coil: short HV ≤ half of long HV
RANGE_ATR_MAX   = 2.0       # skip if yesterday range > 2× ATR(14)
MIN_PRICE       = 1.0
MIN_VOL_MA      = 100_000
LOOKBACK_PERIOD = "400d"


def _hv(logret: pd.Series, n: int) -> pd.Series:
    """Annualised historical volatility over an n-day window."""
    return logret.rolling(n).std() * np.sqrt(252)


def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    logret = np.log(c / c.shift(1))
    df["hv_s"]     = _hv(logret, HV_SHORT)
    df["hv_l"]     = _hv(logret, HV_LONG)
    df["hv_ratio"] = df["hv_s"] / df["hv_l"].replace(0, np.nan)
    df["range"]    = h - l
    tr = pd.concat(
        [(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["atr14"]    = tr.ewm(alpha=1/14, adjust=False).mean()
    df["sma50"]    = _sma(c, 50)
    df["sma200"]   = _sma(c, 200)
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    return df


def _coil_yesterday(df: pd.DataFrame, idx: int) -> bool:
    """Yesterday's HV ratio ≤ threshold (stock coiled relative to its own history)."""
    if idx < HV_LONG + 5:
        return False
    y = df.iloc[idx - 1]
    ratio = float(y["hv_ratio"])
    return (not pd.isna(ratio)) and ratio <= HV_RATIO_MAX


def _side_today(df: pd.DataFrame, idx: int) -> Optional[str]:
    """Return 'long' / 'short' if today broke yesterday's range; else None."""
    if idx < 2:
        return None
    today = df.iloc[idx]
    yday  = df.iloc[idx - 1]
    c = float(today["Close"])
    hi = float(yday["High"])
    lo = float(yday["Low"])
    if pd.isna(c) or pd.isna(hi) or pd.isna(lo):
        return None
    # At EOD only one side can fire; if both somehow true (gap through), skip.
    above = c > hi
    below = c < lo
    if above and below:
        return None
    if above:
        return "long"
    if below:
        return "short"
    return None


def _range_ok(df: pd.DataFrame, idx: int) -> bool:
    """Yesterday's range must be ≤ 2× ATR(14) so the stop is not absurdly wide."""
    y = df.iloc[idx - 1]
    rng = float(y["range"])
    atr = float(y["atr14"])
    if pd.isna(rng) or pd.isna(atr) or atr <= 0:
        return False
    return rng <= RANGE_ATR_MAX * atr


def _score(df: pd.DataFrame, idx: int, side: str) -> Optional[dict]:
    if idx < HV_LONG + 5:
        return None
    row  = df.iloc[idx]
    yday = df.iloc[idx - 1]
    c = float(row["Close"])
    if pd.isna(c) or c < MIN_PRICE:
        return None
    vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    if vol_ma < MIN_VOL_MA:
        return None
    if not _coil_yesterday(df, idx):
        return None
    if not _range_ok(df, idx):
        return None

    rsi = float(row["rsi"]); adx = float(row["adx"])
    if pd.isna(rsi) or pd.isna(adx):
        return None

    hv_l   = float(yday["hv_l"])
    hv_rat = float(yday["hv_ratio"])
    y_hi   = float(yday["High"])
    y_lo   = float(yday["Low"])
    stop   = y_lo if side == "long" else y_hi
    risk_pct = abs(c - stop) / c * 100 if c else 0

    vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
    conf = {
        "VOL>avg":   vol_ratio >= 1.0,
        "ADX≥16":    adx >= 16,
        "COIL≤0.5":  hv_rat <= HV_RATIO_MAX,
        "RISK≤2ATR": True,   # already gated by _range_ok
    }
    # Directional RSI sanity: longs prefer not-overbought; shorts not-oversold
    if side == "long":
        conf["RSI≤75"] = rsi <= 75
    else:
        conf["RSI≥25"] = rsi >= 25

    score = sum(conf.values())
    fresh = ["HVBRK", "LONG" if side == "long" else "SHORT"]
    return {
        "score":       score,
        "fresh":       fresh,
        "conf":        [k for k, v in conf.items() if v],
        "side":        side,
        "minervini":   0,   # not a trend-template strategy
        "rsi":         round(rsi, 1),
        "adx":         round(adx, 1),
        "price":       round(c, 2),
        "vol_ratio":   round(vol_ratio, 2),
        "hv100":       round(hv_l * 100, 1),   # % annualised
        "hv_ratio":    round(hv_rat, 3),
        "stop_price":  round(stop, 2),
        "risk_pct":    round(risk_pct, 2),
        "y_high":      round(y_hi, 2),
        "y_low":       round(y_lo, 2),
    }


def _signed_ret(entry: float, exit_: float, side: str) -> float:
    if side == "short":
        return (entry - exit_) / entry * 100
    return (exit_ - entry) / entry * 100


def run_backtest(df: pd.DataFrame) -> dict:
    """Walk history; only count trades that would pass the high-vol gate locally
    via HV100 percentile vs the stock's own history is not available cross-sectionally
    here, so backtest uses the coil+breakout rules alone (same as live entry logic
    minus the universe quartile filter applied in scan())."""
    rets, last = [], -10
    for i in range(HV_LONG + 5, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS:
            continue
        if not _coil_yesterday(df, i):
            continue
        if not _range_ok(df, i):
            continue
        side = _side_today(df, i)
        if not side:
            continue
        entry = float(df.iloc[i]["Close"])
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        if pd.isna(entry) or pd.isna(exit_) or entry <= 0:
            continue
        rets.append(_signed_ret(entry, exit_, side))
        last = i
    if not rets:
        return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {
        "n":   len(a),
        "wr":  round(100 * (a > 0).mean(), 1),
        "avg": round(float(a.mean()), 2),
        "med": round(float(np.median(a)), 2),
    }


def _analyze(ticker: str, with_backtest: bool) -> Optional[dict]:
    """Return {hv100, signal?} for quartile ranking + optional breakout."""
    try:
        with _quiet():
            raw = yf.download(ticker, period=LOOKBACK_PERIOD, interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < HV_LONG + 20:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
            return None
        raw = raw.dropna(subset=["Close", "High", "Low", "Volume"])
        if len(raw) < HV_LONG + 20:
            return None
        df = _build(raw.copy())
        last = len(df) - 1
        hv100 = float(df.iloc[last - 1]["hv_l"]) if last >= 1 else float("nan")
        if pd.isna(hv100) or hv100 <= 0:
            return None

        out: dict = {"ticker": ticker, "hv100": hv100, "signal": None}
        side = _side_today(df, last)
        if side:
            sig = _score(df, last, side)
            if sig:
                result = {"ticker": ticker, **sig, "hold_days": HOLD_DAYS}
                for sfx, mkt in {".L": "UK", ".DE": "DE", ".PA": "FR",
                                 ".AS": "NL", ".TO": "CA",
                                 ".NS": "IN", ".BO": "IN"}.items():
                    if ticker.endswith(sfx):
                        result["mkt"] = mkt
                        break
                else:
                    result["mkt"] = "US"
                if with_backtest:
                    result.update(run_backtest(df))
                out["signal"] = result
        return out
    except Exception:
        return None


def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    """Single pass: measure HV100 across the universe, keep top-quartile breakouts."""
    tickers = list(universe.keys())
    hv_map: dict[str, float] = {}
    candidates: list = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_analyze, t, with_backtest): t for t in tickers}
        for f in as_completed(futs, timeout=300):
            try:
                row = f.result(timeout=30)
                if not row:
                    continue
                hv_map[row["ticker"]] = row["hv100"]
                if row.get("signal"):
                    candidates.append(row["signal"])
            except Exception:
                pass

    if len(hv_map) < 8:
        return []

    cutoff = float(np.percentile(list(hv_map.values()), 75))
    results = []
    for r in candidates:
        hv = hv_map.get(r["ticker"])
        if hv is None or hv < cutoff:
            continue
        r["strategy"] = "hv_breakout"
        r["hv100"] = round(hv * 100, 1)
        results.append(r)

    results.sort(key=lambda x: (
        0 if x.get("side") == "long" else 1,
        -(x.get("wr") or 0),
        -x.get("score", 0),
    ))
    return results


def main():
    from momentum_scanner import build_universe, compute_bench_returns
    import time
    wb = "--no-backtest" not in sys.argv
    uni = build_universe()
    bench = compute_bench_returns(set(uni.values()))
    t0 = time.time()
    res = scan(uni, bench, wb)
    print(f"\nHV Breakout Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} {r.get('side','?'):<5} score={r['score']}  "
              f"hv100={r.get('hv100')}%  ratio={r.get('hv_ratio')}  "
              f"rsi={r['rsi']}  stop={r.get('stop_price')}  risk={r.get('risk_pct')}%")


if __name__ == "__main__":
    main()
