#!/usr/bin/env python3
"""
Combo Scanner  |  Pocket Pivot + EMA Ribbon Confluence
───────────────────────────────────────────────────────
Highest-conviction momentum setup: fires only when BOTH pocket pivot AND EMA
ribbon signals trigger on the same ticker on the same day. EMAs 8/13/21/34/55
must be stacked and expanding; price pulled back to touch EMA8 in last 3 bars
then closes above it. Minervini ≥6, ADX 18-40, RSI 45-72, vol ≥1.3×.

python3 combo_scanner.py                # full scan + backtest
python3 combo_scanner.py --no-backtest  # signals only
"""

import os, sys, warnings, logging
import numpy  as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from scanner_utils import _adx, _ema, _quiet, _rsi, _sma

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

# ── CONFIG ────────────────────────────────────────────────────────────────────
HOLD_DAYS      = 7
MAX_WORKERS    = 25
MIN_PRICE      = 1.0
MIN_AVG_VOL    = 100_000
MIN_MINERVINI  = 6
MIN_ADX        = 18
MAX_ADX        = 40
MIN_VOL_RATIO  = 1.3
EMA_PERIODS    = (8, 13, 21, 34, 55)


def _get_mkt(ticker: str) -> str:
    for sfx, mkt in {".L": "UK", ".DE": "DE", ".PA": "FR",
                      ".AS": "NL", ".TO": "CA"}.items():
        if ticker.endswith(sfx):
            return mkt
    return "US"


def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    df["vol_ma20"] = v.rolling(20).mean()
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    for p in EMA_PERIODS:
        df[f"ema{p}"] = _ema(c, p)
    return df


def _minervini(df: pd.DataFrame, idx: int) -> int:
    row = df.iloc[idx]
    c   = float(row["Close"])
    return sum([
        c > row["sma150"],
        c > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        c > row["sma50"],
        c >= 1.30 * row["52w_low"],
        c >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[max(0, idx - 20)]["sma200"],
    ])


def _pocket_pivot_fires(df: pd.DataFrame, idx: int) -> bool:
    """
    True if day idx is a pocket pivot:
    - Closes up vs previous day
    - Today's volume > max down-day volume in prior 10 sessions
    """
    if idx < 10:
        return False
    row  = df.iloc[idx]
    prev = df.iloc[idx - 1]
    if float(row["Close"]) <= float(prev["Close"]):
        return False
    prior = df.iloc[idx - 10:idx]
    down_days = prior[prior["Close"] < prior["Close"].shift(1)]
    if down_days.empty:
        return True
    max_down_vol = float(down_days["Volume"].max())
    return float(row["Volume"]) > max_down_vol


def _ema_ribbon_ok(df: pd.DataFrame, idx: int) -> bool:
    """
    True when:
    - EMAs 8 > 13 > 21 > 34 > 55 (stacked)
    - Spread (ema8 - ema55) is wider than 3 bars ago (ribbon expanding)
    - Price touched EMA8 (low ≤ ema8 × 1.02) in at least one of the last 3 bars,
      and today's close is above ema8
    """
    if idx < 3:
        return False
    row = df.iloc[idx]

    e8  = float(row["ema8"])
    e13 = float(row["ema13"])
    e21 = float(row["ema21"])
    e34 = float(row["ema34"])
    e55 = float(row["ema55"])

    # All must be valid numbers
    if any(pd.isna(v) for v in [e8, e13, e21, e34, e55]):
        return False

    # Stacked
    if not (e8 > e13 > e21 > e34 > e55):
        return False

    # Ribbon expanding (spread wider than 3 bars ago)
    row3 = df.iloc[idx - 3]
    e8_3  = float(row3["ema8"])
    e55_3 = float(row3["ema55"])
    if pd.isna(e8_3) or pd.isna(e55_3):
        return False
    spread_now  = e8  - e55
    spread_past = e8_3 - e55_3
    if spread_now <= spread_past:
        return False

    # Price pulled back to touch EMA8 in last 3 bars (including today)
    touched = False
    for k in range(idx - 2, idx + 1):
        low_k = float(df.iloc[k]["Low"])
        ema8_k = float(df.iloc[k]["ema8"])
        if pd.isna(low_k) or pd.isna(ema8_k):
            continue
        if low_k <= ema8_k * 1.02:
            touched = True
            break
    if not touched:
        return False

    # Closes above EMA8 today
    if float(row["Close"]) <= e8:
        return False

    return True


def _both_signals_fire(df: pd.DataFrame, idx: int) -> bool:
    """Returns True if both pocket pivot and EMA ribbon fire at idx."""
    return _pocket_pivot_fires(df, idx) and _ema_ribbon_ok(df, idx)


def _score_ticker(df: pd.DataFrame) -> Optional[dict]:
    if len(df) < 220:
        return None

    last_idx = len(df) - 1
    row = df.iloc[last_idx]

    price = float(row["Close"])
    if price < MIN_PRICE:
        return None

    vol_ma20 = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0.0
    if vol_ma20 < MIN_AVG_VOL:
        return None

    adx_val = float(row["adx"])
    if pd.isna(adx_val) or adx_val < MIN_ADX or adx_val > MAX_ADX:
        return None

    rsi_val = float(row["rsi"])
    if pd.isna(rsi_val) or rsi_val < 45 or rsi_val > 72:
        return None

    vol_ratio = float(row["Volume"]) / vol_ma20 if vol_ma20 > 0 else 0.0
    if vol_ratio < MIN_VOL_RATIO:
        return None

    m = _minervini(df, last_idx)
    if m < MIN_MINERVINI:
        return None

    if not _both_signals_fire(df, last_idx):
        return None

    # Confirmation signals
    e8  = float(row["ema8"])
    e55 = float(row["ema55"])
    row3 = df.iloc[last_idx - 3]
    e8_3  = float(row3["ema8"])
    e55_3 = float(row3["ema55"])
    spread_now  = e8  - e55
    spread_past = e8_3 - e55_3

    # Pullback touch: any of last 3 bars had low ≤ ema8 × 1.02
    pullback_touch = any(
        float(df.iloc[k]["Low"]) <= float(df.iloc[k]["ema8"]) * 1.02
        for k in range(last_idx - 2, last_idx + 1)
        if not pd.isna(df.iloc[k]["ema8"])
    )

    prev_row = df.iloc[last_idx - 1]
    prev_adx = float(prev_row["adx"]) if not pd.isna(prev_row["adx"]) else adx_val

    conf = {
        "RIBBON_EXP": spread_now > spread_past,
        "PP_VOL":     _pocket_pivot_fires(df, last_idx),
        "PULLBACK":   pullback_touch,
        "RSI50-65":   50 <= rsi_val <= 65,
        "ADX18-40":   MIN_ADX <= adx_val <= MAX_ADX,
    }

    # Base score = 3 (combo premium) + sum of confirmations
    score = 3 + sum(conf.values())

    return {
        "score":     score,
        "fresh":     ["PP", "EMA_RIBBON"],
        "conf":      [k for k, v in conf.items() if v],
        "minervini": m,
        "rsi":       round(rsi_val, 1),
        "adx":       round(adx_val, 1),
        "price":     round(price, 2),
        "vol_ratio": round(vol_ratio, 2),
    }


def _run_backtest(df: pd.DataFrame) -> dict:
    rets, last = [], -HOLD_DAYS
    for i in range(220, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS:
            continue
        if not _both_signals_fire(df, i):
            continue
        row  = df.iloc[i]
        vm20 = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0.0
        if vm20 < MIN_AVG_VOL:
            continue
        adx_v = float(row["adx"]) if not pd.isna(row["adx"]) else 0.0
        if adx_v < MIN_ADX or adx_v > MAX_ADX:
            continue
        rsi_v = float(row["rsi"]) if not pd.isna(row["rsi"]) else 0.0
        if rsi_v < 45 or rsi_v > 72:
            continue
        vr = float(row["Volume"]) / vm20 if vm20 > 0 else 0.0
        if vr < MIN_VOL_RATIO:
            continue
        m = _minervini(df, i)
        if m < MIN_MINERVINI:
            continue
        entry = float(df.iloc[i]["Close"])
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100)
        last = i
    if not rets:
        return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n":   len(a),
            "wr":  round(100 * (a > 0).mean(), 1),
            "avg": round(float(a.mean()), 2),
            "med": round(float(np.median(a)), 2)}


def _analyze(ticker: str, with_backtest: bool) -> Optional[dict]:
    try:
        with _quiet():
            raw = yf.download(ticker, period="400d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 220:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(raw.columns):
            return None
        raw = raw.dropna(subset=["Close", "High", "Low", "Volume"])
        if len(raw) < 220:
            return None

        df  = _build(raw.copy())
        sig = _score_ticker(df)
        if not sig:
            return None

        result = {"ticker": ticker, "mkt": _get_mkt(ticker),
                  **sig, "hold_days": HOLD_DAYS}
        if with_backtest:
            result.update(_run_backtest(df))
        return result
    except Exception:
        return None


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    """Run combo (PP + EMA ribbon) scan across universe; return list of result dicts."""
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_analyze, t, with_backtest): t for t in universe}
        for f in as_completed(futs, timeout=180):
            try:
                r = f.result(timeout=30)
            except Exception:
                continue
            if r:
                r["strategy"] = "combo"
                results.append(r)
    results.sort(key=lambda x: (-(x.get("wr") or 0), -x["score"]))
    return results


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    from momentum_scanner import build_universe, compute_bench_returns
    import time
    wb  = "--no-backtest" not in sys.argv
    uni = build_universe()
    bench = compute_bench_returns(set(uni.values()))
    t0  = time.time()
    res = scan(uni, bench, wb)
    print(f"\nCombo Scanner (PP + EMA Ribbon) — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} score={r['score']}  m={r['minervini']}  "
              f"rsi={r['rsi']}  adx={r['adx']}  vol_ratio={r['vol_ratio']}  "
              f"fresh={r['fresh']}  conf={r['conf']}")


if __name__ == "__main__":
    main()
