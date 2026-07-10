#!/usr/bin/env python3
"""
Analyst Upgrade Cluster Scanner
────────────────────────────────
Fires when ≥3 distinct analyst firms upgrade a stock to Buy-equivalent within
5 trading days, with at least 1 from a tier-1 institution. Signal: coordinated
re-rating not driven by a single analyst's model refresh.

Filters:
  - No large gap-up (>5%) in last 3 days — avoids buying earnings pile-ons
  - <75% existing buy coverage — room for further re-rating
  - Minervini ≥4 — some trend structure

Uses yfinance Ticker.recommendations (firm-level history) + recommendations_summary.

python3 analyst_upgrade_scanner.py --no-backtest
python3 analyst_upgrade_scanner.py
"""

import os, sys, warnings, logging, contextlib
from datetime import datetime, timedelta
import numpy  as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from scanner_utils import _quiet, _sma

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

HOLD_DAYS       = 7
MAX_WORKERS     = 20   # lower — recommendations API is heavier
LOOKBACK_DAYS   = 5    # trading days window to count upgrades
MIN_UPGRADES    = 3    # distinct firms must upgrade
GAP_UP_LIMIT    = 0.05 # reject if stock gapped >5% in last 3 days

BUY_GRADES = {
    "buy", "strong buy", "overweight", "outperform",
    "accumulate", "add", "long-term buy", "positive",
    "sector outperform", "market outperform",
}

TIER1_KEYWORDS = [
    "goldman", "morgan stanley", "jpmorgan", "j.p. morgan", "jp morgan",
    "bank of america", "bofa", "merrill", "citigroup", "citi",
    "barclays", "ubs", "wells fargo", "deutsche bank",
    "rbc", "jefferies", "raymond james", "piper sandler",
    "credit suisse", "hsbc", "nomura",
]

def _is_buy_grade(grade: str) -> bool:
    return str(grade).lower().strip() in BUY_GRADES

def _is_tier1(firm: str) -> bool:
    f = str(firm).lower()
    return any(kw in f for kw in TIER1_KEYWORDS)

def _minervini(df: pd.DataFrame, idx: int) -> int:
    row = df.iloc[idx]; c = float(row["Close"])
    return sum([
        c > row["sma150"], c > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        c > row["sma50"],
        c >= 1.30 * row["52w_low"],
        c >= 0.75 * row["52w_high"],
        row["sma200"] > df.iloc[idx - 20]["sma200"],
    ])

def _build_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    df["sma50"]  = _sma(c, 50); df["sma150"] = _sma(c, 150); df["sma200"] = _sma(c, 200)
    df["52w_high"] = c.rolling(252).max(); df["52w_low"] = c.rolling(252).min()
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    return df

def _get_upgrades(ticker: str) -> tuple:
    """Returns (upgrade_count, has_tier1, firms_list). Uses yfinance recommendations."""
    try:
        t = yf.Ticker(ticker)
        with _quiet():
            rec = t.recommendations
        if rec is None or rec.empty:
            return 0, False, []
        # Normalise index to tz-naive date
        if hasattr(rec.index, 'tz') and rec.index.tz is not None:
            rec.index = rec.index.tz_localize(None)
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=8))  # ~5 trading days buffer
        recent = rec[rec.index >= cutoff].copy()
        if recent.empty:
            return 0, False, []
        # Filter: action == upgrade or initiation, to a buy-equivalent grade
        upgrades = recent[
            recent.get("Action", pd.Series(dtype=str)).str.lower().isin(["up", "init"])
            & recent.get("To Grade", pd.Series(dtype=str)).apply(_is_buy_grade)
        ]
        if upgrades.empty:
            return 0, False, []
        firms = upgrades.get("Firm", upgrades.get("firm", pd.Series(dtype=str))).tolist()
        distinct_firms = list(dict.fromkeys(str(f) for f in firms if f))  # dedup order-preserving
        has_tier1 = any(_is_tier1(f) for f in distinct_firms)
        return len(distinct_firms), has_tier1, distinct_firms
    except Exception:
        return 0, False, []

def _get_buy_pct(ticker: str) -> Optional[float]:
    """Fraction of analysts currently rating as buy-equivalent. Returns None on failure."""
    try:
        t = yf.Ticker(ticker)
        with _quiet():
            summary = t.recommendations_summary
        if summary is None or summary.empty:
            return None
        row = summary.iloc[0]
        strong_buy = int(row.get("strongBuy", 0) or 0)
        buy        = int(row.get("buy",       0) or 0)
        hold       = int(row.get("hold",      0) or 0)
        sell       = int(row.get("sell",      0) or 0)
        strong_sell= int(row.get("strongSell",0) or 0)
        total = strong_buy + buy + hold + sell + strong_sell
        if total == 0: return None
        return (strong_buy + buy) / total
    except Exception:
        return None

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
        df = _build_ohlcv(raw.copy())

        c       = float(df.iloc[-1]["Close"])
        vol_ma  = float(df.iloc[-1]["vol_ma20"]) if not pd.isna(df.iloc[-1]["vol_ma20"]) else 0
        if c < 1.0 or vol_ma < 100_000: return None

        m = _minervini(df, len(df) - 1)
        if m < 4: return None

        # Gap-up guard: reject if stock surged >5% in last 3 days (earnings pile-on)
        if len(df) >= 4:
            price_3d_ago = float(df.iloc[-4]["Close"])
            if price_3d_ago > 0 and (c - price_3d_ago) / price_3d_ago > GAP_UP_LIMIT:
                return None

        # Fetch recommendations (extra API call per ticker)
        n_upgrades, has_tier1, firms = _get_upgrades(ticker)
        if n_upgrades < MIN_UPGRADES: return None
        if not has_tier1: return None   # must have at least one tier-1

        # Buy coverage saturation check
        buy_pct = _get_buy_pct(ticker)
        if buy_pct is not None and buy_pct > 0.75: return None

        vol_ratio = float(df.iloc[-1]["Volume"]) / vol_ma if vol_ma > 0 else 0

        # Score: upgrades + tier1 presence + structural quality
        score = min(n_upgrades, 6) + (2 if has_tier1 else 0) + max(0, m - 4)

        # mkt tag
        mkt = "US"
        for sfx, mk in {".L":"UK",".DE":"DE",".PA":"FR",".AS":"NL",".TO":"CA"}.items():
            if ticker.endswith(sfx): mkt = mk; break

        firms_str = ", ".join(firms[:5])
        return {
            "ticker":      ticker,
            "mkt":         mkt,
            "price":       round(c, 2),
            "score":       score,
            "minervini":   m,
            "rsi":         0.0,   # not computed — not core signal
            "adx":         0.0,
            "vol_ratio":   round(vol_ratio, 2),
            "fresh":       [f"UPGRADES×{n_upgrades}"],
            "conf":        [firms_str[:40]],
            "n_upgrades":  n_upgrades,
            "has_tier1":   has_tier1,
            "hold_days":   HOLD_DAYS,
        }
    except Exception:
        return None

def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
                for t, b in universe.items()}
        for f in as_completed(futs, timeout=240):
            try:
                r = f.result(timeout=45)
                if r:
                    r["strategy"] = "analyst_upgrade"
                    results.append(r)
            except Exception:
                pass
    results.sort(key=lambda x: -x["score"])
    return results

def main():
    from momentum_scanner import build_universe, compute_bench_returns
    import time
    wb  = "--no-backtest" not in sys.argv
    uni = build_universe()
    bench = compute_bench_returns(set(uni.values()))
    t0 = time.time()
    res = scan(uni, bench, wb)
    print(f"\nAnalyst Upgrade Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} upgrades={r['n_upgrades']}  tier1={r['has_tier1']}  "
              f"m={r['minervini']}  score={r['score']}  firms: {r['conf']}")

if __name__ == "__main__":
    main()
