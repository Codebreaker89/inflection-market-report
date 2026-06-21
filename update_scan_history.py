#!/usr/bin/env python3
"""
update_scan_history.py
──────────────────────
1. Reads last_scan.json (produced by scan.py)
2. Appends new rows to scan_history.csv
3. Backfills ret_d5 / ret_d10 / spy_ret_d5 / spy_ret_d10 / excess_ret_d5/d10
   / hit_stop_loss_d5/d10 / max_drawdown_d10 for rows where enough time has passed

Run: python3 update_scan_history.py
"""

import csv, json, math, warnings, logging, contextlib, os, sys
from pathlib     import Path
from datetime    import datetime, date, timedelta
from typing      import Optional

import yfinance as yf
import pandas   as pd

HERE     = Path(__file__).parent
HISTORY  = HERE / "scan_history.csv"
SCAN_JSON= HERE / "last_scan.json"

STOP_LOSS_PCT = 0.03   # 3% — must match config
D5, D10       = 5, 10  # calendar days

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

@contextlib.contextmanager
def _quiet():
    devnull = open(os.devnull,"w"); old=sys.stderr; sys.stderr=devnull
    try:    yield
    finally: sys.stderr=old; devnull.close()

# ── Schema ────────────────────────────────────────────────────────────────────
FIELDNAMES = [
    "scan_date","ticker","company","strategy","strategies_count",
    "price_at_scan","score","wr","avg","adx","rsi","vol_ratio",
    # filled later
    "price_d5","ret_d5","spy_ret_d5","excess_ret_d5","hit_stop_loss_d5",
    "price_d10","ret_d10","spy_ret_d10","excess_ret_d10","hit_stop_loss_d10",
    "max_drawdown_d10",
]

_EMPTY = {f: "" for f in FIELDNAMES}

# ── Price helpers ─────────────────────────────────────────────────────────────
_px_cache: dict = {}

def _fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    key = (ticker, start, end)
    if key in _px_cache:
        return _px_cache[key]
    try:
        with _quiet():
            df = yf.Ticker(ticker).history(
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=5)).strftime("%Y-%m-%d"),
                interval="1d", auto_adjust=True
            )
        df.index = pd.to_datetime(df.index).date
        _px_cache[key] = df
        return df
    except Exception:
        return pd.DataFrame()

def _price_on_or_after(ticker: str, target: date, scan_date: date) -> Optional[float]:
    """Closest closing price on or after target date (up to +5 trading days)."""
    df = _fetch_history(ticker, scan_date, target + timedelta(days=14))
    if df.empty: return None
    future = df[df.index >= target]
    if future.empty: return None
    return float(future["Close"].iloc[0])

def _min_price_between(ticker: str, start: date, end: date) -> Optional[float]:
    """Lowest closing price in [start, end] for drawdown calc."""
    df = _fetch_history(ticker, start, end + timedelta(days=5))
    if df.empty: return None
    window = df[(df.index >= start) & (df.index <= end)]
    if window.empty: return None
    return float(window["Low"].min())

def _spy_return(scan_date: date, target: date) -> Optional[float]:
    df = _fetch_history("SPY", scan_date, target + timedelta(days=14))
    if df.empty: return None
    base = df[df.index >= scan_date]
    tgt  = df[df.index >= target]
    if base.empty or tgt.empty: return None
    p0 = float(base["Close"].iloc[0])
    p1 = float(tgt["Close"].iloc[0])
    return round((p1/p0 - 1)*100, 2) if p0 else None

def _safe(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return ""
    return v

# ── Load / save CSV ───────────────────────────────────────────────────────────
def load_history() -> list[dict]:
    if not HISTORY.exists(): return []
    with open(HISTORY, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_history(rows: list[dict]):
    with open(HISTORY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

# ── Append new scan rows ──────────────────────────────────────────────────────
def append_new_rows(rows: list[dict]) -> list[dict]:
    if not SCAN_JSON.exists():
        print("  last_scan.json not found — run scan.py first"); return rows

    data = json.loads(SCAN_JSON.read_text())
    scan_date = data.get("scan_date", date.today().isoformat())
    rbs       = data.get("results_by_strategy", {})

    # Existing (scan_date, ticker, strategy) combos — no duplicates
    existing = {(r["scan_date"], r["ticker"], r["strategy"]) for r in rows}

    # Cross-strategy count per ticker
    ticker_strats: dict[str, set] = {}
    for strat, results in rbs.items():
        for r in results:
            ticker_strats.setdefault(r["ticker"], set()).add(strat)

    added = 0
    for strat, results in rbs.items():
        for r in results:
            key = (scan_date, r["ticker"], strat)
            if key in existing: continue
            row = {**_EMPTY}
            row["scan_date"]        = scan_date
            row["ticker"]           = r.get("ticker","")
            row["company"]          = r.get("company","")
            row["strategy"]         = strat
            row["strategies_count"] = len(ticker_strats.get(r["ticker"], set()))
            row["price_at_scan"]    = _safe(r.get("price"))
            row["score"]            = _safe(r.get("score"))
            row["wr"]               = _safe(r.get("wr"))
            row["avg"]              = _safe(r.get("avg"))
            row["adx"]              = _safe(r.get("adx"))
            row["rsi"]              = _safe(r.get("rsi"))
            row["vol_ratio"]        = _safe(r.get("vol_ratio"))
            rows.append(row)
            added += 1

    print(f"  Added {added} new row(s) from {scan_date}")
    return rows

# ── Backfill returns ──────────────────────────────────────────────────────────
def backfill_returns(rows: list[dict]) -> list[dict]:
    today = date.today()
    to_fill_d5  = [r for r in rows if not r.get("ret_d5")  and r.get("scan_date")
                   and (today - date.fromisoformat(r["scan_date"])).days >= D5]
    to_fill_d10 = [r for r in rows if not r.get("ret_d10") and r.get("scan_date")
                   and (today - date.fromisoformat(r["scan_date"])).days >= D10]

    if not to_fill_d5 and not to_fill_d10:
        print("  Nothing to backfill yet"); return rows

    print(f"  Backfilling: {len(to_fill_d5)} d5 rows, {len(to_fill_d10)} d10 rows")

    # d5
    for r in to_fill_d5:
        sd      = date.fromisoformat(r["scan_date"])
        target  = sd + timedelta(days=D5)
        ticker  = r["ticker"]
        try:
            p0 = float(r["price_at_scan"])
            p5 = _price_on_or_after(ticker, target, sd)
            if p5 and p0:
                ret5  = round((p5/p0 - 1)*100, 2)
                spy5  = _spy_return(sd, target)
                ex5   = round(ret5 - spy5, 2) if spy5 is not None else ""
                sl5   = 1 if p5 <= p0*(1-STOP_LOSS_PCT) else 0
                r["price_d5"]          = round(p5, 4)
                r["ret_d5"]            = ret5
                r["spy_ret_d5"]        = _safe(spy5)
                r["excess_ret_d5"]     = ex5
                r["hit_stop_loss_d5"]  = sl5
        except Exception:
            pass

    # d10
    for r in to_fill_d10:
        sd      = date.fromisoformat(r["scan_date"])
        target  = sd + timedelta(days=D10)
        ticker  = r["ticker"]
        try:
            p0  = float(r["price_at_scan"])
            p10 = _price_on_or_after(ticker, target, sd)
            pmin= _min_price_between(ticker, sd, target)
            if p10 and p0:
                ret10  = round((p10/p0 - 1)*100, 2)
                spy10  = _spy_return(sd, target)
                ex10   = round(ret10 - spy10, 2) if spy10 is not None else ""
                sl10   = 1 if (pmin is not None and pmin <= p0*(1-STOP_LOSS_PCT)) else 0
                mdd    = round((pmin/p0 - 1)*100, 2) if pmin and p0 else ""
                r["price_d10"]         = round(p10, 4)
                r["ret_d10"]           = ret10
                r["spy_ret_d10"]       = _safe(spy10)
                r["excess_ret_d10"]    = ex10
                r["hit_stop_loss_d10"] = sl10
                r["max_drawdown_d10"]  = mdd
        except Exception:
            pass

    return rows

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n  Updating scan history...")
    rows = load_history()
    rows = append_new_rows(rows)
    rows = backfill_returns(rows)
    save_history(rows)
    print(f"  scan_history.csv → {len(rows)} total rows")

if __name__ == "__main__":
    main()
