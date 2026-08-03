#!/usr/bin/env python3
"""
company_cache.py — Persistent ticker → company name lookup.

Stores names in company_names.csv (committed to repo).
Once a name is fetched it is never re-fetched, making
notify.py and scan.py reliable on GitHub Actions.

Usage:
    from company_cache import get_names, update_cache

    names = get_names(["AAPL", "INFY.NS", "RELIANCE.NS"])
    # {"AAPL": "Apple Inc.", "INFY.NS": "Infosys Limited", ...}
"""

import csv, contextlib, os, sys, warnings, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

HERE       = Path(__file__).parent
CACHE_FILE = HERE / "company_names.csv"

@contextlib.contextmanager
def _quiet():
    devnull = open(os.devnull, "w"); old = sys.stderr; sys.stderr = devnull
    try:    yield
    finally: sys.stderr = old; devnull.close()


def _load() -> dict:
    """Load cache from disk → {ticker: name}."""
    if not CACHE_FILE.exists():
        return {}
    with open(CACHE_FILE, newline="", encoding="utf-8") as f:
        return {row["ticker"]: row["name"] for row in csv.DictReader(f) if row.get("name")}


def _save(cache: dict):
    """Persist cache to disk (sorted)."""
    with open(CACHE_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "name"])
        w.writeheader()
        for ticker in sorted(cache):
            w.writerow({"ticker": ticker, "name": cache[ticker]})


def _fetch_one(ticker: str) -> tuple:
    """Fetch company name from yfinance. Returns (ticker, name)."""
    try:
        import yfinance as yf
        with _quiet():
            info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ""
        if name:
            return ticker, name
        # Fallback: fast_info
        with _quiet():
            fi = yf.Ticker(ticker).fast_info
        name = getattr(fi, "longName", "") or getattr(fi, "shortName", "") or ""
        return ticker, name or ticker
    except Exception:
        return ticker, ticker


def get_names(tickers: list, max_workers: int = 15) -> dict:
    """
    Return {ticker: company_name} for all tickers.
    Uses disk cache first; fetches only unknowns from yfinance.
    Writes any new names back to disk automatically.
    """
    cache   = _load()
    missing = [t for t in tickers if t not in cache]

    if missing:
        fetched = {}
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_fetch_one, t): t for t in missing}
                for fut in as_completed(futures):
                    t, name = fut.result()
                    fetched[t] = name
        except Exception:
            for t in missing:
                fetched[t] = t

        cache.update(fetched)
        try:
            _save(cache)
        except Exception:
            pass   # read-only FS in CI — cache still works in-memory for this run

    return {t: cache.get(t, t) for t in tickers}


def update_cache(tickers: list, max_workers: int = 15):
    """Fetch and persist names for a list of tickers (called from scan.py)."""
    get_names(tickers, max_workers=max_workers)
