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
    """Fetch company name from yfinance. Returns (ticker, name-or-EMPTY).

    Returns "" — never the ticker — when the lookup fails. A failure must not
    look like a successful result, because anything non-empty gets written to
    company_names.csv and then permanently satisfies the cache, so the name
    would never be retried. That bug left 23 of 44 cached entries stuck as bare
    ticker symbols in the digest regardless of network health.
    """
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
        return ticker, name or ""
    except Exception:
        return ticker, ""


def get_names(tickers: list, max_workers: int = 15) -> dict:
    """
    Return {ticker: company_name} for all tickers.
    Uses disk cache first; fetches only unknowns from yfinance.
    Writes any new names back to disk automatically.
    """
    cache = _load()
    # A cached value equal to the ticker is a poisoned entry from an older
    # failed fetch, not a real name — treat it as missing so it gets retried.
    missing = [t for t in tickers
               if t not in cache or cache[t].strip().upper() == t.strip().upper()]

    if missing:
        fetched = {}
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_fetch_one, t): t for t in missing}
                for fut in as_completed(futures):
                    t, name = fut.result()
                    if name:                 # only real names enter the cache
                        fetched[t] = name
        except Exception:
            pass                             # nothing fetched; retry next run

        if fetched:
            cache.update(fetched)
            try:
                _save(cache)
            except Exception:
                pass   # read-only FS in CI — cache still works in-memory for this run

    # Callers still get the ticker as a display fallback — it just isn't persisted.
    return {t: cache.get(t) or t for t in tickers}


def update_cache(tickers: list, max_workers: int = 15):
    """Fetch and persist names for a list of tickers (called from scan.py)."""
    get_names(tickers, max_workers=max_workers)


def purge_fallbacks() -> int:
    """Drop cache rows whose name is just the ticker. Returns rows removed.

    Those are poisoned entries from failed lookups. Removing them lets the next
    run with working network fetch the real name.
    """
    cache = _load()
    bad = [t for t, n in cache.items() if n.strip().upper() == t.strip().upper()]
    for t in bad:
        del cache[t]
    if bad:
        _save(cache)
    return len(bad)


if __name__ == "__main__":
    import sys
    if "--purge" in sys.argv:
        n = purge_fallbacks()
        print(f"Purged {n} ticker-fallback row(s). They will re-fetch on the next run.")
    else:
        c = _load()
        bad = [t for t, n in c.items() if n.strip().upper() == t.strip().upper()]
        print(f"{len(c)} cached name(s), {len(bad)} poisoned: {bad}")
        print("Run with --purge to remove them.")
