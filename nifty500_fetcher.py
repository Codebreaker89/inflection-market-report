#!/usr/bin/env python3
"""
Nifty 500 Universe Fetcher
──────────────────────────
Returns a list of Yahoo Finance tickers (with .NS suffix) for the Nifty 500.

Usage:
    from nifty500_fetcher import get_nifty500, INDIA_BENCH
    tickers = get_nifty500()   # e.g. ['RELIANCE.NS', 'TCS.NS', ...]
"""

import io, sys, os
import requests
import pandas as pd

INDIA_BENCH = "^NSEI"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nifty500-fetcher/1.0)"}

# ── ANSI helpers (mirror pocket_pivot_scanner.py) ─────────────────────────────
_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _color else text
DIM = lambda t: _c("2", t)

# ── Fallback hardcoded top-50 Nifty stocks ────────────────────────────────────
_TOP50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "SUNPHARMA", "WIPRO", "ULTRACEMCO", "NESTLEIND", "TECHM",
    "POWERGRID", "NTPC", "ONGC", "BAJFINANCE", "BAJAJFINSV",
    "ADANIENT", "ADANIPORTS", "TATASTEEL", "TATAMOTORS", "JSWSTEEL",
    "HINDALCO", "COALINDIA", "GRASIM", "DIVISLAB", "CIPLA",
    "DRREDDY", "APOLLOHOSP", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO",
    "HCLTECH", "BRITANNIA", "INDUSINDBK", "M&M", "SBILIFE",
    "HDFCLIFE", "VEDL", "PIDILITIND", "DABUR", "MUTHOOTFIN",
]


def _clean_ns(symbols: list) -> list[str]:
    """Normalise raw symbol strings and append .NS suffix."""
    out = []
    seen = set()
    for s in symbols:
        if not isinstance(s, str):
            continue
        s = s.strip().upper()
        if not s or s == "-" or " " in s:
            continue
        ticker = s if s.endswith(".NS") else s + ".NS"
        if len(ticker) > 15:
            continue
        if ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


def _from_nse_csv() -> list[str]:
    """Primary: NSE official Nifty 500 constituent CSV."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return _clean_ns(df["Symbol"].dropna().tolist())


def _from_inda_etf() -> list[str]:
    """Fallback 1: iShares INDA ETF (India large-cap) holdings CSV from BlackRock."""
    url = (
        "https://www.ishares.com/us/products/239659/ishares-msci-india-etf/"
        "1467271812596.ajax?fileType=csv&fileName=INDA_holdings&dataType=fund"
    )
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), skiprows=9)
    # Ticker column contains Bloomberg tickers like "RELIANCE IN" — strip suffix
    raw = []
    for t in df["Ticker"].dropna():
        if not isinstance(t, str):
            continue
        parts = t.strip().split()
        raw.append(parts[0])
    return _clean_ns(raw)


def _from_hardcoded() -> list[str]:
    """Fallback 2: hardcoded top-50 Nifty stocks."""
    return _clean_ns(_TOP50)


def get_nifty500() -> list[str]:
    """
    Return Nifty 500 tickers with .NS suffix for Yahoo Finance.

    Attempts sources in order:
      1. NSE India official CSV
      2. iShares INDA ETF holdings (BlackRock)
      3. Hardcoded top-50 fallback
    """
    sources = [
        ("NSE CSV",       _from_nse_csv),
        ("iShares INDA",  _from_inda_etf),
        ("hardcoded top-50", _from_hardcoded),
    ]
    for name, fn in sources:
        try:
            tickers = fn()
            if tickers:
                print(DIM(f"  Nifty universe: {len(tickers)} tickers loaded via {name}"), flush=True)
                return tickers
        except Exception as exc:
            print(DIM(f"  [{name}] failed: {exc}"), flush=True)

    # Should never reach here, but be safe
    return []


if __name__ == "__main__":
    t = get_nifty500()
    print(f"Total: {len(t)}  First 5: {t[:5]}  Bench: {INDIA_BENCH}")
