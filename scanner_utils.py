"""
scanner_utils.py — shared technical indicator utilities for all *_scanner.py files.

Every scanner imports the functions it needs:
    from scanner_utils import _quiet, _sma, _ema, _rsi, _adx
    from scanner_utils import _quiet, _sma, _ema, _rsi, _adx, _fetch_html  # if it fetches Wikipedia

Do NOT add scanner-specific logic here. Universe loaders (get_universe, get_sp500_with_sectors, etc.)
stay in each scanner because they have scanner-specific variants.
"""

import os, sys, io, contextlib
import numpy  as np
import pandas as pd
import requests

# Generic User-Agent for Wikipedia HTML fetches (used by _fetch_html only).
# Each scanner keeps its own HEADERS constant for its other requests.get() calls.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; swing-scanner/1.0)"}


@contextlib.contextmanager
def _quiet():
    """Redirect stderr to /dev/null — suppresses yfinance 'Failed download' noise."""
    devnull = open(os.devnull, "w")
    old_err = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_err
        devnull.close()


# ── Technical indicators ───────────────────────────────────────────────────────

def _sma(s, n):
    return s.rolling(n).mean()


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def _adx(high, low, close, n=14):
    tr  = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    up  = (high - high.shift()).clip(lower=0)
    dn  = (low.shift() - low).clip(lower=0)
    dmp = up.where(up > dn, 0).ewm(alpha=1/n, adjust=False).mean()
    dmm = dn.where(dn > up, 0).ewm(alpha=1/n, adjust=False).mean()
    dip = 100 * dmp / atr
    dim = 100 * dmm / atr
    dx  = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


# ── HTML fetch (Wikipedia universe loaders) ───────────────────────────────────

def _fetch_html(url):
    """Fetch URL and return list of DataFrames (pd.read_html). Used by universe loaders."""
    r = requests.get(url, headers=_HEADERS, timeout=15)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))
