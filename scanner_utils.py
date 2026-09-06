"""
scanner_utils.py — shared technical indicator utilities for all *_scanner.py files.

Every scanner imports the functions it needs:
    from scanner_utils import _quiet, _sma, _ema, _rsi, _adx
    from scanner_utils import _quiet, _sma, _ema, _rsi, _adx, _fetch_html  # if it fetches Wikipedia

Do NOT add scanner-specific logic here. Universe loaders (get_universe, get_sp500_with_sectors, etc.)
stay in each scanner because they have scanner-specific variants.
"""

import os, sys, io, contextlib, time
import numpy  as np
import pandas as pd
import requests
import yfinance as yf

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


# ── Shared price cache ──────────────────────────────────────────────────────
#
# scan.py runs 33 scanner files, each with its own analyze_ticker() that calls
# `yf.download(ticker, period=f"{LOOKBACK_DAYS}d", interval="1d", ...)`
# independently. With ~90 tickers surviving the pre-filters, that's up to
# 33 x 90 ≈ 3,000 individual network calls in one scan run — measured at
# 164s in a real CI run, almost entirely network wait, not analysis time.
# LOOKBACK_DAYS clusters at 400-420 across scanners, so one generous fetch
# per ticker (500 calendar days) covers every scanner that would otherwise
# have re-fetched it.
#
# Implemented as a monkeypatch on yfinance.download rather than a parameter
# threaded through 33 files' function signatures — every scanner already does
# `import yfinance as yf`, so all of them read the same module-level
# `download` attribute at call time. This intercepts exactly the call shape
# scanners use (single ticker, period="Nd", interval="1d", no start/end) and
# leaves every other call pattern (multi-ticker batch fetches in
# update_scan_history.py, RRG engine, sector pulse, FX rates — all of which
# use different argument shapes) completely untouched, going straight to the
# real yf.download unmodified.
#
# Extra cached history beyond what a given scanner asked for is safe: every
# scanner's indicators are rolling-window calculations read off the LAST row
# (`iloc[-1]` / `len(df)-1`) — more history further back doesn't change what
# a rolling mean/RSI/ADX reads at the current row, it only adds warm-up
# context. The one caveat: a backtest loop that walks every historical bar
# will see more sample rows than before, which can only add data points, not
# change the ones it already had — not a correctness risk, just occasionally
# a slightly larger backtest sample than pre-cache.
_PRICE_CACHE_MAX_DAYS = 500
_price_cache: dict = {}          # ticker -> DataFrame (or None for a failed fetch)
_price_cache_stats = {"unique_fetches": 0, "hits": 0, "passthrough": 0}
_real_yf_download = yf.download   # captured once, before any monkeypatch


def _cached_yf_download(tickers, *args, **kwargs):
    """Drop-in replacement for yf.download that dedupes the scanner call shape.

    Falls through to the real yf.download, unmodified, for anything that isn't
    exactly `yf.download("TICKER", period="Nd", interval="1d")` with no other
    positional/keyword args — that's the one shape all 33 scanners use, and
    the only one this cache understands.
    """
    period   = kwargs.get("period")
    interval = kwargs.get("interval", "1d")
    is_scanner_shape = (
        isinstance(tickers, str)
        and not args
        and isinstance(period, str) and period.endswith("d") and period[:-1].isdigit()
        and interval == "1d"
        and "start" not in kwargs and "end" not in kwargs
        and int(period[:-1]) <= _PRICE_CACHE_MAX_DAYS
    )
    if not is_scanner_shape:
        _price_cache_stats["passthrough"] += 1
        return _real_yf_download(tickers, *args, **kwargs)

    ticker = tickers
    if ticker in _price_cache:
        _price_cache_stats["hits"] += 1
        cached = _price_cache[ticker]
        return cached.copy() if cached is not None else cached

    _price_cache_stats["unique_fetches"] += 1
    fetch_kwargs = dict(kwargs)
    fetch_kwargs["period"] = f"{_PRICE_CACHE_MAX_DAYS}d"
    try:
        df = _real_yf_download(ticker, **fetch_kwargs)
    except Exception:
        df = None
    _price_cache[ticker] = df
    return df.copy() if df is not None else df


def install_price_cache():
    """Activate the shared price cache for this process. Call once, before any
    scanner runs. Idempotent — safe to call more than once (e.g. re-running
    main() in a REPL); re-installing just repoints yf.download at the same
    wrapper and does not reset already-cached tickers."""
    yf.download = _cached_yf_download


def reset_price_cache():
    """Clear cached prices and stats. Not called automatically — a fresh
    process (the normal case: one `python3 scan.py` invocation) never needs
    this, but it keeps re-runs in a long-lived process (tests, a REPL)
    honest rather than serving stale data silently."""
    _price_cache.clear()
    for k in _price_cache_stats:
        _price_cache_stats[k] = 0


def price_cache_report() -> str:
    """One-line summary of how much duplicate fetching the cache avoided."""
    s = _price_cache_stats
    return (f"price cache: {s['unique_fetches']} ticker(s) fetched once, "
            f"{s['hits']} redundant fetch(es) avoided, "
            f"{s['passthrough']} call(s) passed through unmodified")
