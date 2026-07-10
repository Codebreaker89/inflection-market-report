#!/usr/bin/env python3
"""
Chokepoint Inflection Scanner
──────────────────────────────────────────────────────────────────────────────
Macro chokepoint → commodity/sector inflection → correlated stock lag detector.

Logic:
  1. Commodity/cyclical basket inflects (5d return > threshold AND accelerating)
  2. yfinance news on that commodity confirms macro trigger keyword
  3. Find stocks in universe with 60d rolling correlation > 0.55 to commodity
  4. Filter: stock 5d return < commodity 5d return × 0.15  (stock hasn't moved yet)
  5. Apply Minervini ≥ 4 + price > SMA50 hard filters
  6. Score by: correlation strength × lag gap × volume

Commodity basket (Tier 1):
  CL=F   Crude oil            → energy producers, refiners
  NG=F   Natural gas          → LNG, utilities, fertiliser
  HG=F   Copper               → miners, EV makers, grid infra
  GC=F   Gold                 → gold miners, royalty cos
  ZW=F   Wheat                → grain processors, ag equipment
  ALI=F  Aluminum             → smelters, pure-play mfrs
  URA    Uranium ETF          → nuclear operators, U miners
  XME    Metals & Mining ETF  → broad metals basket
  SOXX   Semiconductors ETF   → fabs, EDA, equipment
  REMX   Rare Earth ETF       → rare earth miners, magnets
  BDRY   Dry Bulk Shipping    → shipping lines, port operators

python3 chokepoint_inflection_scanner.py --no-backtest
python3 chokepoint_inflection_scanner.py
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

HOLD_DAYS    = 5
MAX_WORKERS  = 25
FRESH_WINDOW = 2   # signal valid if fired in last 2 days

# ── Commodity basket ──────────────────────────────────────────────────────────
COMMODITY_BASKET = {
    "CL=F":  {"name": "Crude Oil",          "keywords": ["oil", "opec", "hormuz", "iraq", "iran", "venezuela", "embargo", "pipeline", "brent", "wti", "energy", "war", "sanctions"]},
    "NG=F":  {"name": "Natural Gas",         "keywords": ["gas", "lng", "pipeline", "storage", "winter", "heating", "europe", "russia", "supply", "freeze"]},
    "HG=F":  {"name": "Copper",              "keywords": ["copper", "chile", "peru", "mine", "strike", "supply", "ev", "grid", "shortage", "china"]},
    "GC=F":  {"name": "Gold",                "keywords": ["gold", "inflation", "fed", "rate", "war", "safe haven", "dollar", "central bank"]},
    "ZW=F":  {"name": "Wheat",               "keywords": ["wheat", "grain", "black sea", "ukraine", "russia", "drought", "crop", "food", "supply"]},
    "ALI=F": {"name": "Aluminum",            "keywords": ["aluminum", "aluminium", "smelter", "energy", "russia", "rusal", "bauxite", "supply"]},
    "URA":   {"name": "Uranium",             "keywords": ["uranium", "nuclear", "reactor", "kazakh", "cameco", "enrichment", "fuel", "energy"]},
    "XME":   {"name": "Metals & Mining",     "keywords": ["metal", "mining", "commodity", "supply", "tariff", "china", "iron", "ore", "steel"]},
    "SOXX":  {"name": "Semiconductors",      "keywords": ["semiconductor", "chip", "tsmc", "taiwan", "asml", "export", "restriction", "china", "shortage", "wafer", "fab"]},
    "REMX":  {"name": "Rare Earths",         "keywords": ["rare earth", "lithium", "cobalt", "nickel", "magnet", "china", "export", "ban", "critical", "mineral"]},
    "BDRY":  {"name": "Dry Bulk Shipping",   "keywords": ["shipping", "freight", "suez", "panama", "canal", "port", "vessel", "congestion", "baltic", "container"]},
}

# Thresholds
COMMODITY_5D_THRESH  = 0.04    # commodity must be up ≥ 4% in 5 days
CORR_MIN             = 0.55    # 60d rolling correlation minimum
LAG_RATIO_MAX        = 0.15    # stock 5d return < commodity 5d return × this
MINERVINI_MIN        = 4
NEWS_LOOKBACK_DAYS   = 5       # scan news from last N days
CORR_LOOKBACK        = 60      # days for rolling correlation
ACCEL_DAYS           = 3       # confirm acceleration over N days

# ── Helpers ───────────────────────────────────────────────────────────────────

def _minervini(row) -> int:
    c = float(row["Close"])
    m = sum([
        c > row["sma150"], c > row["sma200"],
        row["sma150"] > row["sma200"],
        row["sma50"]  > row["sma150"],
        c > row["sma50"],
        c >= 1.30 * row["52w_low"],
        c >= 0.75 * row["52w_high"],
        row["sma200"] > row["sma200_20d_ago"],
    ])
    return m

# ── Step 1: Fetch commodity prices + detect inflection ───────────────────────

def _fetch_commodity(ticker: str) -> Optional[pd.Series]:
    """Returns Close series for the commodity, or None on failure."""
    try:
        with _quiet():
            raw = yf.download(ticker, period="120d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 20: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        return raw["Close"].dropna()
    except Exception:
        return None

def _commodity_inflected(prices: pd.Series) -> Optional[dict]:
    """
    Returns inflection info dict if commodity qualifies, else None.
    Checks:
      - 5d return > COMMODITY_5D_THRESH
      - Accelerating: today's 5d ROC > 3-day-ago 5d ROC
    """
    if len(prices) < 10: return None
    p_now   = float(prices.iloc[-1])
    p_5d    = float(prices.iloc[-6])  if len(prices) >= 6  else None
    p_5d_3a = float(prices.iloc[-9])  if len(prices) >= 9  else None

    if p_5d is None or p_5d_3a is None or p_5d <= 0 or p_5d_3a <= 0:
        return None

    ret_5d      = (p_now - p_5d) / p_5d
    ret_5d_3ago = (prices.iloc[-4] - p_5d_3a) / p_5d_3a if len(prices) >= 9 else 0.0

    if ret_5d < COMMODITY_5D_THRESH: return None
    # Acceleration: today's 5d ROC > 3-day-ago 5d ROC
    if float(ret_5d) <= float(ret_5d_3ago): return None

    return {
        "ret_5d":    round(ret_5d * 100, 2),
        "accel":     round((ret_5d - ret_5d_3ago) * 100, 2),
        "price_now": round(p_now, 4),
    }

# ── Step 2: News keyword confirmation ─────────────────────────────────────────

def _news_confirmed(commodity_ticker: str, keywords: list) -> bool:
    """
    Scan yfinance news headlines for the commodity ticker.
    Returns True if any headline contains a macro keyword.
    """
    try:
        t = yf.Ticker(commodity_ticker)
        news = t.news
        if not news: return False
        from datetime import datetime, timezone, timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=NEWS_LOOKBACK_DAYS)).timestamp()
        found = False
        for item in news:
            # Accept item if within lookback window (or no timestamp — include anyway)
            ts = item.get("providerPublishTime", item.get("publish_time", None))
            if ts and ts < cutoff: continue
            title   = (item.get("title", "") or "").lower()
            summary = (item.get("summary", "") or "").lower()
            text    = title + " " + summary
            if any(kw in text for kw in keywords):
                found = True
                break
        return found
    except Exception:
        return False

# ── Step 3: Compute 60d rolling correlation for universe ─────────────────────

def _fetch_stock_prices(ticker: str) -> Optional[pd.Series]:
    try:
        with _quiet():
            raw = yf.download(ticker, period="180d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 65: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        return raw["Close"].dropna()
    except Exception:
        return None

def _rolling_correlation(stock_prices: pd.Series, commodity_prices: pd.Series) -> float:
    """60-day rolling correlation using the most recent CORR_LOOKBACK days."""
    # Align on dates
    aligned = pd.concat([stock_prices.rename("s"), commodity_prices.rename("c")], axis=1).dropna()
    if len(aligned) < CORR_LOOKBACK: return 0.0
    recent = aligned.tail(CORR_LOOKBACK)
    corr = recent["s"].corr(recent["c"])
    return float(corr) if not pd.isna(corr) else 0.0

# ── Step 4 + 5: Score a single stock against an inflected commodity ───────────

def _score_stock(ticker: str, commodity_ticker: str, commodity_info: dict,
                 commodity_prices: pd.Series) -> Optional[dict]:
    """
    Full check for one stock against one inflected commodity.
    Returns result dict or None.
    """
    try:
        with _quiet():
            raw = yf.download(ticker, period="400d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 220: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return None
        raw = raw.dropna(subset=["Close","High","Low","Volume"])
        if len(raw) < 220: return None

        df = raw.copy()
        c_ser = df["Close"]

        # Build indicators
        df["sma50"]    = _sma(c_ser, 50)
        df["sma150"]   = _sma(c_ser, 150)
        df["sma200"]   = _sma(c_ser, 200)
        df["52w_high"] = c_ser.rolling(252).max()
        df["52w_low"]  = c_ser.rolling(252).min()
        df["sma200_20d_ago"] = df["sma200"].shift(20)
        df["vol_ma20"] = df["Volume"].rolling(20).mean()
        df["rsi"]      = _rsi(c_ser, 14)
        df["adx"]      = _adx(df["High"], df["Low"], c_ser, 14)

        row = df.iloc[-1]
        c   = float(row["Close"])
        if pd.isna(c) or c < 1.0: return None
        vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
        if vol_ma < 100_000: return None

        # Hard filter: price > SMA50 (uptrending stock only)
        if pd.isna(row["sma50"]) or c <= float(row["sma50"]): return None

        # Hard filter: Minervini ≥ MINERVINI_MIN
        m = _minervini(row)
        if m < MINERVINI_MIN: return None

        # Correlation check
        stock_close = df["Close"]
        corr = _rolling_correlation(stock_close, commodity_prices)
        if corr < CORR_MIN: return None

        # Lag check: stock hasn't priced it in
        if len(df) < 6: return None
        stock_5d_ret = (c - float(df.iloc[-6]["Close"])) / float(df.iloc[-6]["Close"])
        comm_5d_ret  = commodity_info["ret_5d"] / 100.0
        lag_threshold = comm_5d_ret * LAG_RATIO_MAX
        if stock_5d_ret >= lag_threshold: return None

        # Score: corr × lag_gap (bigger gap = more upside) × vol confirmation
        lag_gap   = comm_5d_ret - stock_5d_ret
        vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
        score_raw = corr * lag_gap * 100
        score     = min(int(round(score_raw)), 10)

        conf_flags = {
            "CORR>0.7":  corr > 0.70,
            "VOL1.5x":   vol_ratio > 1.5,
            "M≥5":       m >= 5,
            "ADX>15":    float(row["adx"]) > 15 if not pd.isna(row["adx"]) else False,
        }
        conf_true = [k for k, v in conf_flags.items() if v]

        commodity_name = COMMODITY_BASKET[commodity_ticker]["name"]

        return {
            "ticker":        ticker,
            "commodity":     commodity_ticker,
            "commodity_name": commodity_name,
            "comm_ret_5d":   commodity_info["ret_5d"],
            "comm_accel":    commodity_info["accel"],
            "stock_5d_ret":  round(stock_5d_ret * 100, 2),
            "lag_gap":       round(lag_gap * 100, 2),
            "correlation":   round(corr, 3),
            "price":         round(c, 2),
            "rsi":           round(float(row["rsi"]), 1) if not pd.isna(row["rsi"]) else 0,
            "adx":           round(float(row["adx"]), 1) if not pd.isna(row["adx"]) else 0,
            "vol_ratio":     round(vol_ratio, 2),
            "minervini":     m,
            "score":         score,
            "fresh":         [f"CP:{commodity_ticker}", f"+{commodity_info['ret_5d']}%"],
            "conf":          conf_true,
            "hold_days":     HOLD_DAYS,
        }
    except Exception:
        return None


# ── Main scan entry point ─────────────────────────────────────────────────────

def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    """
    1. Check each commodity for inflection (fast — 11 tickers).
    2. News-confirm inflected commodities.
    3. For confirmed commodities, scan stock universe for correlated laggards.
    """
    results = []

    # Step 1+2: Find inflected + news-confirmed commodities
    active_commodities = {}   # ticker → {inflection_info, prices}

    for cticker, cinfo in COMMODITY_BASKET.items():
        prices = _fetch_commodity(cticker)
        if prices is None: continue
        inflection = _commodity_inflected(prices)
        if inflection is None: continue
        # News confirmation
        confirmed = _news_confirmed(cticker, cinfo["keywords"])
        if not confirmed: continue
        active_commodities[cticker] = {"info": inflection, "prices": prices}

    if not active_commodities:
        return []

    # Step 3: Scan stocks against each active commodity
    # Use ThreadPoolExecutor per commodity to avoid too many concurrent yfinance calls
    seen = set()   # deduplicate: keep highest score per ticker

    for cticker, cdata in active_commodities.items():
        comm_prices = cdata["prices"]
        comm_info   = cdata["info"]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {
                pool.submit(_score_stock, ticker, cticker, comm_info, comm_prices): ticker
                for ticker in universe
            }
            for f in as_completed(futs, timeout=180):
                try:
                    r = f.result(timeout=30)
                    if r is None: continue
                    t = r["ticker"]
                    # Keep highest score if same ticker appears for multiple commodities
                    existing = next((x for x in results if x["ticker"] == t), None)
                    if existing:
                        if r["score"] > existing["score"]:
                            results.remove(existing)
                            results.append(r)
                    else:
                        results.append(r)
                except Exception:
                    pass

    for r in results:
        r["strategy"] = "chokepoint_inflection"
        # Attach market tag
        for sfx, mkt in {".L":"UK",".DE":"DE",".PA":"FR",".AS":"NL",".TO":"CA"}.items():
            if r["ticker"].endswith(sfx):
                r["mkt"] = mkt; break
        else:
            r.setdefault("mkt", "US")

    results.sort(key=lambda x: (-x["score"], -x["lag_gap"]))
    return results


def main():
    from momentum_scanner import build_universe, compute_bench_returns
    import time
    wb = "--no-backtest" not in sys.argv
    uni = build_universe()
    bench = compute_bench_returns(set(uni.values()))
    t0 = time.time()
    res = scan(uni, bench, wb)
    elapsed = time.time() - t0
    print(f"\nChokepoint Inflection Scanner — {len(res)} signals in {elapsed:.0f}s")

    if not res:
        print("  No active commodity inflections with news confirmation today.")
        return

    for r in res[:20]:
        print(f"  {r['ticker']:<10} [{r['commodity_name']:<22}]  "
              f"comm+{r['comm_ret_5d']}%  stock{r['stock_5d_ret']:+.1f}%  "
              f"corr={r['correlation']}  score={r['score']}  m={r['minervini']}")


if __name__ == "__main__":
    main()
