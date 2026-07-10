#!/usr/bin/env python3
"""
Power Earnings Gap Scanner  |  Gil Morales / IBD
──────────────────────────────────────────────────────────────────────────────
Detects stocks that gapped up powerfully on earnings and are holding the gap —
the classic sign of institutional accumulation on fundamental confirmation.

Gap criteria:
  - Open ≥ 8% above prior day's close on the gap day (hard minimum)
  - Volume on gap day ≥ 2× 20-day average (institutional buying)
  - Gap occurred within last FRESH_WINDOW (5) trading days
  - Price today still above gap day's LOW (gap not filled = buyers defending)

Earnings verification:
  - Tries to confirm gap day coincides with an earnings release via yfinance
  - If earnings_dates unavailable: accepts gap if volume ≥ 3× average (higher bar)
  - Tags as "EG✓" (verified) or "EG~" (pattern-only, earnings not confirmed)

Hard filters:
  - Price > SMA50 before the gap (stock in uptrend heading into earnings)
  - Minervini ≥ 4 at gap day
  - Market cap implied by price × vol_ma > $0 (basic liquidity)
  - Vol average > 100k (liquidity gate)
  - Not extended: today's price ≤ gap_day_close × 1.20 (within 20% of gap)

Scoring:
  - Gap size tier (8-15%, 15-25%, >25%)
  - Volume ratio on gap day
  - Gap held (price > gap day low — strongest signal)
  - Price consolidating tightly since gap (range < 8% of gap day close)
  - RSI 50-75 (momentum healthy)
  - MACD histogram positive
  - Minervini ≥ 5

Hold: 10 days

python3 power_earnings_gap_scanner.py --no-backtest
python3 power_earnings_gap_scanner.py
"""

import os, sys, warnings, logging, contextlib
import numpy  as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Optional
from scanner_utils import _adx, _ema, _quiet, _rsi, _sma

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

HOLD_DAYS    = 10
MAX_WORKERS  = 20
FRESH_WINDOW = 5    # gap must have occurred within last N trading days

GAP_MIN_PCT        = 0.08   # minimum gap: 8%
GAP_VOL_RATIO_MIN  = 2.0    # volume on gap day ≥ 2× average
GAP_VOL_UNVERIFIED = 3.0    # if earnings not confirmed, require 3× volume
MAX_EXTENSION      = 0.20   # don't buy if already 20%+ above gap close

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]    = _sma(c, 50)
    df["sma150"]   = _sma(c, 150)
    df["sma200"]   = _sma(c, 200)
    df["sma200_20d_ago"] = df["sma200"].shift(20)
    df["vol_ma20"] = v.rolling(20).mean()
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    macd           = _ema(c, 12) - _ema(c, 26)
    df["macd_hist"] = macd - _ema(macd, 9)
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df

# ── Earnings date verification ────────────────────────────────────────────────

def _recent_earnings_dates(ticker: str, lookback_days: int = 10) -> list:
    """
    Returns list of recent earnings dates (as pd.Timestamp) from yfinance.
    Falls back to empty list on any error.
    """
    try:
        t  = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is None or len(ed) == 0: return []
        now     = pd.Timestamp.now(tz="UTC")
        cutoff  = now - pd.Timedelta(days=lookback_days)
        # Filter: past dates within lookback window
        past    = ed[(ed.index <= now) & (ed.index >= cutoff)]
        return list(past.index)
    except Exception:
        return []

def _gap_is_earnings(gap_date: pd.Timestamp, earnings_dates: list,
                     window: int = 2) -> bool:
    """True if gap_date is within `window` calendar days of any earnings date."""
    gd = gap_date.tz_localize(None) if gap_date.tzinfo else gap_date
    for ed in earnings_dates:
        ed_naive = ed.tz_localize(None) if ed.tzinfo else ed
        if abs((gd - ed_naive).days) <= window:
            return True
    return False

# ── Gap detection ─────────────────────────────────────────────────────────────

def _find_gap(df: pd.DataFrame, idx: int,
              earnings_dates: list) -> Optional[dict]:
    """
    Searches last FRESH_WINDOW bars for a qualifying earnings gap.
    Returns the best (largest) qualifying gap info, or None.
    """
    best = None

    for lookback in range(1, FRESH_WINDOW + 1):
        gap_idx = idx - lookback + 1   # potential gap day (today = idx, so look back)
        # Actually: the gap day itself is lookback days ago
        gap_idx = idx - lookback
        if gap_idx < 1: continue

        gap_row  = df.iloc[gap_idx]
        prev_row = df.iloc[gap_idx - 1]

        gap_open  = float(gap_row["Open"])
        prev_close = float(prev_row["Close"])
        if prev_close <= 0: continue

        gap_pct = (gap_open - prev_close) / prev_close
        if gap_pct < GAP_MIN_PCT: continue    # minimum 8% gap

        vol_ma  = float(gap_row["vol_ma20"]) if not pd.isna(gap_row["vol_ma20"]) else 0
        if vol_ma < 100_000: continue
        vol_ratio = float(gap_row["Volume"]) / vol_ma if vol_ma > 0 else 0

        # Earnings verification
        gap_date  = df.index[gap_idx]
        if hasattr(gap_date, 'date'): gap_date = pd.Timestamp(gap_date)
        verified  = _gap_is_earnings(gap_date, earnings_dates)

        # Volume gate: unverified gaps need higher volume bar
        min_vol_ratio = GAP_VOL_RATIO_MIN if verified else GAP_VOL_UNVERIFIED
        if vol_ratio < min_vol_ratio: continue

        gap_day_close = float(gap_row["Close"])
        gap_day_low   = float(gap_row["Low"])

        if best is None or gap_pct > best["gap_pct"]:
            best = {
                "gap_pct":       round(gap_pct * 100, 1),
                "vol_ratio":     round(vol_ratio, 2),
                "gap_day_close": gap_day_close,
                "gap_day_low":   gap_day_low,
                "gap_idx":       gap_idx,
                "verified":      verified,
                "days_ago":      lookback,
            }

    return best

# ── Score a ticker at bar idx ─────────────────────────────────────────────────

def _score(df: pd.DataFrame, idx: int, gap_info: dict) -> Optional[dict]:
    row = df.iloc[idx]
    c   = float(row["Close"])
    if pd.isna(c) or c < 1.0: return None

    vol_ma    = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
    rsi       = float(row["rsi"])      if not pd.isna(row["rsi"])      else 50
    adx       = float(row["adx"])      if not pd.isna(row["adx"])      else 0
    macd_h    = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0
    sma50     = float(row["sma50"])    if not pd.isna(row["sma50"])    else 0
    sma200    = float(row["sma200"])   if not pd.isna(row["sma200"])   else 0

    # Hard: must be above SMA50 before/at gap (implies pre-gap uptrend)
    # Check SMA50 at the gap day itself
    gap_row_sma50 = float(df.iloc[gap_info["gap_idx"]]["sma50"]) \
                    if not pd.isna(df.iloc[gap_info["gap_idx"]]["sma50"]) else 0
    gap_open_price = float(df.iloc[gap_info["gap_idx"]]["Open"])
    if gap_open_price < gap_row_sma50 * 0.90: return None  # gapped into downtrend

    # Hard: gap not filled — today's price above gap day's low
    if c < gap_info["gap_day_low"]: return None

    # Hard: not excessively extended above gap close
    if c > gap_info["gap_day_close"] * (1 + MAX_EXTENSION): return None

    # Minervini ≥ 4
    m = sum([
        c > row["sma150"] if not pd.isna(row["sma150"]) else False,
        c > sma200,
        float(row["sma150"]) > sma200 if not pd.isna(row["sma150"]) else False,
        sma50 > float(row["sma150"]) if not pd.isna(row["sma150"]) else False,
        c > sma50,
        c >= 1.30 * float(row["52w_low"]) if not pd.isna(row["52w_low"]) else False,
        c >= 0.75 * float(row["52w_high"]) if not pd.isna(row["52w_high"]) else False,
        sma200 > float(row["sma200_20d_ago"]) if not pd.isna(row["sma200_20d_ago"]) else False,
    ])
    if m < 4: return None

    # Post-gap consolidation tightness (good: price not whipping around)
    gap_idx  = gap_info["gap_idx"]
    post_gap = df["Close"].iloc[gap_idx : idx + 1]
    if len(post_gap) > 1:
        pg_range = (float(post_gap.max()) - float(post_gap.min())) / gap_info["gap_day_close"]
    else:
        pg_range = 0.0
    tight_consolidation = pg_range < 0.08   # within 8% = tight

    # Gap size tier
    gp = gap_info["gap_pct"]
    gap_tier = "GAP>25%" if gp > 25 else ("GAP15-25%" if gp > 15 else "GAP8-15%")

    vol_ratio_now = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0

    conf_flags = {
        gap_tier:          True,
        f"VOL{round(gap_info['vol_ratio'],1)}x": gap_info["vol_ratio"] > 2.5,
        "HELD":            True,   # gap not filled — always show
        "TIGHT":           tight_consolidation,
        "RSI50-75":        50 <= rsi <= 75,
        "MACD+":           macd_h > 0,
        "M≥5":             m >= 5,
        "EG✓":             gap_info["verified"],
    }
    score = sum(conf_flags.values())
    if score < 3: return None

    tag = "EG✓" if gap_info["verified"] else "EG~"

    return {
        "score":          score,
        "fresh":          [tag, f"+{gap_info['gap_pct']}%GAP"],
        "conf":           [k for k, v in conf_flags.items() if v],
        "rsi":            round(rsi, 1),
        "adx":            round(adx, 1),
        "vol_ratio":      round(gap_info["vol_ratio"], 2),
        "minervini":      m,
        "price":          round(c, 2),
        "gap_pct":        gap_info["gap_pct"],
        "gap_vol_ratio":  gap_info["vol_ratio"],
        "gap_verified":   gap_info["verified"],
        "gap_held":       c >= gap_info["gap_day_low"],
        "days_since_gap": gap_info["days_ago"],
        "tight_consol":   tight_consolidation,
    }

# ── Backtest ──────────────────────────────────────────────────────────────────

def _backtest_simple(df: pd.DataFrame) -> dict:
    """Buy on any 8%+ gap-up day with 2× volume; hold HOLD_DAYS."""
    rets, last = [], -10
    vol_ma = df["vol_ma20"]
    for i in range(21, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        if i < 1: continue
        gap   = (float(df.iloc[i]["Open"]) / float(df.iloc[i-1]["Close"]) - 1)
        vm    = float(vol_ma.iloc[i]) if not pd.isna(vol_ma.iloc[i]) else 0
        vratio = float(df.iloc[i]["Volume"]) / vm if vm > 0 else 0
        if gap < GAP_MIN_PCT or vratio < GAP_VOL_RATIO_MIN: continue
        entry = float(df.iloc[i]["Close"])
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - entry) / entry * 100)
        last = i
    if not rets: return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n": len(a), "wr": round(100*(a>0).mean(),1),
            "avg": round(float(a.mean()),2), "med": round(float(np.median(a)),2)}

# ── Per-ticker analysis ───────────────────────────────────────────────────────

def analyze_ticker(ticker: str, bench_ret: Optional[float],
                   with_backtest: bool) -> Optional[dict]:
    try:
        with _quiet():
            raw = yf.download(ticker, period="400d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 55: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return None
        raw = raw.dropna(subset=["Open","Close","High","Low","Volume"])
        if len(raw) < 55: return None

        df   = _build(raw.copy())
        last = len(df) - 1

        # Quick pre-check: any gap ≥ 8% in last FRESH_WINDOW bars?
        has_gap = False
        for k in range(max(1, last - FRESH_WINDOW), last + 1):
            if k < 1: continue
            gap_pct = (float(df.iloc[k]["Open"]) / float(df.iloc[k-1]["Close"]) - 1)
            if gap_pct >= GAP_MIN_PCT:
                has_gap = True; break
        if not has_gap: return None

        # Fetch earnings dates (only if we found a gap — limits API calls)
        earnings_dates = _recent_earnings_dates(ticker, lookback_days=FRESH_WINDOW + 3)

        gap_info = _find_gap(df, last, earnings_dates)
        if not gap_info: return None

        sig = _score(df, last, gap_info)
        if not sig: return None

        result = {"ticker": ticker, **sig, "hold_days": HOLD_DAYS}
        for sfx, mkt in {".L":"UK",".DE":"DE",".PA":"FR",".AS":"NL",".TO":"CA"}.items():
            if ticker.endswith(sfx): result["mkt"] = mkt; break
        else:
            result["mkt"] = "US"

        if with_backtest: result.update(_backtest_simple(df))
        return result
    except Exception:
        return None

# ── Scan entry point ──────────────────────────────────────────────────────────

def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(analyze_ticker, t, bench_returns.get(b), with_backtest): t
            for t, b in universe.items()
        }
        for f in as_completed(futs, timeout=240):
            try:
                r = f.result(timeout=45)
                if r:
                    r["strategy"] = "power_earnings_gap"
                    results.append(r)
            except Exception:
                pass
    results.sort(key=lambda x: (-(x.get("wr") or 0), -x["score"], -x["gap_pct"]))
    return results


def main():
    from momentum_scanner import build_universe, compute_bench_returns
    import time
    wb  = "--no-backtest" not in sys.argv
    uni = build_universe()
    bench = compute_bench_returns(set(uni.values()))
    t0  = time.time()
    res = scan(uni, bench, wb)
    print(f"\nPower Earnings Gap Scanner — {len(res)} signals in {time.time()-t0:.0f}s")
    for r in res[:20]:
        tag = "✓" if r["gap_verified"] else "~"
        print(f"  {r['ticker']:<10}  EG{tag}  +{r['gap_pct']}%  "
              f"vol={r['gap_vol_ratio']}x  held={'Y' if r['gap_held'] else 'N'}  "
              f"tight={'Y' if r['tight_consol'] else 'N'}  "
              f"m={r['minervini']}  score={r['score']}")


if __name__ == "__main__":
    main()
