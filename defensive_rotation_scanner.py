#!/usr/bin/env python3
"""
Defensive Rotation Scanner  |  Meb Faber / Sector Rotation
──────────────────────────────────────────────────────────────────────────────
Detects when institutional money rotates from momentum/growth into defensive
sectors, then finds the individual stock leaders within those sectors.

Step 1 — Rotation detection (ETF level, fast):
  XLU / XLP / XLV / GLD each checked:
    - 20d return > SPY 20d return + 3%  (strict outperformance)
    - Recent 5d outperformance > prior 5d outperformance  (accelerating)
  Only confirmed rotating sectors proceed to Step 2.

Step 2 — Individual stock selection within rotating sectors:
  Hard filters:
    - Price > SMA50  (uptrend even within defensive)
    - Stock 20d return > sector ETF 20d return  (leader within sector)
    - RSI 35–70  (momentum without being overbought)
    - ADX > 12  (some directional trend)
    - Vol > 0.8× average  (participation present)
  Scoring: RS vs SPY, RS vs sector ETF, volume expansion, MACD+,
           near 52w high, Minervini ≥ 3, price > SMA200

Hold: 10 days (defensive rotations move slowly, need time to play out)

python3 defensive_rotation_scanner.py --no-backtest
python3 defensive_rotation_scanner.py
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

HOLD_DAYS    = 10
MAX_WORKERS  = 25
FRESH_WINDOW = 3

# Rotation threshold: sector ETF must beat SPY by this margin (10d)
# Lowered 3% → 1.5% + window 20d → 10d to catch early-stage rotations
ROTATION_THRESHOLD = 0.015  # 1.5 percentage points
ROTATION_WINDOW    = 10     # trading days for outperformance measurement

# ── Sector definitions ────────────────────────────────────────────────────────

SECTORS = {
    "XLU": {
        "name": "Utilities",
        "tickers_us": [
            "NEE", "SO", "DUK", "AEP", "EXC", "SRE", "D", "PEG", "XEL",
            "ES", "WEC", "ETR", "FE", "PPL", "AEE", "CMS", "NI", "EVRG",
            "PNW", "NWE", "AVA", "IDA", "POR",
        ],
        "tickers_intl": ["SSE.L", "NG.L", "SVT.L", "UU.L"],
    },
    "XLP": {
        "name": "Consumer Staples",
        "tickers_us": [
            "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL", "KMB",
            "SYY", "TSN", "HRL", "CAG", "GIS", "MKC", "K", "CPB", "SJM",
            "CHD", "CLX", "KR", "TGT", "MDLZ",
        ],
        "tickers_intl": ["ULVR.L", "BATS.L", "DGE.L"],
    },
    "XLV": {
        "name": "Healthcare",
        "tickers_us": [
            "JNJ", "UNH", "ABT", "MDT", "PFE", "MRK", "BMY", "CVS", "CI",
            "HUM", "TMO", "DHR", "EW", "ZBH", "BAX", "BDX", "BSX", "SYK",
            "IQV", "A", "DGX", "LH", "HCA", "CNC", "MOH", "ISRG", "ELV",
        ],
        "tickers_intl": ["AZN.L", "GSK.L", "SAN.PA"],
    },
    "GLD": {
        "name": "Gold",
        "tickers_us": [
            "NEM", "GOLD", "AEM", "WPM", "KGC", "AGI", "BTG", "DRD",
            "GFI", "HMY", "AU", "OR", "FNV", "RGLD",
        ],
        "tickers_intl": [],
    },
    # ── EU defensive ETFs (STOXX Europe 600 sub-indices, Xetra-listed) ────────
    "EXV1.DE": {
        "name": "EU Utilities",
        "tickers_us": [],
        "tickers_intl": [
            "ENEL.MI", "ENGI.PA", "IBE.MC", "RWE.DE", "EONGn.DE",
            "EDP.LS", "VIE.PA", "SSE.L", "NG.L", "SVT.L", "UU.L",
            "ELE.MC", "RED.MC", "BKW.SW",
        ],
    },
    "EXH1.DE": {
        "name": "EU Healthcare",
        "tickers_us": [],
        "tickers_intl": [
            "ROG.SW", "NOVN.SW", "AZN.L", "GSK.L", "SAN.PA",
            "BAYN.DE", "SHL.DE", "FRE.DE", "SOON.SW", "GIVN.SW",
            "COHU.DE", "FME.DE", "EVO.DE",
        ],
    },
    "EXH3.DE": {
        "name": "EU Consumer Staples",
        "tickers_us": [],
        "tickers_intl": [
            "NESN.SW", "UNA.AS", "ULVR.L", "DGE.L", "BATS.L",
            "HEIN.AS", "BN.PA", "CPR.MI", "LONN.SW", "ABI.BR",
            "TREIF.SW", "BEIA.AS", "DANO.PA",
        ],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_etf(ticker: str, period: str = "60d") -> Optional[pd.Series]:
    try:
        with _quiet():
            raw = yf.download(ticker, period=period, interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 25: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        return raw["Close"].dropna()
    except Exception:
        return None

# ── Step 1: Rotation detection at ETF level ───────────────────────────────────

def detect_rotating_sectors(spy_prices: pd.Series) -> dict:
    """
    Returns dict of sector_etf → rotation_info for confirmed rotating sectors.
    Gate: 10d outperformance > 1.5% AND recent 5d outperformance > prior 5d.
    EU ETFs compared against SPY-equivalent (price-based approximation).
    """
    rotating = {}
    W = ROTATION_WINDOW + 1   # iloc offset for W trading days back

    spy_nd   = float(spy_prices.iloc[-1] / spy_prices.iloc[-W]  - 1) if len(spy_prices) >= W  else 0
    spy_5d_r = float(spy_prices.iloc[-1] / spy_prices.iloc[-6]  - 1) if len(spy_prices) >= 6  else 0
    spy_5d_p = float(spy_prices.iloc[-6] / spy_prices.iloc[-11] - 1) if len(spy_prices) >= 11 else 0

    for etf_ticker in SECTORS:
        etf = _fetch_etf(etf_ticker, period="60d")
        if etf is None or len(etf) < W: continue

        etf_nd   = float(etf.iloc[-1] / etf.iloc[-W] - 1)
        etf_5d_r = float(etf.iloc[-1] / etf.iloc[-6]  - 1) if len(etf) >= 6  else 0
        etf_5d_p = float(etf.iloc[-6] / etf.iloc[-11] - 1) if len(etf) >= 11 else 0

        rel_nd        = etf_nd   - spy_nd      # 10d relative perf vs SPY
        rel_5d_recent = etf_5d_r - spy_5d_r    # recent 5d relative perf
        rel_5d_prior  = etf_5d_p - spy_5d_p    # prior 5d relative perf

        # Gate: >1.5% outperformance over 10 days
        if rel_nd < ROTATION_THRESHOLD: continue
        # Acceleration gate: recent 5d better than prior 5d
        if rel_5d_recent <= rel_5d_prior: continue

        rotating[etf_ticker] = {
            "name":        SECTORS[etf_ticker]["name"],
            "rel_nd":      round(rel_nd * 100, 2),
            "rel_5d":      round(rel_5d_recent * 100, 2),
            "etf_nd_ret":  round(etf_nd * 100, 2),
            "spy_nd_ret":  round(spy_nd * 100, 2),
            "etf_prices":  etf,
        }

    return rotating

# ── Step 2: Score individual stock within a rotating sector ───────────────────

def _build_stock(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["sma50"]    = _sma(c, 50)
    df["sma200"]   = _sma(c, 200)
    df["vol_ma20"] = v.rolling(20).mean()
    df["rsi"]      = _rsi(c, 14)
    df["adx"]      = _adx(h, l, c, 14)
    macd           = _ema(c, 12) - _ema(c, 26)
    df["macd_hist"] = macd - _ema(macd, 9)
    df["52w_high"] = c.rolling(252).max()
    df["52w_low"]  = c.rolling(252).min()
    return df

def score_stock(ticker: str, sector_etf: str, rotation_info: dict,
                spy_prices: pd.Series, with_backtest: bool) -> Optional[dict]:
    try:
        with _quiet():
            raw = yf.download(ticker, period="400d", interval="1d",
                              progress=False, auto_adjust=True, threads=False)
        if raw is None or len(raw) < 55: return None
        if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.droplevel(1)
        if not {"Open","High","Low","Close","Volume"}.issubset(raw.columns): return None
        raw = raw.dropna(subset=["Close","High","Low","Volume"])
        if len(raw) < 55: return None

        df  = _build_stock(raw.copy())
        row = df.iloc[-1]
        c   = float(row["Close"])
        if pd.isna(c) or c < 1.0: return None

        vol_ma = float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else 0
        if vol_ma < 50_000: return None

        # ── Hard filters ──────────────────────────────────────────────────────
        sma50 = float(row["sma50"])
        if pd.isna(sma50) or c <= sma50: return None   # must be above SMA50

        rsi = float(row["rsi"]) if not pd.isna(row["rsi"]) else 50
        if rsi < 35 or rsi > 70: return None            # momentum zone only

        adx = float(row["adx"]) if not pd.isna(row["adx"]) else 0
        if adx < 12: return None                         # needs some direction

        vol_ratio = float(row["Volume"]) / vol_ma if vol_ma > 0 else 0
        if vol_ratio < 0.8: return None                  # participation required

        # ── Relative strength vs sector ETF ──────────────────────────────────
        etf_prices = rotation_info["etf_prices"]
        # Align dates between stock and ETF
        stock_close = df["Close"]
        aligned     = pd.concat([stock_close.rename("s"),
                                  etf_prices.rename("e"),
                                  spy_prices.rename("spy")], axis=1).dropna()
        if len(aligned) < 21: return None

        W = ROTATION_WINDOW + 1
        if len(aligned) < W: return None
        stock_nd = float(aligned["s"].iloc[-1] / aligned["s"].iloc[-W] - 1)
        etf_nd   = float(aligned["e"].iloc[-1] / aligned["e"].iloc[-W] - 1)
        spy_nd   = float(aligned["spy"].iloc[-1] / aligned["spy"].iloc[-W] - 1)

        # Hard filter: stock must outperform its own sector ETF
        if stock_nd <= etf_nd: return None

        rs_vs_spy    = stock_nd - spy_nd
        rs_vs_sector = stock_nd - etf_nd

        # ── Scoring ───────────────────────────────────────────────────────────
        sma200    = float(row["sma200"]) if not pd.isna(row["sma200"]) else 0
        w52h      = float(row["52w_high"]) if not pd.isna(row["52w_high"]) else 0
        macd_h    = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0

        # Minervini-lite (relaxed for defensives — they rarely score 6+)
        m = sum([
            c > sma50,
            c > sma200 if sma200 > 0 else False,
            sma50 > sma200 if sma200 > 0 else False,
            c >= 1.20 * float(row["52w_low"]) if not pd.isna(row["52w_low"]) else False,
            c >= 0.80 * w52h if w52h > 0 else False,
        ])

        conf_flags = {
            f"RS+{round(rs_vs_spy*100,1)}%":    rs_vs_spy > 0.01,   # beats SPY by 1%+
            f"vsETF+{round(rs_vs_sector*100,1)}%": rs_vs_sector > 0.005,
            "VOL↑":     vol_ratio > 1.2,
            "MACD+":    macd_h > 0,
            ">SMA200":  c > sma200 if sma200 > 0 else False,
            "52wHI":    w52h > 0 and c >= 0.90 * w52h,
            "M≥3":      m >= 3,
        }
        score = sum(conf_flags.values())
        if score < 2: return None   # need at least some confirmation

        fresh_tag = f"ROT:{sector_etf}"
        mkt = "US"
        for sfx, m_tag in {".L":"UK",".DE":"DE",".PA":"FR",".AS":"NL",".TO":"CA"}.items():
            if ticker.endswith(sfx):
                mkt = m_tag; break

        result = {
            "ticker":       ticker,
            "sector":       rotation_info["name"],
            "sector_etf":   sector_etf,
            "score":        score,
            "fresh":        [fresh_tag, f"ETF+{rotation_info['rel_nd']}%vsS&P"],
            "conf":         [k for k, v in conf_flags.items() if v],
            "rsi":          round(rsi, 1),
            "adx":          round(adx, 1),
            "vol_ratio":    round(vol_ratio, 2),
            "minervini":    m,
            "price":        round(c, 2),
            "rs_vs_spy":    round(rs_vs_spy  * 100, 2),
            "rs_vs_sector": round(rs_vs_sector* 100, 2),
            "mkt":          mkt,
            "hold_days":    HOLD_DAYS,
        }

        if with_backtest:
            result.update(_backtest(df))

        return result
    except Exception:
        return None

# ── Backtest ──────────────────────────────────────────────────────────────────

def _backtest(df: pd.DataFrame) -> dict:
    """Buy when price > SMA50 and RSI 35-70; hold HOLD_DAYS."""
    rets, last = [], -10
    rsi_s = _rsi(df["Close"])
    sma50 = _sma(df["Close"], 50)
    for i in range(50, len(df) - HOLD_DAYS - 1):
        if i - last < HOLD_DAYS: continue
        c_ = float(df.iloc[i]["Close"])
        r_ = float(rsi_s.iloc[i]) if not pd.isna(rsi_s.iloc[i]) else 50
        s_ = float(sma50.iloc[i])  if not pd.isna(sma50.iloc[i])  else 0
        if c_ <= s_ or r_ < 35 or r_ > 70: continue
        exit_ = float(df.iloc[i + HOLD_DAYS]["Close"])
        rets.append((exit_ - c_) / c_ * 100)
        last = i
    if not rets: return {"n": 0, "wr": None, "avg": None, "med": None}
    a = np.array(rets)
    return {"n": len(a), "wr": round(100*(a>0).mean(),1),
            "avg": round(float(a.mean()),2), "med": round(float(np.median(a)),2)}

# ── Main scan ─────────────────────────────────────────────────────────────────

def scan(universe: dict, bench_returns: dict, with_backtest: bool = True) -> list:
    """
    1. Fetch SPY prices (shared reference).
    2. Detect which defensive sectors are rotating in.
    3. Scan all stock tickers for each active sector.
    """
    # Step 1: SPY
    spy = _fetch_etf("SPY", period="60d")
    if spy is None:
        return []

    # Step 2: Which sectors are rotating?
    rotating = detect_rotating_sectors(spy)
    if not rotating:
        return []   # no active rotation today

    # Step 3: Build ticker → sector map from hardcoded defensive universe
    # Map each ticker to its sector ETF (a ticker can appear in only one sector)
    ticker_sector_map: dict[str, str] = {}
    for etf_ticker, info in rotating.items():
        all_tickers = (SECTORS[etf_ticker]["tickers_us"] +
                       SECTORS[etf_ticker]["tickers_intl"])
        for t in all_tickers:
            if t not in ticker_sector_map:
                ticker_sector_map[t] = etf_ticker

    if not ticker_sector_map:
        return []

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(score_stock, ticker, etf_ticker, rotating[etf_ticker],
                        spy, with_backtest): ticker
            for ticker, etf_ticker in ticker_sector_map.items()
        }
        for f in as_completed(futs, timeout=240):
            try:
                r = f.result(timeout=45)
                if r:
                    r["strategy"] = "defensive_rotation"
                    results.append(r)
            except Exception:
                pass

    results.sort(key=lambda x: (-(x.get("wr") or 0), -x["score"], -x["rs_vs_sector"]))
    return results


def main():
    import time
    wb = "--no-backtest" not in sys.argv

    # Fetch SPY for rotation check
    spy = _fetch_etf("SPY", period="60d")
    if spy is None:
        print("Could not fetch SPY — aborting.")
        return

    rotating = detect_rotating_sectors(spy)
    if not rotating:
        print("\nDefensive Rotation Scanner — no sectors rotating in today.")
        print("  (Need sector ETF to outperform SPY by >3% over 20d AND accelerate)")
        return

    print(f"\nDefensive Rotation Scanner — {len(rotating)} sector(s) active:")
    for etf, info in rotating.items():
        print(f"  {etf} ({info['name']}): ETF +{info['etf_20d_ret']}% vs "
              f"SPY +{info['spy_20d_ret']}% (rel: +{info['rel_20d']}%,  "
              f"5d accel: +{info['rel_5d']}%)")

    # Use hardcoded defensive universe
    dummy_universe = {}
    dummy_bench    = {}
    t0 = time.time()
    res = scan(dummy_universe, dummy_bench, wb)
    print(f"\n  {len(res)} stock signal(s) in {time.time()-t0:.0f}s")
    for r in res[:20]:
        print(f"  {r['ticker']:<10} [{r['sector']:<18}]  "
              f"RS-SPY={r['rs_vs_spy']:+.1f}%  RS-ETF={r['rs_vs_sector']:+.1f}%  "
              f"rsi={r['rsi']}  score={r['score']}")


if __name__ == "__main__":
    main()
