"""
╔══════════════════════════════════════════════════════════════════╗
║  DAILY MARKET REPORT — Sector Rotation + Stock Picks            ║
║  Run once each morning for auto-refreshed data + full analysis  ║
╠══════════════════════════════════════════════════════════════════╣
║  §1  Config           tickers, params, company names            ║
║  §2  Data Layer       auto-download / refresh all tickers       ║
║  §3  Indicator Engine RRG, StealthTrail, 4-Factor, backtests    ║
║      ├─ Quadrant Transition Analysis (QTA)  — sector backtest   ║
║      └─ Signal Forward Return Analysis (SFRA) — stock backtest  ║
║  §4  Sector Rotation  scoring → ROTATING / WATCH / AVOID        ║
║  §5  Stock Drill-Down picks + reasons for rotating sectors      ║
║  §6  Charts           RRG scatter, score bars, stock score bars ║
║  §7  HTML Report      self-contained dashboard → market_report  ║
║  §8  Console Output   structured text summary                   ║
║  §9  Main             orchestrates all steps                    ║
╚══════════════════════════════════════════════════════════════════╝
Usage:
    python market_report.py
"""

import os, json, base64, io, warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yfinance as yf


# ══════════════════════════════════════════════════════════════════
#  §1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DATA_DIR   = "data"
START_DATE = "2010-01-01"
BENCHMARK  = "SPY"

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLY":  "Consumer Disc.",
    "XLP":  "Consumer Staples",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLC":  "Comm. Services",
    "SOXX": "Semiconductors",
    "XRT":  "Retail",
    "IBB":  "Biotech",
    "KRE":  "Reg. Banks",
    "ITA":  "Aerospace/Defense",
    "XOP":  "Oil & Gas E&P",
    "OIH":  "Oil Services",
    "GDX":  "Gold Miners",
    "ICLN": "Clean Energy",
    "JETS": "Airlines",
}

STOCK_UNIVERSE = {
    "XLE":  ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","DVN",
             "HAL","BKR","FANG","KMI","WMB","EQT","TRGP","OKE"],
    "XLU":  ["NEE","SO","DUK","SRE","AEP","D","EXC","PCG","XEL","ED",
             "WEC","EIX","ETR","AWK","ES","PPL","AES","CNP","NI","CMS"],
    "XLK":  ["AAPL","MSFT","NVDA","AMD","AVGO","INTC","QCOM","CRM","ADBE","NOW","INTU"],
    "SOXX": ["NVDA","AMD","AVGO","INTC","QCOM","AMAT","LRCX","KLAC","MU","TSM","SMCI"],
    "XLV":  ["JNJ","UNH","ABBV","MRK","LLY","PFE","TMO","ABT","MDT","ISRG","ELV"],
    "XLF":  ["JPM","BAC","WFC","GS","MS","BLK","C","AXP","COF","SCHW"],
    "XLY":  ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","RL","TPR"],
    "XLC":  ["META","GOOGL","NFLX","DIS","VZ","T","EA"],
    "XLI":  ["CAT","HON","UPS","RTX","LMT","DE","GE","NOC","EMR","ETN"],
    "XLB":  ["LIN","APD","SHW","NEM","FCX","NUE","ALB","PKG"],
    "IBB":  ["AMGN","GILD","BIIB","REGN","VRTX","MRNA"],
    "XRT":  ["TGT","WMT","COST","TJX","ROST"],
}

# Full company name lookup — shown in brackets next to ticker in reports
COMPANY_NAMES = {
    # Benchmark & ETFs
    "SPY":"S&P 500 ETF","XLK":"Technology SPDR","XLF":"Financials SPDR",
    "XLY":"Consumer Disc. SPDR","XLP":"Consumer Staples SPDR","XLE":"Energy SPDR",
    "XLV":"Health Care SPDR","XLI":"Industrials SPDR","XLB":"Materials SPDR",
    "XLU":"Utilities SPDR","XLRE":"Real Estate SPDR","XLC":"Comm. Services SPDR",
    "SOXX":"iShares Semiconductors","XRT":"SPDR Retail ETF","IBB":"iShares Biotech",
    "KRE":"SPDR Reg. Banks","ITA":"iShares Aerospace/Defense","XOP":"SPDR Oil & Gas E&P",
    "OIH":"VanEck Oil Services","GDX":"VanEck Gold Miners",
    "ICLN":"iShares Clean Energy","JETS":"US Global Jets ETF",
    # Energy
    "XOM":"ExxonMobil","CVX":"Chevron","COP":"ConocoPhillips",
    "EOG":"EOG Resources","SLB":"Schlumberger","MPC":"Marathon Petroleum",
    "PSX":"Phillips 66","VLO":"Valero Energy","OXY":"Occidental Petroleum",
    "DVN":"Devon Energy","HAL":"Halliburton","BKR":"Baker Hughes",
    "FANG":"Diamondback Energy","KMI":"Kinder Morgan","WMB":"Williams Cos.",
    "EQT":"EQT Corp","TRGP":"Targa Resources","OKE":"ONEOK",
    # Utilities
    "NEE":"NextEra Energy","SO":"Southern Company","DUK":"Duke Energy",
    "SRE":"Sempra Energy","AEP":"Amer. Electric Power","D":"Dominion Energy",
    "EXC":"Exelon","PCG":"PG&E","XEL":"Xcel Energy",
    "ED":"Consol. Edison","WEC":"WEC Energy","EIX":"Edison International",
    "ETR":"Entergy","AWK":"Amer. Water Works","ES":"Eversource Energy",
    "PPL":"PPL Corp","AES":"AES Corp","CNP":"CenterPoint Energy",
    "NI":"NiSource","CMS":"CMS Energy",
    # Technology
    "AAPL":"Apple","MSFT":"Microsoft","NVDA":"NVIDIA","AMD":"AMD",
    "AVGO":"Broadcom","INTC":"Intel","QCOM":"Qualcomm",
    "CRM":"Salesforce","ADBE":"Adobe","NOW":"ServiceNow",
    "INTU":"Intuit","ORCL":"Oracle",
    # Semiconductors
    "AMAT":"Applied Materials","LRCX":"Lam Research","KLAC":"KLA Corp",
    "MU":"Micron Technology","TSM":"TSMC","SMCI":"Super Micro Computer",
    # Health Care
    "JNJ":"Johnson & Johnson","UNH":"UnitedHealth","ABBV":"AbbVie",
    "MRK":"Merck","LLY":"Eli Lilly","PFE":"Pfizer",
    "TMO":"Thermo Fisher","ABT":"Abbott Labs","MDT":"Medtronic",
    "ISRG":"Intuitive Surgical","ELV":"Elevance Health",
    # Financials
    "JPM":"JPMorgan Chase","BAC":"Bank of America","WFC":"Wells Fargo",
    "GS":"Goldman Sachs","MS":"Morgan Stanley","BLK":"BlackRock",
    "C":"Citigroup","AXP":"American Express","COF":"Capital One","SCHW":"Charles Schwab",
    # Consumer Discretionary
    "AMZN":"Amazon","TSLA":"Tesla","HD":"Home Depot",
    "MCD":"McDonald's","NKE":"Nike","SBUX":"Starbucks",
    "LOW":"Lowe's","TJX":"TJX Companies","BKNG":"Booking Holdings",
    "RL":"Ralph Lauren","TPR":"Tapestry",
    # Communication Services
    "META":"Meta Platforms","GOOGL":"Alphabet","NFLX":"Netflix",
    "DIS":"Walt Disney","VZ":"Verizon","T":"AT&T","EA":"Electronic Arts",
    # Industrials
    "CAT":"Caterpillar","HON":"Honeywell","UPS":"UPS",
    "RTX":"RTX Corp","LMT":"Lockheed Martin","DE":"John Deere",
    "GE":"GE Aerospace","NOC":"Northrop Grumman","EMR":"Emerson Electric","ETN":"Eaton Corp",
    # Materials
    "LIN":"Linde","APD":"Air Products","SHW":"Sherwin-Williams",
    "DD":"DuPont","NEM":"Newmont","FCX":"Freeport-McMoRan",
    "NUE":"Nucor","ALB":"Albemarle","CE":"Celanese","PKG":"Packaging Corp",
    # Biotech
    "AMGN":"Amgen","GILD":"Gilead Sciences","BIIB":"Biogen",
    "REGN":"Regeneron","VRTX":"Vertex Pharma","MRNA":"Moderna",
    "BNTX":"BioNTech","ILMN":"Illumina",
    # Retail
    "COST":"Costco","WMT":"Walmart","TGT":"Target","ROST":"Ross Stores",
    # Previous analysis stocks
    "LULU":"lululemon","CPRI":"Capri Holdings","PVH":"PVH Corp",
    "VFC":"VF Corporation","URBN":"Urban Outfitters",
    "WDAY":"Workday","TEAM":"Atlassian",
    "CRWD":"CrowdStrike","PANW":"Palo Alto Networks","ZS":"Zscaler",
    "FTNT":"Fortinet","OKTA":"Okta","CYBR":"CyberArk",
}

# StealthTrail parameters (tuned on SPY daily)
ST_ATR_LEN, ST_BASE_MULT, ST_ADAPT_SMOOTH = 13, 2.5, 55
ST_MIN_MULT, ST_MAX_MULT                   = 1.0, 5.0
ST_RSI_LEN, ST_RSI_THRESH                  = 13, 45

# Scoring thresholds
SECTOR_DRILL_THRESHOLD = 6
STOCK_STRONG_BUY       = 12
STOCK_BUY              = 9
STOCK_WATCH            = 6

# Backtest holding periods (trading days)
BT_PERIODS = [5, 10, 20, 30]


# ══════════════════════════════════════════════════════════════════
#  §2  DATA LAYER
# ══════════════════════════════════════════════════════════════════

def _all_tickers():
    tickers = {BENCHMARK} | set(SECTOR_ETFS.keys())
    for stocks in STOCK_UNIVERSE.values():
        tickers.update(stocks)
    return sorted(tickers)

def refresh_data(verbose=True):
    """Download / update all tickers to latest trading day."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tickers = _all_tickers()
    today   = datetime.today().date()
    end_str = str(today + timedelta(days=1))
    updated, skipped, failed = 0, 0, 0
    for t in tickers:
        path        = f"{DATA_DIR}/{t}_1d.csv"
        need_refresh = True
        if os.path.exists(path):
            try:
                last_idx = pd.read_csv(path, index_col=0).index[-1]
                last     = str(last_idx)[:10]
                if (today - pd.Timestamp(last).date()).days <= 1:
                    need_refresh = False
            except Exception:
                pass
        if not need_refresh:
            skipped += 1
            continue
        try:
            df = yf.download(t, start=START_DATE, end=end_str,
                             auto_adjust=True, progress=False)
            if df.empty:
                if verbose: print(f"  [SKIP] {t}: no data returned")
                failed += 1
                continue
            df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                          for c in df.columns]
            df.index.name = "Date"
            df = df[["open","high","low","close","volume"]].dropna()
            df.to_csv(path)
            updated += 1
            if verbose: print(f"  [OK]   {t:6s} → {len(df):,} rows, last={df.index[-1][:10]}")
        except Exception as e:
            if verbose: print(f"  [ERR]  {t}: {e}")
            failed += 1
    print(f"\n  Data refresh: {updated} updated | {skipped} up-to-date | {failed} failed\n")

def load_df(ticker):
    path = f"{DATA_DIR}/{ticker}_1d.csv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        df = df[["open","high","low","close","volume"]].dropna()
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        return df
    except Exception:
        return None

def ticker_label(ticker):
    """Returns 'TICKER (Company Name)' or just 'TICKER' if no name found."""
    name = COMPANY_NAMES.get(ticker)
    return f"{ticker} ({name})" if name else ticker


# ══════════════════════════════════════════════════════════════════
#  §3  INDICATOR ENGINE + BACKTESTS
# ══════════════════════════════════════════════════════════════════

# ── Core indicators ──────────────────────────────────────────────

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _rsi(close, n=14):
    d  = close.diff()
    g  = d.clip(lower=0); l = (-d).clip(lower=0)
    ag = g.ewm(alpha=1/n, adjust=False).mean()
    al = l.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + ag / (al + 1e-9))

def _atr(df, n=14):
    hi = df["high"]; lo = df["low"]; cl = df["close"]
    tr = pd.concat([hi - lo,
                    (hi - cl.shift()).abs(),
                    (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

# ── RRG ──────────────────────────────────────────────────────────

def compute_rrg(stock_close, bench_close):
    both  = pd.concat([stock_close.rename("s"), bench_close.rename("b")],
                      axis=1).dropna()
    rs    = 100 * both["s"] / both["b"]
    rs1   = _ema(rs, 10); rs2 = _ema(rs, 26)
    ratio = 100 + (rs1 - rs2) / rs2 * 100
    mom   = 100 + (_ema(ratio, 10) - _ema(ratio, 26)) / _ema(ratio, 26) * 100
    return ratio, mom

def quadrant(ratio_val, mom_val):
    if   ratio_val >= 100 and mom_val >= 100: return "Leading"
    elif ratio_val >= 100 and mom_val <  100: return "Weakening"
    elif ratio_val <  100 and mom_val >= 100: return "Improving"
    else:                                      return "Lagging"

# ── StealthTrail ─────────────────────────────────────────────────

def _stealth_trend_series(df):
    """Full historical StealthTrail trend (+1 bull / -1 bear) + RSI series."""
    if len(df) < 200:
        return None, None
    hi = df["high"]; lo = df["low"]; cl = df["close"]
    a     = _atr(df, ST_ATR_LEN)
    r     = _rsi(cl, ST_RSI_LEN)
    v_ema = _ema(cl, ST_ADAPT_SMOOTH)
    v_rat = (a / v_ema).fillna(0)
    v_min = v_rat.rolling(50).min(); v_max = v_rat.rolling(50).max()
    mult  = (ST_MIN_MULT + (ST_MAX_MULT - ST_MIN_MULT) *
             (v_rat - v_min) / (v_max - v_min + 1e-9)).clip(ST_MIN_MULT, ST_MAX_MULT)
    hl2   = (hi + lo) / 2
    up    = hl2 - mult * a
    dn    = hl2 + mult * a
    trend = pd.Series(1, index=cl.index)
    for i in range(1, len(cl)):
        up.iat[i] = max(up.iat[i], up.iat[i-1]) if cl.iat[i-1] > up.iat[i-1] else up.iat[i]
        dn.iat[i] = min(dn.iat[i], dn.iat[i-1]) if cl.iat[i-1] < dn.iat[i-1] else dn.iat[i]
        if   cl.iat[i] > dn.iat[i-1]: trend.iat[i] = 1
        elif cl.iat[i] < up.iat[i-1]: trend.iat[i] = -1
        else:                          trend.iat[i] = trend.iat[i-1]
    return trend, r

def stealth_signal(df):
    """Return ('BUY'|'FLAT', vol_surge_bool) from latest StealthTrail bar."""
    trend, r = _stealth_trend_series(df)
    if trend is None:
        return "FLAT", False
    in_buy    = (trend.iloc[-1] == 1) and (r.iloc[-1] > ST_RSI_THRESH)
    vol_surge = df["volume"].iloc[-1] > df["volume"].rolling(20).mean().iloc[-1] * 1.5
    return ("BUY" if in_buy else "FLAT"), bool(vol_surge)

# ── 4-Factor confirmation ────────────────────────────────────────

def four_factor(df, rs_mom_series):
    cl  = df["close"]; vol = df["volume"]
    price_trend = bool(cl.iloc[-1] > _ema(cl, 50).iloc[-1])
    vol_ok      = bool(vol.iloc[-1] > vol.rolling(20).mean().iloc[-1] * 1.3)
    roc         = cl.pct_change(10) * 100
    roc_acc     = bool(roc.iloc[-1] > roc.iloc[-3])
    rs_ok       = bool(rs_mom_series.iloc[-1] > 100 and
                       rs_mom_series.iloc[-1] > rs_mom_series.iloc[-5])
    sigs  = [price_trend, vol_ok, roc_acc, rs_ok]
    n     = sum(sigs)
    level = "STRONG" if n >= 4 else "ALERT" if n >= 3 else "WATCH" if n >= 2 else "NONE"
    return level, n, sigs

# ── Volatility & return helpers ──────────────────────────────────

def vol_bucket(df):
    ann_vol = df["close"].pct_change().std() * np.sqrt(252) * 100
    bucket  = "Low" if ann_vol < 20 else "Medium" if ann_vol < 35 else "High"
    return bucket, round(ann_vol, 1)

def ret_1y(df):
    return round((df["close"].iloc[-1] / df["close"].iloc[-252] - 1) * 100, 1) \
        if len(df) >= 252 else None

# ── Reason builder ───────────────────────────────────────────────

def build_reason(rrg_spy, rrg_sec, st_sig, alert, n_sigs, r1y, is_sector=False):
    parts = []
    if rrg_spy in ("Leading","Improving"):
        parts.append(f"Outperforming SPY ({rrg_spy})")
    elif rrg_spy in ("Weakening","Lagging"):
        parts.append(f"Underperforming SPY ({rrg_spy})")
    if not is_sector:
        if rrg_sec in ("Leading","Improving"):
            parts.append(f"Top within sector ({rrg_sec})")
        elif rrg_sec in ("Weakening","Lagging"):
            parts.append(f"Lagging within sector ({rrg_sec})")
    if st_sig == "BUY":
        parts.append("StealthTrail active")
    else:
        parts.append("No trend signal")
    if alert in ("STRONG","ALERT"):
        parts.append(f"{n_sigs}/4 confirmation signals")
    if r1y is not None and r1y > 20:
        parts.append(f"+{r1y}% 1Y momentum")
    elif r1y is not None and r1y < 0:
        parts.append(f"{r1y}% 1Y (negative)")
    return "; ".join(parts)


# ══════════════════════════════════════════════════════════════════
#  BACKTESTS
#  ┌─ Quadrant Transition Analysis (QTA) — sectors
#  └─ Signal Forward Return Analysis (SFRA) — stocks
# ══════════════════════════════════════════════════════════════════

def _forward_returns(close_series, entry_indices, periods):
    """Compute forward returns for a list of integer entry positions."""
    n   = len(close_series)
    out = {p: [] for p in periods}
    for idx in entry_indices:
        ep = close_series.iat[idx]
        for p in periods:
            fi = idx + p
            if fi < n:
                out[p].append((close_series.iat[fi] / ep - 1) * 100)
    return out

def _summarise(rets_dict, periods):
    stats = {}
    for p in periods:
        vals = rets_dict[p]
        if vals:
            stats[f"hit_{p}d"]    = round(sum(v > 0 for v in vals) / len(vals) * 100, 1)
            stats[f"avg_{p}d"]    = round(float(np.mean(vals)), 2)
            stats[f"n_obs_{p}d"]  = len(vals)
    return stats

def backtest_qta(df, spy_df, periods=BT_PERIODS):
    """
    Quadrant Transition Analysis (QTA):
    Find every historical date a sector ETF entered the 'Leading' quadrant
    (coming from a different quadrant) and measure forward returns.

    Returns dict with keys:
      n_entries, hit_Xd, avg_Xd (for each period),
      days_in_leading (days since current Leading run began)
    """
    comm = df.index.intersection(spy_df.index)
    if len(comm) < 300:
        return None
    df_a  = df.loc[comm]; spy_a = spy_df.loc[comm]
    ratio, mom = compute_rrg(df_a["close"], spy_a["close"])
    quads  = pd.Series([quadrant(r, m) for r, m in zip(ratio, mom)], index=ratio.index)
    # Transition: previous bar was NOT Leading, current bar IS Leading
    is_entry    = (quads == "Leading") & (quads.shift(1) != "Leading")
    entry_dates = quads.index[is_entry]
    if len(entry_dates) < 3:
        return None
    close       = df_a["close"]
    entry_locs  = [close.index.get_loc(d) for d in entry_dates if d in close.index]
    rets        = _forward_returns(close, entry_locs, periods)
    stats       = _summarise(rets, periods)
    stats["n_entries"] = len(entry_locs)
    # Days since current Leading run started
    if quads.iloc[-1] == "Leading":
        recent = [d for d in entry_dates if d <= quads.index[-1]]
        if recent:
            stats["days_in_leading"] = (quads.index[-1] - recent[-1]).days
    return stats

def backtest_sfra(df, periods=BT_PERIODS):
    """
    Signal Forward Return Analysis (SFRA):
    Find every historical StealthTrail BUY entry (trend flip from non-bull → bull
    AND RSI above threshold) and measure forward returns.

    Returns dict with keys:
      n_signals, hit_Xd, avg_Xd (for each period),
      days_active (days since current signal triggered, or None)
    """
    trend, rsi_s = _stealth_trend_series(df)
    if trend is None:
        return None
    is_entry   = (trend == 1) & (trend.shift(1) != 1) & (rsi_s > ST_RSI_THRESH)
    entry_dates = df.index[is_entry]
    if len(entry_dates) < 3:
        return None
    close      = df["close"]
    entry_locs = [close.index.get_loc(d) for d in entry_dates]
    rets       = _forward_returns(close, entry_locs, periods)
    stats      = _summarise(rets, periods)
    stats["n_signals"] = len(entry_dates)
    # Current signal age + actual return since signal fired
    if trend.iloc[-1] == 1:
        recent = [d for d in entry_dates if d <= trend.index[-1]]
        if recent:
            last_entry  = recent[-1]
            stats["days_active"]   = (trend.index[-1] - last_entry).days
            entry_price            = close.loc[last_entry]
            current_price          = close.iloc[-1]
            stats["current_ret"]   = round((current_price / entry_price - 1) * 100, 2)
    return stats


# ══════════════════════════════════════════════════════════════════
#  §4  SECTOR ROTATION ENGINE
# ══════════════════════════════════════════════════════════════════

def _sector_score(rrg_quad, st_sig, alert_level, n_sigs):
    s  = {"Leading":4,"Improving":3,"Weakening":1,"Lagging":0}.get(rrg_quad, 0)
    s += 3 if st_sig == "BUY" else 0
    s += {"STRONG":4,"ALERT":2,"WATCH":1,"NONE":0}.get(alert_level, 0)
    s += n_sigs
    return s

def _sector_alert(score):
    if score >= 12: return "ROTATING ★★★"
    if score >= 9:  return "ROTATING ★★"
    if score >= 6:  return "WATCH ★"
    return "AVOID"

def run_sector_rotation(spy_df):
    """Score all sector ETFs vs SPY. Includes QTA backtest for each."""
    results = []
    for etf, name in SECTOR_ETFS.items():
        df = load_df(etf)
        if df is None or len(df) < 200:
            continue
        comm = df.index.intersection(spy_df.index)
        if len(comm) < 200:
            continue
        df_a  = df.loc[comm]; spy_a = spy_df.loc[comm]
        ratio, mom = compute_rrg(df_a["close"], spy_a["close"])
        quad      = quadrant(ratio.iloc[-1], mom.iloc[-1])
        st, vs    = stealth_signal(df)
        alert, n_sigs, flags = four_factor(df_a, mom)
        score     = _sector_score(quad, st, alert, n_sigs)
        r1        = ret_1y(df)
        vb, av    = vol_bucket(df)
        reason    = build_reason(quad, "N/A", st, alert, n_sigs, r1, is_sector=True)
        # QTA backtest
        qta       = backtest_qta(df, spy_df)
        results.append({
            "etf": etf, "name": name,
            "quad": quad, "rs_ratio": round(ratio.iloc[-1],3),
            "rs_mom": round(mom.iloc[-1],3),
            "st_signal": st, "vol_surge": vs,
            "4f_alert": alert, "4f_n": n_sigs, "4f_flags": flags,
            "score": score, "alert": _sector_alert(score),
            "ret_1y": r1, "vol_bucket": vb, "ann_vol": av,
            "reason": reason, "qta": qta,
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════
#  §5  STOCK DRILL-DOWN ENGINE
# ══════════════════════════════════════════════════════════════════

def _stock_score(rrg_spy_q, rrg_sec_q, st_sig, alert_level, n_sigs):
    s  = {"Leading":3,"Improving":2,"Weakening":1,"Lagging":0}.get(rrg_spy_q, 0)
    s += {"Leading":2,"Improving":2,"Weakening":1,"Lagging":0}.get(rrg_sec_q, 0)
    s += 3 if st_sig == "BUY" else 0
    s += {"STRONG":4,"ALERT":2,"WATCH":1,"NONE":0}.get(alert_level, 0)
    s += n_sigs
    return s

def _stock_rec(score):
    if score >= STOCK_STRONG_BUY: return "★★★ STRONG BUY"
    if score >= STOCK_BUY:        return "★★  BUY"
    if score >= STOCK_WATCH:      return "★   WATCH"
    return "    AVOID"

def run_stock_drilldown(rotating_sectors, spy_df):
    """For each rotating sector, score its stock universe. Includes SFRA backtest."""
    drilldown = {}
    for sec in rotating_sectors:
        etf = sec["etf"]
        if etf not in STOCK_UNIVERSE:
            continue
        etf_df   = load_df(etf)
        sec_rows = []
        for ticker in STOCK_UNIVERSE[etf]:
            df = load_df(ticker)
            if df is None or len(df) < 200:
                continue
            comm = df.index.intersection(spy_df.index)
            if len(comm) < 200:
                continue
            df_a  = df.loc[comm]; spy_a = spy_df.loc[comm]
            ratio, mom = compute_rrg(df_a["close"], spy_a["close"])
            quad_spy   = quadrant(ratio.iloc[-1], mom.iloc[-1])
            quad_sec   = "N/A"
            if etf_df is not None:
                comm2 = df.index.intersection(etf_df.index)
                if len(comm2) > 200:
                    r2, m2   = compute_rrg(df.loc[comm2]["close"], etf_df.loc[comm2]["close"])
                    quad_sec = quadrant(r2.iloc[-1], m2.iloc[-1])
            st, vs     = stealth_signal(df)
            alert, n, _ = four_factor(df_a, mom)
            sc         = _stock_score(quad_spy, quad_sec, st, alert, n)
            r1         = ret_1y(df)
            vb, av     = vol_bucket(df)
            reason     = build_reason(quad_spy, quad_sec, st, alert, n, r1)
            # SFRA backtest
            sfra       = backtest_sfra(df)
            sec_rows.append({
                "ticker": ticker,
                "name": COMPANY_NAMES.get(ticker, ""),
                "quad_spy": quad_spy, "quad_sec": quad_sec,
                "st_signal": st, "vol_surge": vs,
                "4f_alert": alert, "4f_n": n,
                "score": sc, "rec": _stock_rec(sc),
                "ret_1y": r1, "vol_bucket": vb, "ann_vol": av,
                "reason": reason, "sfra": sfra,
            })
        sec_rows.sort(key=lambda x: x["score"], reverse=True)
        drilldown[etf] = sec_rows
    return drilldown


# ══════════════════════════════════════════════════════════════════
#  §6  CHART GENERATION
# ══════════════════════════════════════════════════════════════════

Q_COLORS = {
    "Leading":"#10b981","Weakening":"#f59e0b",
    "Improving":"#3b82f6","Lagging":"#ef4444",
}

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def chart_rrg_scatter(sector_results):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0a0f1e"); ax.set_facecolor("#111827")
    ratios = [r["rs_ratio"] for r in sector_results]
    moms   = [r["rs_mom"]   for r in sector_results]
    xlim   = [min(min(ratios)-0.5, 99.0), max(max(ratios)+0.5, 101.0)]
    ylim   = [min(min(moms)-0.5,   99.0), max(max(moms)+0.5,   101.0)]
    for (x1,x2),(y1,y2),col,lbl in [
        ((100,xlim[1]),(100,ylim[1]),"#10b981","Leading"),
        ((100,xlim[1]),(ylim[0],100),"#f59e0b","Weakening"),
        ((xlim[0],100),(100,ylim[1]),"#3b82f6","Improving"),
        ((xlim[0],100),(ylim[0],100),"#ef4444","Lagging"),
    ]:
        ax.fill_between([x1,x2],[y1,y1],[y2,y2],color=col,alpha=0.07)
        ax.text((x1+x2)/2,(y1+y2)/2,lbl,ha="center",va="center",
                fontsize=10,color=col,alpha=0.25,fontweight="bold")
    ax.axhline(100,color="white",lw=0.8,alpha=0.4)
    ax.axvline(100,color="white",lw=0.8,alpha=0.4)
    for r in sector_results:
        c = Q_COLORS.get(r["quad"],"#aaaaaa")
        ax.scatter(r["rs_ratio"],r["rs_mom"],s=160,c=c,alpha=0.9,
                   edgecolors="white",linewidths=0.5,zorder=4)
        ax.annotate(r["etf"],(r["rs_ratio"],r["rs_mom"]),
                    textcoords="offset points",xytext=(5,4),
                    color="white",fontsize=8,fontweight="bold")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("RS-Ratio  (>100 = outperforming SPY)",color="#9ca3af",fontsize=10)
    ax.set_ylabel("RS-Momentum  (>100 = accelerating)",color="#9ca3af",fontsize=10)
    ax.tick_params(colors="#9ca3af")
    for sp in ax.spines.values(): sp.set_edgecolor("#374151")
    handles = [mpatches.Patch(color=v,label=k) for k,v in Q_COLORS.items()]
    ax.legend(handles=handles,facecolor="#111827",labelcolor="white",fontsize=9,loc="upper left")
    ax.set_title("Live RRG Snapshot — All Sectors vs SPY",
                 color="white",fontsize=13,pad=12,fontweight="bold")
    return _fig_to_b64(fig)

def chart_sector_scores(sector_results):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0a0f1e"); ax.set_facecolor("#111827")
    rev   = list(reversed(sector_results))
    labels = [f"{r['etf']} — {r['name']}" for r in rev]
    scores = [r["score"] for r in rev]
    colors = [Q_COLORS.get(r["quad"],"#aaaaaa") for r in rev]
    ax.barh(labels,scores,color=colors,alpha=0.85,height=0.65)
    ax.axvline(12,color="#10b981",lw=1.2,ls="--",alpha=0.6,label="Rotating ≥12")
    ax.axvline(9, color="#f59e0b",lw=1.2,ls="--",alpha=0.6,label="Watch ≥9")
    ax.axvline(6, color="#3b82f6",lw=1.2,ls="--",alpha=0.6,label="Watch ≥6")
    for i,sc in enumerate(scores):
        ax.text(sc+0.1,i,f" {sc}",va="center",color="white",fontsize=8)
    ax.set_xlim(0,18); ax.set_xlabel("Score (max ~15)",color="#9ca3af")
    ax.tick_params(colors="#9ca3af")
    for sp in ax.spines.values(): sp.set_edgecolor("#374151")
    ax.legend(facecolor="#111827",labelcolor="white",fontsize=9)
    ax.set_title("Sector Rotation Scores",color="white",fontsize=13,
                 pad=12,fontweight="bold")
    return _fig_to_b64(fig)

def chart_stock_scores(drilldown, top_n=25):
    all_stocks = [{**s,"etf":etf} for etf,rows in drilldown.items() for s in rows]
    all_stocks.sort(key=lambda x: x["score"],reverse=True)
    top = all_stocks[:top_n]
    if not top:
        return None
    fig,ax = plt.subplots(figsize=(11,max(6,len(top)*0.42)))
    fig.patch.set_facecolor("#0a0f1e"); ax.set_facecolor("#111827")
    rev    = list(reversed(top))
    labels = [f"{s['ticker']} ({s['etf']})" for s in rev]
    scores = [s["score"] for s in rev]
    rmap   = {"★★★ STRONG BUY":"#10b981","★★  BUY":"#3b82f6",
              "★   WATCH":"#f59e0b","AVOID":"#ef4444"}
    colors = [rmap.get(s["rec"].strip(),rmap.get(s["rec"],"#aaaaaa")) for s in rev]
    ax.barh(labels,scores,color=colors,alpha=0.85,height=0.65)
    ax.axvline(STOCK_STRONG_BUY,color="#10b981",lw=1.2,ls="--",
               alpha=0.6,label=f"Strong Buy ≥{STOCK_STRONG_BUY}")
    ax.axvline(STOCK_BUY,color="#3b82f6",lw=1.2,ls="--",
               alpha=0.6,label=f"Buy ≥{STOCK_BUY}")
    for i,sc in enumerate(scores):
        ax.text(sc+0.1,i,f" {sc}",va="center",color="white",fontsize=8)
    ax.set_xlim(0,20); ax.set_xlabel("Score (max ~16)",color="#9ca3af")
    ax.tick_params(colors="#9ca3af")
    for sp in ax.spines.values(): sp.set_edgecolor("#374151")
    ax.legend(facecolor="#111827",labelcolor="white",fontsize=9)
    ax.set_title(f"Top {top_n} Stock Picks — Rotating Sectors",
                 color="white",fontsize=13,pad=12,fontweight="bold")
    return _fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════
#  §7  HTML REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════

def _q_badge(q):
    c = {"Leading":"#10b981","Improving":"#3b82f6",
         "Weakening":"#f59e0b","Lagging":"#ef4444","N/A":"#6b7280"}.get(q,"#6b7280")
    return f'<span style="background:{c};color:#fff;padding:2px 7px;border-radius:4px;font-size:11px">{q}</span>'

def _alert_badge(a):
    c = {"ROTATING ★★★":"#10b981","ROTATING ★★":"#34d399",
         "WATCH ★":"#f59e0b","AVOID":"#ef4444"}.get(a,"#6b7280")
    return f'<span style="background:{c};color:#fff;padding:2px 7px;border-radius:4px;font-size:11px;white-space:nowrap">{a}</span>'

def _rec_badge(r):
    c = {"★★★ STRONG BUY":"#10b981","★★  BUY":"#3b82f6",
         "★   WATCH":"#f59e0b","    AVOID":"#ef4444"}.get(r,"#6b7280")
    return f'<span style="background:{c};color:#fff;padding:2px 7px;border-radius:4px;font-size:11px;white-space:nowrap">{r.strip()}</span>'

def _img_tag(b64):
    if b64 is None: return ""
    return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:8px;margin-top:8px">'

def _ret_color(v):
    if v is None: return "#9ca3af"
    return "#10b981" if v >= 0 else "#ef4444"

def _fmt_ret(v):
    if v is None: return "N/A"
    return f"+{v}%" if v >= 0 else f"{v}%"

def _qta_cell(qta):
    """Compact QTA backtest cell HTML."""
    if qta is None:
        return '<td style="color:#6b7280;font-size:11px">—</td>'
    avg20 = qta.get("avg_20d")
    hit20 = qta.get("hit_20d")
    n     = qta.get("n_entries", 0)
    days  = qta.get("days_in_leading")
    avg_c = _ret_color(avg20)
    parts = [f'<span style="color:#9ca3af">n={n}</span>']
    if avg20 is not None:
        parts.append(f'Avg 20d: <span style="color:{avg_c};font-weight:600">{_fmt_ret(avg20)}</span>')
    if hit20 is not None:
        hc = "#10b981" if hit20 >= 60 else "#f59e0b" if hit20 >= 45 else "#ef4444"
        parts.append(f'Hit: <span style="color:{hc};font-weight:600">{hit20}%</span>')
    if days is not None:
        parts.append(f'<span style="color:#60a5fa">{days}d in run</span>')
    return f'<td style="font-size:11px;line-height:1.7">{"<br>".join(parts)}</td>'

def _sfra_cell(sfra):
    """Compact SFRA backtest cell HTML."""
    if sfra is None:
        return '<td style="color:#6b7280;font-size:11px">—</td>'
    avg20       = sfra.get("avg_20d")
    hit20       = sfra.get("hit_20d")
    n           = sfra.get("n_signals", 0)
    days        = sfra.get("days_active")
    current_ret = sfra.get("current_ret")
    avg_c = _ret_color(avg20)
    parts = [f'<span style="color:#9ca3af">n={n} signals</span>']
    if avg20 is not None:
        parts.append(f'Hist avg 20d: <span style="color:{avg_c};font-weight:600">{_fmt_ret(avg20)}</span>')
    if hit20 is not None:
        hc = "#10b981" if hit20 >= 60 else "#f59e0b" if hit20 >= 45 else "#ef4444"
        parts.append(f'Hit rate: <span style="color:{hc};font-weight:600">{hit20}%</span>')
    if days is not None and current_ret is not None:
        cr_c = _ret_color(current_ret)
        # Context: how far through the historical avg move are we already?
        if avg20 is not None and avg20 != 0:
            pct_of_avg = current_ret / avg20 * 100
            if pct_of_avg > 120:
                ctx = ' <span style="color:#f59e0b;font-size:10px">⚠ extended vs hist avg</span>'
            elif pct_of_avg < 20 and days <= 20:
                ctx = ' <span style="color:#10b981;font-size:10px">✓ room to run</span>'
            else:
                ctx = ''
        else:
            ctx = ''
        parts.append(
            f'Since signal ({days}d ago): '
            f'<span style="color:{cr_c};font-weight:700">{_fmt_ret(current_ret)}</span>{ctx}'
        )
    elif days is not None:
        parts.append(f'<span style="color:#60a5fa">Signal: {days}d old, return N/A</span>')
    return f'<td style="font-size:11px;line-height:1.8">{"<br>".join(parts)}</td>'

def _sector_table_rows(sector_results):
    html = ""
    for r in sector_results:
        html += f"""
        <tr>
          <td><strong>{r['etf']}</strong></td>
          <td style="color:#9ca3af">{r['name']}</td>
          <td>{_alert_badge(r['alert'])}</td>
          <td><strong>{r['score']}</strong></td>
          <td>{_q_badge(r['quad'])}</td>
          <td style="font-size:11px">{r['rs_ratio']:.2f}</td>
          <td style="font-size:11px">{r['rs_mom']:.2f}</td>
          <td>{'🟢' if r['st_signal']=='BUY' else '⚪'} {r['st_signal']}</td>
          <td>{r['4f_alert']} ({r['4f_n']}/4)</td>
          <td style="color:{_ret_color(r['ret_1y'])}">{_fmt_ret(r['ret_1y'])}</td>
          {_qta_cell(r.get('qta'))}
          <td style="font-size:11px;color:#9ca3af;max-width:260px">{r['reason']}</td>
        </tr>"""
    return html

def _stock_table_rows(stock_rows):
    html = ""
    for s in stock_rows:
        name_part = f'<br><span style="color:#9ca3af;font-size:10px">{s["name"]}</span>' if s["name"] else ""
        html += f"""
        <tr class="stock-row" data-ticker="{s['ticker'].lower()}" data-name="{s['name'].lower()}">
          <td><strong>{s['ticker']}</strong>{name_part}</td>
          <td>{_rec_badge(s['rec'])}</td>
          <td><strong>{s['score']}</strong></td>
          <td>{_q_badge(s['quad_spy'])}</td>
          <td>{_q_badge(s['quad_sec'])}</td>
          <td>{'🟢' if s['st_signal']=='BUY' else '⚪'} {s['st_signal']}</td>
          <td>{s['4f_alert']} ({s['4f_n']}/4)</td>
          <td style="color:#9ca3af">{s['vol_bucket']} ({s['ann_vol']}%)</td>
          <td style="color:{_ret_color(s['ret_1y'])}">{_fmt_ret(s['ret_1y'])}</td>
          {_sfra_cell(s.get('sfra'))}
          <td style="font-size:11px;color:#9ca3af;max-width:280px">{s['reason']}</td>
        </tr>"""
    return html

def _rrg_chartjs_data(sector_results):
    by_quad = {"Leading":[],"Weakening":[],"Improving":[],"Lagging":[]}
    for r in sector_results:
        by_quad[r["quad"]].append({"x":r["rs_ratio"],"y":r["rs_mom"],"label":r["etf"]})
    colors = {"Leading":"rgba(16,185,129,0.85)","Weakening":"rgba(245,158,11,0.85)",
              "Improving":"rgba(59,130,246,0.85)","Lagging":"rgba(239,68,68,0.85)"}
    return json.dumps([{"label":q,"data":pts,"backgroundColor":colors[q],
                        "pointRadius":8,"pointHoverRadius":11}
                       for q,pts in by_quad.items() if pts])

def _build_signal_summary(drilldown):
    """
    Scan all BUY-rated stocks across drilldown and bucket into 4 groups:
      1. New Signals     — signal fired ≤ 10 trading days ago
      2. Room to Run     — current return < 40% of historical avg (still early)
      3. Extended        — current return > 120% of historical avg (move mostly done)
      4. Best Quality    — hit rate ≥ 65% AND historical avg 20d ≥ 2%
    Returns HTML string for a summary card.
    """
    new_signals, room_to_run, extended, best_quality = [], [], [], []

    for etf, rows in drilldown.items():
        for s in rows:
            if "AVOID" in s["rec"].strip():
                continue
            sfra = s.get("sfra") or {}
            days = sfra.get("days_active")
            cur  = sfra.get("current_ret")
            avg  = sfra.get("avg_20d")
            hit  = sfra.get("hit_20d")
            entry = dict(ticker=s["ticker"], name=s["name"], etf=etf,
                         rec=s["rec"].strip(), score=s["score"],
                         days=days, cur=cur, avg=avg, hit=hit)

            if days is not None and days <= 10:
                new_signals.append(entry)

            if (days is not None and cur is not None and avg is not None
                    and days <= 30 and avg > 0.3 and cur < avg * 0.4):
                room_to_run.append(entry)

            if (cur is not None and avg is not None and avg > 0
                    and cur / avg > 1.2 and cur > 5):
                extended.append(entry)

            if hit is not None and avg is not None and hit >= 65 and avg >= 2.0:
                best_quality.append(entry)

    new_signals.sort(key=lambda x: x["days"] or 999)
    room_to_run.sort(key=lambda x: (x["cur"] or 0) - (x["avg"] or 0))
    extended.sort(key=lambda x: -(x["cur"] or 0))
    best_quality.sort(key=lambda x: -(x["hit"] or 0) * 10 - (x["avg"] or 0))

    def chip(e, bucket):
        rec_c = ("#10b981" if "STRONG" in e["rec"] else
                 "#3b82f6" if "BUY" in e["rec"] else "#f59e0b")
        if bucket == "new":
            sub = f"{e['days']}d old"
            sub2 = (f"now {e['cur']:+.1f}% vs hist avg {e['avg']:+.1f}%"
                    if e['cur'] is not None and e['avg'] is not None else "")
        elif bucket == "room":
            sub  = f"now {e['cur']:+.1f}%"
            sub2 = f"hist avg {e['avg']:+.1f}% → {round((e['avg'] or 0)-(e['cur'] or 0),1):+.1f}% left"
        elif bucket == "ext":
            sub  = f"now {e['cur']:+.1f}%"
            sub2 = f"hist avg was {e['avg']:+.1f}%"
        else:  # quality
            sub  = f"Hit {e['hit']}%"
            sub2 = f"avg 20d {e['avg']:+.1f}%"
        return f"""
        <div style="background:#1a2235;border:1px solid #1e3a5f;border-radius:8px;
                    padding:10px 13px;min-width:155px">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
            <strong style="color:#f3f4f6;font-size:13px">{e['ticker']}</strong>
            <span style="background:{rec_c};color:#fff;font-size:9px;padding:1px 5px;
                         border-radius:4px;white-space:nowrap">{e['rec'].replace('★','').strip()}</span>
            <span style="color:#6b7280;font-size:10px">{e['etf']}</span>
          </div>
          <div style="color:#9ca3af;font-size:11px">{e['name']}</div>
          <div style="color:#e5e7eb;font-size:12px;margin-top:4px;font-weight:600">{sub}</div>
          <div style="color:#6b7280;font-size:10px;margin-top:1px">{sub2}</div>
        </div>"""

    def bucket_section(title, icon, color, items, bucket, empty_msg):
        chips = "".join(chip(e, bucket) for e in items[:8])
        count_badge = (f'<span style="background:{color}22;color:{color};font-size:11px;'
                       f'padding:2px 8px;border-radius:10px;font-weight:600">{len(items)}</span>')
        content = (f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px">{chips}</div>'
                   if items else
                   f'<div style="color:#6b7280;font-size:12px;margin-top:10px">{empty_msg}</div>')
        return f"""
      <div style="flex:1;min-width:260px;background:#111827;border:1px solid #1e3a5f;
                  border-radius:10px;padding:14px 16px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px">
          <span style="font-size:16px">{icon}</span>
          <span style="font-weight:700;color:#f3f4f6;font-size:13px">{title}</span>
          {count_badge}
        </div>
        <div style="color:#6b7280;font-size:11px">{_bucket_desc(bucket)}</div>
        {content}
      </div>"""

    return f"""
    <div style="background:#0d1526;border:1px solid #1e3a5f;border-radius:12px;
                padding:18px 20px;margin-bottom:20px">
      <div style="font-size:14px;font-weight:700;color:#f3f4f6;margin-bottom:14px">
        🎯 Signal Action Summary
        <span style="color:#6b7280;font-size:12px;font-weight:400;margin-left:8px">
          — stocks worth acting on today, across all rotating sectors</span>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:12px">
        {bucket_section("New Signals","🆕","#3b82f6", new_signals,"new","No fresh signals today")}
        {bucket_section("Room to Run","✅","#10b981", room_to_run,"room","No stocks early-stage today")}
        {bucket_section("Extended","⚠️","#f59e0b", extended,"ext","No extended moves today")}
        {bucket_section("Best Quality","🏆","#a78bfa", best_quality,"quality","None meet quality threshold")}
      </div>
    </div>"""

def _bucket_desc(b):
    return {
        "new":   "Signal fired ≤ 10 days ago — fresh entry window",
        "room":  "Current gain &lt; 40% of hist avg — move still early",
        "ext":   "Current gain &gt; 120% of hist avg — consider waiting",
        "quality":"Hit rate ≥ 65% &amp; avg 20d return ≥ 2% historically",
    }.get(b, "")

def generate_html_report(sector_results, drilldown, run_dt, data_as_of,
                         img_rrg=None, img_scores=None, img_stocks=None):
    n_rotating   = sum(1 for r in sector_results if "ROTATING" in r["alert"])
    n_improving  = sum(1 for r in sector_results if r["quad"]=="Improving" and "ROTATING" not in r["alert"])
    n_weakening  = sum(1 for r in sector_results if r["quad"]=="Weakening")
    n_lagging    = sum(1 for r in sector_results if r["quad"]=="Lagging")
    rotating     = [r for r in sector_results if r["score"] >= SECTOR_DRILL_THRESHOLD]
    signal_summary_html = _build_signal_summary(drilldown)

    # Build full-universe lookup map (all stocks, not just rotating sectors)
    _stock_map = {}
    for etf, tickers in STOCK_UNIVERSE.items():
        sec_info = next((r for r in sector_results if r["etf"] == etf), None)
        for ticker in tickers:
            _stock_map[ticker.upper()] = {
                "name":        COMPANY_NAMES.get(ticker, ticker),
                "sector_etf":  etf,
                "sector_name": SECTOR_ETFS.get(etf, etf),
                "rotating":    bool(sec_info and sec_info["score"] >= SECTOR_DRILL_THRESHOLD),
                "quad":        sec_info["quad"]  if sec_info else "Unknown",
                "score":       sec_info["score"] if sec_info else 0,
                "alert":       sec_info["alert"] if sec_info else "N/A",
            }
    stock_map_json = json.dumps(_stock_map)

    drill_html = ""
    for sec in rotating:
        etf  = sec["etf"]
        if etf not in drilldown or not drilldown[etf]:
            continue
        rows   = drilldown[etf]
        buys   = [s["ticker"] for s in rows if "BUY" in s["rec"]]
        avoids = [s["ticker"] for s in rows if "AVOID" in s["rec"].strip()]
        # SFRA summary for the sector card header
        sfra_vals = [s["sfra"] for s in rows if s.get("sfra")]
        avg_hit   = round(np.mean([s["hit_20d"] for s in sfra_vals if "hit_20d" in s]),1) \
                    if sfra_vals else None
        avg_ret   = round(np.mean([s["avg_20d"] for s in sfra_vals if "avg_20d" in s]),2) \
                    if sfra_vals else None
        sfra_summary = ""
        if avg_hit is not None:
            sfra_summary = (f'<span style="color:#9ca3af;font-size:12px">'
                            f'SFRA avg across picks — Hit Rate 20d: '
                            f'<strong style="color:{"#10b981" if avg_hit>=60 else "#f59e0b"}">'
                            f'{avg_hit}%</strong> | Avg 20d Return: '
                            f'<strong style="color:{"#10b981" if avg_ret and avg_ret>=0 else "#ef4444"}">'
                            f'{_fmt_ret(avg_ret)}</strong></span>')
        drill_html += f"""
        <div class="card" style="margin-bottom:16px">
          <div class="card-toggle" onclick="toggleCard('card-{etf}')"
               style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;
                      padding:4px 2px;margin-bottom:10px">
            <h3 style="margin:0;color:#f3f4f6">{etf} — {sec['name']}</h3>
            {_alert_badge(sec['alert'])}
            <span style="color:#9ca3af;font-size:12px">Score: {sec['score']} | RRG: {sec['quad']}</span>
            <span style="color:#9ca3af;font-size:12px;margin-left:4px">
              BUY: <strong style="color:#10b981">{len(buys)}</strong> &nbsp;
              Avoid: <strong style="color:#ef4444">{len(avoids)}</strong> &nbsp;
              Top: <strong style="color:#3b82f6">{', '.join(buys[:4]) or '—'}</strong>
            </span>
            {sfra_summary}
            <span class="card-chevron" id="chev-card-{etf}">▼</span>
          </div>
          <div class="collapsible-body" id="card-{etf}">
          <div class="table-wrap">
          <table>
            <thead><tr>
              <th>Ticker</th><th>Recommendation</th><th>Score</th>
              <th>RRG vs SPY</th><th>RRG vs Sector</th>
              <th>StealthTrail</th><th>4-Factor</th>
              <th>Volatility</th><th>1Y Return</th>
              <th>SFRA Backtest <span title="Signal Forward Return Analysis: historical forward returns after each StealthTrail BUY signal">ⓘ</span></th>
              <th>Why</th>
            </tr></thead>
            <tbody>{_stock_table_rows(rows)}</tbody>
          </table>
          </div>
          </div>
        </div>"""

    rrg_data = _rrg_chartjs_data(sector_results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Market Report — {run_dt}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0f1e;color:#e5e7eb;padding:0}}
  .header{{background:linear-gradient(135deg,#1e3a5f 0%,#0f2027 100%);
           padding:28px 36px;border-bottom:1px solid #1e3a5f}}
  .header h1{{font-size:26px;color:#f9fafb}}
  .header .sub{{color:#9ca3af;font-size:13px;margin-top:6px}}
  .container{{max-width:1500px;margin:0 auto;padding:28px 24px}}
  .section-title{{font-size:18px;font-weight:700;color:#f3f4f6;margin:32px 0 14px;
                  padding-bottom:8px;border-bottom:2px solid #1e3a5f;
                  display:flex;align-items:center;gap:10px}}
  .section-title .pill{{background:#1e3a5f;color:#93c5fd;padding:2px 10px;
                        border-radius:12px;font-size:12px;font-weight:600}}
  .card{{background:#111827;border:1px solid #1e3a5f;border-radius:12px;padding:20px}}
  .stats-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}
  .stat-card{{background:#111827;border:1px solid #1e3a5f;border-radius:10px;
              padding:16px;text-align:center}}
  .stat-val{{font-size:32px;font-weight:800}}
  .stat-label{{color:#9ca3af;font-size:12px;margin-top:4px}}
  .charts-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}}
  .table-wrap{{overflow-x:auto}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#1f2937;color:#9ca3af;padding:10px 12px;text-align:left;
      font-weight:600;white-space:nowrap}}
  td{{padding:9px 12px;border-bottom:1px solid #1f2937;vertical-align:top}}
  tr:hover td{{background:#1a2335}}
  .mini-stat{{border-radius:8px;padding:10px 16px;min-width:110px}}
  .mini-label{{color:#9ca3af;font-size:11px;text-transform:uppercase;letter-spacing:.5px}}
  .mini-val{{font-size:18px;font-weight:700;margin-top:4px}}
  [title]{{cursor:help}}
  /* ── Collapsible sections ── */
  .collapsible-toggle{{cursor:pointer;user-select:none}}
  .collapsible-toggle:hover .section-chevron{{color:#60a5fa}}
  .section-chevron{{margin-left:auto;font-size:16px;color:#6b7280;
                   transition:transform 0.25s ease;display:inline-block}}
  .section-chevron.closed{{transform:rotate(-90deg)}}
  .collapsible-body{{overflow:hidden;transition:max-height 0.35s ease,
                    opacity 0.25s ease;max-height:9999px;opacity:1}}
  .collapsible-body.closed{{max-height:0 !important;opacity:0}}
  .card-toggle{{cursor:pointer;user-select:none;border-radius:8px}}
  .card-toggle:hover{{background:rgba(255,255,255,0.03)}}
  .card-chevron{{margin-left:auto;font-size:14px;color:#6b7280;
                transition:transform 0.2s ease;flex-shrink:0}}
  .card-chevron.closed{{transform:rotate(-90deg)}}
  @media(max-width:900px){{.stats-grid{{grid-template-columns:repeat(2,1fr)}}
    .charts-grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="header">
  <h1>📊 Daily Market Report</h1>
  <div class="sub">Generated: {run_dt} &nbsp;|&nbsp; Data as of: {data_as_of}
    &nbsp;|&nbsp; Benchmark: {BENCHMARK} &nbsp;|&nbsp; Sectors tracked: {len(sector_results)}
  </div>
</div>
<div class="container">

<!-- Quick Stats -->
<div class="stats-grid">
  <div class="stat-card"><div class="stat-val" style="color:#10b981">{n_rotating}</div>
    <div class="stat-label">Rotating Sectors</div></div>
  <div class="stat-card"><div class="stat-val" style="color:#3b82f6">{n_improving}</div>
    <div class="stat-label">Improving (Watch)</div></div>
  <div class="stat-card"><div class="stat-val" style="color:#f59e0b">{n_weakening}</div>
    <div class="stat-label">Weakening</div></div>
  <div class="stat-card"><div class="stat-val" style="color:#ef4444">{n_lagging}</div>
    <div class="stat-label">Lagging / Avoid</div></div>
</div>

<!-- §1 Sector Rotation Alerts -->
<div class="section-title collapsible-toggle" onclick="toggleSection('s1')">
  §1 &nbsp; Sector Rotation Alerts
  <span class="pill">{len(sector_results)} sectors</span>
  <span class="section-chevron" id="chev-s1">▼</span>
</div>
<div class="collapsible-body" id="s1">
<div class="card">
  <div class="table-wrap">
  <table>
    <thead><tr>
      <th>ETF</th><th>Sector</th><th>Alert</th><th>Score</th>
      <th>RRG Quadrant</th><th>RS-Ratio</th><th>RS-Mom</th>
      <th>StealthTrail</th><th>4-Factor</th><th>1Y Return</th>
      <th>QTA Backtest <span title="Quadrant Transition Analysis: forward returns after sector enters Leading quadrant">ⓘ</span></th>
      <th>Reason</th>
    </tr></thead>
    <tbody>{_sector_table_rows(sector_results)}</tbody>
  </table>
  </div>
</div>
</div>

<!-- §2 Live RRG Snapshot -->
<div class="section-title">§2 &nbsp; Live RRG Snapshot
  <span class="pill">RS-Ratio vs RS-Momentum</span></div>
<div class="charts-grid">
  <div class="card">
    <p style="color:#9ca3af;font-size:12px;margin-bottom:12px">
      X = RS-Ratio (&gt;100 = beating SPY) &nbsp;|&nbsp;
      Y = RS-Momentum (&gt;100 = accelerating) &nbsp;|&nbsp; Hover for values
    </p>
    <div style="position:relative;height:380px"><canvas id="rrgChart"></canvas></div>
  </div>
  <div class="card">{_img_tag(img_scores)}</div>
</div>

<!-- §3 Stock Drill-Down -->
<div class="section-title collapsible-toggle" onclick="toggleSection('s3')">
  §3 &nbsp; Stock Drill-Down
  <span class="pill">Rotating sectors only — 3-layer filter + SFRA backtest</span>
  <span class="section-chevron" id="chev-s3">▼</span>
</div>
<div class="collapsible-body" id="s3">

<!-- Stock Search Box -->
<div style="background:#111827;border:1px solid #1e3a5f;border-radius:12px;
            padding:16px 20px;margin-bottom:20px">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
    <span style="font-size:14px;color:#9ca3af;white-space:nowrap">🔍 Look up a stock:</span>
    <input id="stock-search" type="text"
           placeholder="Type ticker or company name… e.g. OXY, Netflix, Valero"
           oninput="searchStock()"
           style="flex:1;min-width:240px;background:#1f2937;border:1px solid #374151;
                  border-radius:8px;padding:9px 14px;color:#f3f4f6;font-size:14px;
                  outline:none;transition:border-color 0.2s"
           onfocus="this.style.borderColor='#3b82f6'"
           onblur="this.style.borderColor='#374151'">
    <button onclick="clearSearch()"
            style="background:#1f2937;border:1px solid #374151;border-radius:8px;
                   padding:9px 16px;color:#9ca3af;font-size:13px;cursor:pointer;
                   white-space:nowrap">✕ Clear</button>
  </div>
  <div id="search-result-box" style="display:none;margin-top:14px"></div>
</div>

{signal_summary_html}

<p style="color:#9ca3af;font-size:12px;margin-bottom:16px">
  <strong style="color:#d1d5db">SFRA (Signal Forward Return Analysis):</strong>
  For each stock, we scanned its full history and found every past StealthTrail BUY signal.
  We then measured how much the stock rose over the next 5 / 10 / 20 / 30 trading days after each signal.
  <em>Hit Rate</em> = % of past signals where the 20-day return was positive.
  <em>Avg 20d</em> = average 20-day forward return across all past signals.
  This tells you how <strong>reliable</strong> a signal has been for this specific stock historically.
</p>
{drill_html if drill_html else '<div class="card" style="color:#9ca3af">No rotating sectors above drill-down threshold.</div>'}
</div>

<!-- §4 Visualisations -->
<div class="section-title">§4 &nbsp; Visualisations</div>
<div class="charts-grid">
  <div class="card">
    <h4 style="color:#d1d5db;margin-bottom:8px">RRG Scatter (Static)</h4>
    {_img_tag(img_rrg)}
  </div>
  <div class="card">
    <h4 style="color:#d1d5db;margin-bottom:8px">Top Stock Picks — Score Chart</h4>
    {_img_tag(img_stocks)}
  </div>
</div>

<!-- How to read -->
<div class="card" style="margin-top:24px;border-color:#374151">
  <h4 style="color:#d1d5db;margin-bottom:12px">📖 How to Read This Report</h4>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;
              color:#9ca3af;font-size:13px">
    <div><strong style="color:#10b981">ROTATING ★★★ (≥12)</strong><br>
      Leading SPY + StealthTrail active + 3–4 confirmation signals. Act now.</div>
    <div><strong style="color:#34d399">ROTATING ★★ (9–11)</strong><br>
      Improving vs SPY with 2+ signals. Enter smaller, add on confirmation.</div>
    <div><strong style="color:#f59e0b">WATCH ★ (6–8)</strong><br>
      Early rotation — not yet fully confirmed. Set price alerts.</div>
    <div><strong style="color:#ef4444">AVOID (&lt;6)</strong><br>
      Lagging or Weakening vs SPY. No new positions.</div>
    <div><strong style="color:#d1d5db">3-Layer Stock Filter</strong><br>
      1. RRG vs SPY → beating market?<br>
      2. RRG vs Sector → best in sector?<br>
      3. StealthTrail + 4-Factor → entry confirmed?</div>
    <div><strong style="color:#d1d5db">Backtest Columns</strong><br>
      <em>QTA</em>: Quadrant Transition Analysis — after sector enters Leading.<br>
      <em>SFRA</em>: Signal Forward Return Analysis — after stock's BUY trigger.<br>
      Both show <em>n</em> (past signals), Avg 20d return, Hit Rate.</div>
  </div>
</div>
</div>

<script>
/* ── Full stock universe map (all sectors, injected from Python) ── */
const STOCK_MAP = {stock_map_json};

/* ── Stock Search ── */
function searchStock() {{
  const raw = document.getElementById('stock-search').value.trim().toLowerCase();
  const box = document.getElementById('search-result-box');

  if (!raw) {{ clearSearch(); return; }}

  let matches = [];
  document.querySelectorAll('#s3 .stock-row').forEach(row => {{
    const t = row.dataset.ticker || '';
    const n = row.dataset.name   || '';
    const hit = t === raw || t.startsWith(raw) || n.includes(raw);
    row.style.display = hit ? '' : 'none';
    if (hit) matches.push(row);
  }});

  /* expand / collapse sector cards based on visible rows */
  document.querySelectorAll('#s3 [id^="card-"]').forEach(body => {{
    const hasVisible = Array.from(body.querySelectorAll('.stock-row'))
                            .some(r => r.style.display !== 'none');
    body.classList.toggle('closed', !hasVisible);
    const chev = document.getElementById('chev-' + body.id);
    if (chev) chev.classList.toggle('closed', !hasVisible);
  }});

  if (matches.length === 0) {{
    /* Fall back to full universe map */
    const query = raw.toUpperCase();
    let foundKey = null, foundInfo = null;
    for (const [ticker, info] of Object.entries(STOCK_MAP)) {{
      if (ticker === query || ticker.startsWith(query) || info.name.toLowerCase().includes(raw)) {{
        foundKey = ticker; foundInfo = info; break;
      }}
    }}
    if (foundInfo) {{
      const rotating  = foundInfo.rotating;
      const statusBg  = rotating ? '#10b981' : '#6b7280';
      const statusLbl = rotating ? 'ROTATING ★' : 'NOT ROTATING TODAY';
      const quadColor = foundInfo.quad === 'Leading'   ? '#10b981'
                      : foundInfo.quad === 'Improving' ? '#3b82f6'
                      : foundInfo.quad === 'Weakening' ? '#f59e0b' : '#ef4444';
      box.innerHTML = `
        <div style="border-top:1px solid #1e3a5f;padding-top:14px">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px">
            <span style="font-size:20px;font-weight:800;color:#f3f4f6">${{foundKey}}</span>
            <span style="color:#9ca3af;font-size:13px">${{foundInfo.name}}</span>
            <span style="background:${{statusBg}};color:#fff;padding:3px 10px;border-radius:6px;
                         font-size:12px;font-weight:600">${{statusLbl}}</span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px">
            <div style="background:#1f2937;border-radius:8px;padding:10px">
              <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:.5px">Sector ETF</div>
              <div style="color:#f3f4f6;font-weight:600;margin-top:5px">${{foundInfo.sector_etf}}</div>
              <div style="color:#9ca3af;font-size:11px;margin-top:2px">${{foundInfo.sector_name}}</div>
            </div>
            <div style="background:#1f2937;border-radius:8px;padding:10px">
              <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:.5px">RRG Quadrant</div>
              <div style="color:${{quadColor}};font-weight:600;margin-top:5px">${{foundInfo.quad}}</div>
            </div>
            <div style="background:#1f2937;border-radius:8px;padding:10px">
              <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:.5px">Sector Score</div>
              <div style="color:#f3f4f6;font-weight:600;margin-top:5px">${{foundInfo.score}} / 16</div>
            </div>
            <div style="background:#1f2937;border-radius:8px;padding:10px">
              <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:.5px">Alert</div>
              <div style="color:#f3f4f6;font-weight:600;margin-top:5px">${{foundInfo.alert}}</div>
            </div>
          </div>
          ${{!rotating ? `<div style="margin-top:12px;padding:10px 14px;background:#1f2937;border-radius:8px;
              color:#f59e0b;font-size:12px;line-height:1.6">
              ⚠ <strong>${{foundKey}}</strong> is in <strong>${{foundInfo.sector_etf}} (${{foundInfo.sector_name}})</strong>
              which is currently in <strong style="color:${{quadColor}}">${{foundInfo.quad}}</strong> quadrant
              and not meeting the rotation threshold today. Individual stock drill-down is only run
              for rotating sectors. Check back when ${{foundInfo.sector_etf}} enters Leading/Improving.
            </div>` : ''}}
        </div>`;
    }} else {{
      box.innerHTML = `<div style="color:#ef4444;font-size:13px;padding:4px 0">
        "<strong>${{raw}}</strong>" is not in our stock universe.</div>`;
    }}
  }} else if (matches.length === 1) {{
    box.innerHTML = buildStockCard(matches[0]);
    matches[0].scrollIntoView({{behavior:'smooth', block:'center'}});
    matches[0].style.outline = '2px solid #3b82f6';
    matches[0].style.background = '#1e3358';
    setTimeout(() => {{
      matches[0].style.outline = '';
      matches[0].style.background = '';
    }}, 2500);
  }} else {{
    box.innerHTML = `<div style="color:#9ca3af;font-size:13px;padding:4px 0">
      <strong style="color:#f3f4f6">${{matches.length}}</strong> stocks match
      "<strong style="color:#f3f4f6">${{raw}}</strong>" — results highlighted below.</div>`;
    matches.forEach(r => {{
      r.style.outline = '2px solid #3b82f6';
      setTimeout(() => r.style.outline = '', 3000);
    }});
  }}
  box.style.display = 'block';
}}

function clearSearch() {{
  document.getElementById('stock-search').value = '';
  document.getElementById('search-result-box').style.display = 'none';
  document.getElementById('search-result-box').innerHTML = '';
  document.querySelectorAll('#s3 .stock-row').forEach(r => {{
    r.style.display = ''; r.style.outline = ''; r.style.background = '';
  }});
  document.querySelectorAll('#s3 [id^="card-"]').forEach(body => {{
    body.classList.remove('closed');
    const chev = document.getElementById('chev-' + body.id);
    if (chev) chev.classList.remove('closed');
  }});
}}

function buildStockCard(row) {{
  const c      = row.cells;
  const ticker = row.dataset.ticker.toUpperCase();
  const name   = row.dataset.name;
  const rec    = c[1].innerText.trim();
  const score  = c[2].innerText.trim();
  const rrgSpy = c[3].innerText.trim();
  const rrgSec = c[4].innerText.trim();
  const st     = c[5].innerText.trim();
  const ff     = c[6].innerText.trim();
  const vol    = c[7].innerText.trim();
  const ret1y  = c[8].innerText.trim();
  const sfra   = c[9].innerHTML;
  const why    = c[10].innerText.trim();
  const rc     = rec.includes('STRONG') ? '#10b981'
               : rec.includes('BUY')    ? '#3b82f6'
               : rec.includes('WATCH')  ? '#f59e0b' : '#ef4444';
  const sectorCard = row.closest('[id^="card-"]');
  const sectorLabel = sectorCard ? sectorCard.previousElementSibling
                                             .querySelector('h3')?.innerText || '' : '';
  return `
  <div style="border-top:1px solid #1e3a5f;padding-top:14px">
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px">
      <span style="font-size:20px;font-weight:800;color:#f3f4f6">${{ticker}}</span>
      <span style="color:#9ca3af;font-size:13px">${{name}}</span>
      <span style="background:${{rc}};color:#fff;padding:3px 10px;border-radius:6px;
                   font-size:12px;font-weight:600">${{rec}}</span>
      <span style="color:#9ca3af;font-size:12px">Score: <strong style="color:#f3f4f6">${{score}}/16</strong></span>
      ${{sectorLabel ? `<span style="color:#6b7280;font-size:12px">in ${{sectorLabel}}</span>` : ''}}
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-bottom:12px">
      ${{[['RRG vs SPY',rrgSpy],['RRG vs Sector',rrgSec],['StealthTrail',st],
          ['4-Factor',ff],['Volatility',vol],['1Y Return',ret1y]].map(([lbl,val]) =>
        `<div style="background:#1f2937;border-radius:8px;padding:10px">
           <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:.5px">${{lbl}}</div>
           <div style="color:#f3f4f6;font-weight:600;margin-top:5px;font-size:13px">${{val}}</div>
         </div>`).join('')}}
    </div>
    <div style="background:#1f2937;border-radius:8px;padding:12px;margin-bottom:10px">
      <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">
        SFRA Backtest — Signal Forward Return Analysis</div>
      <div style="font-size:13px">${{sfra}}</div>
    </div>
    <div style="background:#1f2937;border-radius:8px;padding:12px">
      <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">Why this recommendation</div>
      <div style="color:#d1d5db;font-size:13px">${{why}}</div>
    </div>
  </div>`;
}}

/* ── Collapsible helpers ── */
function toggleSection(id) {{
  const body  = document.getElementById(id);
  const chev  = document.getElementById('chev-' + id);
  const open  = !body.classList.contains('closed');
  body.classList.toggle('closed', open);
  chev.classList.toggle('closed', open);
}}
function toggleCard(id) {{
  const body  = document.getElementById(id);
  const chev  = document.getElementById('chev-' + id);
  const open  = !body.classList.contains('closed');
  body.classList.toggle('closed', open);
  if (chev) chev.classList.toggle('closed', open);
}}

Chart.register(ChartDataLabels);
const rrg_data = {rrg_data};
const ctx = document.getElementById('rrgChart').getContext('2d');
const quadrantPlugin = {{
  id:'quadrantBg',
  beforeDraw(chart){{
    const {{ctx,scales:{{x,y}}}} = chart;
    const x100=x.getPixelForValue(100),y100=y.getPixelForValue(100);
    const {{left,right,top,bottom}} = chart.chartArea;
    [[x100,right,top,y100,'rgba(16,185,129,0.07)'],
     [x100,right,y100,bottom,'rgba(245,158,11,0.07)'],
     [left,x100,top,y100,'rgba(59,130,246,0.07)'],
     [left,x100,y100,bottom,'rgba(239,68,68,0.07)']
    ].forEach(([x1,x2,y1,y2,col])=>{{
      ctx.fillStyle=col; ctx.fillRect(x1,y1,x2-x1,y2-y1);
    }});
    ctx.save(); ctx.strokeStyle='rgba(255,255,255,0.25)';
    ctx.lineWidth=1; ctx.setLineDash([4,4]);
    ctx.beginPath();ctx.moveTo(x100,top);ctx.lineTo(x100,bottom);ctx.stroke();
    ctx.beginPath();ctx.moveTo(left,y100);ctx.lineTo(right,y100);ctx.stroke();
    ctx.restore();
  }}
}};
new Chart(ctx,{{
  type:'scatter', data:{{datasets:rrg_data}},
  options:{{
    responsive:true, maintainAspectRatio:false, animation:{{duration:600}},
    plugins:{{
      legend:{{labels:{{color:'#9ca3af',font:{{size:11}}}}}},
      tooltip:{{
        callbacks:{{label:(ctx)=>`${{ctx.raw.label}}: RS-Ratio=${{ctx.raw.x.toFixed(2)}}, RS-Mom=${{ctx.raw.y.toFixed(2)}}`}},
        backgroundColor:'#1f2937',titleColor:'#f3f4f6',bodyColor:'#d1d5db',
      }},
      datalabels:{{color:'#ffffff',font:{{size:10,weight:'bold'}},
        formatter:(val)=>val.label,anchor:'end',align:'top',offset:2}}
    }},
    scales:{{
      x:{{title:{{display:true,text:'RS-Ratio (>100 = outperforming SPY)',
          color:'#9ca3af',font:{{size:11}}}},
         grid:{{color:'rgba(255,255,255,0.05)'}},ticks:{{color:'#9ca3af'}}}},
      y:{{title:{{display:true,text:'RS-Momentum (>100 = accelerating)',
          color:'#9ca3af',font:{{size:11}}}},
         grid:{{color:'rgba(255,255,255,0.05)'}},ticks:{{color:'#9ca3af'}}}}
    }}
  }}, plugins:[quadrantPlugin]
}});
</script>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════
#  §8  CONSOLE OUTPUT
# ══════════════════════════════════════════════════════════════════

def print_console_report(sector_results, drilldown, run_dt, data_as_of):
    W = 80
    print("\n" + "═"*W)
    print(f"  MARKET REPORT — {run_dt}  |  Data as of: {data_as_of}")
    print("═"*W)
    print("\n  §1  SECTOR ROTATION ALERTS")
    print("  " + "─"*(W-2))
    rotating = [r for r in sector_results if r["score"] >= SECTOR_DRILL_THRESHOLD]
    avoiding = [r for r in sector_results if r["score"] < SECTOR_DRILL_THRESHOLD]
    if rotating:
        print("  ROTATING / WATCH:")
        for r in rotating:
            qta    = r.get("qta") or {}
            avg20  = f"{qta.get('avg_20d',0):+.1f}%" if "avg_20d" in qta else "  N/A "
            hit20  = f"{qta.get('hit_20d',0):.0f}%" if "hit_20d" in qta else "N/A"
            days   = f"{qta.get('days_in_leading')}d" if qta.get("days_in_leading") else "—"
            print(f"    {r['alert']:15s}  {r['etf']:5s}  {r['name']:22s}  "
                  f"Score={r['score']:2d}  {r['quad']:10s}  "
                  f"QTA: avg20={avg20} hit={hit20} run={days}")
    if avoiding:
        print("\n  AVOID:")
        for r in avoiding:
            print(f"    {'AVOID':15s}  {r['etf']:5s}  {r['name']:22s}  "
                  f"Score={r['score']:2d}  {r['quad']:10s}  1Y={_fmt_ret(r['ret_1y'])}")
    print(f"\n  §2  LIVE RRG SNAPSHOT (as of {data_as_of})")
    print("  " + "─"*(W-2))
    print(f"  {'ETF':6s}  {'Sector':24s}  {'Quadrant':12s}  {'RS-Ratio':9s}  {'RS-Mom':9s}")
    for r in sector_results:
        print(f"  {r['etf']:6s}  {r['name']:24s}  {r['quad']:12s}  "
              f"{r['rs_ratio']:9.3f}  {r['rs_mom']:9.3f}")
    print(f"\n  §3  STOCK PICKS — ROTATING SECTORS")
    print("  " + "─"*(W-2))
    for sec in rotating:
        etf = sec["etf"]
        if etf not in drilldown or not drilldown[etf]:
            continue
        print(f"\n  ─── {etf} {sec['name']} [{sec['alert']}] ───")
        for s in drilldown[etf]:
            sfra    = s.get("sfra") or {}
            avg20   = f"{sfra.get('avg_20d',0):+.1f}%" if "avg_20d" in sfra else " N/A"
            hit20   = f"{sfra.get('hit_20d',0):.0f}%"  if "hit_20d" in sfra else "N/A"
            cur_ret = sfra.get("current_ret")
            days    = sfra.get("days_active")
            cur_str = f"  now={cur_ret:+.1f}% ({days}d)" if cur_ret is not None else ""
            name    = f" ({s['name']})" if s["name"] else ""
            print(f"    {s['ticker']:6s}{name:28s}  {s['rec']:20s}  "
                  f"Score={s['score']:2d}  SFRA: avg20={avg20} hit={hit20}{cur_str}")
    print("\n" + "═"*W)


# ══════════════════════════════════════════════════════════════════
#  §9  MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def main():
    run_dt = datetime.now().strftime("%a %b %d, %Y  %H:%M")
    print("\n" + "═"*60)
    print("  STEP 1 — Refreshing market data …")
    print("═"*60)
    refresh_data(verbose=True)
    spy_df = load_df(BENCHMARK)
    if spy_df is None:
        print("ERROR: Could not load SPY data."); return
    data_as_of = str(spy_df.index[-1].date())

    print("  STEP 2 — Running sector rotation engine + QTA backtests …")
    sector_results = run_sector_rotation(spy_df)

    rotating_sectors = [r for r in sector_results if r["score"] >= SECTOR_DRILL_THRESHOLD]
    print(f"  STEP 3 — Drilling into {len(rotating_sectors)} rotating sector(s) + SFRA backtests …")
    drilldown = run_stock_drilldown(rotating_sectors, spy_df)

    print("  STEP 4 — Generating charts …")
    img_rrg    = chart_rrg_scatter(sector_results)
    img_scores = chart_sector_scores(sector_results)
    img_stocks = chart_stock_scores(drilldown)

    print_console_report(sector_results, drilldown, run_dt, data_as_of)

    print("\n  STEP 5 — Generating HTML report …")
    html     = generate_html_report(sector_results, drilldown, run_dt, data_as_of,
                                    img_rrg, img_scores, img_stocks)
    out_path = "market_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓  Saved → {out_path}")
    print(f"  Open: file://{os.path.abspath(out_path)}\n")

if __name__ == "__main__":
    main()
