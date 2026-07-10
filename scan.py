#!/usr/bin/env python3
"""
Unified Swing Trading Scanner
──────────────────────────────
Runs all five strategy scanners against a shared universe, merges results,
displays them grouped by strategy, and lets you add any ticker to the tracker.

Usage:
  python3 scan.py                                          # all strategies + backtest
  python3 scan.py --no-backtest                            # signals only (faster)
  python3 scan.py --strategies momentum,breakout           # subset of strategies
  python3 scan.py --strategies pocket_pivot,connors_rsi2,ema_ribbon --no-backtest

Available strategies: momentum, breakout, pocket_pivot, connors_rsi2, ema_ribbon
"""

import sys, os, time, json, warnings as _warnings, contextlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

HERE = Path(__file__).parent
LAST_SCAN_JSON = HERE / "last_scan.json"

# ── SECTOR → ETF MAP ──────────────────────────────────────────────────────────
SECTOR_ETF = {
    "Technology":             "XLK",
    "Industrials":            "XLI",
    "Healthcare":             "XLV",
    "Financial Services":     "XLF",
    "Consumer Cyclical":      "XLY",
    "Consumer Defensive":     "XLP",
    "Basic Materials":        "XLB",
    "Energy":                 "XLE",
    "Communication Services": "XLC",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
}
_ETF_SHORT = {v: k[:5] for k, v in SECTOR_ETF.items()}  # XLK → "Techn"


def _fetch_sector_pulse() -> tuple:
    """Return (excess_dict, spy_10d_ret). excess = ETF 10d return minus SPY 10d return."""
    if not _HAS_YF:
        return {}, 0.0
    _warnings.filterwarnings("ignore")
    syms = ["SPY"] + list(SECTOR_ETF.values())
    rets = {}
    for sym in syms:
        try:
            with contextlib.suppress(Exception):
                df = yf.download(sym, period="20d", interval="1d",
                                 progress=False, auto_adjust=True, threads=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                c = df["Close"].dropna()
                if len(c) >= 11:
                    rets[sym] = float((c.iloc[-1] - c.iloc[-11]) / c.iloc[-11] * 100)
        except Exception:
            pass
    spy = rets.get("SPY", 0.0)
    excess = {sym: round(v - spy, 2) for sym, v in rets.items() if sym != "SPY"}
    return excess, round(spy, 2)


def _print_sector_pulse(excess: dict, spy_ret: float):
    """Print compact sector heat strip above scan output."""
    if not excess:
        return
    ranked = sorted(excess.items(), key=lambda x: -x[1])
    parts = []
    for etf, ex in ranked:
        label = _ETF_SHORT.get(etf, etf)
        s = f"{label}({etf}) {'+' if ex >= 0 else ''}{ex:.1f}%"
        parts.append(GRN(s) if ex >= 1.0 else (RED(s) if ex <= -1.0 else DIM(s)))
    spy_s = f"SPY 10d: {'+' if spy_ret >= 0 else ''}{spy_ret:.1f}%"
    print()
    print(BOLD(f"  📊  SECTOR PULSE  ·  {spy_s}  ·  vs SPY:"))
    print("    " + "   ".join(parts))
    # Build underperforming set for caller use
    return {etf for etf, ex in excess.items() if ex <= -1.5}


def _thematic_check():
    """Read trades.csv → show open & recent closed position concentration by sector."""
    trades_path = HERE / "trades.csv"
    if not trades_path.exists():
        return
    try:
        df = pd.read_csv(trades_path)
        open_t   = df[df["status"] == "OPEN"] if "status" in df.columns else pd.DataFrame()
        if not open_t.empty:
            sectors = open_t["sector"].dropna().value_counts() if "sector" in open_t.columns else pd.Series()
            total   = len(open_t)
            print()
            print(BOLD(f"  💼  OPEN POSITIONS ({total}) by sector:"))
            print("    " + "   ".join(f"{s}:{c}" for s, c in sectors.items()))
            for sec, cnt in sectors.items():
                if cnt / total > 0.40 and total >= 3:
                    print(YLW(f"    ⚠  {sec} at {cnt/total:.0%} of book — new {sec} signals are HIGH-RISK"))
    except Exception:
        pass

# ── IMPORTS ───────────────────────────────────────────────────────────────────
from momentum_scanner       import build_universe, compute_bench_returns  # scan disabled: 0% WR
from breakout_scanner       import scan as scan_breakout
from pocket_pivot_scanner   import scan as scan_pocket_pivot
from connors_rsi2_scanner   import scan as scan_connors
from ema_ribbon_scanner     import scan as scan_ema_ribbon
from nr7_scanner             import scan as scan_nr7
# bb_squeeze_scanner — disabled: WR 33.3%, high variance
# high_tight_flag_scanner — disabled: 0% WR
# stage4_short_scanner — disabled: tracking inverted, true WR=11.4% (L020)
from analyst_upgrade_scanner import scan as scan_analyst_upgrade
from signal_velocity_scanner      import scan as scan_signal_velocity
from chokepoint_inflection_scanner import scan as scan_chokepoint
from defensive_rotation_scanner    import scan as scan_defensive_rotation
from cup_handle_scanner            import scan as scan_cup_handle
from power_earnings_gap_scanner    import scan as scan_peg
from darvas_box_scanner            import scan as scan_darvas
from rs_line_scanner               import scan as scan_rs_line
from vcp_scanner                   import scan as scan_vcp
from elder_impulse_scanner         import scan as scan_elder
from raschke_holy_grail_scanner    import scan as scan_holy_grail
from connors_3down_scanner         import scan as scan_3down
from williams_pct_r_scanner        import scan as scan_williams_r
from bollinger_pctb_scanner        import scan as scan_bb_pctb
from connors_r3_scanner            import scan as scan_r3
from connors_tps_scanner           import scan as scan_tps
from turtle_soup_scanner           import scan as scan_turtle_soup
from raschke_8020_scanner          import scan as scan_8020
from wyckoff_spring_scanner        import scan as scan_wyckoff_spring
from weinstein_stage2_scanner      import scan as scan_weinstein_stage2
from show_tracker                  import (add_trade_interactive, load_trades, save_trades,
                                           next_id, ticker_ccy, fetch_fx_on_date,
                                           fetch_company_name, fetch_sector, fetch_market_regime,
                                           trade_stop_loss, trade_hold_days, biz_days_add,
                                           _yf_ticker as _yf_sym)
from scanner_utils                 import _quiet as _quiet_ctx

# ANSI helpers (inline — no shared module dependency)
import os as _os
_color = sys.stdout.isatty() and not _os.environ.get("NO_COLOR")
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _color else text
GRN  = lambda t: _c("32", t);  YLW  = lambda t: _c("33", t)
CYN  = lambda t: _c("36", t);  BOLD = lambda t: _c("1",  t)
DIM  = lambda t: _c("2",  t);  RED  = lambda t: _c("31", t)

def ret_fmt(v):
    if v is None: return DIM("   ─  ")
    return GRN(f"+{v:.2f}%") if v >= 0 else RED(f"{v:.2f}%")

def wr_fmt(v):
    if v is None: return DIM("  ─ ")
    return GRN(f"{v:.0f}%") if v >= 60 else (YLW(f"{v:.0f}%") if v >= 45 else RED(f"{v:.0f}%"))


# ── STRATEGY REGISTRY ─────────────────────────────────────────────────────────
ALL_STRATEGIES = ["breakout", "pocket_pivot", "connors_rsi2",
                  "ema_ribbon", "nr7", "high_tight_flag",
                  "analyst_upgrade", "signal_velocity", "chokepoint_inflection",
                  "stage4_short", "defensive_rotation",
                  "cup_handle", "power_earnings_gap",
                  "darvas_box", "rs_line", "vcp", "elder_impulse",
                  "holy_grail", "connors_3down", "williams_pct_r", "bollinger_pctb",
                  "connors_r3", "connors_tps", "turtle_soup", "raschke_8020",
                  "wyckoff_spring", "weinstein_stage2"]

SCANNER_MAP = {
    # "momentum":        scan_momentum,  # DISABLED: 0% WR, avg -4.40% across 5 signals (Jun 30 backtest)
    "breakout":        scan_breakout,
    "pocket_pivot":    scan_pocket_pivot,
    "connors_rsi2":    scan_connors,
    "ema_ribbon":      scan_ema_ribbon,
    "nr7":             scan_nr7,
    # "bb_squeeze": scan_bb_squeeze,   # DISABLED: WR 33.3% avg +0.91% n=21 — high variance, avoid
    # "high_tight_flag": scan_htf,   # DISABLED: 0% WR, avg -5.05% across 5 signals (backtest)

    "analyst_upgrade":      scan_analyst_upgrade,
    "signal_velocity":      scan_signal_velocity,
    "chokepoint_inflection": scan_chokepoint,
    # "stage4_short": scan_stage4_short,  # DISABLED: tracking inverted — true WR=11.4% (L020)
    "defensive_rotation":    scan_defensive_rotation,
    "cup_handle":            scan_cup_handle,
    "power_earnings_gap":    scan_peg,
    "darvas_box":            scan_darvas,
    "rs_line":               scan_rs_line,
    "vcp":                   scan_vcp,
    "elder_impulse":         scan_elder,
    "holy_grail":            scan_holy_grail,
    "connors_3down":         scan_3down,
    "williams_pct_r":        scan_williams_r,
    "bollinger_pctb":        scan_bb_pctb,
    "connors_r3":            scan_r3,
    "connors_tps":           scan_tps,
    "turtle_soup":           scan_turtle_soup,
    "raschke_8020":          scan_8020,
    "wyckoff_spring":        scan_wyckoff_spring,
    "weinstein_stage2":      scan_weinstein_stage2,
}

STRATEGY_LABELS = {
    "momentum":       "🟢  MOMENTUM  (O'Neil / IBD crossover signals)",
    "breakout":       "🔭  BREAKOUT  (VCP / coil pre-breakout)",
    "pocket_pivot":   "🟠  POCKET PIVOT  (Morales & Kacher)",
    "connors_rsi2":   "🔵  CONNORS RSI(2)  (mean reversion in uptrend)",
    "ema_ribbon":     "🟣  EMA RIBBON  (8/13/21/34/55 expansion pullback)",
    "nr7":             "⚡  NR7  (Toby Crabel — narrowest range compression)",
    "bb_squeeze":      "🔲  BB SQUEEZE  (TTM Squeeze — Bollinger / John Carter)",
    "high_tight_flag": "🚀  HIGH TIGHT FLAG  (Minervini / O'Neil — pole + flag)",
    "analyst_upgrade": "📊  ANALYST UPGRADE  (≥3 firms upgrade in 5 days, tier-1 required)",
    "signal_velocity":       "⚙️   SIGNAL VELOCITY  (TV-style indicator convergence acceleration)",
    "chokepoint_inflection": "🌐  CHOKEPOINT INFLECTION  (macro event → commodity spike → correlated stock lag)",
    "stage4_short":          "🔻  STAGE 4 SHORT  (Weinstein/Minervini — confirmed distribution, failed rally entry)",
    "defensive_rotation":    "🛡️   DEFENSIVE ROTATION  (Faber — sector ETF outperforms SPY >3% + accelerating → stock leaders)",
    "cup_handle":            "☕  CUP & HANDLE  (O'Neil / IBD — rounded base + tight handle at pivot)",
    "power_earnings_gap":    "⚡  POWER EARNINGS GAP  (Gil Morales — 8%+ gap on earnings, 2× volume, gap held)",
    "darvas_box":            "📦  DARVAS BOX  (Nicolas Darvas — 52w high → tight box → volume breakout)",
    "rs_line":               "📈  RS LINE NEW HIGH  (O'Neil/IBD — RS line vs SPY makes new 52w high before price)",
    "vcp":                   "🌀  VCP  (Minervini — ≥3 volatility contractions, each tighter, volume drying)",
    "elder_impulse":         "💚  ELDER IMPULSE  (Alexander Elder — EMA13 + MACD-hist both rising = green bar)",
    "holy_grail":            "🏆  HOLY GRAIL  (Raschke — ADX peaked >30, pullback to EMA20, bounce)",
    "connors_3down":         "📉  CONNORS 3-DOWN  (Connors — 3 consecutive lower closes in uptrend, RSI2<20)",
    "williams_pct_r":        "📊  WILLIAMS %R  (Larry Williams — %R crosses above -80 from oversold)",
    "bollinger_pctb":        "🎯  BOLLINGER %B  (John Bollinger — %B<0.20 + MFI<35 + bouncing, sideways specialist)",
    "connors_r3":            "🔁  CONNORS R3  (Connors — RSI2 drops 3 days from <60, RSI2<10, mean reversion)",
    "connors_tps":           "📐  CONNORS TPS  (Connors — Time/Price Scale-In, 3-7 lower closes, RSI2<25)",
    "turtle_soup":           "🐢  TURTLE SOUP  (Raschke — false 20d low breakdown → reversal close above)",
    "raschke_8020":          "🔄  RASCHKE 80-20  (Raschke — open bottom-20% yesterday's range, close top-50%)",
    "wyckoff_spring":        "🌱  WYCKOFF SPRING  (Wyckoff — false break below support on low vol → shakeout, markup begins)",
    "weinstein_stage2":      "📈  WEINSTEIN STAGE 2  (Weinstein — 30-week MA turns up, price crosses above it on volume surge)",
}

STRATEGY_DESCRIPTIONS = {
    "momentum": (
        "Finds stocks that have JUST entered momentum — MACD, RSI(14), and EMA9/21 crossovers\n"
        "  must have fired within the last 3 bars. Requires ADX≥22 (trend present) and\n"
        "  Minervini Trend Template ≥6/8 (healthy structure). Best used when SPY is in BULL\n"
        "  regime. Hold 5 days. Source: IBD / William O'Neil CANSLIM methodology."
    ),
    "breakout": (
        "Finds stocks COILING before a breakout — VCP (Volatility Contraction Pattern).\n"
        "  Looks for: price range tightening, volume drying up, price near pivot high,\n"
        "  ADX curling up from a low base. BREAK phase = volume/price confirms it's starting.\n"
        "  Enter before the crowd notices. Hold 5 days. Source: Mark Minervini SEPA."
    ),
    "pocket_pivot": (
        "Fires when today's UP-day volume exceeds the HIGHEST down-day volume in the prior\n"
        "  10 sessions — a sign that institutions are quietly accumulating before a move.\n"
        "  Earlier signal than a full breakout; stock must be in a base (not extended).\n"
        "  Hold 7 days. Source: Gil Morales & Chris Kacher ('Trade Like an O'Neil Disciple')."
    ),
    "connors_rsi2": (
        "Counter-trend dip-buyer in an uptrend. RSI(2) drops below 10 (deeply oversold\n"
        "  short-term) while price stays above its 200-day SMA (long-term uptrend intact).\n"
        "  Edge disappears after day 5 — exit when RSI(2) recovers above 65, not on a fixed\n"
        "  calendar date. Hold 5 days. Source: Larry Connors, 'Short Term Trading Strategies\n"
        "  That Work' — one of the most statistically verified short-term strategies."
    ),
    "ema_ribbon": (
        "Trend-following re-entry. EMAs 8/13/21/34/55 must be perfectly stacked AND the gap\n"
        "  between EMA8 and EMA55 must be WIDENING (trend accelerating). Price pulls back to\n"
        "  touch the 8-EMA then closes above it — optimal low-risk entry back into the trend.\n"
        "  Hold 7 days. Used by SMB Capital, Warrior Trading, and quantitative CTAs."
    ),
    "nr7": (
        "Today's high-low range is the NARROWEST of the last 7 days — maximum volatility\n"
        "  compression. Compression precedes expansion. Price above 50d SMA for trend context.\n"
        "  Very short hold (3 days) — just riding the volatility burst. Works in any market.\n"
        "  Source: Toby Crabel, 'Day Trading with Short Term Price Patterns' (1990)."
    ),
    "bb_squeeze": (
        "Bollinger Bands collapse INSIDE Keltner Channels (TTM Squeeze) — the market is\n"
        "  coiling. Fire = squeeze releases + MACD histogram turns positive (momentum direction).\n"
        "  Outperforms in sideways/low-volatility markets where momentum strategies struggle.\n"
        "  Hold 7 days. Source: John Carter 'Mastering the Trade', Larry Connors BB research."
    ),
    "high_tight_flag": (
        "Rare, extreme momentum setup. Stock surges ≥90% in ≤8 weeks (the 'pole'), then\n"
        "  consolidates ≤25% from the peak in a tight flag. Enter when still within 15% of\n"
        "  the 8-week high. O'Neil called this the most powerful pattern in bull markets.\n"
        "  Hold 10 days. Source: William O'Neil, Mark Minervini — high-conviction bull signal."
    ),
    "analyst_upgrade": (
        "Coordinated re-rating signal. Fires when ≥3 distinct analyst firms upgrade a stock\n"
        "  to Buy/Overweight/Outperform within 5 trading days, with ≥1 from a tier-1 firm\n"
        "  (GS, MS, JPM, BofA, Citi, Barclays, etc). Rejects earnings pile-ons (gap-up >5%)\n"
        "  and saturated coverage (>75% already buy). Uses yfinance recommendations API.\n"
        "  Hold 7 days. Source: Womack (1996), Barber et al. (2001) — post-upgrade drift."
    ),
    "signal_velocity": (
        "Inflection point detector. Computes a TradingView-style indicator score (15 signals:\n"
        "  7 MAs + RSI + Stochastic + CCI + Williams%R + MACD + BBP + UO). Fires when the\n"
        "  net buy/sell score gains ≥6 points over 3 consecutive days — multiple indicators\n"
        "  flipping bullish simultaneously. Catches transitions BEFORE crossovers confirm.\n"
        "  Hold 5 days. Inspired by TradingView technical summary rate-of-change."
    ),
    "chokepoint_inflection": (
        "Macro event → commodity/cyclical spike → correlated stock lag detector.\n"
        "  Step 1: 11-ticker basket (crude, gas, copper, gold, wheat, aluminium, uranium,\n"
        "  metals, semiconductors, rare earths, dry bulk) fires when 5d return >4% AND\n"
        "  accelerating. Step 2: yfinance news headlines confirm macro keyword (war, sanctions,\n"
        "  shortage, strait, embargo, chip, rare earth…). Step 3: stocks with 60d rolling\n"
        "  correlation >0.55 to the commodity that have NOT yet moved (stock 5d ret <15% of\n"
        "  commodity move). Minervini ≥4 + price>SMA50 hard filters. Score = corr × lag gap.\n"
        "  Hold 5 days. Zero signals on calm days is correct — fires only on genuine macro events."
    ),
    "stage4_short": (
        "SHORT ONLY. Finds stocks in confirmed Stage 4 distribution (Weinstein) ready to\n"
        "  fall further. Hard filters: full bearish SMA stack (price<SMA50<SMA150<SMA200),\n"
        "  SMA200 declining, price ≤70% of 52w high, ADX≥20, market cap>$500M, no biotech,\n"
        "  no earnings within 5 days. Entry trigger: failed rally (stock bounced toward SMA\n"
        "  then got rejected) OR new 20-day low OR distribution cluster (3+ heavy vol down\n"
        "  days). Backtest return = short return (positive = stock fell as expected).\n"
        "  Hold 7 days. Works in bear AND bull markets — finds individual stock blow-ups."
    ),
    "defensive_rotation": (
        "Fires when institutional money rotates from growth into defensive sectors. Step 1:\n"
        "  XLU/XLP/XLV/GLD each checked — ETF must outperform SPY by >3% over 20 days AND\n"
        "  recent 5d outperformance must exceed prior 5d (accelerating). Step 2: within\n"
        "  confirmed rotating sectors, scans ~100 known defensive names for individual leaders\n"
        "  (stock must outperform its own sector ETF, price>SMA50, RSI 35–70). Zero signals\n"
        "  on most bull days — correct. Fires in late-cycle bear market transitions.\n"
        "  Hold 10 days. Source: Meb Faber 'GTAA', sector rotation academic literature."
    ),
    "cup_handle": (
        "Detects the classic Cup & Handle base pattern near the pivot breakout point.\n"
        "  Cup: 12–35% depth, 30–200 days duration, rounded bottom (low in middle 60% of\n"
        "  cup duration — filters V-bottoms), right lip within 5% of left lip. Handle:\n"
        "  5–25 days, ≤12% depth, sits in upper half of cup, volume drying up. Entry:\n"
        "  price within 3% below handle high (pivot). Minervini ≥5, price >SMA50+SMA200.\n"
        "  Hold 10 days. Source: William O'Neil 'How to Make Money in Stocks' — IBD."
    ),
    "power_earnings_gap": (
        "Stocks that gap ≥8% on earnings with 2× volume — institutional validation of\n"
        "  fundamentals. Three conditions must all hold: (1) Gap fired within last 5 days,\n"
        "  (2) price still above gap day's low (gap not filled = buyers defending), (3) stock\n"
        "  not extended >20% above gap close. Earnings verified via yfinance; unverified gaps\n"
        "  require 3× volume. Tagged EG✓ (confirmed) or EG~ (pattern-only).\n"
        "  Hold 10 days. Source: Gil Morales 'Power Earnings Gaps', IBD gap-up research."
    ),
    "darvas_box": (
        "Nicolas Darvas (1960) — stock makes new 52w high, consolidates in a tight box\n"
        "  (≤15% width) for ≥3 bars, then breaks above box top on volume ≥1.5× avg.\n"
        "  ADX 16-35, Minervini ≥5. Hold 5 days."
    ),
    "rs_line": (
        "O'Neil / IBD — Relative Strength line (stock / SPY) makes new 52w high.\n"
        "  Leading indicator: RS line new high before price = institutional accumulation.\n"
        "  ADX 16-35, price within 15% of 52w high, Minervini ≥5. Hold 7 days."
    ),
    "vcp": (
        "Minervini Volatility Contraction Pattern — ≥3 price contractions, each smaller\n"
        "  than the last, on drying volume. Final contraction ≤10%. Price near pivot.\n"
        "  ADX 16-35, Minervini ≥6. Hold 10 days. Source: 'Trade Like a Stock Market Wizard'."
    ),
    "elder_impulse": (
        "Alexander Elder — EMA(13) slope AND MACD histogram both rising = green bar.\n"
        "  Signal: 2 consecutive green bars (confirmed). ADX 16-35, RSI 45-75, Minervini ≥5.\n"
        "  SPY regime gate: suppressed in CHOPPY/BEAR market. Hold 5 days. Source: 'Come Into My Trading Room'."
    ),
    "holy_grail": (
        "Linda Bradford Raschke ('Street Smarts') — ADX(14) peaked above 30 in last 10 bars,\n"
        "  stock pulls back to EMA(20) for ≥2 bars (volume drying), then bounces. ADX floor 16,\n"
        "  RSI 40-65. Works in trending AND slowing markets. Hold 5 days."
    ),
    "connors_3down": (
        "Larry Connors ('Short-Term Trading Strategies That Work') — 3 consecutive lower closes\n"
        "  in a stock above 200d + 50d SMA. RSI(2) < 20 (short-term oversold). ADX 16-40.\n"
        "  Mean-reversion snap-back in any market. Hold 3 days."
    ),
    "williams_pct_r": (
        "Larry Williams ('Long-Term Secrets to Short-Term Trading') — %R drops below -80 then\n"
        "  crosses back above (oversold reversal). Stock above 50d + 200d SMA. ADX 16-40.\n"
        "  Works in sideways + mild uptrend. Hold 3 days."
    ),
    "bollinger_pctb": (
        "John Bollinger ('Bollinger on Bollinger Bands') — %B < 0.20 (price near lower band)\n"
        "  AND MFI < 35 (money flowing out) AND %B rising (bounce starting). Above 200d SMA.\n"
        "  ADX floor 12 — sideways market specialist. Hold 5 days."
    ),
    "connors_r3": (
        "Larry Connors ('High Probability ETF Trading' 2009) — RSI(2) drops 3 consecutive days,\n"
        "  first day from below 60 (not from overbought peak), final RSI(2) < 10. Price above\n"
        "  200d SMA. Pure mean reversion — strongest in sideways/choppy markets. Hold 3 days.\n"
        "  Backtested 90% win rate on SPY since 1993 per Connors Research."
    ),
    "connors_tps": (
        "Larry Connors ('High Probability ETF Trading' 2009) — Time/Price Scale-In strategy.\n"
        "  3-7 consecutive lower closes, RSI(2) declining each day, RSI(2) < 25 on entry.\n"
        "  Volume declining (orderly pullback, not panic). Price above 200d + 50d SMA.\n"
        "  Explicitly designed for choppy/sideways markets. Hold 4 days."
    ),
    "turtle_soup": (
        "Linda Raschke & Larry Connors ('Street Smarts' 1996) — Stock makes new 20-day low\n"
        "  (suckers short sellers in) but CLOSES back above the prior 20d low the same day.\n"
        "  False breakdown reversal. Volume confirms. Works in sideways markets where false\n"
        "  breakouts dominate. Price above 200d SMA. Hold 3 days."
    ),
    "raschke_8020": (
        "Linda Raschke ('Street Smarts' 1996) — Bullish 80-20 Rule. Stock opens in BOTTOM 20%\n"
        "  of yesterday's range (bearish gap/open) but closes ABOVE yesterday's midpoint.\n"
        "  Failed breakdown = next 1-2 days trend up. Works in choppy/sideways markets.\n"
        "  Pure price-action pattern, any ADX regime. Hold 2 days."
    ),
}

HOLD_DAYS_MAP = {
    "momentum":       5,
    "breakout":       5,
    "pocket_pivot":   7,
    "connors_rsi2":   5,
    "ema_ribbon":     7,
    "nr7":             3,
    "bb_squeeze":      7,
    "high_tight_flag": 10,
    "analyst_upgrade": 7,
    "signal_velocity":       5,
    "chokepoint_inflection": 5,
    "stage4_short":          7,
    "defensive_rotation":    10,
    "cup_handle":            10,
    "darvas_box":            5,
    "rs_line":               7,
    "vcp":                   10,
    "elder_impulse":         5,
    "holy_grail":            5,
    "connors_3down":         3,
    "williams_pct_r":        3,
    "bollinger_pctb":        5,
    "power_earnings_gap":    10,
    "connors_r3":            3,
    "connors_tps":           4,
    "turtle_soup":           3,
    "raschke_8020":          2,
    "wyckoff_spring":        5,
    "weinstein_stage2":      10,
}

W = 110


# ── CONVICTION TIER ───────────────────────────────────────────────────────────
# Criteria derived from scan_history backtest (285 rows, 4 scan dates):
#   HIGH  = strat_count≥2  AND score≤5  AND RSI 50-70  AND ADX 16-35
#   MED   = strat_count≥2  OR  (score≤5 AND RSI 50-70 AND ADX 16-35)
#   LOW   = everything else

def _conviction_tier(r: dict, multi_tickers: set) -> tuple:
    """Returns (tier_str, color_fn) for a result row."""
    is_multi  = bool(multi_tickers and r.get("ticker") in multi_tickers)
    score     = r.get("score", 99)
    rsi       = r.get("rsi",   0)
    adx       = r.get("adx",   0)
    good_score = score <= 5
    good_rsi   = 50 <= rsi <= 70
    good_adx   = 16 <= adx <= 35

    if is_multi and good_score and good_rsi and good_adx:
        return ("★★★ HIGH", GRN)
    if is_multi or (good_score and good_rsi and good_adx):
        return ("★★  MED ", YLW)
    return ("★   LOW ", DIM)


# PROVEN EDGE strategies (WR≥60%, n≥10 from scan_history backtest)
PROVEN_EDGE = {"pocket_pivot", "ema_ribbon", "cup_handle",
               "signal_velocity", "connors_rsi2"}
# stage4_short REMOVED from PROVEN_EDGE: tracking was inverted (ret_d5>0 = price UP = SHORT LOSS).
# True WR for stage4_short as a short strategy = 11.4% (9/79). Disabled scanner.

# Regime-strategy fit: which strategies thrive in which market regime
# Trend-following: need confirmed uptrend (BULL)
_TREND_STRATS     = {"ema_ribbon", "pocket_pivot", "cup_handle", "breakout",
                     "signal_velocity", "weinstein_stage2", "vcp", "power_earnings_gap"}
# Mean reversion: work in sideways/oversold conditions (NEUTRAL/BEAR)
_REVERSION_STRATS = {"connors_rsi2", "nr7", "wyckoff_spring", "raschke_8020",
                     "connors_3down", "bollinger_pctb"}


def _rank_score(r: dict, strats_fired: list, elder_count: int = 0) -> float:
    """
    Score each HIGH-conviction pick so we can label top 1/2/3.
    Higher = better. Criteria (max ~19 pts):
      +3  any PROVEN_EDGE strategy fired
      +1  ADX 20-35  (data: -3.1% WR delta, reduced from +2)
      +1  ADX 16-20 or 35-45
      +2  RSI 50-65  (momentum without overextension)
          RSI 65-70  REMOVED (data: -2.3% WR)
      +3  3+ strategies (data: 74% WR)
      +2  2 strategies  (data: 63% WR)
      +1  score ≤ 3
      +2  vol 1.5-2x   (data: 68.4% WR)
      +1  vol ≥ 2x     (data: 66.2% WR)
      +2  persistent 3+ scan dates (L013 — 61% vs 47% WR)
      +1  persistent 2 scan dates
      +1  RS positive vs SPY 10d
      +1  regime-strategy fit (trend strats in BULL, reversion strats in NEUTRAL)
    Calibrated via optimize_weights.py on 1119 signals (10 scan dates).
    """
    pts = 0.0
    if any(s in PROVEN_EDGE for s in strats_fired):
        pts += 3
    adx = r.get("adx", 0) or 0
    if 20 <= adx <= 35:
        pts += 1
    elif 16 <= adx < 20 or 35 < adx <= 45:
        pts += 1
    rsi = r.get("rsi", 0) or 0
    if 50 <= rsi <= 65:
        pts += 2
    n_strats = len(strats_fired)
    if n_strats >= 3:
        pts += 3
    elif n_strats == 2:
        pts += 2
    if (r.get("score") or 99) <= 3:
        pts += 1
    vr = r.get("vol_ratio", 0) or 0
    if 1.5 <= vr < 2.0:
        pts += 2
    elif vr >= 2.0:
        pts += 1
    days_seen = _PERSISTENCE.get(r.get("ticker", ""), 0)
    if days_seen >= 3:
        pts += 2
    elif days_seen >= 2:
        pts += 1
    if r.get("rs_vs_spy", 0) and float(r.get("rs_vs_spy", 0)) > 0:
        pts += 1
    # Regime fit: +1 if strategy type matches current market regime
    is_bull    = elder_count >= 15
    is_neutral = 5 <= elder_count < 15
    has_trend  = any(s in _TREND_STRATS     for s in strats_fired)
    has_revert = any(s in _REVERSION_STRATS for s in strats_fired)
    if (is_bull and has_trend) or (is_neutral and has_revert):
        pts += 1
    return pts


def _get_ticker_sector_tag(ticker: str, sector_excess: dict) -> str:
    """Return '🔥' if ticker's sector is top 3 vs SPY, '❄' if bottom 3, else ''."""
    if not sector_excess or not _HAS_YF:
        return ""
    try:
        info = yf.Ticker(_yf_sym(ticker)).fast_info
        sector = getattr(info, "sector", None) or ""
    except Exception:
        return ""
    etf = SECTOR_ETF.get(sector, "")
    ex = sector_excess.get(etf)
    if ex is None:
        return ""
    ranked = sorted(sector_excess.values(), reverse=True)
    if ex >= ranked[2]:   # top 3
        return GRN(" 🔥")
    if ex <= ranked[-3]:  # bottom 3
        return RED(" ❄")
    return ""


def _print_high_conviction(results_by_strategy: dict, multi_tickers: set, with_backtest: bool,
                           sector_excess: dict = None, elder_count: int = 0):
    """Lead with ACT ON THESE — make the actionable signal unmissable."""
    seen = {}
    tier_rank = {"★★★ HIGH": 0, "★★  MED ": 1, "★   LOW ": 2}

    for strategy, results in results_by_strategy.items():
        for r in results:
            t = r["ticker"]
            tier, _ = _conviction_tier(r, multi_tickers)
            rank = tier_rank[tier]
            if tier == "★   LOW ":
                continue
            if t not in seen or rank < seen[t][0]:
                seen[t] = (rank, r, strategy)

    if not seen:
        return

    picks = sorted(seen.values(), key=lambda x: (x[0], x[1].get("score", 99)))
    high_picks = [p for p in picks if p[0] == 0]
    med_picks  = [p for p in picks if p[0] == 1]

    # ── ACT ON THESE (HIGH conviction) ────────────────────────────────────────
    print()
    print("╔" + "═"*(W-2) + "╗")
    if high_picks:
        print("║" + GRN(BOLD(f"  🎯  ACT ON THESE  ·  {len(high_picks)} stock(s)  ·  ★★★ HIGH CONVICTION")).ljust(W+16) + "║")
    else:
        print("║" + YLW(BOLD(f"  👀  NO HIGH CONVICTION TODAY  —  see WATCHLIST below")).ljust(W+12) + "║")
    print("╚" + "═"*(W-2) + "╝")

    if high_picks:
        # Pre-compute strats_fired and rank scores for all HIGH picks
        enriched = []
        for _, r, strategy in high_picks:
            strats_fired = sorted(
                [s for s, res in results_by_strategy.items()
                 if any(x["ticker"] == r["ticker"] for x in res)],
                key=lambda s: -_HIST_STATS.get(s, {}).get("wr", 0)
            )
            rs = _rank_score(r, strats_fired, elder_count)
            enriched.append((rs, r, strategy, strats_fired))

        # Sort by rank score descending
        enriched.sort(key=lambda x: -x[0])

        # Assign medal labels to top 3
        medals = {0: GRN(BOLD(" #1 ")), 1: GRN(BOLD(" #2 ")), 2: YLW(BOLD(" #3 "))}

        # Header
        print(BOLD(f"  {'RANK':<4}  {'TICKER':<10}  {'COMPANY':<22}  {'STRATEGIES':<28}  {'HIST WR':>7}  {'SCORE':>5}  {'RSI':>5}  {'ADX':>5}  {'PRICE':>8}"))
        print("  " + "─"*(W-2))
        for idx, (rs, r, strategy, strats_fired) in enumerate(enriched):
            best_strat = next((s for s in strats_fired if _HIST_STATS.get(s,{}).get("n",0)>=5), strategy)
            proven_badge = " ✦PROVEN" if any(s in PROVEN_EDGE for s in strats_fired) else ""
            strat_str = "+".join(s.replace("_"," ").upper()[:6] for s in strats_fired[:3])
            hist = _HIST_STATS.get(best_strat, {})
            wr_s = f"{hist['wr']:.0f}%WR" if hist.get("n",0)>=5 else "─"
            wr_col = GRN(f"{wr_s:>7}") if hist.get("wr",0)>=60 else YLW(f"{wr_s:>7}")
            proven_s = GRN(proven_badge) if proven_badge else ""
            company = ""
            for s in strats_fired:
                for rx in results_by_strategy.get(s, []):
                    if rx["ticker"] == r["ticker"] and rx.get("company"):
                        company = str(rx["company"])[:22]; break
                if company: break
            rank_label = medals.get(idx, "     ")
            ticker_fmt = f"{r['ticker']:<10}"
            sec_tag = _get_ticker_sector_tag(r["ticker"], sector_excess or {})
            # Persistence badge
            days_seen = _PERSISTENCE.get(r["ticker"], 0)
            pers_badge = (GRN(" 🔁PERSIST") if days_seen >= 3
                          else (DIM(" 🔁x2") if days_seen >= 2 else ""))
            # Volume surge badge
            vr = r.get("vol_ratio", 0) or 0
            vol_badge = GRN(" ⚡VOL") if vr >= 2.0 else (YLW(" ⚡") if vr >= 1.5 else "")
            # RS badge
            rs_badge = GRN(" ↑RS") if r.get("rs_vs_spy", 0) and float(r.get("rs_vs_spy",0)) > 0 else ""
            print(f"  {rank_label}  {GRN(BOLD(ticker_fmt))}{sec_tag}  {company:<22}  {strat_str:<28}{proven_s}  {wr_col}  {r.get('score',0):>5}  {r.get('rsi',0):>5.1f}  {r.get('adx',0):>5.1f}  {r.get('price',0):>8.2f}{pers_badge}{vol_badge}{rs_badge}")
        print()

    # ── WATCHLIST (MED conviction) ─────────────────────────────────────────────
    if med_picks:
        print(BOLD(f"  👀  WATCHLIST  ·  {len(med_picks)} stock(s)  ·  ★★ MED — confirm before entering"))
        print("  " + "─"*(W-2))
        for _, r, strategy in med_picks[:15]:
            strats_fired = sorted(
                [s for s, res in results_by_strategy.items()
                 if any(x["ticker"] == r["ticker"] for x in res)],
                key=lambda s: -_HIST_STATS.get(s, {}).get("wr", 0)
            )
            best_strat = next((s for s in strats_fired if _HIST_STATS.get(s,{}).get("n",0)>=5), strategy)
            strat_str = "+".join(s.replace("_"," ").upper()[:5] for s in strats_fired[:2])
            proven_badge = " ✦" if any(s in PROVEN_EDGE for s in strats_fired) else ""
            hist = _HIST_STATS.get(best_strat, {})
            wr_s = f"{hist['wr']:.0f}%WR" if hist.get("n",0)>=5 else "─"
            ticker_fmt2 = f"{r['ticker']:<10}"
            print(f"  {YLW(BOLD(ticker_fmt2))}  {str(r.get('company','') or '')[:22]:<22}  {strat_str+proven_badge:<30}  {wr_s:>6}  score={r.get('score',0)}  RSI={r.get('rsi',0):.0f}  ADX={r.get('adx',0):.0f}")
        if len(med_picks) > 15:
            print(DIM(f"  ... and {len(med_picks)-15} more"))
        print()

    print(DIM(f"  {len(high_picks)} HIGH  ·  {len(med_picks)} MED  ·  Focus: HIGH only unless 2+ strategies confirmed"))
    print()




def _load_persistence_counts() -> dict:
    """Return {ticker: n_unique_scan_dates} from scan_history.csv — used for persistence badge."""
    import csv as _csv
    from collections import defaultdict
    from pathlib import Path as _Path
    p = _Path(__file__).parent / "scan_history.csv"
    if not p.exists(): return {}
    ticker_dates = defaultdict(set)
    try:
        with open(p, newline="") as f:
            for row in _csv.DictReader(f):
                t = row.get("ticker","").strip()
                sd = row.get("scan_date","").strip()
                if t and sd:
                    ticker_dates[t].add(sd)
    except Exception:
        return {}
    return {t: len(d) for t, d in ticker_dates.items()}

_PERSISTENCE: dict = {}  # {ticker: n_scan_dates} — loaded in main()

def _load_hist_stats() -> dict:
    import csv as _csv, math as _math
    from collections import defaultdict
    from pathlib import Path as _Path
    p = _Path(__file__).parent / "scan_history.csv"
    if not p.exists():
        return {}
    stats = defaultdict(lambda: {"wins": 0, "total": 0, "s": 0.0})
    try:
        with open(p, newline="") as f:
            for row in _csv.DictReader(f):
                strat = row.get("strategy", "").strip()
                if not strat: continue
                try:
                    ret = float(row["ret_d5"])
                    if _math.isnan(ret): continue
                    stats[strat]["total"] += 1
                    stats[strat]["s"] += ret
                    if ret > 0: stats[strat]["wins"] += 1
                except (ValueError, TypeError, KeyError): pass
    except Exception:
        return {}
    out = {}
    for s, d in stats.items():
        n = d["total"]
        if n < 1: continue
        out[s] = {"n": n, "wr": round(100*d["wins"]/n, 1), "avg": round(d["s"]/n, 2)}
    return out

_HIST_STATS: dict = {}

# ── DISPLAY ───────────────────────────────────────────────────────────────────

def _print_group(strategy: str, results: list, with_backtest: bool, multi_tickers: set = None):
    """Print a single strategy group. multi_tickers = tickers passing ≥2 strategies (promoted to top)."""
    label = STRATEGY_LABELS.get(strategy, strategy.upper())
    hold  = HOLD_DAYS_MAP.get(strategy, 5)
    count = len(results)
    desc  = STRATEGY_DESCRIPTIONS.get(strategy, "")

    h      = _HIST_STATS.get(strategy, {})
    h_n    = h.get("n", 0)
    h_wr   = h.get("wr", 0.0)
    h_avg  = h.get("avg", 0.0)
    if h_n >= 3:
        wr_col   = GRN if h_wr >= 60 else (YLW if h_wr >= 45 else RED)
        proven   = "  ★ PROVEN EDGE" if h_n >= 10 and h_wr >= 60 else ""
        wr_part  = wr_col(str(round(h_wr)) + "%WR")
        avg_sign = "+" if h_avg >= 0 else ""
        avg_part = avg_sign + str(round(h_avg, 2)) + "%avg"
        hist_tag = "  [" + wr_part + " · " + avg_part + " · n=" + str(h_n) + proven + "]"
    else:
        hist_tag = ""

    print()
    print("┌" + "─"*(W-2) + "┐")
    hdr_line = "  " + BOLD(label) + "  ·  " + str(count) + " signal(s)  ·  hold=" + str(hold) + "d" + hist_tag
    print("│" + hdr_line.ljust(W+20) + "│")
    if desc:
        for line in desc.split("\n"):
            print("│" + DIM("  " + line).ljust(W+8) + "│")
    print("└" + "─"*(W-2) + "┘")

    if not results:
        print(DIM("  (no signals)"))
        return

    print()
    if with_backtest:
        hdr = (f"  {'#':>3}  {'CV':<2}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}  "
               f"{'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'M':>3}  {'SCR':>3}  "
               f"{'#BT':>3}  {'WIN%':>5}  {'AVG':>7}  {'MED':>7}  SIGNALS")
    else:
        hdr = (f"  {'#':>3}  {'CV':<2}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}  "
               f"{'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'M':>3}  {'SCR':>3}  SIGNALS")
    print(BOLD(hdr))
    print("  " + "─"*(W-2))

    # Multi-strategy tickers always surface first (highest conviction).
    # Within tier: breakout sorts BREAK→COIL; others sort by WR then score.
    def _sort_key(r):
        multi_rank = 0 if (multi_tickers and r.get("ticker") in multi_tickers) else 1
        if strategy == "breakout":
            phase_rank = 0 if r.get("phase") == "BREAK" else 1
            return (multi_rank, phase_rank, -(r.get("wr") or 0), -r.get("score", 0))
        return (multi_rank, -(r.get("wr") or 0), -r.get("score", 0))

    sorted_r = sorted(results, key=_sort_key)

    _prev_phase = None
    for rank, r in enumerate(sorted_r[:50], 1):
        # Inject BREAK / COIL section header for breakout strategy
        cur_phase = r.get("phase")
        if strategy == "breakout" and cur_phase != _prev_phase:
            if cur_phase == "BREAK":
                print(f"\n  {GRN(BOLD('▶  BREAKING OUT NOW  —  volume + price confirming, act today'))}")
            elif cur_phase == "COIL" and _prev_phase is not None:
                print(f"\n  {YLW(BOLD('◎  COILING / WATCHLIST  —  setup building, do NOT trade yet'))}")
            elif cur_phase == "COIL":
                print(f"\n  {YLW(BOLD('◎  COILING / WATCHLIST  —  setup building, do NOT trade yet'))}")
            _prev_phase = cur_phase
        fresh_str = " ".join(r.get("fresh", []))
        conf_str  = ("  · " + " ".join(r.get("conf", []))) if r.get("conf") else ""
        sig_str   = CYN(fresh_str) + DIM(conf_str)
        mkt_s     = YLW(f"{r['mkt']:<3}") if r["mkt"] != "US" else DIM(f"{r['mkt']:<3}")
        ticker_s  = BOLD(f"{r['ticker']:<8}")
        tier, tier_color = _conviction_tier(r, multi_tickers)
        tier_badge = tier_color(tier[:2])   # ★★★ / ★★  / ★
        base = (f"  {rank:>3}  {tier_badge}  {mkt_s}  {ticker_s}  {r.get('price', 0):>9.2f}"
                f"  {r.get('rsi', 0):>5.1f}  {r.get('adx', 0):>5.1f}"
                f"  {r.get('vol_ratio', 0):>5.1f}"
                f"  {r.get('minervini', 0):>3}  {r.get('score', 0):>3}  ")
        if with_backtest and r.get("n", 0) > 0:
            row = (base + f"{r['n']:>3}  "
                   + f"{wr_fmt(r.get('wr')):>5}  "
                   + f"{ret_fmt(r.get('avg')):>7}  "
                   + f"{ret_fmt(r.get('med')):>7}  "
                   + sig_str)
        elif with_backtest:
            row = base + DIM("  ─     ─      ─      ─   ") + sig_str
        else:
            row = base + sig_str
        print(row)


def _print_header(strategies: list, total: int, with_backtest: bool):
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    strat_str = ", ".join(strategies)
    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + f"  UNIFIED SCANNER  ·  {now}  ·  {total} total signals  ·  backtest={'ON' if with_backtest else 'OFF'}".ljust(W-2) + "║")
    print("║" + f"  Strategies:  {strat_str}".ljust(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")


def _nan_safe(d: dict) -> dict:
    """Replace NaN/inf/numpy scalars so the dict is JSON-serialisable."""
    import math, numpy as np
    out = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = None if (math.isnan(v) or math.isinf(v)) else float(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


# ── CROSS-STRATEGY MATRIX ─────────────────────────────────────────────────────

def _print_matrix(results_by_strategy: dict, strategies: list, upgrade_tickers: set = None):
    """Compact multi-strategy view — only show tickers firing 2+ scanners."""
    all_tickers = {}
    for strat, results in results_by_strategy.items():
        for r in results:
            t = r["ticker"]
            if t not in all_tickers:
                all_tickers[t] = {}
            all_tickers[t][strat] = r

    # Only show multi-strategy tickers here — single-strategy are in group view
    multi = {t: m for t, m in all_tickers.items() if len(m) >= 2}
    if not multi:
        print()
        print(DIM("  No multi-strategy overlaps today."))
        return

    sorted_t = sorted(multi.items(), key=lambda kv: -len(kv[1]))

    print()
    print(BOLD(f"  ⚡  MULTI-STRATEGY OVERLAPS  ·  {len(multi)} ticker(s)  —  highest conviction pool"))
    print(BOLD(f"  {'STRATS':>6}  {'TICKER':<10}  {'COMPANY':<24}  {'FIRED ON':<45}  {'SCORE':>5}  {'RSI':>5}  {'ADX':>5}"))
    print("  " + "─"*(W-2))

    for ticker, strat_map in sorted_t[:30]:
        company = ""
        best_r = None
        best_score = 99
        for s, r in strat_map.items():
            if not company: company = r.get("company","") or ""
            sc = r.get("score", 99)
            if sc < best_score:
                best_score = sc
                best_r = r

        n_strats = len(strat_map)
        up_star  = "⭐" if (upgrade_tickers and ticker in upgrade_tickers) else ""
        strat_names = sorted(strat_map.keys(), key=lambda s: -_HIST_STATS.get(s,{}).get("wr",0))
        chips = "  ".join(
            (GRN if _HIST_STATS.get(s,{}).get("wr",0)>=60 else YLW)(
                f"[{s.replace('_',' ').upper()[:8]} {_HIST_STATS.get(s,{}).get('wr',0):.0f}%]"
                if _HIST_STATS.get(s,{}).get("n",0)>=5
                else f"[{s.replace('_',' ').upper()[:8]}]"
            )
            for s in strat_names[:4]
        )
        ticker_s = GRN(BOLD(f"{ticker+up_star:<10}")) if n_strats >= 3 else YLW(BOLD(f"{ticker+up_star:<10}"))
        r = best_r or {}
        print(f"  {n_strats:>6}  {ticker_s}  {str(company)[:24]:<24}  {chips:<45}  {r.get('score',0):>5}  {r.get('rsi',0):>5.1f}  {r.get('adx',0):>5.1f}")

    print()


# ── TRACKER PROMPT ────────────────────────────────────────────────────────────

def _tracker_prompt(all_results: list):
    """Interactive prompt to add tickers to the tracker."""
    # Build a lookup: ticker → strategy (last write wins if duplicated)
    ticker_to_strategy = {}
    for r in all_results:
        ticker_to_strategy[r["ticker"].upper()] = r.get("strategy", "momentum")

    print()
    print(BOLD("─" * W))
    print(BOLD("  Add to tracker?  Enter ticker(s) comma-separated (or Enter to skip):"))
    try:
        raw = input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not raw:
        return

    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    for ticker in tickers:
        strategy = ticker_to_strategy.get(ticker, "momentum")
        print(f"  Adding {BOLD(ticker)} ({strategy})...")
        try:
            add_trade_interactive(["--ticker", ticker, "--strategy", strategy])
        except Exception as e:
            print(f"  Error adding {ticker}: {e}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args_raw      = sys.argv[1:]
    with_backtest = "--no-backtest" not in args_raw

    # Parse --strategies flag
    strategies = list(ALL_STRATEGIES)  # default: all
    for arg in args_raw:
        if arg.startswith("--strategies="):
            strategies = [s.strip() for s in arg.split("=", 1)[1].split(",") if s.strip()]
        elif arg == "--strategies" and args_raw.index(arg) + 1 < len(args_raw):
            idx = args_raw.index(arg)
            strategies = [s.strip() for s in args_raw[idx+1].split(",") if s.strip()]

    # Validate
    invalid = [s for s in strategies if s not in ALL_STRATEGIES]
    if invalid:
        print(f"Unknown strategies: {invalid}")
        print(f"Available: {ALL_STRATEGIES}")
        sys.exit(1)

    # Friday warning — signals have 45% WR historically (vs Mon 65%, Wed 71%)
    is_friday = datetime.now().weekday() == 4
    if is_friday:
        print()
        print("╔" + "═"*(W-2) + "╗")
        print("║" + RED(BOLD("  ⚠  FRIDAY SCAN — historical WR=45% avg -0.31%  ·  nr7/connors/elder all <42% WR on Fri")).ljust(W+12) + "║")
        print("║" + YLW("  💡  Hold entry signals until Monday. Practice trades will NOT be auto-added today.").ljust(W+8) + "║")
        print("╚" + "═"*(W-2) + "╝")
        print()

    print(DIM(f"  Running: {', '.join(strategies)}  ·  backtest={'ON' if with_backtest else 'OFF'}"))
    print()

    # Build shared universe once
    t0 = time.time()
    universe     = build_universe()
    bench_returns = compute_bench_returns(set(universe.values()))

    bt_label = "backtest ON" if with_backtest else "backtest OFF"
    print(DIM(f"  Universe: {len(universe)} tickers  ·  {bt_label}"))
    print()

    # Run scanners sequentially (avoid yfinance rate limit from concurrent universe fetches)
    all_results = []
    results_by_strategy = {}

    for strategy in strategies:
        scanner = SCANNER_MAP.get(strategy)
        if scanner is None:
            continue  # disabled strategy (e.g. high_tight_flag)
        print(DIM(f"  [{strategy}] scanning..."), flush=True)
        t1 = time.time()
        try:
            res = scanner(universe, bench_returns, with_backtest)
        except Exception as e:
            print(f"  [{strategy}] ERROR: {e}")
            res = []
        elapsed = time.time() - t1
        print(DIM(f"  [{strategy}] done in {elapsed:.0f}s — {len(res)} signal(s)"))
        results_by_strategy[strategy] = res
        all_results.extend(res)

    # Score cap: score>=6 WR drops to 25-56%, avg turns negative (Jun 30 backtest, 620 trades)
    # Exception: signal_velocity uses 0-13 scale so cap doesn't apply
    SCORE_CAP_EXEMPT = {"signal_velocity"}
    for strat in results_by_strategy:
        if strat in SCORE_CAP_EXEMPT:
            continue
        results_by_strategy[strat] = [r for r in results_by_strategy[strat]
                                       if (r.get("score") or 0) <= 5]
    all_results = [r for r in all_results
                   if r.get("strategy") in SCORE_CAP_EXEMPT
                   or (r.get("score") or 0) <= 5]

    # RSI floor: drop RSI<50 for non-mean-reversion strategies (dead zone confirmed WR 48%)
    RSI_FLOOR_EXEMPT = {"connors_rsi2", "connors_3down", "connors_r3", "connors_tps",
                        "williams_pct_r", "bollinger_pctb", "raschke_8020", "turtle_soup",
                        "signal_velocity", "wyckoff_spring", "weinstein_stage2"}
    for strat in list(results_by_strategy.keys()):
        if strat in RSI_FLOOR_EXEMPT:
            continue
        results_by_strategy[strat] = [r for r in results_by_strategy[strat]
                                       if (r.get("rsi") or 0) >= 50]
    all_results = [r for r in all_results
                   if r.get("strategy") in RSI_FLOOR_EXEMPT
                   or (r.get("rsi") or 0) >= 50]

    # connors_rsi2 RSI cap: strategy designed for oversold bounces; Jul data shows it
    # misfiring at RSI 70-94 → WR collapses. Cap at 75 (data: RSI 60-75 = 64% WR, RSI>75 = weak)
    results_by_strategy["connors_rsi2"] = [
        r for r in results_by_strategy.get("connors_rsi2", [])
        if (r.get("rsi") or 0) <= 75
    ]

    # Relative strength vs SPY: compute 10d return for HIGH-conviction tickers only.
    # Stock beating SPY before signal = actual alpha, not just beta. +1pt in rank_score.
    def _fetch_rs_vs_spy(ticker: str, spy_10d: float) -> float:
        """Return stock 10d return minus SPY 10d return. Positive = outperforming."""
        try:
            import yfinance as _yf
            sym = _yf_sym(ticker)
            with _quiet_ctx():
                df = _yf.Ticker(sym).history(period="15d", interval="1d", auto_adjust=True)
            if df is None or len(df) < 5: return 0.0
            ret = (df["Close"].iloc[-1] / df["Close"].iloc[-10] - 1) * 100 if len(df) >= 10 else 0.0
            return round(ret - spy_10d, 2)
        except Exception:
            return 0.0

    # Compute RS only for multi-strategy tickers (HIGH/MED) to avoid slowing full scan
    from collections import Counter as _Counter
    _tc = _Counter(r["ticker"] for r in all_results)
    _multi_for_rs = {t for t, n in _tc.items() if n >= 2}
    _spy_10d = 0.0
    try:
        import yfinance as _yf2
        with _quiet_ctx():
            _sp = _yf2.Ticker("SPY").history(period="15d", interval="1d", auto_adjust=True)
        if _sp is not None and len(_sp) >= 10:
            _spy_10d = (_sp["Close"].iloc[-1] / _sp["Close"].iloc[-10] - 1) * 100
    except Exception:
        pass
    _rs_cache: dict = {}
    if _multi_for_rs and _HAS_YF:
        print(DIM(f"  Computing RS vs SPY for {len(_multi_for_rs)} conviction tickers..."), flush=True)
        for _t in _multi_for_rs:
            _rs_cache[_t] = _fetch_rs_vs_spy(_t, _spy_10d)
    # Stamp rs_vs_spy onto each result dict
    for strat in results_by_strategy:
        for r in results_by_strategy[strat]:
            if r["ticker"] in _rs_cache:
                r["rs_vs_spy"] = _rs_cache[r["ticker"]]

    # Load persistence counts (must be before rank scoring)
    global _PERSISTENCE
    _PERSISTENCE = _load_persistence_counts()

    # Earnings block: drop tickers with earnings within 14 days (gap risk destroys stop)
    EARNINGS_EXEMPT: set = set()  # no exemptions — stage4_short disabled
    def _near_earnings(ticker: str, days: int = 14) -> bool:
        try:
            t_obj = yf.Ticker(_yf_sym(ticker))
            cal = t_obj.calendar
            if cal is None or cal.empty:
                return False
            # calendar may be dict or DataFrame depending on yfinance version
            if hasattr(cal, "T"):
                dates = cal.T.get("Earnings Date", pd.Series()).dropna()
            else:
                dates = pd.Series(cal.get("Earnings Date", []))
            today = pd.Timestamp.today().normalize()
            for d in dates:
                diff = (pd.Timestamp(d).normalize() - today).days
                if -2 <= diff <= days:
                    return True
        except Exception:
            pass
        return False

    print(DIM("  Checking earnings windows..."), flush=True)
    earnings_blocked = set()
    for strat, res in results_by_strategy.items():
        if strat in EARNINGS_EXEMPT:
            continue
        for r in res:
            t = r["ticker"]
            if t not in earnings_blocked and _near_earnings(t):
                earnings_blocked.add(t)

    if earnings_blocked:
        print(YLW(f"  ⚠  Earnings block ({len(earnings_blocked)} tickers removed): "
                  + ", ".join(sorted(earnings_blocked))))
        for strat in list(results_by_strategy.keys()):
            if strat in EARNINGS_EXEMPT:
                continue
            results_by_strategy[strat] = [r for r in results_by_strategy[strat]
                                           if r["ticker"] not in earnings_blocked]
        all_results = [r for r in all_results
                       if r.get("strategy") in EARNINGS_EXEMPT
                       or r["ticker"] not in earnings_blocked]

    total_time = time.time() - t0

    # Sector pulse (fetch 10d sector ETF returns vs SPY)
    print(DIM("  Fetching sector pulse..."), flush=True)
    sector_excess, spy_ret = _fetch_sector_pulse()

    # Analyst upgrade tickers (for cross-reference star in matrix)
    upgrade_tickers = {r["ticker"] for r in results_by_strategy.get("analyst_upgrade", [])}

    # Multi-strategy tickers: appear in ≥2 strategies (highest conviction — float to top)
    from collections import Counter
    ticker_strat_count = Counter(r["ticker"] for r in all_results)
    multi_tickers = {t for t, n in ticker_strat_count.items() if n >= 2}

    # Load historical WR stats (must happen before any display function)
    global _HIST_STATS
    _HIST_STATS = _load_hist_stats()

    # Display
    _print_header(strategies, len(all_results), with_backtest)
    _print_sector_pulse(sector_excess, spy_ret)
    _thematic_check()

    # ── LEAD WITH CONVICTION — what to act on ─────────────────────────────────
    elder_count = len(results_by_strategy.get("elder_impulse", []))
    _print_high_conviction(results_by_strategy, multi_tickers, with_backtest, sector_excess, elder_count)

    # ── MULTI-STRATEGY OVERLAP MATRIX ─────────────────────────────────────────
    _print_matrix(results_by_strategy, strategies, upgrade_tickers)

    # ── FULL STRATEGY DETAIL (reference) ──────────────────────────────────────
    print()
    print(DIM("  ── FULL SCAN DETAIL ──────────────────────────────────────────────────────"))
    for strategy in strategies:
        if strategy not in results_by_strategy:
            continue
        _print_group(strategy, results_by_strategy[strategy], with_backtest, multi_tickers)

    print()
    print("─" * W)
    print(DIM(f"  Total time: {total_time:.0f}s  ·  {len(all_results)} signals across {len(strategies)} strategies"))
    print("─" * W)

    # Persist latest scan results for notify.py / update_scan_history.py
    try:
        payload = {
            "scan_date": datetime.now().strftime("%Y-%m-%d"),
            "strategies": strategies,
            "results_by_strategy": {
                s: [_nan_safe(r) for r in res]
                for s, res in results_by_strategy.items()
            },
            "sector_excess": sector_excess,
            "spy_ret": spy_ret,
            "elder_impulse_count": elder_count,
            "market_regime": "BULL" if elder_count >= 15 else ("NEUTRAL" if elder_count >= 5 else "BEAR"),
        }
        LAST_SCAN_JSON.write_text(json.dumps(payload, indent=2))
        print(DIM(f"  Saved last_scan.json ({sum(len(v) for v in payload['results_by_strategy'].values())} results)"))
    except Exception as e:
        print(f"  WARNING: could not save last_scan.json — {e}")

    # Auto-add HIGH conviction picks as practice trades
    _auto_add_practice_trades(results_by_strategy, multi_tickers, is_friday=is_friday)

    # Tracker prompt
    if all_results:
        _tracker_prompt(all_results)


def _auto_add_practice_trades(results_by_strategy: dict, multi_tickers: set, is_friday: bool = False):
    """
    Auto-add every HIGH conviction pick as a practice trade.
    Skips tickers already OPEN in trades.csv (any trade type).
    Skips entirely on Friday (WR=45%, not worth entering).
    Runs silently at end of scan — no user input needed.
    """
    if is_friday:
        print(DIM("  [practice-auto] Friday scan — adding practice trades (WR=45%, paper only)."))

    try:
        existing_trades = load_trades()
    except Exception:
        existing_trades = []

    open_tickers = {t["ticker"] for t in existing_trades if t.get("status") == "OPEN"}
    today = date.today()
    scan_date_str = today.strftime("%Y-%m-%d")
    added = []
    skipped = []

    # Collect HIGH conviction picks (same logic as _print_high_conviction)
    seen: dict = {}
    tier_rank = {"★★★ HIGH": 0, "★★  MED ": 1, "★   LOW ": 2}
    for strategy, results in results_by_strategy.items():
        for r in results:
            t = r["ticker"]
            tier, _ = _conviction_tier(r, multi_tickers)
            rank = tier_rank[tier]
            if tier != "★★★ HIGH":
                continue
            if t not in seen or rank < seen[t][0]:
                seen[t] = (rank, r, strategy)

    if not seen:
        return

    # Sort by rank score (best first)
    picks = sorted(seen.values(), key=lambda x: (x[0], x[1].get("score", 99)))

    for _, r, strategy in picks:
        ticker = r["ticker"]

        if ticker in open_tickers:
            skipped.append(ticker)
            continue

        try:
            ccy      = ticker_ccy(ticker)
            fx       = fetch_fx_on_date(ccy, today)
            price    = float(r.get("price") or 0)
            if not price:
                continue
            sl       = trade_stop_loss(price)
            hold_d   = trade_hold_days(strategy)
            exit_dt  = biz_days_add(today, hold_d)
            qty      = round(1000.0 * fx / price, 4)
            inv_eur  = round(qty * price / fx, 2)
            company  = r.get("company") or fetch_company_name(ticker)
            sector   = fetch_sector(ticker)
            regime   = fetch_market_regime(today)

            # Build signals string from scan result
            strats_fired = [s for s, res in results_by_strategy.items()
                            if any(x["ticker"] == ticker for x in res)]
            proven = any(s in PROVEN_EDGE for s in strats_fired)
            sig = ("+".join(strats_fired) +
                   (f" · ✦PROVEN ({strategy} {int(_HIST_STATS.get(strategy,{}).get('wr',0))}%WR)" if proven else "") +
                   f" · score={r.get('score',0)} · RSI={r.get('rsi',0):.1f} · ADX={r.get('adx',0):.1f} · ★★★ HIGH conviction")

            trade = {
                "id":                  next_id(existing_trades),
                "entry_date":          scan_date_str,
                "ticker":              ticker,
                "company":             company,
                "currency":            ccy,
                "buy_price":           price,
                "stop_loss_price":     sl,
                "fx_at_entry":         round(fx, 6),
                "qty":                 qty,
                "investment_eur":      inv_eur,
                "trade_type":          "practice",
                "strategy":            strategy,
                "hold_days":           hold_d,
                "target_exit_date":    exit_dt.strftime("%Y-%m-%d"),
                "signals":             sig,
                "status":              "OPEN",
                "actual_sell_date":    "",
                "exit_price":          "",
                "rsi_at_entry":        r.get("rsi", ""),
                "adx_at_entry":        r.get("adx", ""),
                "minervini_at_entry":  r.get("minervini", ""),
                "vol_ratio_entry":     r.get("vol_ratio", ""),
                "atr_ratio_entry":     r.get("atr_ratio", ""),
                "market_regime_entry": regime,
                "sector":              sector,
                "rsi_at_exit":         "",
                "adx_at_exit":         "",
                "minervini_at_exit":   "",
                "vol_ratio_exit":      "",
                "max_dd_1wk":          "",
                "exit_reason":         "",
            }
            existing_trades.append(trade)
            open_tickers.add(ticker)
            added.append(ticker)
        except Exception as e:
            print(DIM(f"  [practice-auto] skipped {ticker}: {e}"))

    if added or skipped:
        try:
            save_trades(existing_trades)
        except Exception as e:
            print(f"  WARNING: could not save practice trades — {e}")
            return
        if added:
            print(GRN(f"\n  ✅  Auto-practice: added {len(added)} trade(s) → {', '.join(added)}"))
        if skipped:
            print(DIM(f"  ↩  Already open, skipped: {', '.join(skipped)}"))


if __name__ == "__main__":
    main()
