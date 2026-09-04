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
from datetime import datetime, date
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
# from power_earnings_gap_scanner  import scan as scan_peg   # DISABLED: WR 40.0% n=5 — below threshold
from rs_line_scanner              import scan as scan_rs_line  # v2: RS-leads-price O'Neil impl
# from raschke_holy_grail_scanner  import scan as scan_holy_grail  # DISABLED: WR 42.9% n=7
# from connors_3down_scanner       import scan as scan_3down  # DISABLED: WR 46.2% n=13, avg -1.15%
from darvas_box_scanner            import scan as scan_darvas
from vcp_scanner                   import scan as scan_vcp
# from elder_impulse_scanner       import scan as scan_elder  # DISABLED: WR 50.0% avg -0.57% n=64 — coin flip
from williams_pct_r_scanner        import scan as scan_williams_r
from bollinger_pctb_scanner        import scan as scan_bb_pctb
from connors_r3_scanner            import scan as scan_r3
from connors_tps_scanner           import scan as scan_tps
from turtle_soup_scanner           import scan as scan_turtle_soup
from raschke_8020_scanner          import scan as scan_8020
from wyckoff_spring_scanner        import scan as scan_wyckoff_spring
from weinstein_stage2_scanner      import scan as scan_weinstein_stage2
from momentum_burst_scanner        import scan as scan_momentum_burst
from ma50_reclaim_scanner          import scan as scan_ma50_reclaim
from turnover_momentum_scanner     import scan as scan_turnover_momentum
from three_weeks_tight_scanner     import scan as scan_3wt
from episodic_pivot_scanner        import scan as scan_ep
from combo_scanner                 import scan as scan_combo
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
                  "wyckoff_spring", "weinstein_stage2",
                  "momentum_burst", "ma50_reclaim", "turnover_momentum",
                  "three_weeks_tight", "episodic_pivot", "combo_pp_ribbon"]

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
    # "power_earnings_gap": scan_peg,   # DISABLED: WR 40%
    "rs_line":           scan_rs_line,  # v2: RS-leads-price O'Neil impl
    # "holy_grail":        scan_holy_grail, # DISABLED: WR 42.9%
    # "connors_3down":     scan_3down,   # DISABLED: WR 46.2%, avg -1.15%
    "darvas_box":            scan_darvas,
    "vcp":                   scan_vcp,
    # "elder_impulse":       scan_elder,  # DISABLED: WR 50.0% avg -0.57% n=64
    "williams_pct_r":        scan_williams_r,
    "bollinger_pctb":        scan_bb_pctb,
    "connors_r3":            scan_r3,
    "connors_tps":           scan_tps,
    "turtle_soup":           scan_turtle_soup,
    "raschke_8020":          scan_8020,
    "wyckoff_spring":        scan_wyckoff_spring,
    "weinstein_stage2":      scan_weinstein_stage2,
    "momentum_burst":        scan_momentum_burst,
    "ma50_reclaim":          scan_ma50_reclaim,
    "turnover_momentum":     scan_turnover_momentum,
    "three_weeks_tight":     scan_3wt,
    "episodic_pivot":        scan_ep,
    "combo_pp_ribbon":       scan_combo,
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
    "momentum_burst":        "💥  MOMENTUM BURST  (Stockbee — first explosive day ≥4% after NR compression, vol≥1.5x, fresh move)",
    "ma50_reclaim":          "📍  50 SMA RECLAIM  (Minervini/IBD — price reclaims 50 SMA after pullback, institutions add here)",
    "turnover_momentum":     "🔵  TURNOVER MOMENTUM  (Medhat & Schmeling RFS 2022 — top-33% 12-1m momentum × below-median turnover)",
    "three_weeks_tight":     "🗜  3-WEEKS-TIGHT  (O'Neil/IBD — 3 weekly closes within 1.5% + volume drying = pre-breakout coil)",
    "episodic_pivot":        "⚡  EPISODIC PIVOT  (Gil Morales/Kacher — catalyst gap ≥8% on 2.5× volume, permanently repriced, entry on pullback)",
    "combo_pp_ribbon":       "🏆  COMBO PP+RIBBON  (Premium setup — Pocket Pivot AND EMA Ribbon fire simultaneously; highest-conviction momentum)",
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
    "turnover_momentum": (
        "Medhat & Schmeling (RFS 2022) — buys stocks in top-third of 12-1m momentum "
        "that also have BELOW-median share turnover. Low turnover = uncrowded position, "
        "dramatically reduces momentum crash risk. Hold 5 days."
    ),
    "three_weeks_tight": (
        "O'Neil / IBD — Three weekly closing prices within 1.5% of each other, with volume\n"
        "  declining week-over-week. The stock is digesting a prior move in a tight, orderly\n"
        "  fashion — institutions holding, not distributing. Entry just below the weekly high.\n"
        "  Hold 7 days. One of O'Neil's highest-WR continuation setups."
    ),
    "episodic_pivot": (
        "Gil Morales & Chris Kacher ('Trade Like an O'Neil Disciple' 2010) — A single news\n"
        "  catalyst (earnings beat, FDA approval, contract win) causes a permanent institutional\n"
        "  repricing: gap ≥8% on ≥2.5× volume. Gap must hold for 3+ days (no fill). Enter on\n"
        "  the first constructive pullback — not immediately into the gap. Hold 10 days.\n"
        "  Historical WR ~70% when gap holds 3 days. NVDA Dec 2023 and MRNA COVID approval\n"
        "  were textbook Episodic Pivots."
    ),
    "combo_pp_ribbon": (
        "Premium setup: Pocket Pivot AND EMA Ribbon fire simultaneously on the same ticker.\n"
        "  PP = institutional accumulation signal (volume surge on up-day).\n"
        "  Ribbon = 8/13/21/34/55 EMAs all stacked and expanding + price pulls back and bounces.\n"
        "  When both align: institutional buyers entering INTO a strengthening trend structure.\n"
        "  Minervini ≥6 required. Hold 7 days. Gil Morales & Kacher's preferred entry combination."
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
    "momentum_burst":        5,
    "ma50_reclaim":          7,
    "turnover_momentum":     5,
}

# Short display aliases for strategy names (replaces garbled [:6] truncation)
STRAT_ALIAS = {
    "pocket_pivot":          "PP",
    "ema_ribbon":            "EMA",
    "cup_handle":            "CUP",
    "connors_rsi2":          "RSI2",
    "connors_r3":            "R3",
    "connors_tps":           "TPS",
    "signal_velocity":       "SV",
    "breakout":              "BKT",
    "vcp":                   "VCP",
    "darvas_box":            "DARV",
    "rs_line":               "RSL",
    "nr7":                   "NR7",
    "williams_pct_r":        "WR%",
    "bollinger_pctb":        "BB%B",
    "turtle_soup":           "TSUP",
    "raschke_8020":          "R820",
    "wyckoff_spring":        "WYK",
    "weinstein_stage2":      "W2",
    "momentum_burst":        "MBST",
    "ma50_reclaim":          "MA50",
    "holy_grail":            "HGRL",
    "analyst_upgrade":       "UPGR",
    "chokepoint_inflection": "CHKPT",
    "defensive_rotation":    "DEFR",
    "momentum":              "MOM",
    "elder_impulse":         "EIMP",
    "stage4_short":          "S4SH",
    "turnover_momentum":     "TMOM",
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
                     "signal_velocity", "weinstein_stage2", "vcp",
                     "momentum_burst", "ma50_reclaim"}
# Mean reversion: work in sideways/oversold conditions (NEUTRAL/BEAR)
_REVERSION_STRATS = {"connors_rsi2", "nr7", "wyckoff_spring", "raschke_8020",
                     "connors_3down", "bollinger_pctb"}


def _rank_score(r: dict, strats_fired: list, elder_count: int = 0,
                sector_excess: dict = None, ticker_etf: str = "") -> float:
    """
    Score each HIGH-conviction pick so we can label top 1/2/3.
    Higher = better. Criteria (max ~17 pts):
      +3  any PROVEN_EDGE strategy fired
          ADX REMOVED (data: -3.1% WR delta, n=588 — noise, not signal)
      +2  RSI 50-65  (momentum without overextension)
      +3  3+ strategies with quality gate (WR≥50%)
      +2  2 strategies with quality gate
      +1  score ≤ 3
      -1  score 7-10 (overextended/noise: 33.3% WR)
      +2  vol >2x    (71.1% WR)
      +1  vol 1.5-2x (64.2% WR)
      +2  persistent 3+ scan dates (61% vs 47% WR)
      +1  persistent 2 scan dates
      +1  RS positive vs SPY 10d
      +1  regime-strategy fit (trend in BULL, reversion in NEUTRAL)
      -1  ticker sector is bottom-3 vs SPY 10d (sector headwind)
    """
    pts = 0.0
    if any(s in PROVEN_EDGE for s in strats_fired):
        pts += 3
    rsi = r.get("rsi", 0) or 0
    # Breakout/momentum strategies can legitimately fire at RSI 65-75 (trending stocks).
    # Keep tighter 50-65 cap for mean-reversion (those need low RSI for a bounce).
    _BROAD_RSI_STRATS = {"breakout", "darvas_box", "pocket_pivot", "ema_ribbon", "vcp",
                         "cup_handle", "weinstein_stage2", "momentum", "momentum_burst",
                         "power_earnings_gap", "rs_line", "signal_velocity"}
    _rsi_hi = 75 if any(s in _BROAD_RSI_STRATS for s in strats_fired) else 65
    if 50 <= rsi <= _rsi_hi:
        pts += 2
    has_quality = any(_HIST_STATS.get(s, {}).get("wr", 0) >= 50 for s in strats_fired)
    n_strats = len(strats_fired)
    if has_quality:
        if n_strats >= 3:
            pts += 3
        elif n_strats == 2:
            pts += 2
    if (r.get("score") or 99) <= 3:
        pts += 1
    if 7 <= (r.get("score") or 0) <= 10:
        pts -= 1
    vr = r.get("vol_ratio", 0) or 0
    # 2026-09 recal: vol≥2x WR=47.5% (n=158, -7.1% delta) — high vol often means news/gap; no extra bonus.
    # vol 1.5-2x WR=60.1% (n=148, +6.5% delta) — sweet spot, keep +1.
    if vr >= 2.0:
        pts += 1   # was +2; recalibrated 2026-09
    elif 1.5 <= vr < 2.0:
        pts += 1
    days_seen = _PERSISTENCE.get(r.get("ticker", ""), 0)
    if days_seen >= 3:
        pts += 2
    elif days_seen >= 2:
        pts += 1
    if r.get("rs_vs_spy", 0) and float(r.get("rs_vs_spy", 0)) > 0:
        pts += 1
    is_bull    = elder_count >= 15
    is_neutral = 5 <= elder_count < 15
    has_trend  = any(s in _TREND_STRATS     for s in strats_fired)
    has_revert = any(s in _REVERSION_STRATS for s in strats_fired)
    if (is_bull and has_trend) or (is_neutral and has_revert):
        pts += 1
    # Sector headwind penalty: bottom-3 sectors vs SPY 10d
    if sector_excess and ticker_etf and ticker_etf in sector_excess:
        ranked_vals = sorted(sector_excess.values())
        if sector_excess[ticker_etf] <= ranked_vals[2]:  # bottom 3
            pts -= 1
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

    # ── Batch-fetch company name + sector ETF for ALL tickers in results ─────
    display_tickers = list(dict.fromkeys(
        [p[1]["ticker"] for p in high_picks + med_picks]  # conviction picks first
        + [r["ticker"] for res in results_by_strategy.values() for r in res]  # then rest
    ))
    _company_cache: dict[str, str] = {}
    _sector_etf_cache: dict[str, str] = {}  # ticker → ETF code e.g. "XLF"

    def _fetch_info(t: str):
        # Prefer company already in result dict
        existing_co = next((r.get("company") for _, r, _ in picks
                            if r["ticker"] == t and r.get("company")), None)
        etf, name = "", existing_co or ""
        try:
            import yfinance as _yf
            info = _yf.Ticker(t).info
            if not name:
                name = info.get("shortName") or info.get("longName") or t
            etf = SECTOR_ETF.get(info.get("sector", ""), "")
        except Exception:
            if not name:
                name = t
        return t, name or t, etf

    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as ex:
            for ticker, name, etf in ex.map(_fetch_info, display_tickers):
                _company_cache[ticker]    = name
                _sector_etf_cache[ticker] = etf
    except Exception:
        pass

    # Helper: colorised sector column "XLF+2.9%" / "XLK-5.0%"
    def _sec_col(ticker: str) -> str:
        etf = _sector_etf_cache.get(ticker, "")
        if not etf or not sector_excess or etf not in sector_excess:
            return f"{'─':^11}"
        ex  = sector_excess[etf]
        s   = f"{etf} {'+' if ex>=0 else ''}{ex:.1f}%"
        ranked = sorted(sector_excess.values(), reverse=True)
        if ex >= ranked[2]:   return GRN(f"{s:^11}")   # top 3
        if ex <= sorted(sector_excess.values())[2]: return RED(f"{s:^11}")   # bottom 3
        return f"{s:^11}"

    # ── ACT ON THESE (HIGH conviction) ────────────────────────────────────────
    print()
    print("╔" + "═"*(W-2) + "╗")
    if high_picks:
        print("║" + GRN(BOLD(f"  🎯  ACT ON THESE  ·  {len(high_picks)} stock(s)  ·  ★★★ HIGH CONVICTION")).ljust(W+16) + "║")
    else:
        print("║" + YLW(BOLD(f"  👀  NO HIGH CONVICTION TODAY  —  see WATCHLIST below")).ljust(W+12) + "║")
    print("╚" + "═"*(W-2) + "╝")

    if high_picks:
        enriched = []
        for _, r, strategy in high_picks:
            strats_fired = sorted(
                [s for s, res in results_by_strategy.items()
                 if any(x["ticker"] == r["ticker"] for x in res)],
                key=lambda s: -_HIST_STATS.get(s, {}).get("wr", 0)
            )
            etf = _sector_etf_cache.get(r["ticker"], "")
            rs  = _rank_score(r, strats_fired, elder_count, sector_excess, etf)
            enriched.append((rs, r, strategy, strats_fired))

        enriched.sort(key=lambda x: -x[0])
        medals = {0: GRN(BOLD(" #1 ")), 1: GRN(BOLD(" #2 ")), 2: YLW(BOLD(" #3 "))}

        print(BOLD(f"  {'RANK':<4}  {'TICKER':<10}  {'SECTOR':^11}  {'COMPANY':<22}  {'STRATEGIES':<20}  {'WR':>5}  {'SCR':>4}  {'RSI':>5}  {'ADX':>5}  {'HOLD':>5}  {'PRICE':>8}  {'SL ~':>8}"))
        print("  " + "─"*(W-2))
        for idx, (rs, r, strategy, strats_fired) in enumerate(enriched):
            best_strat   = next((s for s in strats_fired if _HIST_STATS.get(s,{}).get("n",0)>=5), strategy)
            proven_badge = " ✦PROVEN" if any(s in PROVEN_EDGE for s in strats_fired) else ""
            strat_str    = "+".join(STRAT_ALIAS.get(s, s[:4].upper()) for s in strats_fired[:3])
            hist         = _HIST_STATS.get(best_strat, {})
            wr_s         = f"{hist['wr']:.0f}%WR" if hist.get("n",0)>=5 else "─"
            wr_col       = GRN(f"{wr_s:>7}") if hist.get("wr",0)>=60 else YLW(f"{wr_s:>7}")
            proven_s     = GRN(proven_badge) if proven_badge else ""
            company      = str(_company_cache.get(r["ticker"], "") or "")[:22]
            rank_label   = medals.get(idx, "     ")
            ticker_fmt   = f"{r['ticker']:<10}"
            sec_display  = _sec_col(r["ticker"])
            days_seen    = _PERSISTENCE.get(r["ticker"], 0)
            pers_badge   = GRN(" 🔁PERSIST") if days_seen >= 3 else (DIM(" 🔁x2") if days_seen >= 2 else "")
            vr           = r.get("vol_ratio", 0) or 0
            vol_badge    = GRN(" ⚡VOL") if vr >= 2.0 else (YLW(" ⚡") if vr >= 1.5 else "")
            rs_badge     = GRN(" ↑RS") if r.get("rs_vs_spy", 0) and float(r.get("rs_vs_spy",0)) > 0 else ""
            _price  = r.get("price", 0) or 0
            _sl     = _price * 0.97
            _hold   = HOLD_DAYS_MAP.get(best_strat, 5)
            print(f"  {rank_label}  {GRN(BOLD(ticker_fmt))}  {sec_display}  {company:<22}  {strat_str+str(proven_s):<20}  {wr_col}  {r.get('score',0):>4}  {r.get('rsi',0):>5.1f}  {r.get('adx',0):>5.1f}  {_hold:>4}d  {_price:>8.2f}  {RED(f'${_sl:>6.2f}')}{pers_badge}{vol_badge}{rs_badge}")
        print()

    # ── WATCHLIST (MED conviction) ─────────────────────────────────────────────
    if med_picks:
        print(BOLD(f"  👀  WATCHLIST  ·  {len(med_picks)} stock(s)  ·  ★★ MED — wait for stronger signal before buying"))
        print(BOLD(f"  {'TICKER':<10}  {'SECTOR':^11}  {'COMPANY':<22}  {'STRATEGIES':<18}  {'WR':>5}  {'SCR':>4}  {'RSI':>5}  {'ADX':>5}  {'HOLD':>5}  {'PRICE':>8}"))
        print("  " + "─"*(W-2))
        for _, r, strategy in med_picks[:15]:
            strats_fired = sorted(
                [s for s, res in results_by_strategy.items()
                 if any(x["ticker"] == r["ticker"] for x in res)],
                key=lambda s: -_HIST_STATS.get(s, {}).get("wr", 0)
            )
            best_strat   = next((s for s in strats_fired if _HIST_STATS.get(s,{}).get("n",0)>=5), strategy)
            strat_str    = "+".join(STRAT_ALIAS.get(s, s[:4].upper()) for s in strats_fired[:2])
            proven_badge = "✦" if any(s in PROVEN_EDGE for s in strats_fired) else ""
            hist         = _HIST_STATS.get(best_strat, {})
            wr_s         = f"{hist['wr']:.0f}%" if hist.get("n",0)>=5 else "─"
            ticker_fmt2  = f"{r['ticker']:<10}"
            co2          = str(_company_cache.get(r["ticker"], r.get("company","") or ""))[:22]
            sec_display  = _sec_col(r["ticker"])
            _hold2       = HOLD_DAYS_MAP.get(best_strat, 5)
            _price2      = r.get("price", 0) or 0
            print(f"  {YLW(BOLD(ticker_fmt2))}  {sec_display}  {co2:<22}  {(strat_str+proven_badge):<18}  {wr_s:>5}  {r.get('score',0):>4}  {r.get('rsi',0):>5.1f}  {r.get('adx',0):>5.1f}  {_hold2:>4}d  {_price2:>8.2f}")
        if len(med_picks) > 15:
            print(DIM(f"  ... and {len(med_picks)-15} more"))
        print()

    print(DIM(f"  {len(high_picks)} HIGH  ·  {len(med_picks)} MED  ·  Focus: HIGH only unless 2+ strategies confirmed"))
    print()
    return _company_cache




def _apply_trend_template(universe: dict) -> dict:
    """
    Minervini Trend Template — filter universe to Stage 2 stocks only.
    Criteria (Minervini, Trade Like a Stock Market Wizard):
      1. Price > 50MA > 150MA > 200MA  (MA stack aligned)
      2. 200MA trending up (current > value 30 trading days ago)
      3. Price >= 30% above 52-week low
      4. Price <= 25% below 52-week high  (not too extended)
    Stocks failing any criterion are Stage 1/3/4 — skip them.
    """
    if not _HAS_YF:
        return universe
    import yfinance as _yf
    tickers = list(universe.keys())
    passed = {}
    failed = 0
    print(DIM(f"  Applying Trend Template to {len(tickers)} tickers..."), flush=True)
    try:
        with _quiet_ctx():
            raw = _yf.download(
                [_yf_sym(t) for t in tickers],
                period="1y", interval="1d",
                auto_adjust=True, progress=False, threads=True,
            )
        if raw.empty:
            return universe
        # Multi-ticker download has MultiIndex columns: (field, ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"]
        else:
            close_df = raw  # single ticker fallback

        for t in tickers:
            sym = _yf_sym(t)
            try:
                s = close_df[sym].dropna() if sym in close_df.columns else None
                if s is None or len(s) < 60:
                    failed += 1; continue
                price   = float(s.iloc[-1])
                ma50    = float(s.rolling(50).mean().iloc[-1])
                ma150   = float(s.rolling(150).mean().iloc[-1]) if len(s) >= 150 else None
                ma200   = float(s.rolling(200).mean().iloc[-1]) if len(s) >= 200 else None
                ma200_30ago = float(s.rolling(200).mean().iloc[-31]) if len(s) >= 231 else None
                hi52    = float(s[-252:].max()) if len(s) >= 252 else float(s.max())
                lo52    = float(s[-252:].min()) if len(s) >= 252 else float(s.min())

                # Criterion 1: MA stack (require at least 50MA and 150MA)
                if ma150 is None or price < ma50 or ma50 < ma150:
                    failed += 1; continue
                if ma200 is not None and ma150 < ma200:
                    failed += 1; continue
                # Criterion 2: 200MA trending up
                if ma200 is not None and ma200_30ago is not None and ma200 < ma200_30ago:
                    failed += 1; continue
                # Criterion 3: price >= 30% above 52-week low
                if lo52 > 0 and price < lo52 * 1.30:
                    failed += 1; continue
                # Criterion 4: price <= 25% below 52-week high (not too extended)
                if hi52 > 0 and price < hi52 * 0.75:
                    failed += 1; continue
                passed[t] = universe[t]
            except Exception:
                failed += 1
                continue
    except Exception as e:
        print(DIM(f"  TT filter error ({e}) — using full universe"))
        return universe

    print(DIM(f"  Trend Template: {len(passed)} passed / {failed} filtered out  "
              f"({100*len(passed)/max(len(tickers),1):.0f}% Stage 2)"))
    return passed if passed else universe


def _apply_eps_filter(universe: dict) -> dict:
    """
    EPS fundamental filter — applied AFTER Trend Template (~180 tickers).
    Keeps stocks with EPS growth ≥15% YoY OR where data is unavailable (fail-open).
    Uses yfinance .info; skips tickers that time out or error (fail-open).
    """
    if not _HAS_YF:
        return universe
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
    import yfinance as _yf

    EPS_GROWTH_MIN = 0.15  # 15% YoY growth minimum

    def _check_eps(ticker: str) -> tuple[str, bool]:
        """Return (ticker, keep). keep=True if EPS growth OK or data missing."""
        try:
            with _quiet_ctx():
                info = _yf.Ticker(_yf_sym(ticker)).info
            if not info:
                return (ticker, True)  # fail-open
            # Prefer earningsGrowth (analyst-reported YoY EPS growth)
            eg = info.get("earningsGrowth")
            if eg is not None:
                return (ticker, float(eg) >= EPS_GROWTH_MIN)
            # Fallback: trailing EPS vs forward EPS
            t_eps = info.get("trailingEps")
            f_eps = info.get("forwardEps")
            if t_eps and f_eps and float(t_eps) > 0:
                growth = (float(f_eps) - float(t_eps)) / abs(float(t_eps))
                return (ticker, growth >= EPS_GROWTH_MIN)
            return (ticker, True)  # no data → fail-open
        except Exception:
            return (ticker, True)  # fail-open on any error

    tickers = list(universe.keys())
    print(DIM(f"  Applying EPS filter (≥15% growth) to {len(tickers)} tickers..."), flush=True)
    passed = {}
    failed = 0
    with _TPE(max_workers=20) as ex:
        futs = {ex.submit(_check_eps, t): t for t in tickers}
        for f in _ac(futs):
            t, keep = f.result()
            if keep:
                passed[t] = universe[t]
            else:
                failed += 1
    print(DIM(f"  EPS filter: {len(passed)} passed / {failed} filtered out"))
    return passed if passed else universe


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


def _streak_leaders(min_streak: int = 5) -> list[dict]:
    """Return tickers appearing in ≥1 scanner on each of the last min_streak consecutive trading days.
    Returns list of dicts sorted by streak length desc: {ticker, streak, strategies, last_price, last_score}"""
    import csv as _csv
    from collections import defaultdict
    from pathlib import Path as _Path
    p = _Path(__file__).parent / "scan_history.csv"
    if not p.exists(): return []
    # Build {ticker: set of scan_dates} and {date: {ticker: [strategies]}}
    ticker_dates: dict = defaultdict(set)
    date_ticker_info: dict = defaultdict(lambda: defaultdict(list))
    try:
        with open(p, newline="") as f:
            for row in _csv.DictReader(f):
                t  = row.get("ticker","").strip()
                sd = row.get("scan_date","").strip()
                st = row.get("strategy","").strip()
                if t and sd:
                    ticker_dates[t].add(sd)
                    date_ticker_info[sd][t].append({
                        "strategy": st,
                        "score":    row.get("score",""),
                        "price":    row.get("price_at_scan",""),
                    })
    except Exception:
        return []

    all_dates = sorted(ticker_dates[next(iter(ticker_dates), "")] | set().union(*[v.keys() for v in [date_ticker_info]]))
    all_dates = sorted(set(sd for t in ticker_dates for sd in ticker_dates[t]))
    if len(all_dates) < min_streak:
        return []

    # Last N consecutive trading days present in scan_history
    recent_dates = all_dates[-min_streak:]

    results = []
    for ticker, dates_seen in ticker_dates.items():
        if all(d in dates_seen for d in recent_dates):
            # Compute full streak (how many consecutive days back)
            streak = 0
            for d in reversed(all_dates):
                if d in dates_seen:
                    streak += 1
                else:
                    break
            # Gather latest info
            last_date = max(dates_seen)
            entries   = date_ticker_info[last_date][ticker]
            strats    = list({e["strategy"] for e in entries})
            last_price = next((e["price"] for e in entries if e["price"]), "")
            last_score = max((int(e["score"]) for e in entries if e["score"].isdigit()), default=0)
            results.append({
                "ticker":      ticker,
                "streak":      streak,
                "strategies":  strats,
                "last_price":  last_price,
                "last_score":  last_score,
                "last_date":   last_date,
            })

    return sorted(results, key=lambda x: -x["streak"])


def _print_streak_leaders(min_streak: int = 5) -> None:
    leaders = _streak_leaders(min_streak)
    if not leaders:
        return
    print()
    print(BOLD(f"  🔁  PERSISTENCE LEADERS  ·  seen ≥{min_streak} consecutive trading days"))
    print(DIM("  " + "─" * 70))
    for l in leaders:
        strat_str = "+".join(l["strategies"])
        price_str = f"  ${l['last_price']}" if l["last_price"] else ""
        print(f"  {BOLD(l['ticker']): <12}  streak={YLW(str(l['streak'])+'d')}  score={l['last_score']}  "
              f"{DIM(strat_str)}{price_str}")
    print()


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


def _print_header(strategies: list, total: int, with_backtest: bool,
                  elder_count: int = 0, spy_ret: float = 0.0):
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    strat_str = ", ".join(strategies)
    regime_label = "BULL" if elder_count >= 15 else ("NEUTRAL" if elder_count >= 5 else "BEAR")
    regime_col   = GRN if elder_count >= 15 else (YLW if elder_count >= 5 else RED)
    spy_s = f"SPY 10d {'+' if spy_ret >= 0 else ''}{spy_ret:.1f}%"
    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + f"  UNIFIED SCANNER  ·  {now}  ·  {total} total signals  ·  backtest={'ON' if with_backtest else 'OFF'}".ljust(W-2) + "║")
    print("║" + (regime_col(f"  REGIME: {regime_label}  ({spy_s})  ·  {'🔥 Uptrend — favour trend strategies' if elder_count >= 15 else ('⚠ Mixed — high-conviction only' if elder_count >= 5 else '❄️ Weak market — avoid new longs')}")).ljust(W+9) + "║")
    print("║" + DIM(f"  Strategies: {strat_str[:W-17]}").ljust(W+7) + "║")
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
    india_mode    = "--india" in args_raw

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
    if india_mode:
        from nifty500_fetcher import get_nifty500, INDIA_BENCH
        _india_tickers = get_nifty500()
        universe = {t: INDIA_BENCH for t in _india_tickers}
        bench_returns = compute_bench_returns({INDIA_BENCH})
        print(DIM(f"  🇮🇳 India mode — {len(universe)} Nifty 500 tickers  ·  benchmark {INDIA_BENCH}"))
    else:
        universe     = build_universe()
        bench_returns = compute_bench_returns(set(universe.values()))

    # ── Minervini Trend Template pre-filter ───────────────────────────────────
    # Only scan Stage 2 stocks: MA stack aligned, 200MA trending up, near 52w high.
    # Cuts universe ~60-70% while keeping only institutional-grade uptrends.
    if _HAS_YF:
        universe = _apply_trend_template(universe)
        universe = _apply_eps_filter(universe)

    bt_label = "backtest ON" if with_backtest else "backtest OFF"
    print(DIM(f"  Universe (post-TT+EPS filter): {len(universe)} tickers  ·  {bt_label}"))
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
                        "signal_velocity", "wyckoff_spring", "weinstein_stage2",
                        "momentum_burst", "ma50_reclaim"}
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

    # ── Data-driven signal quality filters (Jul 2026 backtest, n=1808) ───────────
    #
    # nr7: vol≥1.5x → WR 51%→67%;  score≥4 → WR 51%→75%. Both gates together.
    results_by_strategy["nr7"] = [
        r for r in results_by_strategy.get("nr7", [])
        if (r.get("vol_ratio") or 0) >= 1.5 and (r.get("score") or 0) >= 4
    ]

    # connors_rsi2: vol≥1.5x → WR 49%→66%. Mean-reversion needs volume confirmation.
    results_by_strategy["connors_rsi2"] = [
        r for r in results_by_strategy.get("connors_rsi2", [])
        if (r.get("vol_ratio") or 0) >= 1.5
    ]

    # momentum: tighten to RSI 55-70 + vol≥1.5x. Fires too broadly in non-trending markets.
    # (scanner stays active — filter is signal quality, not regime gating)
    results_by_strategy["momentum"] = [
        r for r in results_by_strategy.get("momentum", [])
        if 55 <= (r.get("rsi") or 0) <= 70
        and (r.get("vol_ratio") or 0) >= 1.5
    ]

    # rs_line + raschke_8020: add vol≥1.5x + score≥4 quality gate.
    # Both had 14-15% WR on low-quality signals — volume confirmation is minimum bar.
    for _strat in ("rs_line", "raschke_8020"):
        results_by_strategy[_strat] = [
            r for r in results_by_strategy.get(_strat, [])
            if (r.get("vol_ratio") or 0) >= 1.5 and (r.get("score") or 0) >= 4
        ]

    # Sync all_results with updated per-strategy lists
    _keep = {id(r) for res in results_by_strategy.values() for r in res}
    all_results = [r for r in all_results if id(r) in _keep]

    # Dollar-volume floor — $5M/day minimum (illiquid = wide spreads, hard to exit)
    # Fetch 20d avg volume for each unique ticker in signals (quick, batched).
    _DOLLAR_VOL_MIN = 5_000_000
    _all_tickers_dv = list({r["ticker"] for r in all_results})
    _dv_fail: set = set()
    if _HAS_YF and _all_tickers_dv:
        print(DIM(f"  Checking dollar-volume floor ({len(_all_tickers_dv)} tickers)..."), flush=True)
        for _dv_tick in _all_tickers_dv:
            try:
                with _quiet_ctx():
                    _dv_df = yf.Ticker(_yf_sym(_dv_tick)).history(
                        period="30d", interval="1d", auto_adjust=True)
                if _dv_df is None or len(_dv_df) < 5:
                    continue
                _avg_vol = float(_dv_df["Volume"].tail(20).mean())
                _price_c = float(_dv_df["Close"].iloc[-1])
                if _avg_vol * _price_c < _DOLLAR_VOL_MIN:
                    _dv_fail.add(_dv_tick)
            except Exception:
                pass
    if _dv_fail:
        print(DIM(f"  Dollar-volume filter removed {len(_dv_fail)} tickers below $5M/day"))
        for _strat in list(results_by_strategy.keys()):
            results_by_strategy[_strat] = [r for r in results_by_strategy[_strat]
                                           if r["ticker"] not in _dv_fail]
        all_results = [r for r in all_results if r["ticker"] not in _dv_fail]

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

    _earnings_warn_msg = ""
    if earnings_blocked:
        _earnings_warn_msg = (f"  ⚠  Earnings block ({len(earnings_blocked)} tickers removed): "
                              + ", ".join(sorted(earnings_blocked)))
        for strat in list(results_by_strategy.keys()):
            if strat in EARNINGS_EXEMPT:
                continue
            results_by_strategy[strat] = [r for r in results_by_strategy[strat]
                                           if r["ticker"] not in earnings_blocked]
        all_results = [r for r in all_results
                       if r.get("strategy") in EARNINGS_EXEMPT
                       or r["ticker"] not in earnings_blocked]

    total_time = time.time() - t0

    # Sector pulse
    print(DIM("  Fetching sector pulse..."), flush=True)
    if india_mode:
        # Use Nifty 10d return as regime proxy; no sector ETF breakdown for India
        sector_excess = {}
        try:
            with contextlib.suppress(Exception):
                _nifty_df = yf.download("^NSEI", period="20d", interval="1d",
                                        progress=False, auto_adjust=True, threads=False)
                if isinstance(_nifty_df.columns, pd.MultiIndex):
                    _nifty_df.columns = _nifty_df.columns.droplevel(1)
                _nc = _nifty_df["Close"].dropna()
                spy_ret = float((_nc.iloc[-1] - _nc.iloc[-11]) / _nc.iloc[-11] * 100) if len(_nc) >= 11 else 0.0
        except Exception:
            spy_ret = 0.0
    else:
        sector_excess, spy_ret = _fetch_sector_pulse()

    # Derive market regime from SPY 10d return (elder_impulse disabled — WR 50%)
    if spy_ret >= 3.0:       elder_count = 18
    elif spy_ret >= 1.5:     elder_count = 15
    elif spy_ret >= 0.5:     elder_count = 10
    elif spy_ret >= -0.5:    elder_count = 5
    elif spy_ret >= -2.0:    elder_count = 2
    else:                    elder_count = 0

    # Analyst upgrade tickers (for cross-reference star in matrix)
    upgrade_tickers = {r["ticker"] for r in results_by_strategy.get("analyst_upgrade", [])}

    # Strategy families — correlated signals within a family count as ONE independent vote.
    # Prevents connors_rsi2+r3+tps (same bet × 3) from inflating to HIGH conviction.
    _STRAT_FAMILY = {
        "connors_rsi2":   "connors_mr",
        "connors_r3":     "connors_mr",
        "connors_tps":    "connors_mr",
        "turtle_soup":    "false_breakdown",
        "raschke_8020":   "false_breakdown",
        "wyckoff_spring": "false_breakdown",
        "vcp":            "base_pattern",
        "cup_handle":     "base_pattern",
        "breakout":       "breakout_fam",
        "darvas_box":     "breakout_fam",
    }
    # Multi-strategy: ticker must fire in ≥2 INDEPENDENT strategy families
    from collections import Counter, defaultdict as _defaultdict
    _ticker_families: dict = _defaultdict(set)
    for _strat, _res_list in results_by_strategy.items():
        _fam = _STRAT_FAMILY.get(_strat, _strat)   # singleton strats are their own family
        for _r in _res_list:
            _ticker_families[_r["ticker"]].add(_fam)
    multi_tickers = {t for t, fams in _ticker_families.items() if len(fams) >= 2}

    # Load historical WR stats (must happen before any display function)
    global _HIST_STATS
    _HIST_STATS = _load_hist_stats()

    # Display
    _print_header(strategies, len(all_results), with_backtest, elder_count, spy_ret)
    _print_sector_pulse(sector_excess, spy_ret)

    # ── Portfolio heat (show BEFORE signals so trader knows available slots) ──
    try:
        _open_t = [t for t in load_trades() if t.get("status") == "OPEN"]
        _total_inv = sum(float(t.get("investment_eur") or 0) for t in _open_t)
        _slots_used = len(_open_t)
        _heat_col = GRN if _slots_used <= 4 else (YLW if _slots_used <= 5 else RED)
        print(_heat_col(f"  📊  Portfolio heat: {_slots_used}/6 slots open  ·  €{_total_inv:,.0f} deployed"
                        + ("  🔥 FULL — close a position before entering new trades" if _slots_used >= 6 else "")))
    except Exception:
        pass

    # ── Earnings block warning (deferred from scan progress for visibility) ──
    if _earnings_warn_msg:
        print(YLW(_earnings_warn_msg))

    _thematic_check()

    # ── LEAD WITH CONVICTION — what to act on ─────────────────────────────────
    _company_cache = _print_high_conviction(results_by_strategy, multi_tickers, with_backtest, sector_excess, elder_count)

    # ── PERSISTENCE LEADERS (≥5 consecutive trading days) ─────────────────────
    _print_streak_leaders(min_streak=5)

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
        # Update persistent company name cache with all tickers in this scan
        try:
            from company_cache import update_cache as _update_cache
            _scan_tickers = list({r["ticker"] for res in results_by_strategy.values() for r in res})
            _update_cache(_scan_tickers, max_workers=15)
        except Exception:
            pass

        # Enrich results with company names fetched during display
        _co = _company_cache or {}
        for res in results_by_strategy.values():
            for r in res:
                if not r.get("company"):
                    r["company"] = _co.get(r["ticker"], "")
        _out_file = HERE / ("last_scan_india.json" if india_mode else "last_scan.json")
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
            "india_mode": india_mode,
        }
        _out_file.write_text(json.dumps(payload, indent=2))
        LAST_SCAN_JSON = _out_file  # keep reference consistent
        print(DIM(f"  Saved last_scan.json ({sum(len(v) for v in payload['results_by_strategy'].values())} results)"))
    except Exception as e:
        print(f"  WARNING: could not save last_scan.json — {e}")

    # Auto-add HIGH conviction picks as practice trades (includes heat display)
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

    open_trades  = [t for t in existing_trades if t.get("status") == "OPEN"]
    open_tickers = {t["ticker"] for t in open_trades}
    MAX_OPEN     = 6
    total_invested = sum(float(t.get("investment_eur") or 0) for t in open_trades)

    # ── Portfolio heat check ───────────────────────────────────────────────────
    print()
    heat_color = GRN if len(open_trades) <= 4 else (YLW if len(open_trades) <= 5 else RED)
    print(heat_color(f"  📊  Portfolio heat: {len(open_trades)} open positions  ·  €{total_invested:,.0f} deployed"))
    if len(open_trades) >= MAX_OPEN:
        print(RED(f"  🔥  Heat limit reached ({MAX_OPEN} open) — no new trades added. Close a position first."))
        return

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

    # Fetch regime once for bear suppression
    _regime_today = fetch_market_regime(date.today())
    # Was `isinstance(_regime_today, dict) and ...` — fetch_market_regime()
    # returns the STRING "BULL"/"BEAR"/"", never a dict, so that check was
    # always False and this suppression never fired. Confirmed live on
    # 2026-08-17: the India digest was headlined "🔴 MARKET REGIME: BEAR"
    # while still handing out 8 HIGH-conviction trend-strategy longs — this
    # exact guard should have blocked those.
    _is_bear = (_regime_today == "BEAR")

    for _, r, strategy in picks:
        ticker = r["ticker"]

        # Bear regime: skip trend strategies (mean reversion only)
        if _is_bear and strategy in _TREND_STRATS:
            print(YLW(f"  [practice-auto] BEAR regime — skipping trend strategy {strategy} for {ticker}"))
            continue

        if ticker in open_tickers:
            skipped.append(ticker)
            continue

        try:
            ccy      = ticker_ccy(ticker)
            fx       = fetch_fx_on_date(ccy, today)
            price    = float(r.get("price") or 0)
            if not price:
                continue
            # yfinance quotes LSE ordinaries (.L) in pence, but ticker_ccy()
            # labels them "GBP" (pounds). r["price"] here is that raw pence
            # value straight from the scanner's Close column. Storing it
            # unconverted under a "GBP" label broke two things downstream:
            # (1) qty = invest_eur*fx/price came out 100x too small, so a
            #     trade "worth €1000" only deployed ~€10 of real exposure —
            #     confirmed against trades.csv id 15 (DCC.L): qty=0.1351 at
            #     a real price of 63.35 GBP is €10.00, not the claimed €1000.
            # (2) buy_price stored 100x high, so P&L vs the live price (which
            #     IS correctly pence-converted by show_tracker.fetch_live_price)
            #     read as a ~-99% loss on a position that hadn't moved.
            # Converting here, before qty/sl are computed, fixes both at once.
            if ticker.upper().endswith(".L"):
                price = round(price / 100.0, 4)
            sl       = trade_stop_loss(price)
            hold_d   = trade_hold_days(strategy)
            exit_dt  = biz_days_add(today, hold_d)
            qty      = round(1000.0 * fx / price, 4)
            inv_eur  = round(qty * price / fx, 2)
            company  = r.get("company") or fetch_company_name(ticker)
            sector   = fetch_sector(ticker)
            regime   = _regime_today

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
