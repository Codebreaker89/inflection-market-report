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

import sys, os, time, json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
LAST_SCAN_JSON = HERE / "last_scan.json"

# ── IMPORTS ───────────────────────────────────────────────────────────────────
from momentum_scanner       import (scan as scan_momentum,
                                    build_universe, compute_bench_returns,
                                    print_results as _print_momentum)
from breakout_scanner       import scan as scan_breakout
from pocket_pivot_scanner   import scan as scan_pocket_pivot
from connors_rsi2_scanner   import scan as scan_connors
from ema_ribbon_scanner     import scan as scan_ema_ribbon
from nr7_scanner             import scan as scan_nr7
from bb_squeeze_scanner      import scan as scan_bb_squeeze
from high_tight_flag_scanner import scan as scan_htf
from analyst_upgrade_scanner import scan as scan_analyst_upgrade
from signal_velocity_scanner      import scan as scan_signal_velocity
from chokepoint_inflection_scanner import scan as scan_chokepoint
from stage4_short_scanner          import scan as scan_stage4_short
from defensive_rotation_scanner    import scan as scan_defensive_rotation
from cup_handle_scanner            import scan as scan_cup_handle
from power_earnings_gap_scanner    import scan as scan_peg
from show_tracker                  import add_trade_interactive

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
ALL_STRATEGIES = ["momentum", "breakout", "pocket_pivot", "connors_rsi2",
                  "ema_ribbon", "nr7", "bb_squeeze", "high_tight_flag",
                  "analyst_upgrade", "signal_velocity", "chokepoint_inflection",
                  "stage4_short", "defensive_rotation",
                  "cup_handle", "power_earnings_gap"]

SCANNER_MAP = {
    "momentum":        scan_momentum,
    "breakout":        scan_breakout,
    "pocket_pivot":    scan_pocket_pivot,
    "connors_rsi2":    scan_connors,
    "ema_ribbon":      scan_ema_ribbon,
    "nr7":             scan_nr7,
    "bb_squeeze":      scan_bb_squeeze,
    "high_tight_flag": scan_htf,
    "analyst_upgrade":      scan_analyst_upgrade,
    "signal_velocity":      scan_signal_velocity,
    "chokepoint_inflection": scan_chokepoint,
    "stage4_short":          scan_stage4_short,
    "defensive_rotation":    scan_defensive_rotation,
    "cup_handle":            scan_cup_handle,
    "power_earnings_gap":    scan_peg,
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
    "power_earnings_gap":    10,
}

W = 110


# ── DISPLAY ───────────────────────────────────────────────────────────────────

def _print_group(strategy: str, results: list, with_backtest: bool):
    """Print a single strategy group."""
    label = STRATEGY_LABELS.get(strategy, strategy.upper())
    hold  = HOLD_DAYS_MAP.get(strategy, 5)
    count = len(results)
    desc  = STRATEGY_DESCRIPTIONS.get(strategy, "")

    print()
    print("┌" + "─"*(W-2) + "┐")
    print("│" + f"  {BOLD(label)}  ·  {count} signal(s)  ·  hold={hold}d".ljust(W+8) + "│")
    if desc:
        for line in desc.split("\n"):
            print("│" + DIM(f"  {line}").ljust(W+8) + "│")
    print("└" + "─"*(W-2) + "┘")

    if not results:
        print(DIM("  (no signals)"))
        return

    print()
    if with_backtest:
        hdr = (f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}  "
               f"{'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'M':>3}  {'SCR':>3}  "
               f"{'#BT':>3}  {'WIN%':>5}  {'AVG':>7}  {'MED':>7}  SIGNALS")
    else:
        hdr = (f"  {'#':>3}  {'MKT':<3}  {'TICKER':<8}  {'PRICE':>9}  "
               f"{'RSI':>5}  {'ADX':>5}  {'VOL×':>5}  {'M':>3}  {'SCR':>3}  SIGNALS")
    print(BOLD(hdr))
    print("  " + "─"*(W-2))

    # Sort by win rate desc, then score desc
    sorted_r = sorted(results, key=lambda r: (-(r.get("wr") or 0), -r.get("score", 0)))

    for rank, r in enumerate(sorted_r[:50], 1):
        fresh_str = " ".join(r.get("fresh", []))
        conf_str  = ("  · " + " ".join(r.get("conf", []))) if r.get("conf") else ""
        sig_str   = CYN(fresh_str) + DIM(conf_str)
        mkt_s     = YLW(f"{r['mkt']:<3}") if r["mkt"] != "US" else DIM(f"{r['mkt']:<3}")
        ticker_s  = BOLD(f"{r['ticker']:<8}")
        base = (f"  {rank:>3}  {mkt_s}  {ticker_s}  {r.get('price', 0):>9.2f}"
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

def _print_matrix(results_by_strategy: dict, strategies: list):
    """Print ticker × strategy pass/fail matrix."""
    # Collect all tickers that passed at least one strategy
    all_tickers = {}   # ticker → {strategy: result_dict}
    for strat, results in results_by_strategy.items():
        for r in results:
            t = r["ticker"]
            if t not in all_tickers:
                all_tickers[t] = {}
            all_tickers[t][strat] = r

    if not all_tickers:
        return

    # Short column labels
    col_labels = {
        "momentum":       "MNTM",
        "breakout":       "BRKOUT",
        "pocket_pivot":   "PP",
        "connors_rsi2":   "RSI2",
        "ema_ribbon":     "EMARIBN",
        "nr7":             "NR7",
        "bb_squeeze":      "BBSQZ",
        "high_tight_flag": "HTF",
        "analyst_upgrade": "ANUPGRD",
        "signal_velocity":       "SIGVEL",
        "chokepoint_inflection": "CHKPNT",
        "stage4_short":          "S4SHORT",
        "defensive_rotation":    "DEFROT",
        "cup_handle":            "C&H",
        "power_earnings_gap":    "PEG",
    }
    cols = [col_labels.get(s, s[:6].upper()) for s in strategies]
    col_w = [max(len(c), 6) for c in cols]

    print()
    print(BOLD("  CROSS-STRATEGY MATRIX  —  tickers that passed ≥1 scanner"))
    print()

    # Header
    hdr = f"  {'TICKER':<10}  {'COMPANY':<24}"
    for c, w in zip(cols, col_w):
        hdr += f"  {c:^{w}}"
    print(BOLD(hdr))
    print("  " + "─" * (W - 2))

    # Sort: tickers passing most strategies first
    sorted_tickers = sorted(all_tickers.items(),
                            key=lambda kv: -len(kv[1]))

    for ticker, strat_map in sorted_tickers[:60]:
        company = ""
        for r in strat_map.values():
            company = r.get("company", "") or ""
            if company: break
        # Try to get company from any result
        if not company:
            for r in strat_map.values():
                company = r.get("ticker", "")

        passes = len(strat_map)
        ticker_s = RED(BOLD(f"{ticker:<10}")) if passes > 1 else f"{ticker:<10}"
        company_s = RED(f"{str(company)[:24]:<24}") if passes > 1 else f"{str(company)[:24]:<24}"
        row = f"  {ticker_s}  {company_s}"
        for strat, w in zip(strategies, col_w):
            if strat in strat_map:
                r = strat_map[strat]
                wr = r.get("wr")
                if wr is not None:
                    cell = GRN(f"{'✓ '+str(int(wr))+'%':^{w}}")
                else:
                    cell = GRN(f"{'✓':^{w}}")
            else:
                cell = DIM(f"{'─':^{w}}")
            row += f"  {cell}"
        print(row)

    multi = sum(1 for _, m in sorted_tickers if len(m) > 1)
    if multi:
        print()
        print(GRN(f"  ★  {multi} ticker(s) passed multiple strategies — highest conviction"))
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
        scanner = SCANNER_MAP[strategy]
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

    total_time = time.time() - t0

    # Display
    _print_header(strategies, len(all_results), with_backtest)
    for strategy in strategies:
        _print_group(strategy, results_by_strategy[strategy], with_backtest)

    # Cross-strategy matrix
    _print_matrix(results_by_strategy, strategies)

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
            }
        }
        LAST_SCAN_JSON.write_text(json.dumps(payload, indent=2))
        print(DIM(f"  Saved last_scan.json ({sum(len(v) for v in payload['results_by_strategy'].values())} results)"))
    except Exception as e:
        print(f"  WARNING: could not save last_scan.json — {e}")

    # Tracker prompt
    if all_results:
        _tracker_prompt(all_results)


if __name__ == "__main__":
    main()
