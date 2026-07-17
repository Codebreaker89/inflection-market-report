"""
rrg_engine.py — Relative Rotation Graph (RRG) sector analysis
Shared by notify.py (email digest) and market_report.py (deep analysis).

Usage:
    from rrg_engine import run_sector_rrg, chart_rrg_scatter

    results = run_sector_rrg()          # list of sector dicts, sorted by score
    b64_png = chart_rrg_scatter(results) # base64 PNG for email <img> tag
"""

import warnings
warnings.filterwarnings("ignore")

import io, base64
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ── Sector ETF map (ETF → display name) ──────────────────────────────────────
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
}

# Quadrant → colour for charts
QUAD_COLORS = {
    "Leading":   "#10b981",
    "Improving": "#3b82f6",
    "Weakening": "#f59e0b",
    "Lagging":   "#ef4444",
}

# Quadrant → emoji for email text
QUAD_EMOJI = {
    "Leading":   "🟢",
    "Improving": "🔵",
    "Weakening": "🟡",
    "Lagging":   "🔴",
}


# ── Core math ─────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def compute_rrg(stock_close: pd.Series, bench_close: pd.Series):
    """
    Return (rs_ratio, rs_momentum) Series aligned on the same index.
    RS-Ratio  >100 = outperforming SPY; RS-Momentum >100 = acceleration.
    Standard double-EMA smoothing (JdK RS-Ratio methodology).
    """
    both  = pd.concat([stock_close.rename("s"), bench_close.rename("b")], axis=1).dropna()
    rs    = 100 * both["s"] / both["b"]
    rs1   = _ema(rs, 10)
    rs2   = _ema(rs, 26)
    ratio = 100 + (rs1 - rs2) / rs2 * 100
    mom   = 100 + (_ema(ratio, 10) - _ema(ratio, 26)) / _ema(ratio, 26) * 100
    return ratio, mom


def quadrant(ratio_val: float, mom_val: float) -> str:
    if   ratio_val >= 100 and mom_val >= 100: return "Leading"
    elif ratio_val >= 100 and mom_val <  100: return "Weakening"
    elif ratio_val <  100 and mom_val >= 100: return "Improving"
    else:                                      return "Lagging"


def _sector_score(quad: str, rs_ratio: float, rs_mom: float) -> float:
    """Simple numeric score for sorting: quad rank + proximity to Leading."""
    base = {"Leading": 4, "Improving": 3, "Weakening": 2, "Lagging": 1}[quad]
    # Small bonus for how far into outperformance territory
    ratio_bonus = (rs_ratio - 100) / 10 if rs_ratio > 100 else (rs_ratio - 100) / 20
    mom_bonus   = (rs_mom   - 100) / 10 if rs_mom   > 100 else (rs_mom   - 100) / 20
    return round(base + ratio_bonus + mom_bonus, 3)


# ── Live data fetch + RRG computation ─────────────────────────────────────────

def run_sector_rrg(lookback_days: int = 365) -> list[dict]:
    """
    Download SPY + all sector ETFs via yfinance, compute RRG for each.
    Returns list of dicts sorted best→worst:
      {etf, name, quad, rs_ratio, rs_mom, score, trail}
    trail = list of (ratio, mom) tuples for last 8 weekly samples (for chart).
    Fail-open: returns [] if yfinance unavailable.
    """
    try:
        import yfinance as yf
    except ImportError:
        return []

    end   = datetime.today()
    start = end - timedelta(days=lookback_days + 60)  # extra buffer for EMA warmup

    tickers = ["SPY"] + list(SECTOR_ETFS.keys())
    try:
        raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False)
    except Exception:
        return []

    # Handle multi-level columns from yf.download
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            closes = raw["Close"]
        except KeyError:
            closes = raw.xs("Close", axis=1, level=0)
    else:
        closes = raw[["Close"]] if "Close" in raw.columns else raw

    if "SPY" not in closes.columns:
        return []

    spy = closes["SPY"].dropna()
    results = []

    for etf, name in SECTOR_ETFS.items():
        if etf not in closes.columns:
            continue
        etf_close = closes[etf].dropna()
        comm = spy.index.intersection(etf_close.index)
        if len(comm) < 100:
            continue

        ratio, mom = compute_rrg(etf_close.loc[comm], spy.loc[comm])
        if ratio.empty or mom.empty:
            continue

        last_ratio = float(ratio.iloc[-1])
        last_mom   = float(mom.iloc[-1])
        quad       = quadrant(last_ratio, last_mom)
        score      = _sector_score(quad, last_ratio, last_mom)

        # Weekly trail: last 8 weekly samples (every 5 trading days)
        trail_locs = list(range(-40, 0, 5))
        trail = []
        for i in trail_locs:
            if abs(i) <= len(ratio):
                trail.append((round(float(ratio.iloc[i]), 3),
                               round(float(mom.iloc[i]),   3)))
        trail.append((round(last_ratio, 3), round(last_mom, 3)))

        results.append({
            "etf":      etf,
            "name":     name,
            "quad":     quad,
            "rs_ratio": round(last_ratio, 2),
            "rs_mom":   round(last_mom,   2),
            "score":    score,
            "trail":    trail,
        })

    results.sort(key=lambda x: -x["score"])
    return results


# ── Chart generation ──────────────────────────────────────────────────────────

def chart_rrg_scatter(sector_results: list[dict]) -> str | None:
    """
    Generate RRG scatter plot with trailing arrows.
    Returns base64-encoded PNG string, or None if matplotlib unavailable.
    """
    if not sector_results:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#111827")

    # Axis limits from all trail points
    all_x = [pt[0] for r in sector_results for pt in r.get("trail", [(r["rs_ratio"], r["rs_mom"])])]
    all_y = [pt[1] for r in sector_results for pt in r.get("trail", [(r["rs_ratio"], r["rs_mom"])])]
    pad   = 0.8
    xlim  = [min(min(all_x) - pad, 98.5), max(max(all_x) + pad, 101.5)]
    ylim  = [min(min(all_y) - pad, 98.5), max(max(all_y) + pad, 101.5)]

    # Quadrant shading + labels
    for (x1, x2), (y1, y2), col, lbl in [
        ((100, xlim[1]), (100, ylim[1]), "#10b981", "Leading"),
        ((100, xlim[1]), (ylim[0], 100), "#f59e0b", "Weakening"),
        ((xlim[0], 100), (100, ylim[1]), "#3b82f6", "Improving"),
        ((xlim[0], 100), (ylim[0], 100), "#ef4444", "Lagging"),
    ]:
        ax.fill_between([x1, x2], [y1, y1], [y2, y2], color=col, alpha=0.07)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, lbl,
                ha="center", va="center", fontsize=11,
                color=col, alpha=0.25, fontweight="bold")

    ax.axhline(100, color="white", lw=0.8, alpha=0.35)
    ax.axvline(100, color="white", lw=0.8, alpha=0.35)

    for r in sector_results:
        c     = QUAD_COLORS.get(r["quad"], "#aaaaaa")
        trail = r.get("trail", [])
        n     = len(trail)

        if n >= 2:
            for i in range(n - 1):
                x0, y0 = trail[i]
                x1_t, y1_t = trail[i + 1]
                alpha = 0.08 + 0.47 * (i / max(n - 2, 1))
                lw    = 0.7 + 1.3 * (i / max(n - 2, 1))
                ax.plot([x0, x1_t], [y0, y1_t], color=c,
                        alpha=alpha, lw=lw, solid_capstyle="round", zorder=3)
            for i, (tx, ty) in enumerate(trail[:-1]):
                ax.scatter(tx, ty, s=14, c=c,
                           alpha=0.10 + 0.30 * (i / max(n - 2, 1)), zorder=3)
            # Direction arrow
            x0, y0   = trail[-2]
            x1_t, y1_t = trail[-1]
            ax.annotate("", xy=(x1_t, y1_t), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=c,
                                        lw=1.8, mutation_scale=12),
                        zorder=5)

        ax.scatter(r["rs_ratio"], r["rs_mom"], s=160, c=c,
                   alpha=0.95, edgecolors="white", linewidths=0.6, zorder=6)
        ax.annotate(r["etf"], (r["rs_ratio"], r["rs_mom"]),
                    textcoords="offset points", xytext=(6, 4),
                    color="white", fontsize=8, fontweight="bold", zorder=7)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("RS-Ratio  (>100 = outperforming SPY)", color="#9ca3af", fontsize=9)
    ax.set_ylabel("RS-Momentum  (>100 = accelerating)",   color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#374151")

    handles = [mpatches.Patch(color=v, label=k) for k, v in QUAD_COLORS.items()]
    ax.legend(handles=handles, facecolor="#111827", labelcolor="white",
              fontsize=8, loc="upper left", framealpha=0.8)

    today_str = datetime.today().strftime("%b %d, %Y")
    ax.set_title(f"RRG — Sector Rotation vs SPY  ·  {today_str}  (trail = 8 weeks)",
                 color="white", fontsize=11, pad=10, fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()
