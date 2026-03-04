# Daily Market Report — Sector Rotation + Stock Picks

**🌐 Live Report:** [https://hdahake.github.io/inflection-market-report/](https://hdahake.github.io/inflection-market-report/)

> Auto-generated every weekday at 9:00 AM EST via GitHub Actions.
> To trigger manually: go to **Actions tab → Daily Market Report → Run workflow**.

---

A single Python script that runs every morning to produce a self-contained HTML dashboard covering:

- **§1 Sector Rotation Alerts** — 21 sector ETFs scored via RRG + StealthTrail + 4-Factor confirmation → ROTATING / WATCH / AVOID
- **§2 Live RRG Snapshot** — interactive RS-Ratio vs RS-Momentum scatter (Chart.js)
- **§3 Stock Drill-Down** — for every rotating sector, ranked stock picks with SFRA backtests
- **§4 Signal Action Summary** — New Signals / Room to Run / Extended / Best Quality at a glance
- **Stock Search** — type any ticker or company name to get an instant info card

## How It Works

### Three-layer stock filter
1. **RRG vs SPY** — is this stock outperforming the broad market?
2. **RRG vs Sector ETF** — is it the strongest stock *within* its sector?
3. **StealthTrail + 4-Factor** — is there an active trend entry with institutional confirmation?

### Two backtests per signal
- **QTA (Quadrant Transition Analysis)** — historical forward returns every time a sector entered the Leading RRG quadrant
- **SFRA (Signal Forward Return Analysis)** — historical forward returns after every StealthTrail BUY trigger, plus actual return since the current signal fired

## Usage

```bash
# First run — downloads all data (~286MB, ~144 tickers, ~2 mins)
python3 market_report.py

# Every subsequent morning — refreshes only stale data, runs in ~1 min
python3 market_report.py
```

Open `market_report.html` in any browser. Fully self-contained (no server needed).

## Requirements

```bash
pip install pandas numpy matplotlib yfinance
```

## Sectors Covered
21 sector & sub-sector ETFs vs SPY benchmark:
XLK, XLF, XLY, XLP, XLE, XLV, XLI, XLB, XLU, XLRE, XLC, SOXX, XRT, IBB, KRE, ITA, XOP, OIH, GDX, ICLN, JETS

## Stock Universe
~100 stocks across Energy, Utilities, Technology, Semiconductors, Health Care, Financials, Consumer Discretionary, Communication Services, Industrials, Materials, Biotech, Retail

## Key Signals Explained

| Signal | Meaning |
|--------|---------|
| **ROTATING ★★★** | Score ≥12 — Leading vs SPY + StealthTrail active + 3–4 confirmation signals. Act now. |
| **ROTATING ★★** | Score 9–11 — Improving vs SPY with 2+ signals. Enter with smaller size. |
| **WATCH ★** | Score 6–8 — Early rotation, not yet confirmed. Set price alerts. |
| **AVOID** | Score <6 — Lagging or Weakening vs SPY. No new positions. |

## Strategy Behind the Script
The core trend-following engine is **StealthTrail SuperTrend** (adaptive ATR multiplier + RSI filter), combined with **JdK Relative Rotation Graph** methodology for sector-level momentum. Signal quality is validated via event-study backtests (SFRA/QTA) on 15+ years of daily data.
