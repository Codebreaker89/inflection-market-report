# Inflection Market Report

Two daily reports, auto-generated every weekday and served via GitHub Pages — no server, no cost.

| Report | Live Link | What it covers |
|--------|-----------|----------------|
| **Market Report** | [market_report.html](https://hdahake.github.io/inflection-market-report/market_report.html) | Sector rotation alerts, RRG snapshot, stock picks, backtests |
| **Metal Tracker** | [metal_tracker.html](https://hdahake.github.io/inflection-market-report/metal_tracker.html) | Metal spot prices, supply shock events, global miner stocks |

> **Schedule:** Both reports regenerate every weekday at **11:10 AM Berlin / 10:10 UTC / 3:40 PM IST**.
> To trigger manually: **Actions tab → Daily Reports → Run workflow**.

---

## Report 1 — Market Report (`market_report.py`)

A sector rotation dashboard covering US equities:

- **§1 Sector Rotation Alerts** — 21 sector ETFs scored via RRG + StealthTrail + 4-Factor confirmation → ROTATING / WATCH / AVOID
- **§2 Live RRG Snapshot** — interactive RS-Ratio vs RS-Momentum scatter (Chart.js)
- **§3 Stock Drill-Down** — for every rotating sector, ranked stock picks with SFRA backtests
- **§4 Signal Action Summary** — New Signals / Room to Run / Extended / Best Quality at a glance
- **Stock Search** — type any ticker or company name for an instant info card

### Three-layer stock filter
1. **RRG vs SPY** — is this stock outperforming the broad market?
2. **RRG vs Sector ETF** — is it the strongest stock *within* its sector?
3. **StealthTrail + 4-Factor** — is there an active trend entry with institutional confirmation?

### Two backtests per signal
- **QTA (Quadrant Transition Analysis)** — historical forward returns every time a sector entered the Leading RRG quadrant
- **SFRA (Signal Forward Return Analysis)** — historical forward returns after every StealthTrail BUY trigger, plus actual return since the current signal fired

### Sectors Covered
21 sector & sub-sector ETFs vs SPY:
XLK, XLF, XLY, XLP, XLE, XLV, XLI, XLB, XLU, XLRE, XLC, SOXX, XRT, IBB, KRE, ITA, XOP, OIH, GDX, ICLN, JETS

### Stock Universe
~100 stocks across Energy, Utilities, Technology, Semiconductors, Health Care, Financials, Consumer Discretionary, Communication Services, Industrials, Materials, Biotech, Retail

### Key Signals

| Signal | Meaning |
|--------|---------|
| **ROTATING ★★★** | Score ≥12 — Leading vs SPY + StealthTrail active + 3–4 confirmation signals. Act now. |
| **ROTATING ★★** | Score 9–11 — Improving vs SPY with 2+ signals. Enter with smaller size. |
| **WATCH ★** | Score 6–8 — Early rotation, not yet confirmed. Set price alerts. |
| **AVOID** | Score <6 — Lagging or Weakening vs SPY. No new positions. |

---

## Report 2 — Metal Tracker (`metal_tracker.py`)

A cause-and-effect intelligence dashboard for metals investing:

- **Prices** — spot price + 1-month and 2-month % change for 11 metals (Gold, Silver, Copper, Aluminum, Platinum, Palladium, Steel HRC, Crude Oil, + ETF proxies for Rare Earths, Lithium, Cobalt). Inline supply intelligence brief per metal.
- **Supply Shock Events** — up to 70 latest events aggregated from 20 sources (Mining.com, Kitco, Mining Weekly, Northern Miner + 16 targeted Google News RSS feeds). Each event shows: date, event type (Plant Closure / Strike / Geopolitical / …), affected metal(s), country, headline, root cause extracted from the headline, and price pressure direction (↑ / ↓ / Unclear).
- **Global Metal Company Stocks** — ~52 major producers grouped by metal (Aluminum, Copper, Gold, Silver, Steel, Nickel, Zinc, Platinum/Palladium, Rare Earths, Lithium, Cobalt, Diversified), showing 10-day, 20-day, 1-month, and 2-month % change. Covers India (Hindalco, Vedanta, NALCO, Tata Steel, JSW Steel, Hindustan Zinc, SAIL), Americas, Europe, Australia, and Asia.

### Data sources
- Prices: Yahoo Finance futures tickers via `yfinance`
- Events: RSS feeds from mining specialist publications + Google News; Mining.com homepage scrape
- Stocks: Yahoo Finance via `yfinance`, fetched in parallel

### Master filter
A metal filter bar at the top (Gold / Silver / Copper / …) instantly filters all three sections — prices, events, and stocks — to the selected metal.

---

## Running locally

```bash
pip install pandas numpy matplotlib yfinance feedparser beautifulsoup4 requests

# Market report (~2 min first run, ~1 min after)
python3 market_report.py
open market_report.html

# Metal tracker (~1 min)
python3 metal_tracker.py
open metal_tracker.html
```

Both output fully self-contained HTML files — open in any browser, no server needed.

---

## Strategy Behind the Scripts

**Market Report:** Core engine is **StealthTrail SuperTrend** (adaptive ATR multiplier + RSI filter) combined with **JdK Relative Rotation Graph** methodology. Signal quality validated via event-study backtests (SFRA/QTA) on 15+ years of daily data.

**Metal Tracker:** Intent is *cause → effect* intelligence — tracking plant closures, strikes, geopolitical events, and natural disasters that drive supply shocks and subsequent price moves, not just the price data itself.
