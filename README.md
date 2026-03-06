# Inflection Market Report

Two self-contained daily reports, auto-generated every weekday via GitHub Actions and served on GitHub Pages — no server, no cost.

| Report | What it covers |
|--------|---------------|
| [**Market Report**](https://codebreaker89.github.io/inflection-market-report/market_report.html) | Sector rotation alerts · RRG snapshot · stock picks · signal backtests |
| [**Metal Tracker**](https://codebreaker89.github.io/inflection-market-report/metal_tracker.html) | Metal spot prices · supply shock events · global miner stocks |

> Refreshes every weekday at **8:00 AM Berlin / 7:00 AM UTC / 12:30 PM IST**
> To trigger manually: `gh workflow run daily_report.yml --repo Codebreaker89/inflection-market-report`

---

## Contents

- [Market Report](#market-report)
- [Metal Tracker](#metal-tracker)
- [Running Locally](#running-locally)
- [Schedule & Automation](#schedule--automation)

---

## Market Report

Scores 21 US sector ETFs every morning and surfaces the highest-conviction rotation opportunities.

**Sections**
- **Sector Rotation Alerts** — each sector scored via RRG + StealthTrail + 4-Factor confirmation → `ROTATING` / `WATCH` / `AVOID`
- **Live RRG Snapshot** — interactive RS-Ratio vs RS-Momentum scatter (Chart.js)
- **Stock Drill-Down** — for every rotating sector, ranked stock picks with SFRA backtests
- **Signal Action Summary** — New Signals · Room to Run · Extended · Best Quality
- **Stock Search** — type any ticker or company name for an instant signal card

**Signal ratings**

| Rating | Score | Meaning |
|--------|-------|---------|
| `ROTATING ★★★` | ≥ 12 | Leading vs SPY + StealthTrail active + 3–4 confirmation signals |
| `ROTATING ★★` | 9–11 | Improving vs SPY with 2+ signals |
| `WATCH ★` | 6–8 | Early rotation, not yet confirmed |
| `AVOID` | < 6 | Lagging or Weakening vs SPY |

**How signals are filtered**
1. RRG vs SPY — is the stock outperforming the broad market?
2. RRG vs Sector ETF — is it the strongest stock within its sector?
3. StealthTrail + 4-Factor — is there an active trend entry with institutional confirmation?

**Sectors covered** — 21 ETFs vs SPY:
`XLK` `XLF` `XLY` `XLP` `XLE` `XLV` `XLI` `XLB` `XLU` `XLRE` `XLC` `SOXX` `XRT` `IBB` `KRE` `ITA` `XOP` `OIH` `GDX` `ICLN` `JETS`

**Stock universe** — ~100 stocks across Energy, Technology, Semiconductors, Health Care, Financials, Consumer Discretionary, Industrials, Materials, Biotech, Retail, and more.

---

## Metal Tracker

Tracks *why* metal prices move — cause-and-effect intelligence across 11 metals and ~52 global producers.

**Sections**
- **Prices** — spot price · 1-day · 7-day · 1-month · 2-month % change for each metal, with an inline supply intelligence brief
- **Supply Shock Events** — up to 70 latest events from 20 sources (Mining.com, Kitco, Mining Weekly, Northern Miner + 16 Google News RSS feeds). Each event shows event type, affected metal(s), country, headline, extracted root cause, and price pressure direction
- **Global Metal Company Stocks** — ~52 producers grouped by metal, showing 10-day · 20-day · 1-month · 2-month % change

**Event types tracked**

| Type | Price pressure |
|------|---------------|
| Plant / Smelter Closure | ↑ |
| Labor Strike | ↑ |
| Supply Deficit | ↑ |
| Energy Crisis | ↑ |
| Natural Disaster | ↑ |
| Demand Surge | ↑ |
| New Supply / Restart | ↓ |
| Geopolitical / Sanctions | uncertain |
| Regulatory | ↑ |

**Metals covered** — Gold · Silver · Copper · Aluminum · Platinum · Palladium · Steel (HRC) · Crude Oil · Rare Earths (REMX ETF proxy) · Lithium (LIT ETF proxy) · Cobalt (VALE ADR proxy)

**Companies covered** — Hindalco · Vedanta · NALCO · Tata Steel · JSW Steel · Hindustan Zinc · SAIL · Freeport-McMoRan · Newmont · Barrick · Alcoa · Albemarle · SQM · MP Materials · Glencore · Vale · Sibanye · BHP · ArcelorMittal · and more

---

## Running Locally

```bash
# Install dependencies
pip install pandas numpy matplotlib yfinance feedparser beautifulsoup4 requests

# Market report  (~2 min first run, ~1 min after)
python3 market_report.py
open market_report.html

# Metal tracker  (~1 min)
python3 metal_tracker.py
open metal_tracker.html
```

Both scripts produce fully self-contained HTML files — open in any browser, no server needed.

---

## Schedule & Automation

| What | Value |
|------|-------|
| Trigger | GitHub Actions `schedule` + `workflow_dispatch` |
| Runs | Every weekday |
| Time | 8:00 AM Berlin · 7:00 AM UTC · 12:30 PM IST |
| Output | `market_report.html` + `metal_tracker.html` committed to `main` |
| Hosting | GitHub Pages (served from `main` branch root) |

**Manual trigger**
```bash
gh workflow run daily_report.yml --repo Codebreaker89/inflection-market-report
```
