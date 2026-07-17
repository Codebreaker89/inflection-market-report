# Fire — Daily Swing Trade Digest

A daily email digest sent every weekday morning via GitHub Actions. Scans ~620 stocks across US, UK, Germany and France, identifies momentum opportunities, and tracks your open trades.

> Runs every weekday at **8:00 AM Berlin / 7:00 AM UTC / 12:30 PM IST**
> Manual trigger: `gh workflow run fire_scan.yml --repo Codebreaker89/inflection-market-report`

---

## What the email contains

The digest is a single HTML email with these sections, top to bottom:

### 1. Action Alerts
Stop-loss hits, hold-period expirations, profit targets reached, and earnings warnings for your **open trades**.

### 2. Portfolio Snapshot
All open trades with live P&L, entry price vs current price, and holding days.

### 3. Scanner Results

| Section | What it says in plain English |
|---------|-------------------------------|
| **Market Regime** | Is the market in a bull, neutral, or bear phase? |
| **Sector Strength** | Which sectors are hot (outperforming SPY) and which are cold |
| **🎯 ACT ON THESE** | Highest-conviction stock picks — ranked, with entry price and stop loss |
| **👀 WATCHLIST** | Stocks on the radar but not yet fully confirmed |
| **🔁 Persistence Leaders** | Stocks showing up in scans for 5+ consecutive days |
| **Strategy Scorecard** | Which scanning strategies are actually working (win rate, avg return) |
| **Full Scan Detail** | All raw signals by strategy for reference |

### 4. Metals Snapshot
Live spot prices for Gold, Silver, Copper, Crude Oil, Aluminum, Platinum, Rare Earths, and Lithium — with 1-day and 7-day % change. Plus the latest supply shock events (mine closures, strikes, shortages) from news feeds.

### 5. RRG Sector Rotation *(bottom)*
Relative Rotation Graph — shows which sectors are Leading, Improving, Weakening, or Lagging vs SPY. Includes a visual scatter chart with trailing arrows showing direction of movement.

---

## Files

| File | Purpose |
|------|---------|
| `scan.py` | Runs all 20+ scanning strategies against ~620 tickers |
| `notify.py` | Builds and sends the HTML email digest |
| `rrg_engine.py` | RRG math + chart generation (shared module) |
| `update_scan_history.py` | Logs scan outcomes to `scan_history.csv` for live win-rate tracking |
| `show_tracker.py` | Price fetching utilities |
| `trades.csv` | Your open/closed trades *(gitignored)* |
| `config.py` | Gmail credentials and thresholds *(gitignored)* |

---

## GitHub Actions

| Workflow | Schedule | What it does |
|----------|----------|-------------|
| `fire_scan.yml` | Weekdays 8 AM Berlin | Scan → email digest → commit `scan_history.csv` + `last_scan.json` |

*(The old `daily_report.yml` that ran `market_report.py` and `metal_tracker.py` has been decommissioned — that content is now inside the email digest.)*

---

## Running locally

```bash
# Dry run (prints email to terminal, no send)
python3 notify.py

# Actually send
python3 notify.py --send

# Just run the scanner
python3 scan.py
```

**Dependencies:**
```
pip install yfinance pandas numpy matplotlib feedparser beautifulsoup4 requests
```

---

## Strategy overview

20+ strategies covering momentum, mean-reversion, breakout, and defensive patterns:

| Category | Strategies |
|----------|-----------|
| Momentum | `pocket_pivot`, `ema_ribbon`, `elder_impulse`, `momentum` |
| Breakout | `breakout`, `darvas_box`, `vcp`, `cup_handle` |
| Mean-reversion | `connors_rsi2`, `connors_3down`, `williams_pct_r`, `bollinger_pctb` |
| Volatility | `nr7`, `bb_squeeze` |
| Trend-pullback | `holy_grail`, `rs_line`, `defensive_rotation` |
| Event-driven | `power_earnings_gap`, `analyst_upgrade` |
| Short | `stage4_short` |

**Proven Edge set** (strategies with ≥10 trades and ≥60% win rate): `pocket_pivot`, `ema_ribbon`, `cup_handle`, `signal_velocity`, `connors_rsi2`

---

## Security

- `config.py` contains Gmail credentials — **never commit**, listed in `.gitignore`
- `trades.csv` contains trade data — **never commit**, listed in `.gitignore`
