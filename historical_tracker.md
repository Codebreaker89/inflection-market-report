# Historical Scan Tracker

`scan_history.csv` records every ticker flagged by any scanner on any given day,
then backfills its actual price return at day+5 and day+10. Purpose: build a
ground-truth dataset to measure whether each strategy's signals actually make
money, and to identify which signal combinations or market conditions improve hit rate.

---

## Column Reference

| Column | Type | Description |
|---|---|---|
| `scan_date` | date | Date scan.py was run (YYYY-MM-DD) |
| `ticker` | str | yfinance symbol (e.g. `ADI`, `IFX.DE`) |
| `company` | str | Company name |
| `strategy` | str | Strategy that flagged it: `momentum`, `breakout`, `pocket_pivot`, `connors_rsi2`, `ema_ribbon` |
| `strategies_count` | int | How many strategies flagged this ticker on this scan_date. >1 = high conviction |
| `price_at_scan` | float | Closing price on scan_date (native currency) |
| `score` | float | Scanner's composite score (higher = stronger signal) |
| `wr` | float | Backtest win rate % at time of scan (from scanner's own historical backtest) |
| `avg` | float | Backtest average return % at time of scan |
| `adx` | float | ADX value at scan (trend strength; >25 = trending) |
| `rsi` | float | RSI(14) at scan |
| `vol_ratio` | float | Today's volume ÷ 20-day avg volume at scan |
| `price_d5` | float | Actual closing price 5 calendar days after scan_date |
| `ret_d5` | float | % return from price_at_scan to price_d5 |
| `spy_ret_d5` | float | SPY % return over same 5-day window |
| `excess_ret_d5` | float | ret_d5 − spy_ret_d5 (alpha: did the pick beat the market?) |
| `hit_stop_loss_d5` | 0/1 | 1 if price_d5 ≤ price_at_scan × 0.97 |
| `price_d10` | float | Actual closing price 10 calendar days after scan_date |
| `ret_d10` | float | % return from price_at_scan to price_d10 |
| `spy_ret_d10` | float | SPY % return over same 10-day window |
| `excess_ret_d10` | float | ret_d10 − spy_ret_d10 |
| `hit_stop_loss_d10` | 0/1 | 1 if low in d0→d10 window ever hit price_at_scan × 0.97 |
| `max_drawdown_d10` | float | Worst close vs price_at_scan in the 10-day window (negative = loss) |

Blank cells = not enough time has passed yet (backfilled automatically on next run).

---

## Analysis Playbook

Load the CSV with pandas: `df = pd.read_csv("scan_history.csv")`

### 1. Is each strategy actually profitable?
```python
df.groupby("strategy")[["ret_d5","ret_d10","excess_ret_d5","excess_ret_d10"]].mean()
```
If `excess_ret` is near 0 or negative, the strategy isn't generating alpha — it's just
riding the market. You want `excess_ret_d5 > 1%` consistently.

### 2. Does higher scanner score → better return?
```python
import numpy as np
df["score_bucket"] = pd.qcut(df["score"].astype(float), 4, labels=["Q1","Q2","Q3","Q4"])
df.groupby(["strategy","score_bucket"])["ret_d10"].mean()
```
If Q4 (highest score) outperforms Q1, score is a useful signal. If not, reconsider
how the score is computed.

### 3. Do cross-strategy tickers outperform single-strategy?
```python
df.groupby("strategies_count")[["ret_d5","ret_d10","excess_ret_d10"]].mean()
```
Hypothesis: `strategies_count >= 2` should have higher returns and lower stop-loss rate.

### 4. Stop loss hit rate by strategy
```python
df.groupby("strategy")[["hit_stop_loss_d5","hit_stop_loss_d10"]].mean() * 100
```
If a strategy hits stop loss >30% of the time, its signal quality or entry timing is poor.

### 5. Win rate: % of signals with positive excess return
```python
df["win_d10"] = (df["excess_ret_d10"].astype(float) > 0).astype(int)
df.groupby("strategy")["win_d10"].mean() * 100
```
Compare to the scanner's own `wr` (backtest win rate). If actual win rate << `wr`,
the backtest is overfitting or look-ahead biased.

### 6. Best conditions for each strategy (ADX, RSI, vol_ratio filters)
```python
# Example: does high ADX at scan time predict better momentum returns?
df_mom = df[df["strategy"]=="momentum"].copy()
df_mom["adx_bucket"] = pd.cut(df_mom["adx"].astype(float), bins=[0,20,30,100], labels=["weak","medium","strong"])
df_mom.groupby("adx_bucket")["excess_ret_d10"].mean()
```
Use this to tighten filter thresholds in the scanner.

### 7. Seasonality / day-of-week effect
```python
df["weekday"] = pd.to_datetime(df["scan_date"]).dt.day_name()
df.groupby("weekday")["ret_d5"].mean()
```
If Monday scans consistently outperform Friday scans, factor that into when to run.

### 8. Max drawdown vs actual return
```python
df[["max_drawdown_d10","ret_d10"]].astype(float).corr()
```
Stocks with deep drawdowns that recover indicate whipsaw — strategy may need a wider
stop or earlier exit signal.

### 9. Backtest wr vs actual win rate correlation
```python
df["actual_win_d10"] = (df["ret_d10"].astype(float) > 0).astype(int)
df[["wr","actual_win_d10"]].astype(float).corr()
```
High correlation means the scanner's backtest is predictive. Low = backtest is noise.

### 10. Strategy improvement actions (based on analysis)

| Finding | Action |
|---|---|
| excess_ret near 0 | Raise Minervini score threshold or ADX minimum |
| actual win rate << backtest wr | Check for look-ahead bias in scanner backtest window |
| stop loss rate > 30% | Tighten entry criteria (e.g. require vol_ratio > 1.5 instead of 1.2) |
| strategies_count≥2 beats single | Prioritise cross-strategy tickers in daily review |
| Q4 score doesn't beat Q1 | Score formula needs reweighting — run feature importance |
| High drawdown even on winning trades | Widen stop loss or add trailing stop logic |
