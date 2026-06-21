#!/bin/bash
# Run trade digest — called by macOS LaunchAgent daily at 5:30 PM
cd /Users/hemant.dahake/Claude/Projects/fire
LOG=/Users/hemant.dahake/Claude/Projects/fire/notify.log

echo "=== $(date) ===" >> $LOG
echo "--- To monitor progress: tail -f $LOG ---" >> $LOG

# 1. Run all strategy scanners (no-backtest for speed; saves last_scan.json)
echo "[1/3] Running scanners..." >> $LOG
python3 scan.py --no-backtest >> $LOG 2>&1

# 2. Append today's scan results + backfill d5/d10 returns for past rows
echo "[2/3] Updating scan history..." >> $LOG
python3 update_scan_history.py >> $LOG 2>&1

# 3. Send daily email digest + save local preview
echo "[3/3] Sending email digest..." >> $LOG
python3 notify.py --send >> $LOG 2>&1
python3 notify.py >> $LOG 2>&1

echo "=== Done $(date) ===" >> $LOG
