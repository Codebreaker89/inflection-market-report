# Commands Reference

## Daily Workflow

### Run everything (scan + history + email preview)
```bash
cd ~/Claude/Projects/fire && python3 scan.py --no-backtest && python3 update_scan_history.py && python3 notify.py
```

### Open results after above
```bash
open ~/Claude/Projects/fire/tracker.html        # portfolio tracker (P&L, hold days, stop loss)
open ~/Claude/Projects/fire/digest_preview.html # email digest preview (alerts, positions)
open ~/Claude/Projects/fire/scan_history.csv    # historical scan dataset
```

---

## Individual Commands

### Run scan with backtest (~5-10 min, more accurate win%)
```bash
cd ~/Claude/Projects/fire && python3 scan.py
```

### Run scan without backtest (~1 min, faster)
```bash
cd ~/Claude/Projects/fire && python3 scan.py --no-backtest
```

### Update historical dataset (run after scan.py)
```bash
cd ~/Claude/Projects/fire && python3 update_scan_history.py
```

### Open portfolio tracker HTML (shows trades, P&L, 1w/2w returns)
```bash
cd ~/Claude/Projects/fire && python3 show_tracker.py && open tracker.html
```

### Preview email digest locally (no email sent)
```bash
cd ~/Claude/Projects/fire && python3 notify.py && open digest_preview.html
```

---

## Scheduled Job (runs automatically Mon–Fri 10am)

### Trigger manually
```bash
launchctl kickstart gui/$(id -u)/com.hemant.tradedigest
```

### Monitor live log
```bash
tail -f ~/Claude/Projects/fire/notify.log
```

### Reload schedule after plist changes
```bash
cp ~/Claude/Projects/fire/com.hemant.tradedigest.plist ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.hemant.tradedigest.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.hemant.tradedigest.plist
```

---

## Add a Trade to Tracker
```bash
cd ~/Claude/Projects/fire && python3 show_tracker.py
# Follow the interactive prompt at the end
```
