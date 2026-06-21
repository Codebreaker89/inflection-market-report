#!/bin/bash
# ─────────────────────────────────────────────
# Fire Trading System — Setup Script
# Run once on a new machine after cloning repo.
# ─────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

echo ""
echo "=== Fire Trading System Setup ==="
echo ""

# 1. Check Python 3.9+
PY=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo $PY | cut -d. -f1)
MINOR=$(echo $PY | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
    echo "ERROR: Python 3.9+ required. Found: $PY"
    echo "Install from https://www.python.org/downloads/"
    exit 1
fi
echo "[1/3] Python $PY — OK"

# 2. Install dependencies
echo "[2/3] Installing Python packages..."
pip3 install -r requirements.txt --quiet
echo "      Done."

# 3. Config check
if [ ! -f config.py ]; then
    echo ""
    echo "[3/3] config.py not found — creating from template..."
    cat > config.py << 'CONF'
# ── Email ──────────────────────────────────────────────────────────────────────
import os
GMAIL_USER         = os.environ.get("GMAIL_USER",         "dahakehemant@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "YOUR_APP_PASSWORD_HERE")
NOTIFY_TO          = os.environ.get("GMAIL_USER",         "dahakehemant@gmail.com")

# ── Trade defaults ──────────────────────────────────────────────────────────────
STOP_LOSS_PCT   = 0.03
PROFIT_TARGET   = 0.08
EARNINGS_WARN   = 2

# ── Hold days by strategy ───────────────────────────────────────────────────────
HOLD_DAYS = {
    "momentum": 5, "breakout": 5, "pocket_pivot": 7,
    "connors_rsi2": 5, "ema_ribbon": 7, "nr7": 3,
    "bb_squeeze": 7, "high_tight_flag": 10,
    "analyst_upgrade": 7, "signal_velocity": 5,
}
DEFAULT_HOLD_DAYS = 7
CONF
    echo "      config.py created. Fill in GMAIL_APP_PASSWORD before running notify.py."
else
    echo "[3/3] config.py found — OK"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run a scan:       cd $(pwd) && python3 scan.py --no-backtest"
echo "Update history:   python3 update_scan_history.py"
echo "Preview email:    python3 notify.py && open digest_preview.html"
echo ""
