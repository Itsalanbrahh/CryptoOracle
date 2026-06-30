#!/bin/bash
#
# install_kalshi_cron.sh — set up (or tear down) the Kalshi BTC trading cron jobs.
#
# Installs three jobs, all tagged "# kalshi-oracle" so this script can manage
# them idempotently without touching your other crontab entries:
#
#   • Entry scan         every 30 min   kalshi_live_trade.sh
#       Runs the 13-agent ensemble and places NO trades that clear the gates.
#   • Position heartbeat  every 15 min  kalshi_position_heartbeat.sh
#       Stop-loss / take-profit / expiry settlement on open positions.
#   • Confidence calibration  daily 18:00  kalshi_confidence_calibration.sh
#       Reconciles resolved trades and prints the calibration report
#       (incl. the NO-price-band breakdown). 18:00 is intended as UTC, i.e.
#       after the 17:00 UTC daily BTC settlement — if this box is not on UTC,
#       adjust CAL_HOUR below or the box timezone. The job is idempotent, so
#       the exact hour only affects when the daily report lands.
#
# Usage:
#   crypto_oracle/kalshi/install_kalshi_cron.sh            # install / refresh
#   crypto_oracle/kalshi/install_kalshi_cron.sh uninstall  # remove all jobs
#   crypto_oracle/kalshi/install_kalshi_cron.sh list       # show current jobs
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TAG="# kalshi-oracle"
LOG_DIR="$HOME/.hermes/logs"
DEPLOY_DIR="$HOME/.hermes/scripts"

# Schedules (override via env if you want a different cadence)
SCAN_CRON="${KALSHI_SCAN_CRON:-*/30 * * * *}"
HEARTBEAT_CRON="${KALSHI_HEARTBEAT_CRON:-*/15 * * * *}"
CAL_HOUR="${KALSHI_CAL_HOUR:-18}"
CAL_CRON="0 ${CAL_HOUR} * * *"

list_jobs() {
    crontab -l 2>/dev/null | grep "$TAG" || echo "(no kalshi-oracle cron jobs installed)"
}

strip_jobs() {
    # Echo the current crontab with all kalshi-oracle lines removed.
    crontab -l 2>/dev/null | grep -v "$TAG" || true
}

case "${1:-install}" in
    list)
        list_jobs
        exit 0
        ;;
    uninstall)
        strip_jobs | crontab -
        echo "Removed all kalshi-oracle cron jobs."
        list_jobs
        exit 0
        ;;
    install)
        ;;
    *)
        echo "Usage: $0 [install|uninstall|list]" >&2
        exit 1
        ;;
esac

# ── Prepare directories ────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$DEPLOY_DIR"

# ── Deploy the entry-scan / heartbeat scripts where the .sh wrappers expect ─
# The wrappers exec ~/.hermes/scripts/kalshi_*.py, so copy the repo's canonical
# copies there. (The calibration wrapper runs from the repo checkout directly.)
cp "$SCRIPT_DIR/kalshi_live_trade.py"        "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/kalshi_position_heartbeat.py" "$DEPLOY_DIR/"
chmod +x "$SCRIPT_DIR"/kalshi_live_trade.sh \
         "$SCRIPT_DIR"/kalshi_position_heartbeat.sh \
         "$SCRIPT_DIR"/kalshi_confidence_calibration.sh 2>/dev/null || true

# ── Build the new crontab: existing lines (minus ours) + fresh kalshi jobs ──
{
    strip_jobs
    echo "$SCAN_CRON $SCRIPT_DIR/kalshi_live_trade.sh >> $LOG_DIR/kalshi_live_trade.log 2>&1 $TAG"
    echo "$HEARTBEAT_CRON $SCRIPT_DIR/kalshi_position_heartbeat.sh >> $LOG_DIR/kalshi_heartbeat.log 2>&1 $TAG"
    echo "$CAL_CRON $SCRIPT_DIR/kalshi_confidence_calibration.sh >> $LOG_DIR/kalshi_calibration.log 2>&1 $TAG"
} | crontab -

echo "Installed kalshi-oracle cron jobs (repo: $REPO_ROOT):"
echo "  entry scan        $SCAN_CRON"
echo "  position heartbeat $HEARTBEAT_CRON"
echo "  calibration       $CAL_CRON  (hour is intended UTC)"
echo "Logs → $LOG_DIR/kalshi_*.log"
echo
list_jobs
