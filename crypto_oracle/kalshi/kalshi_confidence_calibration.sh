#!/bin/bash
# 1. Reconcile live positions with the Kalshi API
# 2. Resolve paper/filtered decisions against actual BTC settlement
# 3. Run the calibration report
set -e
cd /Users/alanruelas/crypto_oracle
source .venv/bin/activate
python3 crypto_oracle/kalshi/reconcile_positions.py
echo "---"
python3 crypto_oracle/kalshi/resolve_paper_trades.py
echo "---"
python3 crypto_oracle/kalshi/confidence_validate.py
