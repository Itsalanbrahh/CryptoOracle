#!/bin/bash
# Reconcile postmortem with API first, then run calibration
set -e
cd /Users/alanruelas/crypto_oracle
source .venv/bin/activate
python3 crypto_oracle/kalshi/reconcile_positions.py
echo "---"
python3 crypto_oracle/kalshi/confidence_validate.py