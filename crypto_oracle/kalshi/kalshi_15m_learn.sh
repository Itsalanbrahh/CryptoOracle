#!/usr/bin/env bash
# Nightly / hourly learn job for 15m residual + gate retune.
set -euo pipefail
REPO="${KALSHI_REPO:-$HOME/CryptoOracle}"
cd "$REPO"
export PYTHONUNBUFFERED=1
if [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  PY=python3
fi
exec "$PY" -m crypto_oracle.kalshi.learn_15m
