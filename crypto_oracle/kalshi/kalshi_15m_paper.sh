#!/usr/bin/env bash
# Paper cycle for KXBTC15M (Jev + WS feeds + feature logging).
set -euo pipefail
REPO="${KALSHI_REPO:-$HOME/CryptoOracle}"
cd "$REPO"
export PYTHONUNBUFFERED=1
if [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  PY=python3
fi
exec "$PY" -m crypto_oracle.kalshi.paper_15m_runner --learn
