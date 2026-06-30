#!/bin/bash
# Kalshi position heartbeat — checks open positions for stop-loss, take-profit,
# and rebalancing. Does NOT scan for new entries.
exec /Users/alanruelas/crypto_oracle/.venv/bin/python /Users/alanruelas/.hermes/scripts/kalshi_position_heartbeat.py
