#!/usr/bin/env python3
"""Paper/live scan focused on KXBTC15M + Jev (skips hourly ensemble)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


async def main() -> None:
    live = "--live" in sys.argv
    os.environ.setdefault("KALSHI_15M_ENABLED", "1")
    # Force the hourly path to select nothing by limiting volume absurdly high
    # while still running the shared scan (15m is appended inside run_kalshi_scan).
    from crypto_oracle.kalshi.loop import _scan_15m_markets
    from crypto_oracle.kalshi.market_data import fetch_funding_rate, fetch_realized_vol
    from crypto_oracle.polymarket.agents import fetch_spot_price

    spot, vol, funding = await asyncio.gather(
        fetch_spot_price(),
        fetch_realized_vol(hours=24),
        fetch_funding_rate(),
    )
    max_position = float(os.getenv("KALSHI_MAX_POSITION_USD", "5"))
    results, trades, deployed, entries, daily = await _scan_15m_markets(
        spot=spot,
        annual_vol=vol,
        funding_rate=funding,
        live=live,
        max_position=max_position,
        max_daily_risk=float(os.getenv("KALSHI_MAX_DAILY_RISK_USD", "20")),
        max_entries_per_day=int(os.getenv("KALSHI_MAX_ENTRIES_PER_DAY", "20")),
        entries_today=0,
        deployed_today=0.0,
        maker_mode=os.getenv("KALSHI_MAKER_MODE", "1").strip() == "1",
        kalshi_balance_cents=None,
    )
    print(json.dumps({
        "mode": "live" if live else "paper",
        "spot": spot,
        "vol": vol,
        "trades": trades,
        "deployed": deployed,
        "results": results,
    }, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
