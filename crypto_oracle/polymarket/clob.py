from __future__ import annotations

from typing import Any
from urllib.parse import quote

import aiohttp

CLOB_BASE = "https://clob.polymarket.com"
_HEADERS = {"User-Agent": "crypto-oracle/1.0"}


async def _get_json(url: str) -> Any:
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            resp.raise_for_status()
            return await resp.json()


async def get_midpoint(token_id: str) -> float | None:
    data = await _get_json(f"{CLOB_BASE}/midpoint?token_id={quote(token_id)}")
    mid = data.get('mid')
    return float(mid) if mid is not None else None


async def get_spread(token_id: str) -> float | None:
    data = await _get_json(f"{CLOB_BASE}/spread?token_id={quote(token_id)}")
    spread = data.get('spread')
    return float(spread) if spread is not None else None


async def get_orderbook(token_id: str) -> dict[str, Any]:
    return await _get_json(f"{CLOB_BASE}/book?token_id={quote(token_id)}")


async def get_price_history(condition_id: str, *, interval: str = '1w', fidelity: int = 50) -> list[dict[str, Any]]:
    data = await _get_json(
        f"{CLOB_BASE}/prices-history?market={quote(condition_id)}&interval={quote(interval)}&fidelity={fidelity}"
    )
    return data.get('history', [])
