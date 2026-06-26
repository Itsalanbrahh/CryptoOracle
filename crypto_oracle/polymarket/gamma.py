from __future__ import annotations

import urllib.parse
from typing import Any

import aiohttp

from .client import parse_gamma_market, select_btc_markets
from .models import PolymarketMarket

GAMMA_BASE = "https://gamma-api.polymarket.com"
_HEADERS = {"User-Agent": "crypto-oracle/1.0"}


async def _get_json(url: str) -> Any:
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            resp.raise_for_status()
            return await resp.json()


async def search_events(query: str) -> list[dict[str, Any]]:
    q = urllib.parse.quote(query)
    data = await _get_json(f"{GAMMA_BASE}/public-search?q={q}")
    return data.get("events", [])


async def fetch_markets(*, limit: int = 100, active: bool = True, closed: bool = False, order: str = 'volume') -> list[dict[str, Any]]:
    url = (
        f"{GAMMA_BASE}/markets?limit={limit}&active={'true' if active else 'false'}"
        f"&closed={'true' if closed else 'false'}&order={urllib.parse.quote(order)}&ascending=false"
    )
    data = await _get_json(url)
    return data if isinstance(data, list) else []


async def fetch_event_markets(query: str) -> list[PolymarketMarket]:
    events = await search_events(query)
    rows: list[dict[str, Any]] = []
    for event in events:
        event_id = event.get('id')
        for market in event.get('markets', []):
            row = dict(market)
            if event_id is not None and 'eventId' not in row:
                row['eventId'] = event_id
            rows.append(row)
    return [parse_gamma_market(r) for r in rows]


async def fetch_btc_markets(*, limit: int = 200, min_liquidity: float = 1000.0) -> list[PolymarketMarket]:
    # Search by keyword to find BTC markets reliably — the volume-ranked endpoint
    # rarely surfaces intraday BTC markets in its top results.
    events = await search_events('bitcoin')
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for event in events:
        eid = event.get('id')
        for m in event.get('markets', []):
            mid = m.get('id') or m.get('marketId')
            if mid and mid in seen_ids:
                continue
            if mid:
                seen_ids.add(mid)
            row = dict(m)
            if eid and 'eventId' not in row:
                row['eventId'] = eid
            if 'events' not in row:
                row['events'] = [{'title': event.get('title', '')}]
            rows.append(row)
    # Also pull top volume markets as fallback (catches BTC markets not indexed by title)
    volume_rows = await fetch_markets(limit=100, active=True, closed=False, order='volume')
    for row in volume_rows:
        mid = row.get('id') or row.get('marketId')
        if mid and mid not in seen_ids:
            seen_ids.add(mid)
            rows.append(row)
    return select_btc_markets(rows, min_liquidity=min_liquidity)
