"""Settlement labels for KXBTC15M from official Kalshi market results.

Kalshi settled markets expose ``result`` (yes/no) and ``expiration_value``
(closing BRTI 60s average). Prefer that over any spot proxy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import aiohttp

from .client import KalshiClient

_TIMEOUT = aiohttp.ClientTimeout(total=12)


@dataclass
class SettlementLabel:
    ticker: str
    settled_up: bool
    result: str                 # yes | no
    expiration_value: float | None
    floor_strike: float | None
    source: str                 # kalshi_result | spot_proxy


async def fetch_market_raw(ticker: str) -> dict[str, Any] | None:
    """Public market lookup (no auth)."""
    client = KalshiClient()
    try:
        return await client.get_market(ticker)
    except Exception:
        try:
            markets = await client.get_markets(series_ticker="KXBTC15M", status="settled", limit=50)
            for m in markets:
                if m.get("ticker") == ticker:
                    return m
        except Exception:
            return None
    return None


async def fetch_settlement_label(ticker: str, strike: float | None = None) -> SettlementLabel | None:
    raw = await fetch_market_raw(ticker)
    if not raw:
        return None
    result = str(raw.get("result") or "").strip().lower()
    if result not in ("yes", "no"):
        return None
    exp_raw = raw.get("expiration_value") or raw.get("settlement_value") or ""
    try:
        exp = float(exp_raw) if str(exp_raw).strip() else None
    except (TypeError, ValueError):
        exp = None
    floor = raw.get("floor_strike")
    try:
        floor_f = float(floor) if floor is not None else strike
    except (TypeError, ValueError):
        floor_f = strike
    return SettlementLabel(
        ticker=ticker,
        settled_up=(result == "yes"),
        result=result,
        expiration_value=exp,
        floor_strike=floor_f,
        source="kalshi_result",
    )


async def fetch_brti_via_kalshi() -> float | None:
    """CF Benchmarks BRTI via Kalshi authenticated passthrough (needs API key + PEM)."""
    import os

    key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
    if not key_id:
        return None
    client = KalshiClient(key_id=key_id)
    try:
        data = await client._get("/cfbenchmarks/values", params={"id": "BRTI"}, auth=True)
        payload = data.get("data") or data
        # CF payloads vary; try common shapes
        if isinstance(payload, dict):
            inner = payload.get("payload") or payload
            if isinstance(inner, dict):
                for k in ("value", "VALUE", "price", "last", "v"):
                    if k in inner:
                        return float(inner[k])
                # nested values list
                vals = inner.get("values") or inner.get("VALUE")
                if isinstance(vals, list) and vals:
                    last = vals[-1]
                    if isinstance(last, dict):
                        for k in ("value", "v", "price"):
                            if k in last:
                                return float(last[k])
                    try:
                        return float(last)
                    except (TypeError, ValueError):
                        pass
        return None
    except Exception:
        return None
