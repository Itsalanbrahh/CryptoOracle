"""Kalshi REST API client with RSA-SHA256 authentication."""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_BASE_PATH = "/trade-api/v2"  # included in HMAC signature


def _load_private_key():
    pem_path = Path(os.getenv("KALSHI_RSA_KEY_PATH", "/Users/alanruelas/crypto_oracle/.kalshi_private.pem"))
    pem_bytes = pem_path.read_bytes()
    return serialization.load_pem_private_key(pem_bytes, password=None)


def _make_headers(method: str, path: str, key_id: str) -> dict[str, str]:
    ts_ms = str(int(time.time() * 1000))
    # Kalshi requires full path in signature and PSS padding (not PKCS1v15)
    full_path = KALSHI_BASE_PATH + path
    msg = (ts_ms + method.upper() + full_path).encode()
    private_key = _load_private_key()
    pss = asym_padding.PSS(mgf=asym_padding.MGF1(hashes.SHA256()), salt_length=asym_padding.PSS.MAX_LENGTH)
    signature = private_key.sign(msg, pss, hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "Content-Type": "application/json",
    }


class KalshiClient:
    def __init__(self, key_id: str | None = None, base_url: str = KALSHI_BASE_URL):
        self.key_id = key_id or os.getenv("KALSHI_API_KEY_ID", "")
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str, params: dict | None = None, auth: bool = False) -> Any:
        url = self.base_url + path
        headers = _make_headers("GET", path, self.key_id) if auth else {}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _post(self, path: str, body: dict) -> Any:
        url = self.base_url + path
        headers = _make_headers("POST", path, self.key_id)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
                if resp.status >= 400:
                    raise ValueError(f"Kalshi API error {resp.status}: {data}")
                return data

    async def get_markets(self, series_ticker: str = "KXBTCD", status: str = "open", limit: int = 100) -> list[dict]:
        data = await self._get("/markets", params={"series_ticker": series_ticker, "status": status, "limit": limit})
        return data.get("markets", [])

    async def get_balance(self) -> dict:
        return await self._get("/portfolio/balance", auth=True)

    async def place_order(self, ticker: str, side: str, count: int, price_cents: int, order_type: str = "limit") -> dict:
        """
        ticker: e.g. KXBTCD-26JUN2203-T64199.99
        side: 'yes' or 'no'  (converted to 'bid'/'ask' for v2 API)
        count: number of contracts ($1 each)
        price_cents: price in cents (42 = $0.42)

        Uses Kalshi v2 order endpoint (/portfolio/events/orders) with:
          - bid/ask sides (not yes/no)
          - string count and dollar-string price
          - time_in_force and self_trade_prevention_type required
        """
        # v2 API: side is 'bid' (buy YES) or 'ask' (buy NO)
        book_side = "bid" if side == "yes" else "ask"
        # v2 price is a dollar string; bid price is the YES price, ask price is the NO price
        if side == "yes":
            dollar_price = f"{price_cents / 100:.2f}"
        else:
            dollar_price = f"{price_cents / 100:.2f}"
        body = {
            "ticker": ticker,
            "action": "buy",
            "type": order_type,
            "side": book_side,
            "count": str(count),
            "price": dollar_price,
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
        }
        return await self._post("/portfolio/events/orders", body)

    async def close_position(self, ticker: str, count: int, side: str, price_cents: int) -> dict:
        """Close/reduce an existing position by selling contracts back to the market.

        ticker: the market ticker
        count: number of contracts to sell
        side: the original side ('yes' or 'no') — we sell the opposite
        price_cents: limit price in cents for the sell

        To close a YES position: place a sell (ask) on the same ticker.
        To close a NO position: place a sell (bid on the opposite side, ask on NO).
        """
        # Selling YES = placing an ask (sell) order
        # Selling NO  = placing a bid (sell) on NO
        book_side = "ask" if side == "yes" else "bid"
        dollar_price = f"{price_cents / 100:.2f}"
        body = {
            "ticker": ticker,
            "action": "sell",
            "type": "limit",
            "side": book_side,
            "count": str(count),
            "price": dollar_price,
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
        }
        return await self._post("/portfolio/events/orders", body)
