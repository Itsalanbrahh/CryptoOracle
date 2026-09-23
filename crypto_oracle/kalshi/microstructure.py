"""Short-horizon microstructure snapshots for BTC (public REST, no keys).

Uses Coinbase + Kraken + Bybit (US-reachable). Binance often returns HTTP 451
in the US so it is attempted last as an optional enricher.
"""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

_TIMEOUT = aiohttp.ClientTimeout(total=8)


def _ofi_from_levels(bids: list, asks: list, levels: int = 10) -> dict[str, float]:
    bid_qty = ask_qty = 0.0
    best_bid = best_ask = 0.0
    for i, row in enumerate(bids[:levels]):
        px, qty = float(row[0]), float(row[1])
        bid_qty += qty
        if i == 0:
            best_bid = px
    for i, row in enumerate(asks[:levels]):
        px, qty = float(row[0]), float(row[1])
        ask_qty += qty
        if i == 0:
            best_ask = px
    denom = bid_qty + ask_qty
    imbalance = ((bid_qty - ask_qty) / denom) if denom > 0 else 0.0
    spread = (best_ask - best_bid) if best_bid > 0 and best_ask > 0 else 0.0
    mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
    spread_bps = (spread / mid * 10_000.0) if mid > 0 else 0.0
    return {
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "imbalance": imbalance,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "spread_bps": spread_bps,
        "mid": mid,
    }


async def _coinbase_book(session: aiohttp.ClientSession) -> dict[str, float] | None:
    url = "https://api.exchange.coinbase.com/products/BTC-USD/book"
    try:
        async with session.get(url, params={"level": 2}, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
        return _ofi_from_levels(data.get("bids", []), data.get("asks", []))
    except Exception:
        return None


async def _coinbase_trades(session: aiohttp.ClientSession, limit: int = 100) -> dict[str, float] | None:
    url = "https://api.exchange.coinbase.com/products/BTC-USD/trades"
    try:
        async with session.get(url, params={"limit": limit}, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            trades = await resp.json()
        buy_qty = sell_qty = 0.0
        for t in trades[:limit]:
            qty = float(t.get("size") or 0)
            if t.get("side") == "buy":
                buy_qty += qty
            else:
                sell_qty += qty
        total = buy_qty + sell_qty
        return {
            "buy_qty": buy_qty,
            "sell_qty": sell_qty,
            "trade_imbalance": ((buy_qty - sell_qty) / total) if total > 0 else 0.0,
            "n_trades": float(min(len(trades), limit)),
        }
    except Exception:
        return None


async def _kraken_book(session: aiohttp.ClientSession) -> dict[str, float] | None:
    url = "https://api.kraken.com/0/public/Depth"
    try:
        async with session.get(url, params={"pair": "XBTUSD", "count": 15}, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
        result = data.get("result") or {}
        book = result.get("XXBTZUSD") or next(iter(result.values()), None)
        if not book:
            return None
        return _ofi_from_levels(book.get("bids", []), book.get("asks", []))
    except Exception:
        return None


async def _bybit_book(session: aiohttp.ClientSession) -> dict[str, float] | None:
    url = "https://api.bybit.com/v5/market/orderbook"
    try:
        async with session.get(
            url,
            params={"category": "spot", "symbol": "BTCUSDT", "limit": 25},
            timeout=_TIMEOUT,
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
        result = data.get("result") or {}
        return _ofi_from_levels(result.get("b", []), result.get("a", []))
    except Exception:
        return None


async def fetch_microstructure() -> dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        cb_book, cb_trades, kr_book, by_book = await asyncio.gather(
            _coinbase_book(session),
            _coinbase_trades(session),
            _kraken_book(session),
            _bybit_book(session),
        )

    out: dict[str, Any] = {
        "coinbase_imbalance": None,
        "coinbase_spread_bps": None,
        "coinbase_trade_imbalance": None,
        "kraken_imbalance": None,
        "kraken_spread_bps": None,
        "bybit_imbalance": None,
        "bybit_spread_bps": None,
        # Aliases kept for feature schema compatibility
        "binance_imbalance": None,
        "binance_trade_imbalance": None,
        "binance_spread_bps": None,
        "venue_mid_dispersion_bps": None,
        "combined_ofi": 0.0,
        "ok": False,
    }
    mids: list[float] = []
    ofi_parts: list[float] = []

    if cb_book:
        out["coinbase_imbalance"] = round(cb_book["imbalance"], 4)
        out["coinbase_spread_bps"] = round(cb_book["spread_bps"], 3)
        # Map into binance_* slots so FEATURE_KEYS stay stable without retrain break
        out["binance_imbalance"] = out["coinbase_imbalance"]
        out["binance_spread_bps"] = out["coinbase_spread_bps"]
        if cb_book["mid"] > 0:
            mids.append(cb_book["mid"])
        ofi_parts.append(cb_book["imbalance"])
    if cb_trades:
        out["coinbase_trade_imbalance"] = round(cb_trades["trade_imbalance"], 4)
        out["binance_trade_imbalance"] = out["coinbase_trade_imbalance"]
        ofi_parts.append(cb_trades["trade_imbalance"])
    if kr_book:
        out["kraken_imbalance"] = round(kr_book["imbalance"], 4)
        out["kraken_spread_bps"] = round(kr_book["spread_bps"], 3)
        if kr_book["mid"] > 0:
            mids.append(kr_book["mid"])
        ofi_parts.append(kr_book["imbalance"])
    if by_book:
        out["bybit_imbalance"] = round(by_book["imbalance"], 4)
        out["bybit_spread_bps"] = round(by_book["spread_bps"], 3)
        if by_book["mid"] > 0:
            mids.append(by_book["mid"])
        ofi_parts.append(by_book["imbalance"])

    if len(mids) > 1:
        mean_mid = sum(mids) / len(mids)
        if mean_mid > 0:
            out["venue_mid_dispersion_bps"] = round((max(mids) - min(mids)) / mean_mid * 10_000.0, 3)
    if ofi_parts:
        out["combined_ofi"] = round(sum(ofi_parts) / len(ofi_parts), 4)
        out["ok"] = True
    return out
