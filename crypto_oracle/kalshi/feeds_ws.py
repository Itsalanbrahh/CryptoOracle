"""Short-lived websocket collectors for trade flow + optional Kalshi book.

Coinbase / Kraken need no auth. Kalshi orderbook WS requires API key + PEM;
when unavailable we skip and rely on REST market quotes already in the loop.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from .client import _make_headers

_COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
_KRAKEN_WS = "wss://ws.kraken.com"
_KALSHI_WS = os.getenv("KALSHI_WS_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2")


@dataclass
class TradeFlowSnapshot:
    seconds: float
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    n_trades: int = 0
    last_price: float | None = None
    venue: str = ""
    ok: bool = False

    @property
    def trade_imbalance(self) -> float:
        tot = self.buy_qty + self.sell_qty
        return ((self.buy_qty - self.sell_qty) / tot) if tot > 0 else 0.0


@dataclass
class BookTop:
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    ts: float = 0.0
    ok: bool = False
    source: str = "none"


@dataclass
class LiveFeedSnapshot:
    duration_s: float
    coinbase: TradeFlowSnapshot = field(default_factory=lambda: TradeFlowSnapshot(0))
    kraken: TradeFlowSnapshot = field(default_factory=lambda: TradeFlowSnapshot(0))
    kalshi_book: BookTop = field(default_factory=BookTop)
    brti_proxy: float | None = None          # rolling multi-venue mid average
    brti_proxy_n: int = 0
    combined_trade_imbalance: float = 0.0
    ok: bool = False

    def as_micro_overlay(self) -> dict[str, Any]:
        """Merge into microstructure feature aliases."""
        parts = []
        buy = sell = 0.0
        if self.coinbase.ok:
            parts.append(self.coinbase.trade_imbalance)
            buy += self.coinbase.buy_qty
            sell += self.coinbase.sell_qty
        if self.kraken.ok:
            parts.append(self.kraken.trade_imbalance)
            buy += self.kraken.buy_qty
            sell += self.kraken.sell_qty
        imb = (sum(parts) / len(parts)) if parts else 0.0
        tot = buy + sell
        return {
            "ws_trade_imbalance": round(imb, 4),
            "ws_buy_qty": round(buy, 4),
            "ws_sell_qty": round(sell, 4),
            "ws_n_venues": len(parts),
            "ws_ok": bool(parts),
            "brti_proxy": self.brti_proxy,
            "kalshi_yes_bid": self.kalshi_book.yes_bid,
            "kalshi_yes_ask": self.kalshi_book.yes_ask,
            "kalshi_book_ok": self.kalshi_book.ok,
            # Prefer WS trade imbalance when present for combined_ofi blend upstream
            "binance_trade_imbalance": round(imb, 4) if parts else None,
        }


async def _collect_coinbase(seconds: float) -> TradeFlowSnapshot:
    snap = TradeFlowSnapshot(seconds=seconds, venue="coinbase")
    sub = {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channels": ["matches"],
    }
    deadline = time.monotonic() + seconds
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(_COINBASE_WS, heartbeat=20) as ws:
                await ws.send_json(sub)
                while time.monotonic() < deadline:
                    timeout = max(0.05, deadline - time.monotonic())
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
                    except asyncio.TimeoutError:
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                        continue
                    data = json.loads(msg.data)
                    if data.get("type") != "match":
                        continue
                    qty = float(data.get("size") or 0)
                    px = float(data.get("price") or 0)
                    side = data.get("side")  # buy = taker buy = aggressor buy
                    if side == "buy":
                        snap.buy_qty += qty
                    else:
                        snap.sell_qty += qty
                    snap.n_trades += 1
                    if px > 0:
                        snap.last_price = px
        snap.ok = snap.n_trades > 0
    except Exception:
        snap.ok = False
    return snap


async def _collect_kraken(seconds: float) -> TradeFlowSnapshot:
    snap = TradeFlowSnapshot(seconds=seconds, venue="kraken")
    sub = {"event": "subscribe", "pair": ["XBT/USD"], "subscription": {"name": "trade"}}
    deadline = time.monotonic() + seconds
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(_KRAKEN_WS, heartbeat=20) as ws:
                await ws.send_json(sub)
                while time.monotonic() < deadline:
                    timeout = max(0.05, deadline - time.monotonic())
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
                    except asyncio.TimeoutError:
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                        continue
                    data = json.loads(msg.data)
                    # trade channel: [chanId, [[price, volume, time, side, orderType, misc], ...], "trade", "XBT/USD"]
                    if not isinstance(data, list) or len(data) < 4 or data[-2] != "trade":
                        continue
                    for t in data[1]:
                        px = float(t[0])
                        qty = float(t[1])
                        side = t[3]  # b=buy s=sell
                        if side == "b":
                            snap.buy_qty += qty
                        else:
                            snap.sell_qty += qty
                        snap.n_trades += 1
                        snap.last_price = px
        snap.ok = snap.n_trades > 0
    except Exception:
        snap.ok = False
    return snap


async def _collect_kalshi_book(ticker: str, seconds: float) -> BookTop:
    """Authenticated Kalshi orderbook_delta for a few seconds."""
    top = BookTop(source="kalshi_ws")
    key_id = os.getenv("KALSHI_API_KEY_ID", "").strip()
    if not key_id or not ticker:
        return top
    try:
        # WS handshake uses same signed headers as REST on path /trade-api/ws/v2
        headers = _make_headers("GET", "/trade-api/ws/v2", key_id)
    except Exception:
        return top

    yes_levels: dict[str, float] = {}
    no_levels: dict[str, float] = {}
    deadline = time.monotonic() + seconds

    def _best(levels: dict[str, float], reverse: bool) -> float | None:
        priced = [(float(p), q) for p, q in levels.items() if q > 0]
        if not priced:
            return None
        priced.sort(key=lambda x: x[0], reverse=reverse)
        return priced[0][0]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(_KALSHI_WS, headers=headers, heartbeat=20) as ws:
                await ws.send_json({
                    "id": 1,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta"],
                        "market_ticker": ticker,
                    },
                })
                while time.monotonic() < deadline:
                    timeout = max(0.05, deadline - time.monotonic())
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
                    except asyncio.TimeoutError:
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                        continue
                    data = json.loads(msg.data)
                    typ = data.get("type")
                    m = data.get("msg") or {}
                    if typ == "orderbook_snapshot":
                        yes_levels = {str(p): float(q) for p, q in (m.get("yes_dollars_fp") or [])}
                        no_levels = {str(p): float(q) for p, q in (m.get("no_dollars_fp") or [])}
                    elif typ == "orderbook_delta":
                        side = m.get("side")
                        price = str(m.get("price_dollars"))
                        delta = float(m.get("delta_fp") or 0)
                        book = yes_levels if side == "yes" else no_levels
                        book[price] = book.get(price, 0.0) + delta
                    else:
                        continue
                    top.yes_bid = _best(yes_levels, reverse=True)
                    # best ask ≈ 1 - best no bid when no yes asks tracked separately;
                    # Kalshi books are side-specific; approximate ask from complementary no bid.
                    best_no = _best(no_levels, reverse=True)
                    if best_no is not None:
                        top.yes_ask = round(1.0 - best_no, 4)
                        top.no_bid = best_no
                    if top.yes_bid is not None:
                        top.no_ask = round(1.0 - top.yes_bid, 4)
                    top.ts = time.time()
                    top.ok = top.yes_bid is not None or top.yes_ask is not None
    except Exception:
        return top
    return top


async def collect_live_feeds(
    *,
    ticker: str | None = None,
    seconds: float | None = None,
) -> LiveFeedSnapshot:
    """Collect parallel WS samples for one paper/decision cycle."""
    seconds = float(seconds if seconds is not None else os.getenv("KALSHI_15M_WS_SECONDS", "6"))
    seconds = max(2.0, min(seconds, 20.0))

    cb_t = asyncio.create_task(_collect_coinbase(seconds))
    kr_t = asyncio.create_task(_collect_kraken(seconds))
    kal_t = asyncio.create_task(_collect_kalshi_book(ticker or "", seconds)) if ticker else None

    cb, kr = await asyncio.gather(cb_t, kr_t)
    kal = await kal_t if kal_t else BookTop()

    mids = [p for p in (cb.last_price, kr.last_price) if p]
    brti_proxy = (sum(mids) / len(mids)) if mids else None

    out = LiveFeedSnapshot(
        duration_s=seconds,
        coinbase=cb,
        kraken=kr,
        kalshi_book=kal,
        brti_proxy=round(brti_proxy, 2) if brti_proxy else None,
        brti_proxy_n=len(mids),
    )
    parts = []
    if cb.ok:
        parts.append(cb.trade_imbalance)
    if kr.ok:
        parts.append(kr.trade_imbalance)
    if parts:
        out.combined_trade_imbalance = round(sum(parts) / len(parts), 4)
        out.ok = True
    return out
