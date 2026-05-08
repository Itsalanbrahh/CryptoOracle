"""Micro — Market microstructure agent (order book, spreads, depth)."""

from __future__ import annotations

import json
from typing import Any

import aiohttp

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal

_SYSTEM = """You are Micro, a market microstructure agent specialising in order book
analysis, bid-ask spreads, market depth, and liquidity signals.

Respond ONLY in this exact format:
SIGNAL: BULLISH|BEARISH|NEUTRAL
CONFIDENCE: 0.XX
SUMMARY: one or two sentences
DATA_POINTS: point1 | point2 | point3"""


class MicroAgent(BaseAgent):
    name = "Micro"

    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        ticker = _BINANCE_MAP.get(symbol.upper(), f"{symbol.upper()}USDT")
        depth_url = f"https://api.binance.com/api/v3/depth?symbol={ticker}&limit=10"
        ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={ticker}"

        data: dict[str, Any] = {"symbol": symbol}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    depth_url, timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    depth = await r.json()
                bids = [(float(p), float(q)) for p, q in depth.get("bids", [])[:5]]
                asks = [(float(p), float(q)) for p, q in depth.get("asks", [])[:5]]
                best_bid = bids[0][0] if bids else 0
                best_ask = asks[0][0] if asks else 0
                spread_pct = (
                    (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0
                )
                bid_depth = sum(p * q for p, q in bids)
                ask_depth = sum(p * q for p, q in asks)
                data["microstructure"] = {
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread_pct": round(spread_pct, 4),
                    "bid_depth_usd": round(bid_depth, 2),
                    "ask_depth_usd": round(ask_depth, 2),
                    "bid_ask_ratio": round(bid_depth / ask_depth, 3) if ask_depth else 0,
                }
            except Exception as exc:
                self.logger.warning("Binance depth fetch failed: %s", exc)
                data["microstructure"] = {}

            try:
                async with session.get(
                    ticker_url, timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    ticker_data = await r.json()
                data["ticker"] = {
                    "price_change_pct_24h": float(
                        ticker_data.get("priceChangePercent", 0)
                    ),
                    "volume_24h": float(ticker_data.get("volume", 0)),
                    "quote_volume_24h": float(ticker_data.get("quoteVolume", 0)),
                    "count_trades_24h": int(ticker_data.get("count", 0)),
                    "weighted_avg_price": float(ticker_data.get("weightedAvgPrice", 0)),
                }
            except Exception as exc:
                self.logger.warning("Binance ticker fetch failed: %s", exc)
                data["ticker"] = {}

        return data

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        micro = data.get("microstructure", {})
        ticker = data.get("ticker", {})
        if not micro and not ticker:
            return AgentSignal(
                agent_name=self.name,
                signal="NEUTRAL",
                confidence=0.0,
                summary="Data feed failure — Binance depth and ticker both unavailable.",
                data_points=["data_feed_failure"],
            )
        prompt = (
            f"Symbol: {symbol}\n"
            f"Microstructure: {json.dumps(micro, indent=2)}\n"
            f"24h Ticker: {json.dumps(ticker, indent=2)}\n\n"
            "Analyse order book pressure and liquidity, give your signal."
        )
        text = await self._call_claude(_SYSTEM, prompt)
        signal, confidence, summary, data_points = self._parse_signal_from_text(text)
        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            summary=summary,
            data_points=data_points,
        )


_BINANCE_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "ADA": "ADAUSDT",
    "DOGE": "DOGEUSDT",
    "XRP": "XRPUSDT",
}
