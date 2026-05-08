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
        pair = _KRAKEN_MAP.get(symbol.upper(), f"{symbol.upper()}USD")
        depth_url = f"https://api.kraken.com/0/public/Depth?pair={pair}&count=10"
        ticker_url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"

        data: dict[str, Any] = {"symbol": symbol}

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    depth_url, timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    raw = await r.json()
                result = raw.get("result", {})
                pair_key = next((k for k in result), None)
                if pair_key:
                    bids = [(float(b[0]), float(b[1])) for b in result[pair_key].get("bids", [])[:5]]
                    asks = [(float(a[0]), float(a[1])) for a in result[pair_key].get("asks", [])[:5]]
                    best_bid = bids[0][0] if bids else 0
                    best_ask = asks[0][0] if asks else 0
                    spread_pct = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0
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
                else:
                    data["microstructure"] = {}
            except Exception as exc:
                self.logger.warning("Kraken depth fetch failed: %s", exc)
                data["microstructure"] = {}

            try:
                async with session.get(
                    ticker_url, timeout=aiohttp.ClientTimeout(total=10)
                ) as r:
                    raw = await r.json()
                result = raw.get("result", {})
                pair_key = next((k for k in result), None)
                if pair_key:
                    t = result[pair_key]
                    open_price = float(t["o"])
                    last_price = float(t["c"][0])
                    data["ticker"] = {
                        "last_price": last_price,
                        "volume_24h": float(t["v"][1]),
                        "vwap_24h": float(t["p"][1]),
                        "trades_24h": int(t["t"][1]),
                        "low_24h": float(t["l"][1]),
                        "high_24h": float(t["h"][1]),
                        "price_change_pct_24h": round(
                            (last_price - open_price) / open_price * 100, 2
                        ) if open_price > 0 else 0,
                    }
                else:
                    data["ticker"] = {}
            except Exception as exc:
                self.logger.warning("Kraken ticker fetch failed: %s", exc)
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


_KRAKEN_MAP = {
    "BTC":  "XBTUSD",
    "ETH":  "ETHUSD",
    "SOL":  "SOLUSD",
    "ADA":  "ADAUSD",
    "DOGE": "XDGUSD",
    "XRP":  "XXRPZUSD",
}
