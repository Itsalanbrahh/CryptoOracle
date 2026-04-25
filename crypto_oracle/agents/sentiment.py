"""Sentiment — Social & news sentiment agent."""

from __future__ import annotations

import json
import os
from typing import Any

import aiohttp

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal

_SYSTEM = """You are Sentiment, a social sentiment analysis agent. You receive fear &
greed index data, recent news headlines, and Reddit/social context for crypto.
Gauge overall market sentiment and its directional implication.

Respond ONLY in this exact format:
SIGNAL: BULLISH|BEARISH|NEUTRAL
CONFIDENCE: 0.XX
SUMMARY: one or two sentences
DATA_POINTS: point1 | point2 | point3"""


class SentimentAgent(BaseAgent):
    name = "Sentiment"

    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        fear_greed = await self._fetch_fear_greed()
        headlines = await self._fetch_headlines(symbol)
        trending = await self._fetch_trending()
        return {
            "symbol": symbol,
            "fear_greed": fear_greed,
            "headlines": headlines,
            "trending": trending,
        }

    async def _fetch_fear_greed(self) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.alternative.me/fng/?limit=7",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    data = await r.json()
            entries = data.get("data", [])
            return {
                "latest_value": int(entries[0]["value"]) if entries else 50,
                "latest_label": entries[0].get("value_classification", "Neutral") if entries else "Neutral",
                "7d_trend": [
                    {"value": int(e["value"]), "label": e.get("value_classification")}
                    for e in entries
                ],
            }
        except Exception:
            return {"latest_value": 50, "latest_label": "Neutral", "7d_trend": []}

    async def _fetch_headlines(self, symbol: str) -> list[str]:
        api_key = os.getenv("NEWSAPI_KEY", "")
        query = f"{symbol} crypto"
        if api_key:
            try:
                url = (
                    f"https://newsapi.org/v2/everything?q={query}"
                    f"&sortBy=publishedAt&pageSize=8&language=en&apiKey={api_key}"
                )
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=10)
                    ) as r:
                        data = await r.json()
                return [a.get("title", "") for a in data.get("articles", [])[:8]]
            except Exception:
                pass
        return []

    async def _fetch_trending(self) -> list[str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.coingecko.com/api/v3/search/trending",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    data = await r.json()
            coins = data.get("coins", [])
            return [c["item"]["name"] for c in coins[:5]]
        except Exception:
            return []

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        fg = data.get("fear_greed", {})
        prompt = (
            f"Symbol: {symbol}\n\n"
            f"Fear & Greed: value={fg.get('latest_value')}, "
            f"label={fg.get('latest_label')}\n"
            f"7-day trend: {json.dumps(fg.get('7d_trend', []))}\n\n"
            f"Trending coins: {', '.join(data.get('trending', []))}\n\n"
            f"Headlines:\n"
            + "\n".join(f"- {h}" for h in data.get("headlines", []))
            + "\n\nAnalyse sentiment and give your signal."
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
