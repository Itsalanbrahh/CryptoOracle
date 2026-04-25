"""Macro — Macro-economic context agent (Fed, DXY, rates, news)."""

from __future__ import annotations

import json
import os
from typing import Any

import aiohttp

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal

_SYSTEM = """You are Macro, a macro-economic analysis agent. You receive macro indicators
(DXY, Fed language, interest rate expectations, global risk-on/off signals) and
crypto-relevant news headlines. Assess how the macro environment affects crypto.

Respond ONLY in this exact format:
SIGNAL: BULLISH|BEARISH|NEUTRAL
CONFIDENCE: 0.XX
SUMMARY: one or two sentences
DATA_POINTS: point1 | point2 | point3"""


class MacroAgent(BaseAgent):
    name = "Macro"

    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        headlines = await self._fetch_headlines()
        fear_greed = await self._fetch_fear_greed()
        return {"headlines": headlines, "fear_greed": fear_greed, "symbol": symbol}

    async def _fetch_headlines(self) -> list[str]:
        """Fetch crypto-relevant headlines from NewsAPI or RSS fallback."""
        api_key = os.getenv("NEWSAPI_KEY", "")
        headlines: list[str] = []

        if api_key:
            url = (
                "https://newsapi.org/v2/everything"
                "?q=bitcoin+OR+crypto+OR+federal+reserve+OR+interest+rates"
                "&sortBy=publishedAt&pageSize=10&language=en"
                f"&apiKey={api_key}"
            )
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=10)
                    ) as r:
                        data = await r.json()
                headlines = [
                    a.get("title", "") for a in data.get("articles", [])[:10]
                ]
                return headlines
            except Exception:
                pass

        # RSS fallback — CoinDesk
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.coindesk.com/arc/outboundfeeds/rss/",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    text = await r.text()
            import re
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", text)
            headlines = titles[:10]
        except Exception:
            headlines = ["Could not fetch headlines."]

        return headlines

    async def _fetch_fear_greed(self) -> dict:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.alternative.me/fng/?limit=3",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    data = await r.json()
            entries = data.get("data", [])
            return {
                "current": entries[0] if entries else {},
                "yesterday": entries[1] if len(entries) > 1 else {},
            }
        except Exception:
            return {}

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        prompt = (
            f"Symbol: {symbol}\n\n"
            f"Fear & Greed Index: {json.dumps(data.get('fear_greed', {}), indent=2)}\n\n"
            f"Recent Headlines:\n"
            + "\n".join(f"- {h}" for h in data.get("headlines", []))
            + "\n\nAnalyse the macro environment and give your signal."
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
