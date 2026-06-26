"""StockSentiment — News and sentiment agent for equity stocks."""

from __future__ import annotations

import os
from typing import Any

import aiohttp

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal

_SYSTEM = """You are StockSentiment, a sentiment analysis agent for equity stocks.
You receive recent news headlines, analyst data, and general market fear/greed.
Gauge sentiment for this specific stock and its sector.

BEARISH signals are actionable (short opportunity), not just a warning.

Respond ONLY in this exact format:
SIGNAL: BULLISH|BEARISH|NEUTRAL
CONFIDENCE: 0.XX
SUMMARY: one or two sentences
DATA_POINTS: point1 | point2 | point3"""


class StockSentimentAgent(BaseAgent):
    name = "Sentiment"

    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        headlines = await self._fetch_stock_news(symbol)
        fear_greed = await self._fetch_fear_greed()
        return {"symbol": symbol, "headlines": headlines, "fear_greed": fear_greed}

    async def _fetch_stock_news(self, symbol: str) -> list[str]:
        """Fetch stock news: Finnhub company-news → NewsAPI → yfinance."""
        from datetime import datetime, timedelta

        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        if finnhub_key:
            try:
                to_date = datetime.utcnow().strftime("%Y-%m-%d")
                from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
                url = (
                    f"https://finnhub.io/api/v1/company-news"
                    f"?symbol={symbol.upper()}&from={from_date}&to={to_date}&token={finnhub_key}"
                )
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                        articles = await r.json()
                headlines = [a.get("headline", "") for a in articles[:8] if a.get("headline")]
                if headlines:
                    return headlines
            except Exception:
                pass

        newsapi_key = os.getenv("NEWSAPI_KEY", "")
        if newsapi_key:
            try:
                url = (
                    f"https://newsapi.org/v2/everything"
                    f"?q={symbol}+stock+earnings"
                    f"&sortBy=publishedAt&pageSize=8&language=en&apiKey={newsapi_key}"
                )
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                        data = await r.json()
                headlines = [a.get("title", "") for a in data.get("articles", [])[:8]]
                if headlines:
                    return headlines
            except Exception:
                pass

        # yfinance fallback
        try:
            import asyncio

            def _sync() -> list[str]:
                import yfinance as yf
                ticker = yf.Ticker(symbol.upper())
                news = getattr(ticker, "news", None) or []
                return [
                    item.get("content", {}).get("title", item.get("title", ""))
                    for item in news[:8]
                    if item.get("content", {}).get("title") or item.get("title")
                ]

            return await asyncio.to_thread(_sync)
        except Exception:
            return []

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
                "current_value": int(entries[0]["value"]) if entries else 50,
                "label": entries[0].get("value_classification", "Neutral") if entries else "Neutral",
            }
        except Exception:
            return {"current_value": 50, "label": "Neutral"}

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        fg = data.get("fear_greed", {})
        prompt = (
            f"Stock: {symbol}\n\n"
            f"Market Fear & Greed: {fg.get('current_value', 50)} ({fg.get('label', 'Neutral')})\n\n"
            f"Recent headlines:\n"
            + "\n".join(f"- {h}" for h in data.get("headlines", []))
            + "\n\nAnalyse sentiment. BEARISH = short opportunity."
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
