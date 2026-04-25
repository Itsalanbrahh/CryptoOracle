"""Kronos — Quantitative/Statistical agent using Monte Carlo simulation."""

from __future__ import annotations

import json
import math
import random
from typing import Any

import aiohttp

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal

_SYSTEM = """You are Kronos, a quantitative trading agent specialising in statistical
analysis of crypto price movements. You receive OHLCV data and statistical metrics,
then analyse them using concepts like Monte Carlo paths, Sharpe ratio, volatility
regimes, mean reversion, and momentum signals.

Respond ONLY in this exact format (no extra text):
SIGNAL: BULLISH|BEARISH|NEUTRAL
CONFIDENCE: 0.XX
SUMMARY: one or two sentences about your finding
DATA_POINTS: point1 | point2 | point3"""


class KronosAgent(BaseAgent):
    name = "Kronos"

    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        cg_id = _CG_MAP.get(symbol.upper(), symbol.lower())
        url = (
            f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
            f"?vs_currency=usd&days=30&interval=daily"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                raw = await r.json()

        prices = [p[1] for p in raw.get("prices", [])]
        if len(prices) < 7:
            return {"prices": prices, "stats": {}}

        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
        ]
        mu = sum(returns) / len(returns)
        sigma = math.sqrt(
            sum((r - mu) ** 2 for r in returns) / len(returns)
        )

        # Simplified Monte Carlo: 1 000 paths × 7 days
        current_price = prices[-1]
        paths_up = 0
        paths = 1000
        for _ in range(paths):
            price = current_price
            for _ in range(7):
                price *= math.exp(
                    (mu - 0.5 * sigma ** 2) + sigma * random.gauss(0, 1)
                )
            if price > current_price:
                paths_up += 1

        sharpe = (mu / sigma * math.sqrt(365)) if sigma > 0 else 0
        momentum_7d = (prices[-1] - prices[-7]) / prices[-7] if len(prices) >= 7 else 0

        return {
            "prices": prices[-14:],
            "current_price": current_price,
            "stats": {
                "daily_return_mean": round(mu, 6),
                "daily_return_std": round(sigma, 6),
                "sharpe_annualised": round(sharpe, 3),
                "momentum_7d_pct": round(momentum_7d * 100, 2),
                "monte_carlo_bullish_pct": round(paths_up / paths * 100, 1),
            },
        }

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        stats = data.get("stats", {})
        prompt = (
            f"Symbol: {symbol}\n"
            f"Current price: ${data.get('current_price', 0):,.2f}\n"
            f"Statistical metrics: {json.dumps(stats, indent=2)}\n\n"
            "Analyse the quantitative data above and produce your signal."
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


_CG_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "XRP": "ripple",
}
