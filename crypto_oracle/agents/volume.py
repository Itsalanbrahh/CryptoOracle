"""Volume — Volume profile and OBV analysis agent."""

from __future__ import annotations

import json
import math
from typing import Any

import aiohttp

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal

_SYSTEM = """You are Volume, a volume-analysis trading agent. You receive OHLCV data
including volume metrics, OBV trend, volume-price divergences, and relative volume.
Identify whether volume confirms or contradicts price action.

Respond ONLY in this exact format:
SIGNAL: BULLISH|BEARISH|NEUTRAL
CONFIDENCE: 0.XX
SUMMARY: one or two sentences
DATA_POINTS: point1 | point2 | point3"""


class VolumeAgent(BaseAgent):
    name = "Volume"

    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        cg_id = _CG_MAP.get(symbol.upper(), symbol.lower())
        url = (
            f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
            f"?vs_currency=usd&days=14&interval=daily"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                raw = await r.json()

        prices = [p[1] for p in raw.get("prices", [])]
        volumes = [v[1] for v in raw.get("total_volumes", [])]

        if len(prices) < 7 or len(volumes) < 7:
            return {"prices": prices, "volumes": volumes, "metrics": {}}

        # OBV
        obv = [0.0]
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif prices[i] < prices[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])

        avg_vol_7d = sum(volumes[-7:]) / 7
        avg_vol_prev7d = sum(volumes[-14:-7]) / 7
        relative_volume = avg_vol_7d / avg_vol_prev7d if avg_vol_prev7d > 0 else 1.0

        obv_trend = "rising" if obv[-1] > obv[-7] else "falling"
        price_trend = "rising" if prices[-1] > prices[-7] else "falling"
        divergence = obv_trend != price_trend

        return {
            "current_price": prices[-1],
            "metrics": {
                "avg_volume_7d": round(avg_vol_7d, 2),
                "relative_volume_vs_prior_week": round(relative_volume, 3),
                "obv_trend_7d": obv_trend,
                "price_trend_7d": price_trend,
                "obv_price_divergence": divergence,
                "obv_7d_change_pct": round(
                    (obv[-1] - obv[-7]) / abs(obv[-7]) * 100 if obv[-7] != 0 else 0, 2
                ),
            },
        }

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        prompt = (
            f"Symbol: {symbol}\n"
            f"Current price: ${data.get('current_price', 0):,.2f}\n"
            f"Volume metrics: {json.dumps(data.get('metrics', {}), indent=2)}\n\n"
            "Analyse volume dynamics and give your signal."
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
