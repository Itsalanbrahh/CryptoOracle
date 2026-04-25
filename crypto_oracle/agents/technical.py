"""Technical — Technical analysis agent (EMA, RSI, MACD, Bollinger Bands)."""

from __future__ import annotations

import json
import math
from typing import Any

import aiohttp

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal

_SYSTEM = """You are Technical, a technical analysis trading agent. You receive OHLCV
data and pre-computed technical indicators (EMAs, RSI, MACD, Bollinger Bands).
Identify chart patterns and give your directional signal.

Respond ONLY in this exact format:
SIGNAL: BULLISH|BEARISH|NEUTRAL
CONFIDENCE: 0.XX
SUMMARY: one or two sentences
DATA_POINTS: point1 | point2 | point3"""


def _ema(prices: list[float], period: int) -> list[float]:
    if len(prices) < period:
        return []
    k = 2 / (period + 1)
    ema = [sum(prices[:period]) / period]
    for p in prices[period:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return ema


def _rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [-min(d, 0) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _bollinger(prices: list[float], period: int = 20) -> dict:
    if len(prices) < period:
        return {}
    window = prices[-period:]
    mid = sum(window) / period
    std = math.sqrt(sum((p - mid) ** 2 for p in window) / period)
    return {
        "upper": round(mid + 2 * std, 2),
        "middle": round(mid, 2),
        "lower": round(mid - 2 * std, 2),
        "bandwidth_pct": round(4 * std / mid * 100, 3),
    }


class TechnicalAgent(BaseAgent):
    name = "Technical"

    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        cg_id = _CG_MAP.get(symbol.upper(), symbol.lower())
        url = (
            f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
            f"?vs_currency=usd&days=90&interval=daily"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                raw = await r.json()

        prices = [p[1] for p in raw.get("prices", [])]
        if len(prices) < 30:
            return {"prices": prices, "indicators": {}}

        ema9 = _ema(prices, 9)
        ema21 = _ema(prices, 21)
        ema50 = _ema(prices, 50)

        rsi = _rsi(prices)
        bb = _bollinger(prices)

        # MACD (12-26-9)
        ema12 = _ema(prices, 12)
        ema26 = _ema(prices, 26)
        macd_line = []
        offset = len(ema12) - len(ema26)
        for i in range(len(ema26)):
            macd_line.append(ema12[i + offset] - ema26[i])
        signal_line = _ema(macd_line, 9)
        macd_hist = (
            macd_line[-1] - signal_line[-1] if signal_line else 0.0
        )

        current = prices[-1]
        indicators = {
            "current_price": round(current, 2),
            "ema9": round(ema9[-1], 2) if ema9 else None,
            "ema21": round(ema21[-1], 2) if ema21 else None,
            "ema50": round(ema50[-1], 2) if ema50 else None,
            "ema9_above_ema21": ema9[-1] > ema21[-1] if (ema9 and ema21) else None,
            "ema21_above_ema50": ema21[-1] > ema50[-1] if (ema21 and ema50) else None,
            "rsi_14": round(rsi, 2),
            "macd_histogram": round(macd_hist, 4),
            "bollinger": bb,
            "price_vs_bb_upper_pct": round(
                (current - bb["upper"]) / bb["upper"] * 100, 2
            ) if bb else None,
        }

        return {"prices": prices[-30:], "indicators": indicators}

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        prompt = (
            f"Symbol: {symbol}\n"
            f"Technical indicators: {json.dumps(data.get('indicators', {}), indent=2)}\n\n"
            "Analyse technical indicators and give your signal."
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
