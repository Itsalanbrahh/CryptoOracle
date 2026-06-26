"""StockTechnical — Technical + volume analysis for equities via yfinance."""

from __future__ import annotations

import json
import math
from typing import Any

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal

_SYSTEM = """You are StockTechnical, a technical analysis agent for equities. You receive
OHLCV data and pre-computed indicators (EMAs, RSI, MACD, Bollinger Bands, volume).
This is a stock — analyse chart patterns for both long AND short opportunities.

BEARISH signals are actionable shorts, not just warnings. Be decisive.

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


class StockTechnicalAgent(BaseAgent):
    name = "Technical"

    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        import asyncio

        def _sync() -> dict:
            import yfinance as yf

            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period="90d")
            if hist.empty or len(hist) < 30:
                return {"prices": [], "indicators": {}}

            prices = hist["Close"].tolist()
            volumes = hist["Volume"].tolist()

            ema9 = _ema(prices, 9)
            ema21 = _ema(prices, 21)
            ema50 = _ema(prices, 50)
            rsi = _rsi(prices)
            bb = _bollinger(prices)

            ema12 = _ema(prices, 12)
            ema26 = _ema(prices, 26)
            macd_line: list[float] = []
            offset = len(ema12) - len(ema26)
            for i in range(len(ema26)):
                macd_line.append(ema12[i + offset] - ema26[i])
            signal_line = _ema(macd_line, 9)
            macd_hist = macd_line[-1] - signal_line[-1] if signal_line else 0.0

            current = prices[-1]

            avg_vol_7d = sum(volumes[-7:]) / 7 if len(volumes) >= 7 else 0
            avg_vol_prev7d = sum(volumes[-14:-7]) / 7 if len(volumes) >= 14 else avg_vol_7d
            rel_vol = avg_vol_7d / avg_vol_prev7d if avg_vol_prev7d > 0 else 1.0

            obv: list[float] = [0.0]
            for i in range(1, len(prices)):
                if prices[i] > prices[i - 1]:
                    obv.append(obv[-1] + volumes[i])
                elif prices[i] < prices[i - 1]:
                    obv.append(obv[-1] - volumes[i])
                else:
                    obv.append(obv[-1])
            obv_trend = "rising" if obv[-1] > obv[-7] else "falling"
            price_trend = "rising" if prices[-1] > prices[-7] else "falling"

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
                "volume": {
                    "avg_7d": round(avg_vol_7d, 0),
                    "relative_vs_prior_week": round(rel_vol, 3),
                    "obv_trend": obv_trend,
                    "price_trend": price_trend,
                    "obv_price_divergence": obv_trend != price_trend,
                },
            }
            return {"prices": prices[-30:], "indicators": indicators}

        return await asyncio.to_thread(_sync)

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        prompt = (
            f"Stock symbol: {symbol}\n"
            f"Indicators + volume: {json.dumps(data.get('indicators', {}), indent=2)}\n\n"
            "Analyse and give your signal. BEARISH = actionable short opportunity."
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
