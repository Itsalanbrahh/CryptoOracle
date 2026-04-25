"""Kronos — Probabilistic time-series forecasting via Amazon Chronos-Bolt.

Uses amazon/chronos-bolt-small (48M params, 250x faster than original Chronos)
to generate a 7-day probabilistic price forecast with quantile intervals.

Falls back to pure-Python GBM Monte Carlo if chronos-forecasting is not installed.
Model is cached as a module-level singleton so it loads only once per process.

Model selection (via CHRONOS_MODEL env var, default: amazon/chronos-bolt-small):
  amazon/chronos-bolt-tiny   —  9M params, fastest
  amazon/chronos-bolt-small  — 48M params, good balance (default)
  amazon/chronos-bolt-base   — 205M params, most accurate, needs more RAM

Device auto-detection: cuda → mps (Apple Silicon) → cpu
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any

import aiohttp

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM = """You are Kronos, a quantitative trading agent powered by a probabilistic
time-series forecasting model (Amazon Chronos). You receive a 7-day price forecast
with confidence intervals and derived statistics.

Key metrics to weigh:
- median_return_7d: expected directional move
- bullish_probability: fraction of forecast mass above current price
- ci_width_pct: relative width of the 80% confidence interval (proxy for uncertainty)
- trend_slope: whether the median forecast is accelerating or decelerating

Respond ONLY in this exact format (no extra text):
SIGNAL: BULLISH|BEARISH|NEUTRAL
CONFIDENCE: 0.XX
SUMMARY: one or two sentences referencing specific forecast numbers
DATA_POINTS: point1 | point2 | point3"""

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_pipeline = None
_pipeline_loaded = False


def _get_device_and_dtype():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        if torch.backends.mps.is_available():
            return "mps", torch.float32
    except ImportError:
        pass
    return "cpu", None  # chronos will use float32 on cpu


def _load_pipeline():
    global _pipeline, _pipeline_loaded
    if _pipeline_loaded:
        return _pipeline

    model_name = os.getenv("CHRONOS_MODEL", "amazon/chronos-bolt-small")
    device, dtype = _get_device_and_dtype()

    try:
        import torch
        from chronos import BaseChronosPipeline

        kwargs: dict[str, Any] = {"device_map": device}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype

        logger.info("Loading Chronos model %s on %s …", model_name, device)
        _pipeline = BaseChronosPipeline.from_pretrained(model_name, **kwargs)
        logger.info("Chronos model loaded successfully")
    except Exception as exc:
        logger.warning("Chronos not available (%s) — will use GBM Monte Carlo fallback", exc)
        _pipeline = None

    _pipeline_loaded = True
    return _pipeline


# ---------------------------------------------------------------------------
# CoinGecko symbol map
# ---------------------------------------------------------------------------

_CG_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "XRP": "ripple",
    "AVAX": "avalanche-2",
    "LTC": "litecoin",
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class KronosAgent(BaseAgent):
    name = "Kronos"

    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        cg_id = _CG_MAP.get(symbol.upper(), symbol.lower())
        # 90 days of daily prices gives Chronos solid context
        url = (
            f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
            f"?vs_currency=usd&days=90&interval=daily"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                raw = await r.json()

        prices = [p[1] for p in raw.get("prices", [])]
        if len(prices) < 14:
            return {"prices": prices, "forecast": {}, "method": "insufficient_data"}

        pipeline = _load_pipeline()

        if pipeline is not None:
            forecast = await self._chronos_forecast(pipeline, prices)
            method = "chronos"
        else:
            forecast = _gbm_monte_carlo(prices)
            method = "gbm_monte_carlo"

        return {
            "prices": prices[-14:],
            "current_price": prices[-1],
            "forecast": forecast,
            "method": method,
        }

    async def _chronos_forecast(
        self, pipeline: Any, prices: list[float]
    ) -> dict[str, Any]:
        import asyncio
        import torch

        def _sync() -> dict:
            context = torch.tensor(prices, dtype=torch.float32)

            quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
            quantiles, mean = pipeline.predict_quantiles(
                context,
                prediction_length=7,
                quantile_levels=quantile_levels,
            )
            # quantiles shape: (1, 7, 5)  [batch, timestep, quantile_idx]
            # mean shape: (1, 7)
            q = quantiles[0]  # (7, 5)
            m = mean[0]       # (7,)

            current = prices[-1]

            q10 = q[:, 0].tolist()   # pessimistic bound
            q25 = q[:, 1].tolist()
            q50 = q[:, 2].tolist()   # median
            q75 = q[:, 3].tolist()
            q90 = q[:, 4].tolist()   # optimistic bound
            mean_fc = m.tolist()

            median_end = q50[-1]
            median_return_7d = (median_end - current) / current

            # Fraction of quantile mass above current price at day 7
            # Interpolate: how much of [q10, q90] is above current?
            bounds = [q10[-1], q25[-1], q50[-1], q75[-1], q90[-1]]
            levels = [0.1, 0.25, 0.5, 0.75, 0.9]
            bullish_prob = _interpolate_probability(current, bounds, levels)

            ci_width_pct = (q90[-1] - q10[-1]) / current * 100

            # Trend slope: linear regression of median forecast
            n = len(q50)
            x_mean = (n - 1) / 2
            slope_num = sum((i - x_mean) * (q50[i] - sum(q50) / n) for i in range(n))
            slope_den = sum((i - x_mean) ** 2 for i in range(n))
            trend_slope_pct_per_day = (slope_num / slope_den / current * 100) if slope_den else 0

            return {
                "current_price": round(current, 2),
                "median_price_7d": round(median_end, 2),
                "mean_price_7d": round(mean_fc[-1], 2),
                "median_return_7d_pct": round(median_return_7d * 100, 3),
                "bullish_probability": round(bullish_prob, 3),
                "ci_10_pct": round(q10[-1], 2),
                "ci_25_pct": round(q25[-1], 2),
                "ci_75_pct": round(q75[-1], 2),
                "ci_90_pct": round(q90[-1], 2),
                "ci_width_80_pct": round(ci_width_pct, 2),
                "trend_slope_pct_per_day": round(trend_slope_pct_per_day, 4),
                "median_path": [round(p, 2) for p in q50],
            }

        return await asyncio.to_thread(_sync)

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        fc = data.get("forecast", {})
        method = data.get("method", "unknown")
        current = data.get("current_price", 0)

        prompt = (
            f"Symbol: {symbol}\n"
            f"Current price: ${current:,.2f}\n"
            f"Forecast method: {method}\n"
            f"7-day probabilistic forecast:\n"
            f"{json.dumps(fc, indent=2)}\n\n"
            "Analyse this probabilistic forecast and produce your signal."
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate_probability(
    current: float, bounds: list[float], levels: list[float]
) -> float:
    """Estimate P(price > current) by linear interpolation over quantile bounds."""
    if current <= bounds[0]:
        return 1.0 - levels[0]
    if current >= bounds[-1]:
        return 1.0 - levels[-1]
    for i in range(len(bounds) - 1):
        if bounds[i] <= current <= bounds[i + 1]:
            frac = (current - bounds[i]) / (bounds[i + 1] - bounds[i])
            level_at_current = levels[i] + frac * (levels[i + 1] - levels[i])
            return round(1.0 - level_at_current, 3)
    return 0.5


def _gbm_monte_carlo(prices: list[float]) -> dict[str, Any]:
    """GBM Monte Carlo fallback when Chronos is not installed."""
    returns = [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
    ]
    mu = sum(returns) / len(returns)
    sigma = math.sqrt(sum((r - mu) ** 2 for r in returns) / max(len(returns) - 1, 1))

    current = prices[-1]
    n_paths = 2000
    horizon = 7
    endpoints = []

    for _ in range(n_paths):
        price = current
        for _ in range(horizon):
            price *= math.exp((mu - 0.5 * sigma ** 2) + sigma * random.gauss(0, 1))
        endpoints.append(price)

    endpoints.sort()
    n = len(endpoints)
    q10 = endpoints[int(0.10 * n)]
    q25 = endpoints[int(0.25 * n)]
    q50 = endpoints[int(0.50 * n)]
    q75 = endpoints[int(0.75 * n)]
    q90 = endpoints[int(0.90 * n)]

    bullish_prob = sum(1 for e in endpoints if e > current) / n
    sharpe = (mu / sigma * math.sqrt(365)) if sigma > 0 else 0

    return {
        "current_price": round(current, 2),
        "median_price_7d": round(q50, 2),
        "median_return_7d_pct": round((q50 - current) / current * 100, 3),
        "bullish_probability": round(bullish_prob, 3),
        "ci_10_pct": round(q10, 2),
        "ci_25_pct": round(q25, 2),
        "ci_75_pct": round(q75, 2),
        "ci_90_pct": round(q90, 2),
        "ci_width_80_pct": round((q90 - q10) / current * 100, 2),
        "sharpe_annualised": round(sharpe, 3),
        "note": "GBM Monte Carlo fallback (chronos-forecasting not installed)",
    }
