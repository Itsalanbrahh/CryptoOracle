"""OnChain — On-chain metrics agent (addresses, hash rate, mempool)."""

from __future__ import annotations

import json
from typing import Any

import aiohttp

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal

_SYSTEM = """You are OnChain, a blockchain analytics agent. You receive on-chain metrics
such as active addresses, hash rate, mempool size, exchange flows, and whale movements.
Assess what on-chain data signals about market direction.

Respond ONLY in this exact format:
SIGNAL: BULLISH|BEARISH|NEUTRAL
CONFIDENCE: 0.XX
SUMMARY: one or two sentences
DATA_POINTS: point1 | point2 | point3"""


class OnChainAgent(BaseAgent):
    name = "OnChain"

    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        data: dict[str, Any] = {"symbol": symbol}

        if symbol.upper() == "BTC":
            data.update(await self._fetch_btc_metrics())
        else:
            data["note"] = f"On-chain metrics for {symbol} not available via free tier."

        return data

    async def _fetch_btc_metrics(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        async with aiohttp.ClientSession() as session:
            # Blockchain.com stats — hashrate, mempool, difficulty
            try:
                async with session.get(
                    "https://api.blockchain.info/stats",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    stats = await r.json()
                metrics["blockchain_stats"] = {
                    "hash_rate_gh": round(stats.get("hash_rate", 0) / 1e9, 2),
                    "difficulty": stats.get("difficulty", 0),
                    "total_fees_btc": round(stats.get("total_fees_btc", 0), 4),
                    "n_tx_24h": stats.get("n_tx", 0),
                    "miners_revenue_usd": stats.get("miners_revenue_usd", 0),
                    "estimated_transaction_volume_usd": stats.get(
                        "estimated_transaction_volume_usd", 0
                    ),
                }
            except Exception as exc:
                self.logger.warning("blockchain.info stats failed: %s", exc)
                metrics["blockchain_stats"] = {}

            # Mempool.space — mempool fee rates
            try:
                async with session.get(
                    "https://mempool.space/api/v1/fees/recommended",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    fees = await r.json()
                metrics["mempool_fees"] = fees
            except Exception as exc:
                self.logger.warning("mempool.space fees failed: %s", exc)
                metrics["mempool_fees"] = {}

        return metrics

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        if symbol.upper() == "BTC":
            stats = data.get("blockchain_stats", {})
            fees = data.get("mempool_fees", {})
            if not stats and not fees:
                return AgentSignal(
                    agent_name=self.name,
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary="Data feed failure — blockchain stats and mempool data both unavailable.",
                    data_points=["data_feed_failure"],
                )
            # All-zero stats = API returning placeholder/anomalous values, not real data
            if stats and stats.get("n_tx_24h", 1) == 0 and stats.get("hash_rate_gh", 1) == 0:
                return AgentSignal(
                    agent_name=self.name,
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary="Data anomaly — blockchain stats reporting zero transactions and zero hash rate.",
                    data_points=["data_anomaly", "n_tx_24h=0", "hash_rate=0"],
                )
            # Negative total_fees = corrupted blockchain.info response
            if stats and stats.get("total_fees_btc", 0) < 0:
                return AgentSignal(
                    agent_name=self.name,
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary="Data anomaly — blockchain.info returned negative total fees (corrupted response).",
                    data_points=["data_anomaly", f"total_fees_btc={stats.get('total_fees_btc')}"],
                )
        prompt = (
            f"Symbol: {symbol}\n"
            f"On-chain data: {json.dumps(data, indent=2)}\n\n"
            "Analyse on-chain metrics and give your signal."
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
