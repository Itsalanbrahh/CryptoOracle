"""OnChain — On-chain metrics agent (hash rate, mempool, block activity)."""

from __future__ import annotations

import json
from typing import Any

import aiohttp

from crypto_oracle.agents.base import BaseAgent
from crypto_oracle.models.signals import AgentSignal

_SYSTEM = """You are OnChain, a blockchain analytics agent. You receive on-chain metrics
from mempool.space: network hash rate, mempool congestion, fee pressure, and recent
block transaction throughput. Assess what on-chain data signals about market direction.

High hash rate + low mempool + low fees = healthy/neutral network.
High mempool congestion + rising fees = increased on-chain activity (often bullish).
Falling hash rate = miner capitulation risk (bearish).
Very low fees + empty mempool = low demand/bearish.

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
            # Hash rate (EH/s) from mempool.space — replaces blockchain.info
            try:
                async with session.get(
                    "https://mempool.space/api/v1/mining/hashrate/3d",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    hr_data = await r.json()
                hashrates = hr_data.get("hashrates", [])
                metrics["network"] = {
                    "hash_rate_eh": round(hashrates[-1]["avgHashrate"] / 1e18, 2) if hashrates else 0,
                }
            except Exception as exc:
                self.logger.warning("mempool.space hashrate failed: %s", exc)
                metrics["network"] = {}

            # Mempool state — pending tx count, size, fee pressure
            try:
                async with session.get(
                    "https://mempool.space/api/mempool",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    mp = await r.json()
                metrics["mempool"] = {
                    "pending_tx": mp.get("count", 0),
                    "pending_size_mb": round(mp.get("vsize", 0) / 1e6, 2),
                    "total_pending_fees_btc": round(mp.get("total_fee", 0) / 1e8, 4),
                }
            except Exception as exc:
                self.logger.warning("mempool.space mempool failed: %s", exc)
                metrics["mempool"] = {}

            # Recommended fee rates (sat/vB) — fast/medium/slow
            try:
                async with session.get(
                    "https://mempool.space/api/v1/fees/recommended",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    fees = await r.json()
                metrics["fee_rates_sat_vb"] = fees
            except Exception as exc:
                self.logger.warning("mempool.space fees failed: %s", exc)
                metrics["fee_rates_sat_vb"] = {}

            # Recent block throughput — avg tx/block, block size
            try:
                async with session.get(
                    "https://mempool.space/api/blocks",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    blocks = await r.json()
                if blocks:
                    recent = blocks[:6]
                    metrics["recent_blocks"] = {
                        "latest_height": recent[0].get("height", 0),
                        "avg_tx_per_block": round(
                            sum(b.get("tx_count", 0) for b in recent) / len(recent)
                        ),
                        "avg_size_kb": round(
                            sum(b.get("size", 0) for b in recent) / len(recent) / 1024, 1
                        ),
                    }
            except Exception as exc:
                self.logger.warning("mempool.space blocks failed: %s", exc)
                metrics["recent_blocks"] = {}

        return metrics

    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        if symbol.upper() == "BTC":
            network = data.get("network", {})
            mempool = data.get("mempool", {})
            fees    = data.get("fee_rates_sat_vb", {})
            blocks  = data.get("recent_blocks", {})
            if not any([network, mempool, fees, blocks]):
                return AgentSignal(
                    agent_name=self.name,
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary="Data feed failure — all mempool.space endpoints unavailable.",
                    data_points=["data_feed_failure"],
                )
            # Zero hash rate with no mempool data = suspicious
            if network and network.get("hash_rate_eh", 1) == 0 and not mempool:
                return AgentSignal(
                    agent_name=self.name,
                    signal="NEUTRAL",
                    confidence=0.0,
                    summary="Data anomaly — hash rate reporting zero with no mempool data.",
                    data_points=["data_anomaly", "hash_rate_eh=0"],
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
