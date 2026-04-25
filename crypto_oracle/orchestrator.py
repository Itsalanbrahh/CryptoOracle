"""CryptoOracle Orchestrator — runs all 7 agents and synthesises a MasterRecommendation."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import anthropic

from crypto_oracle.agents.kronos import KronosAgent
from crypto_oracle.agents.macro import MacroAgent
from crypto_oracle.agents.micro import MicroAgent
from crypto_oracle.agents.volume import VolumeAgent
from crypto_oracle.agents.onchain import OnChainAgent
from crypto_oracle.agents.sentiment import SentimentAgent
from crypto_oracle.agents.technical import TechnicalAgent
from crypto_oracle.models.signals import AgentSignal, MasterRecommendation
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

_SYNTH_SYSTEM = """You are the CryptoOracle master analyst. You receive structured signals
from 7 specialist sub-agents and must synthesise them into a single, definitive
trading recommendation.

Be direct and data-driven. Reference the strongest signals explicitly.
Factor in signal agreement, disagreement, and confidence weights.

Respond in this EXACT format (no extra text):
ACTION: BUY|SELL|HOLD
CONFIDENCE: 0.XX
REASONING: 2-3 sentences explaining the recommendation
CATALYSTS: catalyst1 | catalyst2 | catalyst3
RISKS: risk1 | risk2 | risk3
POSITION_SIZE: e.g. "2-3% of portfolio" or "hold current position"
"""


class CryptoOracle:
    """Orchestrates 7 specialist agents and synthesises the master recommendation."""

    def __init__(self) -> None:
        skip_kronos = os.getenv("SKIP_KRONOS", "false").lower() == "true"
        self.agents = [
            KronosAgent() if not skip_kronos else None,
            MacroAgent(),
            MicroAgent(),
            VolumeAgent(),
            OnChainAgent(),
            SentimentAgent(),
            TechnicalAgent(),
        ]
        self.agents = [a for a in self.agents if a is not None]
        self.client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        logger.info("CryptoOracle initialised with %d agents", len(self.agents))

    async def run(self, symbol: str) -> MasterRecommendation:
        """Run all agents concurrently and synthesise recommendation."""
        symbol = symbol.upper()
        logger.info("Oracle run started for %s", symbol)

        tasks = [agent.run(symbol) for agent in self.agents]
        signals: list[AgentSignal] = await asyncio.gather(*tasks, return_exceptions=False)

        valid_signals = [s for s in signals if isinstance(s, AgentSignal)]
        logger.info(
            "Collected %d/%d signals for %s", len(valid_signals), len(self.agents), symbol
        )

        rec = await self._synthesise(symbol, valid_signals)
        rec.agent_signals = valid_signals
        logger.info(
            "Oracle recommendation for %s: %s (%.0f%%)",
            symbol, rec.action, rec.confidence * 100,
        )
        return rec

    async def _synthesise(
        self, symbol: str, signals: list[AgentSignal]
    ) -> MasterRecommendation:
        signals_json = json.dumps(
            [
                {
                    "agent": s.agent_name,
                    "signal": s.signal,
                    "confidence": s.confidence,
                    "summary": s.summary,
                    "data_points": s.data_points,
                }
                for s in signals
            ],
            indent=2,
        )

        user_msg = (
            f"Symbol: {symbol}\n\n"
            f"Agent signals:\n{signals_json}\n\n"
            "Synthesise these signals into a master recommendation."
        )

        msg = await self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_SYNTH_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = msg.content[0].text
        return self._parse_master(symbol, text, signals)

    def _parse_master(
        self, symbol: str, text: str, signals: list[AgentSignal]
    ) -> MasterRecommendation:
        lines = {}
        for line in text.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                lines[k.strip().upper()] = v.strip()

        action = lines.get("ACTION", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"

        try:
            confidence = float(lines.get("CONFIDENCE", "0.5"))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.5

        reasoning = lines.get("REASONING", "Insufficient data for high-confidence signal.")
        catalysts = [c.strip() for c in lines.get("CATALYSTS", "").split("|") if c.strip()]
        risks = [r.strip() for r in lines.get("RISKS", "").split("|") if r.strip()]
        position_size = lines.get("POSITION_SIZE", "")

        return MasterRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            key_catalysts=catalysts,
            key_risks=risks,
            suggested_position_size=position_size,
            agent_signals=signals,
        )
