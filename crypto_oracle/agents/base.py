"""Base class for all CryptoOracle agents."""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any

import anthropic

from crypto_oracle.models.signals import AgentSignal, AgentName
from crypto_oracle.utils.logger import get_logger

# Limit concurrent Claude calls across all agents to avoid 529 overloaded errors
_claude_sem = asyncio.Semaphore(3)


class BaseAgent(ABC):
    """Abstract base for all 7 specialist agents."""

    name: AgentName

    def __init__(self) -> None:
        self.client = anthropic.AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_retries=4,
        )
        self.logger = get_logger(f"agent.{self.name}")

    async def run(self, symbol: str) -> AgentSignal:
        """Fetch data, call Claude, return structured signal."""
        self.logger.info("Running %s agent for %s", self.name, symbol)
        try:
            try:
                from crypto_oracle.models.db import get_agent_config
                config = await get_agent_config(self.name)
                self._db_system_prompt: Optional[str] = (config["system_prompt"] or None) if config else None
                self._db_agent_config: dict = config["config"] if config else {}
                if self._db_system_prompt:
                    self.logger.debug("%s: using master-updated system prompt", self.name)
                if self._db_agent_config:
                    self.logger.debug("%s: using master-updated config %s", self.name, self._db_agent_config)
            except Exception:
                self._db_system_prompt = None
                self._db_agent_config = {}
            data = await self.fetch_data(symbol)
            signal = await self.analyze(symbol, data)
            self.logger.info(
                "%s → %s (%.0f%%)", self.name, signal.signal, signal.confidence * 100
            )
            return signal
        except Exception as exc:
            self.logger.error("%s agent failed: %s", self.name, exc, exc_info=True)
            return AgentSignal(
                agent_name=self.name,
                signal="NEUTRAL",
                confidence=0.0,
                summary=f"Agent failed: {exc}",
            )

    @abstractmethod
    async def fetch_data(self, symbol: str) -> dict[str, Any]:
        """Fetch raw data for this agent's domain."""

    @abstractmethod
    async def analyze(self, symbol: str, data: dict[str, Any]) -> AgentSignal:
        """Use Claude to produce a structured AgentSignal."""

    async def _call_claude(self, system: str, user: str) -> str:
        """Helper: single Claude call, returns text. DB-stored prompt overrides the default."""
        effective_system = getattr(self, "_db_system_prompt", None) or system
        async with _claude_sem:
            msg = await self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=effective_system,
                messages=[{"role": "user", "content": user}],
            )
        return msg.content[0].text

    def _parse_signal_from_text(self, text: str) -> tuple[str, float, str, list[str]]:
        """
        Parse Claude's response for SIGNAL/CONFIDENCE/SUMMARY/DATA_POINTS.
        Expected format:
            SIGNAL: BULLISH|BEARISH|NEUTRAL
            CONFIDENCE: 0.XX
            SUMMARY: one or two sentences
            DATA_POINTS: point1 | point2 | point3
        """
        lines = {
            k.strip().upper(): v.strip()
            for line in text.strip().splitlines()
            if ":" in line
            for k, v in [line.split(":", 1)]
        }
        signal = lines.get("SIGNAL", "NEUTRAL").upper()
        if signal not in ("BULLISH", "BEARISH", "NEUTRAL"):
            signal = "NEUTRAL"
        try:
            confidence = float(lines.get("CONFIDENCE", "0.5"))
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.5
        summary = lines.get("SUMMARY", text[:200])
        raw_points = lines.get("DATA_POINTS", "")
        data_points = [p.strip() for p in raw_points.split("|") if p.strip()]
        return signal, confidence, summary, data_points
