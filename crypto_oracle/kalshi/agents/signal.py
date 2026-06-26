"""Standardized signal wrapper for all Kalshi agents (new and old)."""
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentSignal:
    """Standard interface that the Kalshi loop expects for all agent signals."""
    agent_name: str
    score: float
    confidence: float
    summary: str = ""


def wrap_agent_result(agent_name: str, result: dict) -> AgentSignal:
    """Convert any agent's dict result into an AgentSignal."""
    return AgentSignal(
        agent_name=agent_name,
        score=result.get("score", 0.0),
        confidence=result.get("confidence", 0.0),
        summary=str(result.get("reasoning", result.get("summary", ""))),
    )
