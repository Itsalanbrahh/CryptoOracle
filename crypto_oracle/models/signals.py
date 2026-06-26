"""Pydantic models for agent signals and master recommendations."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


AgentName = Literal[
    "Kronos",
    "Macro",
    "Micro",
    "Volume",
    "OnChain",
    "Sentiment",
    "Technical",
]

SignalType = Literal["BULLISH", "BEARISH", "NEUTRAL"]
ActionType = Literal["BUY", "SELL", "HOLD"]


class AgentSignal(BaseModel):
    agent_name: AgentName
    signal: SignalType
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    data_points: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    extra: dict = Field(default_factory=dict)  # agent-specific payload (e.g. Kronos forecast path)

    # populated after DB save
    recommendation_id: Optional[int] = None


class MasterRecommendation(BaseModel):
    symbol: str
    action: ActionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    key_catalysts: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    suggested_position_size: str = ""
    agent_signals: list[AgentSignal] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    price_at_time: Optional[float] = None   # spot price when recommendation was made

    # populated after DB save
    id: Optional[int] = None

    def to_telegram_message(self) -> str:
        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(self.action, "⚪")
        lines = [
            f"{action_emoji} *CryptoOracle — {self.symbol}*",
            f"Signal: *{self.action}* | Confidence: *{self.confidence*100:.0f}%*",
            "",
            f"_{self.reasoning[:300]}_",
        ]
        if self.key_catalysts:
            lines += ["", "✅ *Key Catalysts:*"] + [f"• {c}" for c in self.key_catalysts[:3]]
        if self.key_risks:
            lines += ["", "⚠️ *Key Risks:*"] + [f"• {r}" for r in self.key_risks[:3]]
        if self.suggested_position_size:
            lines += ["", f"💼 *Position Size:* {self.suggested_position_size}"]
        lines += ["", f"_Updated: {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}_"]
        return "\n".join(lines)
