from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


TradeAction = Literal["BUY_YES", "BUY_NO", "HOLD"]
TrendBias = Literal["BULLISH", "BEARISH", "NEUTRAL"]
SpecialistAgentName = Literal[
    "MicroMarket",
    "MacroMarket",
    "KnowledgeMarket",
    "TechnicalMarket",
    "LinearRegressionMarket",
]


class PolymarketOutcome(BaseModel):
    name: str
    price: float = Field(..., ge=0.0, le=1.0)
    token_id: str | None = None


class PolymarketMarket(BaseModel):
    market_id: str
    event_id: str | None = None
    question: str
    slug: str | None = None
    end_date_iso: str | None = None
    volume: float = 0.0
    liquidity: float = 0.0
    outcomes: list[PolymarketOutcome]
    condition_id: str | None = None
    raw: dict = Field(default_factory=dict)

    @property
    def yes_outcome(self) -> PolymarketOutcome:
        for outcome in self.outcomes:
            if outcome.name.strip().lower() == 'yes':
                return outcome
        raise ValueError('market missing Yes outcome')

    @property
    def no_outcome(self) -> PolymarketOutcome:
        for outcome in self.outcomes:
            if outcome.name.strip().lower() == 'no':
                return outcome
        raise ValueError('market missing No outcome')


class ParsedMarketQuestion(BaseModel):
    comparator: Literal['above', 'below']
    threshold_price: float = Field(..., gt=0.0)
    reference_price: float | None = None
    expiry_iso: str | None = None
    days_to_expiry: float | None = None
    distance_to_threshold_pct: float | None = None
    yes_condition: str


class PolymarketSpecialistSignal(BaseModel):
    agent_name: SpecialistAgentName
    stance: TrendBias
    score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    summary: str
    data_points: list[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)


class PolymarketMasterDecision(BaseModel):
    market_id: str
    question: str
    action: TradeAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    target_outcome: str | None = None
    price: float | None = None
    trend_bias: TrendBias = 'NEUTRAL'
    reasoning: str
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    max_loss_usd: float = 0.0
    position_size_usd: float = 0.0
    expected_edge: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PolymarketRunResult(BaseModel):
    market: PolymarketMarket
    parsed_question: ParsedMarketQuestion
    spot_price: float
    yes_midpoint: float | None = None
    specialist_signals: list[PolymarketSpecialistSignal] = Field(default_factory=list)
    decision: PolymarketMasterDecision
    risk_checks: list[str] = Field(default_factory=list)


class PaperPortfolio(BaseModel):
    cash_usd: float = 1000.0
    yes_positions: dict[str, float] = Field(default_factory=dict)
    no_positions: dict[str, float] = Field(default_factory=dict)
    trade_log: list[dict] = Field(default_factory=list)
