from __future__ import annotations

import asyncio

from crypto_oracle.polymarket.agents import KnowledgeMarketAgent, LinearRegressionMarketAgent, MacroMarketAgent, MicroMarketAgent, TechnicalMarketAgent, fetch_spot_price, parse_market_question
from crypto_oracle.polymarket.models import PolymarketMarket, PolymarketMasterDecision, PolymarketRunResult
from crypto_oracle.polymarket.risk import RiskPolicy, evaluate_risk, load_risk_policy
from crypto_oracle.polymarket.strategy import decide_trade


class PolymarketOrchestrator:
    def __init__(self, policy: RiskPolicy | None = None) -> None:
        self.policy = policy or load_risk_policy()
        self.agents = [
            MicroMarketAgent(),
            MacroMarketAgent(),
            KnowledgeMarketAgent(),
            TechnicalMarketAgent(),
            LinearRegressionMarketAgent(),
        ]

    async def run_market(self, market: PolymarketMarket) -> PolymarketRunResult:
        spot = await fetch_spot_price()
        parsed = parse_market_question(market, spot)
        specialist_signals = await asyncio.gather(*(agent.run(market) for agent in self.agents))
        score_map = {signal.agent_name: signal.score for signal in specialist_signals}
        decision = decide_trade(
            market,
            micro_score=score_map.get('MicroMarket', 0.0),
            macro_score=score_map.get('MacroMarket', 0.0),
            knowledge_score=(
                (score_map.get('KnowledgeMarket', 0.0) * 0.4)
                + (score_map.get('TechnicalMarket', 0.0) * 0.3)
                + (score_map.get('LinearRegressionMarket', 0.0) * 0.3)
            ),
            max_position_usd=self.policy.max_position_usd,
            confidence_threshold=self.policy.min_confidence,
        )
        market_price = market.yes_outcome.price if decision.action == 'BUY_YES' else market.no_outcome.price if decision.action == 'BUY_NO' else market.yes_outcome.price
        implied = market_price if market_price is not None else 0.5
        belief = 0.5 + (decision.confidence / 2 if decision.action == 'BUY_YES' else -decision.confidence / 2 if decision.action == 'BUY_NO' else 0.0)
        decision.expected_edge = round(abs(belief - implied), 4)
        allowed, reasons = await evaluate_risk(market, decision, specialist_signals, self.policy)
        if not allowed and decision.action != 'HOLD':
            decision = PolymarketMasterDecision(
                market_id=decision.market_id,
                question=decision.question,
                action='HOLD',
                confidence=decision.confidence,
                target_outcome=None,
                price=None,
                trend_bias=decision.trend_bias,
                reasoning='Risk gate blocked the trade: ' + '; '.join(reasons),
                catalysts=decision.catalysts,
                risks=decision.risks + reasons,
                max_loss_usd=0.0,
                position_size_usd=0.0,
                expected_edge=decision.expected_edge,
            )
        midpoint = market.raw.get('midpoint_yes') if isinstance(market.raw, dict) else None
        return PolymarketRunResult(
            market=market,
            parsed_question=parsed,
            spot_price=spot,
            yes_midpoint=midpoint,
            specialist_signals=specialist_signals,
            decision=decision,
            risk_checks=reasons,
        )
