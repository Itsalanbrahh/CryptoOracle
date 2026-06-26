from __future__ import annotations

import math

from crypto_oracle.polymarket.agents.base import fetch_spot_history, fetch_spot_price, parse_market_question, realized_volatility
from crypto_oracle.polymarket.models import PolymarketMarket, PolymarketSpecialistSignal


class KnowledgeMarketAgent:
    name = 'KnowledgeMarket'

    async def run(self, market: PolymarketMarket) -> PolymarketSpecialistSignal:
        spot = await fetch_spot_price()
        parsed = parse_market_question(market, spot)
        history = await fetch_spot_history(120)
        vol = realized_volatility(history, 30)
        days = parsed.days_to_expiry or 30.0
        horizon_sigma = vol * math.sqrt(max(days, 1.0))
        distance_ratio = 0.0 if parsed.distance_to_threshold_pct is None else parsed.distance_to_threshold_pct / 100.0
        z = 0.0 if horizon_sigma <= 0 else distance_ratio / horizon_sigma
        score = math.tanh(z)
        confidence = max(0.0, min(1.0, 0.5 + min(0.35, abs(z) * 0.18)))
        stance = 'BULLISH' if score > 0.08 else 'BEARISH' if score < -0.08 else 'NEUTRAL'
        summary = (
            f"Using ~30d realized volatility of {vol:.4f} and {days:.1f} days to expiry, the threshold sits {parsed.distance_to_threshold_pct if parsed.distance_to_threshold_pct is not None else 0:+.2f}% from spot; "
            f"knowledge/range context {'favors' if stance == 'BULLISH' else 'argues against' if stance == 'BEARISH' else 'is mixed on'} resolution at {parsed.yes_condition}."
        )
        return PolymarketSpecialistSignal(
            agent_name='KnowledgeMarket',
            stance=stance,
            score=round(float(score), 4),
            confidence=round(confidence, 4),
            summary=summary,
            data_points=[f'days_to_expiry={days:.1f}', f'realized_vol={vol:.4f}', f'z_score={z:.3f}'],
            evidence={'spot_price': spot, 'parsed_question': parsed.model_dump(), 'realized_volatility': vol, 'z_score': z},
        )
