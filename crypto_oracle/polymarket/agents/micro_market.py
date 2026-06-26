from __future__ import annotations

from crypto_oracle.polymarket.agents.base import (
    fetch_market_microstructure,
    fetch_spot_price,
    parse_market_question,
)
from crypto_oracle.polymarket.models import PolymarketMarket, PolymarketSpecialistSignal


class MicroMarketAgent:
    name = 'MicroMarket'

    async def run(self, market: PolymarketMarket) -> PolymarketSpecialistSignal:
        spot = await fetch_spot_price()
        parsed = parse_market_question(market, spot)
        micro = await fetch_market_microstructure(market)
        midpoint = micro.get('midpoint_yes')
        imbalance = float(micro.get('imbalance') or 0.0)
        spread = float(micro.get('spread_yes') or 0.0)
        momentum = float(micro.get('history_momentum') or 0.0)
        market_edge = 0.0
        if midpoint is not None:
            market_edge = float(midpoint) - 0.5
        score = max(-1.0, min(1.0, (market_edge * 1.2) + (imbalance * 0.9) + (momentum * 1.1) - min(spread, 0.2)))
        if parsed.comparator == 'below':
            score *= -1
        confidence = max(0.0, min(1.0, 0.45 + abs(score) * 0.4 + max(0.0, 0.08 - spread)))
        stance = 'BULLISH' if score > 0.08 else 'BEARISH' if score < -0.08 else 'NEUTRAL'
        summary = (
            f"Yes midpoint {midpoint if midpoint is not None else 'n/a'} with orderbook imbalance {imbalance:+.2f} and short-horizon momentum {momentum:+.2f}; "
            f"microstructure {'supports' if stance == 'BULLISH' else 'leans against' if stance == 'BEARISH' else 'does not strongly support'} the market resolving {parsed.yes_condition}."
        )
        return PolymarketSpecialistSignal(
            agent_name='MicroMarket',
            stance=stance,
            score=round(score, 4),
            confidence=round(confidence, 4),
            summary=summary,
            data_points=[
                f"yes_midpoint={midpoint}",
                f"imbalance={imbalance:+.3f}",
                f"spread={spread:.4f}",
            ],
            evidence={'spot_price': spot, 'parsed_question': parsed.model_dump(), 'micro': micro},
        )
