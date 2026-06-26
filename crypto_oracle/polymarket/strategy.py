from __future__ import annotations

from .models import PolymarketMarket, PolymarketMasterDecision


def decide_trade(
    market: PolymarketMarket,
    *,
    micro_score: float,
    macro_score: float,
    knowledge_score: float,
    max_position_usd: float = 50.0,
    max_loss_pct: float = 0.5,
    confidence_threshold: float = 0.52,
) -> PolymarketMasterDecision:
    """Live trading policy for Polymarket binary BTC markets.

    Scores are in [-1, 1]. Positive = bullish BTC, negative = bearish BTC.
    Converts aggregate to a belief probability [0,1] before comparing to market prices.
    """
    aggregate = (0.4 * micro_score) + (0.35 * macro_score) + (0.25 * knowledge_score)
    confidence = min(0.99, max(0.0, abs(aggregate)))

    # Convert directional score [-1,1] to belief probability [0,1]
    belief_yes = (aggregate + 1.0) / 2.0
    belief_no = 1.0 - belief_yes

    yes_price = market.yes_outcome.price
    no_price = market.no_outcome.price

    # Edge = how much we think the market is mispricing the outcome
    edge_yes = belief_yes - yes_price
    edge_no = belief_no - no_price

    if confidence < confidence_threshold:
        return PolymarketMasterDecision(
            market_id=market.market_id,
            question=market.question,
            action='HOLD',
            confidence=confidence,
            trend_bias='NEUTRAL',
            reasoning='Aggregate conviction is below the minimum threshold.',
            catalysts=['waiting for stronger micro/macro alignment'],
            risks=['low-confidence regime', 'overtrading in noise'],
            max_loss_usd=0.0,
            position_size_usd=0.0,
        )

    # Buy YES when we think YES is underpriced, buy NO when we think NO is underpriced.
    # Don't gate on aggregate direction — belief_yes already encodes that; use pure edge.
    # (edge_yes = -edge_no always, so only one can be positive at a time)
    buy_yes = edge_yes > 0.03
    buy_no = edge_no > 0.03

    # Position sizing: for high-confidence YES (price ≥ 0.75), size up to target ~$5 profit
    best_edge = edge_yes if buy_yes else edge_no if buy_no else 0.0
    if buy_yes and yes_price >= 0.75:
        # profit = (position / yes_price) * (1 - yes_price); solve for $5 target
        target_profit = 5.0
        ideal_position = target_profit * yes_price / (1.0 - yes_price)
        position_size = round(min(max_position_usd, ideal_position), 2)
        # Risk: actual max loss = full position (if NO wins), but probability-weighted
        max_loss = round(position_size * (1.0 - yes_price), 2)  # expected loss at market probability
    else:
        position_size = round(max_position_usd * min(1.0, confidence * (1.0 + best_edge)), 2)
        max_loss = round(position_size * max_loss_pct, 2)

    if buy_yes:
        return PolymarketMasterDecision(
            market_id=market.market_id,
            question=market.question,
            action='BUY_YES',
            confidence=confidence,
            target_outcome='Yes',
            price=yes_price,
            trend_bias='BULLISH',
            reasoning=f'Belief={belief_yes:.2f} > market_price={yes_price:.3f} (edge={edge_yes:.3f}). Signals align bullishly.',
            catalysts=['bullish aggregate', f'Yes underpriced by {edge_yes:.1%}'],
            risks=['BTC reversal', 'binary expiry risk'],
            max_loss_usd=max_loss,
            position_size_usd=position_size,
        )
    if buy_no:
        return PolymarketMasterDecision(
            market_id=market.market_id,
            question=market.question,
            action='BUY_NO',
            confidence=confidence,
            target_outcome='No',
            price=no_price,
            trend_bias='BEARISH',
            reasoning=f'Belief_no={belief_no:.2f} > market_price={no_price:.3f} (edge={edge_no:.3f}). Signals align bearishly.',
            catalysts=['bearish aggregate', f'No underpriced by {edge_no:.1%}'],
            risks=['short squeeze', 'violent BTC upside'],
            max_loss_usd=max_loss,
            position_size_usd=position_size,
        )
    return PolymarketMasterDecision(
        market_id=market.market_id,
        question=market.question,
        action='HOLD',
        confidence=confidence,
        trend_bias='BULLISH' if aggregate > 0 else 'BEARISH',
        reasoning=f'Edge insufficient: edge_yes={edge_yes:.3f}, edge_no={edge_no:.3f}. Market already fairly priced.',
        catalysts=['directional conviction exists'],
        risks=['market priced near conviction', 'thin edge'],
        max_loss_usd=0.0,
        position_size_usd=0.0,
    )
