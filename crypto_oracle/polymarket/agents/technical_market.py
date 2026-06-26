from __future__ import annotations

from crypto_oracle.polymarket.agents.base import fetch_spot_history, fetch_spot_price, parse_market_question
from crypto_oracle.polymarket.models import PolymarketMarket, PolymarketSpecialistSignal


def _ema(prices: list[float], period: int) -> float | None:
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def _rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [-min(d, 0) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


class TechnicalMarketAgent:
    name = 'TechnicalMarket'

    async def run(self, market: PolymarketMarket) -> PolymarketSpecialistSignal:
        spot = await fetch_spot_price()
        parsed = parse_market_question(market, spot)
        prices = await fetch_spot_history(90)
        ema9 = _ema(prices, 9)
        ema21 = _ema(prices, 21)
        ema50 = _ema(prices, 50)
        rsi = _rsi(prices)
        trend = 0.0
        if ema9 and ema21:
            trend += 0.35 if ema9 > ema21 else -0.35
        if ema21 and ema50:
            trend += 0.25 if ema21 > ema50 else -0.25
        if spot > (ema21 or spot):
            trend += 0.2
        else:
            trend -= 0.2
        if rsi > 58:
            trend += 0.15
        elif rsi < 42:
            trend -= 0.15
        threshold_component = 0.0 if parsed.distance_to_threshold_pct is None else max(-0.35, min(0.35, parsed.distance_to_threshold_pct / 20.0))
        score = max(-1.0, min(1.0, trend + threshold_component))
        confidence = max(0.0, min(1.0, 0.46 + abs(score) * 0.34))
        stance = 'BULLISH' if score > 0.08 else 'BEARISH' if score < -0.08 else 'NEUTRAL'
        summary = (
            f"Spot ${spot:,.0f} vs EMA9/21/50 = {ema9 and round(ema9,2)}/{ema21 and round(ema21,2)}/{ema50 and round(ema50,2)}, RSI {rsi:.1f}; "
            f"technical structure {'supports' if stance == 'BULLISH' else 'leans against' if stance == 'BEARISH' else 'is mixed on'} {parsed.yes_condition}."
        )
        return PolymarketSpecialistSignal(
            agent_name='TechnicalMarket',
            stance=stance,
            score=round(score, 4),
            confidence=round(confidence, 4),
            summary=summary,
            data_points=[f'ema9={ema9}', f'ema21={ema21}', f'rsi={rsi:.1f}'],
            evidence={'spot_price': spot, 'parsed_question': parsed.model_dump(), 'ema9': ema9, 'ema21': ema21, 'ema50': ema50, 'rsi': rsi},
        )
