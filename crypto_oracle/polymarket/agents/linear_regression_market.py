from __future__ import annotations

from crypto_oracle.polymarket.agents.base import fetch_spot_history, fetch_spot_price, parse_market_question
from crypto_oracle.polymarket.models import PolymarketMarket, PolymarketSpecialistSignal


def _linear_regression_forecast(prices: list[float], steps_ahead: int = 1) -> tuple[float | None, float]:
    if len(prices) < 5:
        return None, 0.0
    n = len(prices)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(prices) / n
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    if ss_xx == 0:
        return None, 0.0
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, prices))
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    forecast_x = n - 1 + steps_ahead
    forecast = intercept + slope * forecast_x
    ss_tot = sum((y - mean_y) ** 2 for y in prices)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, prices))
    r_squared = 0.0 if ss_tot == 0 else max(0.0, min(1.0, 1 - (ss_res / ss_tot)))
    return forecast, r_squared


class LinearRegressionMarketAgent:
    name = 'LinearRegressionMarket'

    async def run(self, market: PolymarketMarket) -> PolymarketSpecialistSignal:
        spot = await fetch_spot_price()
        parsed = parse_market_question(market, spot)
        prices = await fetch_spot_history(30)
        forecast, r_squared = _linear_regression_forecast(prices[-60:] if len(prices) > 60 else prices, steps_ahead=3)
        if forecast is None:
            score = 0.0
            confidence = 0.0
            stance = 'NEUTRAL'
            summary = 'Linear regression forecast unavailable due to insufficient BTC history.'
            evidence = {'spot_price': spot, 'forecast_price': None, 'r_squared': 0.0}
        else:
            forecast_move_pct = (forecast - spot) / spot if spot > 0 else 0.0
            threshold_component = 0.0 if parsed.distance_to_threshold_pct is None else max(-0.3, min(0.3, parsed.distance_to_threshold_pct / 18.0))
            raw_score = (forecast_move_pct * 8.0) + threshold_component
            score = max(-1.0, min(1.0, raw_score))
            confidence = max(0.0, min(1.0, 0.35 + (abs(score) * 0.35) + (r_squared * 0.25)))
            stance = 'BULLISH' if score > 0.08 else 'BEARISH' if score < -0.08 else 'NEUTRAL'
            summary = (
                f'Linear regression forecast projects BTC to {forecast:,.0f} from spot {spot:,.0f} '
                f'with fit quality r²={r_squared:.2f}; forecast {"supports" if stance == "BULLISH" else "leans against" if stance == "BEARISH" else "is mixed on"} {parsed.yes_condition}.'
            )
            evidence = {
                'spot_price': spot,
                'forecast_price': forecast,
                'forecast_move_pct': round(forecast_move_pct, 6),
                'r_squared': r_squared,
                'parsed_question': parsed.model_dump(),
            }
        return PolymarketSpecialistSignal(
            agent_name='LinearRegressionMarket',
            stance=stance,
            score=round(score, 4),
            confidence=round(confidence, 4),
            summary=summary,
            data_points=[f'forecast={forecast}', f'r_squared={r_squared:.3f}'],
            evidence=evidence,
        )
