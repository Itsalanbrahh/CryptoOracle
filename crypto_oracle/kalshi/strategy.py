"""Trading strategy for Kalshi BTC binary markets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist as _ND
from typing import Literal

from .markets import KalshiMarket


def _gbm_prob(spot: float, strike: float, hours_to_expiry: float, annual_vol: float) -> float:
    """P(BTC > strike at expiry) under lognormal GBM with zero drift."""
    t = max(hours_to_expiry, 0.25) / 8760
    sigma_t = annual_vol * math.sqrt(t)
    if sigma_t < 1e-9:
        return 1.0 if spot >= strike else 0.0
    d2 = math.log(spot / strike) / sigma_t
    return _ND().cdf(d2)


def _gbm_range_prob(spot: float, floor: float, cap: float, hours_to_expiry: float, annual_vol: float) -> float:
    """P(floor <= BTC < cap at expiry) under lognormal GBM with zero drift."""
    return _gbm_prob(spot, floor, hours_to_expiry, annual_vol) - _gbm_prob(spot, cap, hours_to_expiry, annual_vol)


@dataclass
class KalshiDecision:
    ticker: str
    strike: float
    action: Literal["BUY_YES", "BUY_NO", "HOLD"]
    side: Literal["yes", "no", "hold"]
    price: float           # execution price in dollars
    price_cents: int       # execution price in cents (for API)
    count: int             # contracts to buy ($1 each, min 1)
    position_usd: float    # total cost = count * price
    profit_if_win: float   # payoff = count * (1 - price)  for YES, or count * price for NO
    confidence: float
    edge: float
    gbm_baseline: float | None = None  # GBM-implied probability used as anchor
    reasoning: str = ""


def decide_kalshi_trade(
    market: KalshiMarket,
    *,
    aggregate: float,           # [-1, 1], positive = bullish
    confidence: float,          # [0, 1]
    spot: float = 0.0,          # BTC spot price; used for GBM baseline
    annual_vol: float = 0.65,   # realized annualized vol; drives GBM width
    max_position_usd: float = 20.0,
    min_edge: float = 0.03,
    min_confidence: float = 0.52,
    min_strike_distance_pct: float = 0.0,   # skip if strike too close to spot
    momentum_trigger: float = 0.0,          # +1 = strong uptrend, -1 = strong downtrend
    divergence_cut: float = 1.0,            # multiplier for position size (1.0 = no cut)
) -> KalshiDecision:
    """
    Decide whether to buy YES or NO on a Kalshi BTC contract.

    belief_yes is anchored to GBM P(BTC > strike at expiry), then tilted by
    the agent aggregate signal (±15%). This prevents the model from fighting
    efficient market pricing near expiry.
    """
    if spot > 0 and market.strike > 0:
        if market.is_range and market.cap_strike:
            gbm = _gbm_range_prob(spot, market.strike, market.cap_strike, market.hours_to_expiry, annual_vol)
            belief_yes = max(0.02, min(0.98, gbm + aggregate * 0.05))
        else:
            gbm = _gbm_prob(spot, market.strike, market.hours_to_expiry, annual_vol)
            belief_yes = max(0.02, min(0.98, gbm + aggregate * 0.15))
    else:
        belief_yes = max(0.02, min(0.98, (aggregate + 1.0) / 2.0))
    belief_no = 1.0 - belief_yes

    # Use actual execution prices for edge calculation
    # Mid can be misleading when spreads are wide — paying no_ask or yes_ask
    # gives the real edge after accounting for the spread cost
    market_yes = market.mid      # reference only, not used for edge gate
    market_no = 1.0 - market_yes

    # Edge against execution price, not mid: what we'd actually get after spread
    exec_edge_yes = belief_yes - market.yes_ask
    exec_edge_no = belief_no - market.no_ask

    # Keep mid-based edge for display/reference
    edge_yes = belief_yes - market_yes
    edge_no = belief_no - market_no

    def _hold(reason: str) -> KalshiDecision:
        return KalshiDecision(
            ticker=market.ticker, strike=market.strike,
            action="HOLD", side="hold",
            price=market_yes, price_cents=market.mid_cents, count=0,
            position_usd=0.0, profit_if_win=0.0,
            confidence=confidence, edge=max(edge_yes, edge_no),
            gbm_baseline=belief_yes,
            reasoning=reason,
        )

    if confidence < min_confidence:
        return _hold(f"confidence {confidence:.2f} below threshold {min_confidence:.2f}")

    # ── Strike distance gate: skip if strike is too close to spot ──────────
    if spot > 0 and min_strike_distance_pct > 0:
        strike_dist_pct = abs(market.strike - spot) / spot * 100
        if strike_dist_pct < min_strike_distance_pct:
            return _hold(
                f"strike distance {strike_dist_pct:.2f}% below min {min_strike_distance_pct:.1f}% "
                f"(strike=${market.strike:,.0f}, spot=${spot:,.0f})"
            )

    buy_yes = exec_edge_yes > min_edge
    buy_no = exec_edge_no > min_edge

    if not buy_yes and not buy_no:
        return _hold(f"edge insufficient (yes={exec_edge_yes:.3f}, no={exec_edge_no:.3f})")

    if buy_yes:
        # ── Momentum gate: don't buy YES in a strong downtrend ─────────────
        if momentum_trigger < -0.5:
            return _hold(
                f"momentum={momentum_trigger:.2f}: strong downtrend, skipping YES "
                f"(would bet against momentum)"
            )
        exec_price = market.yes_ask   # buy at the ask
        exec_price_cents = int(round(exec_price * 100))
        # Conviction-weighted position sizing: scales $2–5 based on confidence × edge
        # 50% confidence + 0.05 edge → ~$2.00   |   60% confidence + 0.15 edge → $5.00
        conviction_score = max(0.35, min(1.0, (confidence - 0.50) * 8 + exec_edge_yes * 5))
        position_usd = max_position_usd * conviction_score
        # ── Divergence cut: reduce position when agents strongly disagree ──
        if divergence_cut < 1.0:
            position_usd = position_usd * divergence_cut
        count = max(1, int(position_usd / exec_price))
        actual_position = round(count * exec_price, 2)
        profit_if_win = round(count * (1.0 - exec_price), 2)
        return KalshiDecision(
            ticker=market.ticker, strike=market.strike,
            action="BUY_YES", side="yes",
            price=exec_price, price_cents=exec_price_cents, count=count,
            position_usd=actual_position, profit_if_win=profit_if_win,
            confidence=confidence, edge=exec_edge_yes,
            gbm_baseline=belief_yes,
            reasoning=(
                f"{'range' if market.is_range else 'dir'} gbm={belief_yes:.2f} vol={annual_vol:.0%} "
                f"market={market_yes:.2f} edge={exec_edge_yes:.3f} tte={market.hours_to_expiry:.1f}h"
            ),
        )
    else:
        # ── Momentum gate: don't buy NO in a strong uptrend ────────────────
        if momentum_trigger > 0.5:
            return _hold(
                f"momentum={momentum_trigger:.2f}: strong uptrend, skipping NO "
                f"(would bet against momentum)"
            )
        exec_price = market.no_ask
        exec_price_cents = int(round((1.0 - exec_price) * 100))
        # Conviction-weighted position sizing: scales $2–5 based on confidence × edge
        conviction_score = max(0.35, min(1.0, (confidence - 0.50) * 8 + exec_edge_no * 5))
        position_usd = max_position_usd * conviction_score
        # ── Payoff-ratio boost for cheap NO contracts ───────────────────────
        # A NO at $0.12 pays 7.3:1 — Kelly fraction grows with the payoff ratio.
        # Boost is capped at 1.5x to avoid over-sizing on very cheap contracts
        # that are cheap for a reason (near-zero GBM probability of winning).
        payoff_ratio = (1.0 - exec_price) / exec_price if exec_price > 0 else 1.0
        payoff_boost = min(1.5, 1.0 + max(0.0, payoff_ratio - 1.0) * 0.08)
        position_usd = position_usd * payoff_boost
        # ── Divergence cut: reduce position when agents strongly disagree ──
        if divergence_cut < 1.0:
            position_usd = position_usd * divergence_cut
        count = max(1, int(position_usd / exec_price))
        actual_position = round(count * exec_price, 2)
        profit_if_win = round(count * (1.0 - exec_price), 2)
        return KalshiDecision(
            ticker=market.ticker, strike=market.strike,
            action="BUY_NO", side="no",
            price=exec_price, price_cents=exec_price_cents, count=count,
            position_usd=actual_position, profit_if_win=profit_if_win,
            confidence=confidence, edge=exec_edge_no,
            gbm_baseline=belief_yes,
            reasoning=(
                f"{'range' if market.is_range else 'dir'} gbm_yes={belief_yes:.2f} vol={annual_vol:.0%} "
                f"belief_no={belief_no:.2f} market_no={market_no:.2f} edge={exec_edge_no:.3f} tte={market.hours_to_expiry:.1f}h"
            ),
        )
