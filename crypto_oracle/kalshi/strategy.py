"""Trading strategy for Kalshi BTC binary markets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist as _ND
from typing import Literal

from .markets import KalshiMarket


def _gbm_prob(spot: float, strike: float, hours_to_expiry: float, annual_vol: float, drift: float = 0.35) -> float:
    """P(BTC > strike at expiry) under lognormal GBM with BTC long-run drift.

    drift=0.35 (35% annual) reflects BTC's historical upward bias and corrects
    the zero-drift model's systematic underpricing of above-spot YES contracts.
    """
    t = max(hours_to_expiry, 0.25) / 8760
    sigma_t = annual_vol * math.sqrt(t)
    if sigma_t < 1e-9:
        return 1.0 if spot >= strike else 0.0
    d2 = (math.log(spot / strike) + (drift - 0.5 * annual_vol ** 2) * t) / sigma_t
    return _ND().cdf(d2)


def _gbm_range_prob(spot: float, floor: float, cap: float, hours_to_expiry: float, annual_vol: float) -> float:
    """P(floor <= BTC < cap at expiry) under lognormal GBM with zero drift."""
    return _gbm_prob(spot, floor, hours_to_expiry, annual_vol) - _gbm_prob(spot, cap, hours_to_expiry, annual_vol)


# ── Kalshi trading fees ──────────────────────────────────────────────────────
# Taker fee = ceil(0.07 × count × P × (1−P)) in cents, charged on execution.
# Maker fee = 25% of the taker rate (June 2026 schedule); we conservatively
# round UP to the next cent even though small maker fees often round to $0.00.
# The ceil makes 1-contract orders proportionally the most expensive — exactly
# what a small account places — so fees MUST be part of the edge gate.
_TAKER_FEE_RATE = 0.07
_MAKER_FEE_RATE = 0.0175


def _fee_total(price: float, count: int, maker: bool) -> float:
    """Total Kalshi trading fee in dollars for an order of `count` contracts."""
    if price <= 0.0 or price >= 1.0 or count <= 0:
        return 0.0
    rate = _MAKER_FEE_RATE if maker else _TAKER_FEE_RATE
    return math.ceil(rate * count * price * (1.0 - price) * 100.0) / 100.0


def _fee_per_contract(price: float, maker: bool) -> float:
    """Worst-case per-contract fee (count=1, where ceil-to-cent bites hardest)."""
    return _fee_total(price, 1, maker)


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
    maker_mode: bool = True,                # rest inside the spread (maker) vs lift the ask (taker)
    agg_tilt: float = 0.08,                 # max belief tilt from agent aggregate (dollars of prob)
    implied_prob: float | None = None,      # options-implied P(YES) anchor; preferred over GBM when set
) -> KalshiDecision:
    """
    Decide whether to buy YES or NO on a Kalshi BTC contract.

    belief_yes is anchored to GBM P(BTC > strike at expiry), then tilted by
    the agent aggregate signal (±15%). This prevents the model from fighting
    efficient market pricing near expiry.

    maker_mode=True prices entries one cent inside the spread (or joins the
    bid on a 1-cent book) so fills are maker fills: ~75% lower fees and the
    spread is captured instead of paid. Unfilled maker orders are canceled by
    the next scan's stale-entry cleanup, so a stale signal never fills late.
    """
    # The tilt is deliberately small: at 1–24h horizons even sophisticated
    # models barely beat coin-flip direction accuracy, so a large tilt
    # (the old ±0.15) mostly MANUFACTURED edges out of ensemble noise — the
    # perceived edge was the tilt itself, not a market mispricing. Keep the
    # anchor in charge and let agents nudge, not steer.
    #
    # Anchor preference: options-implied probability (Deribit IV surface —
    # the same source professionals price this ladder from) when available;
    # homegrown realized-vol GBM only as fallback. With the implied anchor,
    # "edge" means Kalshi's quote diverges from professional pricing — the
    # documented profitable pattern — rather than from our own model.
    anchor_src = "gbm"
    if spot > 0 and market.strike > 0:
        if market.is_range and market.cap_strike:
            if implied_prob is not None:
                anchor = implied_prob
                anchor_src = "iv"
            else:
                anchor = _gbm_range_prob(spot, market.strike, market.cap_strike, market.hours_to_expiry, annual_vol)
            belief_yes = max(0.02, min(0.98, anchor + aggregate * min(agg_tilt, 0.05)))
        else:
            if implied_prob is not None:
                anchor = implied_prob
                anchor_src = "iv"
            else:
                anchor = _gbm_prob(spot, market.strike, market.hours_to_expiry, annual_vol)
            belief_yes = max(0.02, min(0.98, anchor + aggregate * agg_tilt))
    else:
        belief_yes = max(0.02, min(0.98, (aggregate + 1.0) / 2.0))
    belief_no = 1.0 - belief_yes

    # Use actual execution prices for edge calculation
    # Mid can be misleading when spreads are wide — paying no_ask or yes_ask
    # gives the real edge after accounting for the spread cost
    market_yes = market.mid      # reference only, not used for edge gate
    market_no = 1.0 - market_yes

    # ── Execution price: maker (rest inside the spread) vs taker (lift ask) ──
    # Maker: improve the bid by 1¢ when the spread allows, else join the bid.
    # Never cross the ask — crossing turns the order into a taker fill.
    if maker_mode:
        yes_bid = max(0.01, market.yes_bid)
        yes_exec = max(0.01, min(market.yes_ask - 0.01, yes_bid + 0.01))
        no_bid = max(0.01, 1.0 - market.yes_ask)
        no_exec = max(0.01, min(market.no_ask - 0.01, no_bid + 0.01))
    else:
        yes_exec = market.yes_ask
        no_exec = market.no_ask

    # Per-contract trading fee at the intended execution (worst-case count=1)
    fee_yes = _fee_per_contract(yes_exec, maker=maker_mode)
    fee_no = _fee_per_contract(no_exec, maker=maker_mode)

    # Edge net of spread AND fee: what we would actually keep on a win.
    # Before fees were modeled, every trade with believed edge below the fee
    # (2–7% of stake depending on price) was a slow bleed that the gate passed.
    exec_edge_yes = belief_yes - yes_exec - fee_yes
    exec_edge_no = belief_no - no_exec - fee_no

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
        exec_price = yes_exec
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
        fee_paid = _fee_total(exec_price, count, maker_mode)
        profit_if_win = round(count * (1.0 - exec_price) - fee_paid, 2)
        return KalshiDecision(
            ticker=market.ticker, strike=market.strike,
            action="BUY_YES", side="yes",
            price=exec_price, price_cents=exec_price_cents, count=count,
            position_usd=actual_position, profit_if_win=profit_if_win,
            confidence=confidence, edge=exec_edge_yes,
            gbm_baseline=belief_yes,
            reasoning=(
                f"{'range' if market.is_range else 'dir'} anchor={anchor_src} belief={belief_yes:.2f} vol={annual_vol:.0%} "
                f"market={market_yes:.2f} edge={exec_edge_yes:.3f} fee=${fee_paid:.2f} "
                f"{'maker' if maker_mode else 'taker'} tte={market.hours_to_expiry:.1f}h"
            ),
        )
    else:
        # ── Momentum gate: don't buy NO in a strong uptrend ────────────────
        if momentum_trigger > 0.5:
            return _hold(
                f"momentum={momentum_trigger:.2f}: strong uptrend, skipping NO "
                f"(would bet against momentum)"
            )
        exec_price = no_exec
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
        fee_paid = _fee_total(exec_price, count, maker_mode)
        profit_if_win = round(count * (1.0 - exec_price) - fee_paid, 2)
        return KalshiDecision(
            ticker=market.ticker, strike=market.strike,
            action="BUY_NO", side="no",
            price=exec_price, price_cents=exec_price_cents, count=count,
            position_usd=actual_position, profit_if_win=profit_if_win,
            confidence=confidence, edge=exec_edge_no,
            gbm_baseline=belief_yes,
            reasoning=(
                f"{'range' if market.is_range else 'dir'} anchor={anchor_src} belief_yes={belief_yes:.2f} vol={annual_vol:.0%} "
                f"belief_no={belief_no:.2f} market_no={market_no:.2f} edge={exec_edge_no:.3f} "
                f"fee=${fee_paid:.2f} {'maker' if maker_mode else 'taker'} tte={market.hours_to_expiry:.1f}h"
            ),
        )
