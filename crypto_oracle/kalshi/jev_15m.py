"""Jev-driven entry judgment for Kalshi KXBTC15M (15-minute BTC UP/DOWN).

These markets settle on CF Benchmarks BRTI 60s averages (open vs close of the
interval), not on Coinbase/Kraken last trade. The strategy review's core idea
is probability-dislocation vs the *executable ask* — so Jev's job is to

  1. estimate P(settle UP / YES), and
  2. choose buy_yes / buy_no / hold given that belief and the live book.

Fee-aware edge gates in ``strategy.decide_kalshi_trade`` still own sizing and
whether the edge clears ``KALSHI_15M_MIN_EDGE`` (default 8pp).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Literal

from .jev_client import JevError, jev_enabled, system_one
from .markets import KalshiMarket


ActionChoice = Literal["buy_yes", "buy_no", "hold"]


@dataclass
class Jev15mJudgment:
    """Structured Jev output for one 15m market."""
    p_settle_up: float          # Noul probability YES settles
    action: ActionChoice
    action_confidence: float    # Choice confidence in [0, 1] (0 if unknown)
    action_probs: dict[str, float]
    model: str
    reasoning: str
    state: dict                 # audit trail for postmortem
    raw_answers: dict

    @property
    def belief_yes(self) -> float:
        return self.p_settle_up

    @property
    def aggregate(self) -> float:
        """Map P(YES) → [-1, 1] for agent_signals / postmortem compatibility."""
        return max(-1.0, min(1.0, (self.p_settle_up - 0.5) * 2.0))

    @property
    def confidence(self) -> float:
        """Blend distance-from-coin-flip with Choice confidence when present."""
        sharpness = abs(self.p_settle_up - 0.5) * 2.0  # 0 at 50/50, 1 at 0 or 1
        if self.action_confidence > 0:
            return max(0.20, min(0.95, 0.5 * sharpness + 0.5 * self.action_confidence))
        return max(0.20, min(0.95, sharpness))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw and raw.strip():
        try:
            return float(raw.strip())
        except ValueError:
            pass
    return default


def build_15m_state(
    market: KalshiMarket,
    *,
    spot: float,
    annual_vol: float,
    funding_rate: float | None = None,
    recent_closes: list[float] | None = None,
) -> dict[str, Any]:
    """Encode an auditable, session-aware state blob for Jev."""
    minutes_left = market.hours_to_expiry * 60.0
    target = market.strike
    distance_usd = spot - target if spot > 0 and target > 0 else 0.0
    distance_bps = (distance_usd / target * 10_000.0) if target > 0 else 0.0

    recent = recent_closes or []
    path_change_bps = 0.0
    if len(recent) >= 2 and recent[0] > 0:
        path_change_bps = (recent[-1] - recent[0]) / recent[0] * 10_000.0

    return {
        "market_type": "Kalshi KXBTC15M Bitcoin 15-minute UP/DOWN binary",
        "settlement": (
            "YES (UP) if ending CF Benchmarks BRTI 60-second average "
            ">= opening BRTI target; else NO (DOWN). Spot proxies are not settlement."
        ),
        "session": "crypto_24_7",
        "ticker": market.ticker,
        "opening_brti_target_usd": round(target, 2),
        "spot_proxy_usd": round(spot, 2),
        "distance_usd": round(distance_usd, 2),
        "distance_bps": round(distance_bps, 1),
        "minutes_remaining": round(minutes_left, 2),
        "book": {
            "yes_bid": round(market.yes_bid, 4),
            "yes_ask": round(market.yes_ask, 4),
            "no_bid": round(market.no_bid, 4),
            "no_ask": round(market.no_ask, 4),
            "yes_mid": round(market.mid, 4),
        },
        "realized_vol_annual": round(annual_vol, 4),
        "funding_rate_8h": funding_rate,
        "recent_1m_closes_usd": [round(c, 2) for c in recent[-15:]],
        "path_change_bps_window": round(path_change_bps, 1),
        "edge_policy": (
            f"Only buy when estimated fair probability exceeds executable ask "
            f"by at least {_env_float('KALSHI_15M_MIN_EDGE', 0.08):.0%} after fees."
        ),
    }


def _questions() -> dict[str, dict]:
    return {
        "settle_up": {
            "type": "noul",
            "instructions": (
                "Will the ending CF Benchmarks BRTI 60-second average be at or "
                "above the opening_brti_target_usd for this 15-minute interval? "
                "YES means UP settles; answer as a calibrated probability."
            ),
        },
        "action": {
            "type": "choice",
            "instructions": (
                "Given settle_up probability vs the executable asks in book, "
                "which action maximizes expected value after Kalshi fees? "
                "Prefer hold when edge is thin, time remaining is under 30 seconds, "
                "or the book already prices the likely outcome."
            ),
            "criteria": {
                "buy_yes": (
                    "Buy YES/UP: fair P(UP) clearly exceeds yes_ask after fees; "
                    "spot/path supports finishing above the opening target."
                ),
                "buy_no": (
                    "Buy NO/DOWN: fair P(DOWN)=1-P(UP) clearly exceeds no_ask after fees; "
                    "spot/path supports finishing below the opening target."
                ),
                "hold": (
                    "Do not enter: insufficient edge, late-interval risk, "
                    "or quotes already efficient."
                ),
            },
        },
    }


def _parse_answers(data: dict, state: dict) -> Jev15mJudgment:
    answers = data.get("answers") or {}
    settle = answers.get("settle_up") or {}
    action = answers.get("action") or {}

    p = settle.get("noul")
    if p is None:
        p = settle.get("probability")
    try:
        p_up = float(p) if p is not None else 0.5
    except (TypeError, ValueError):
        p_up = 0.5
    p_up = max(0.0, min(1.0, p_up))

    choice = str(action.get("choice") or "hold").strip().lower()
    if choice not in ("buy_yes", "buy_no", "hold"):
        # Fall back from probabilities if choice label is unexpected
        probs = action.get("probabilities") or {}
        if isinstance(probs, dict) and probs:
            choice = max(probs.items(), key=lambda kv: float(kv[1] or 0))[0]
        else:
            choice = "hold"
    if choice not in ("buy_yes", "buy_no", "hold"):
        choice = "hold"

    probs_raw = action.get("probabilities") or {}
    action_probs = {}
    if isinstance(probs_raw, dict):
        for k, v in probs_raw.items():
            try:
                action_probs[str(k)] = float(v)
            except (TypeError, ValueError):
                pass

    conf = action.get("confidence")
    try:
        action_confidence = float(conf) if conf is not None else 0.0
    except (TypeError, ValueError):
        action_confidence = 0.0
    action_confidence = max(0.0, min(1.0, action_confidence))

    model = str(data.get("model") or DEFAULT_MODEL_LABEL)
    reasoning = (
        f"jev settle_up={p_up:.3f} action={choice} "
        f"action_conf={action_confidence:.2f} "
        f"yes_ask={state['book']['yes_ask']:.3f} no_ask={state['book']['no_ask']:.3f} "
        f"tte={state['minutes_remaining']:.1f}m"
    )
    return Jev15mJudgment(
        p_settle_up=p_up,
        action=choice,  # type: ignore[arg-type]
        action_confidence=action_confidence,
        action_probs=action_probs,
        model=model,
        reasoning=reasoning,
        state=state,
        raw_answers=answers if isinstance(answers, dict) else {},
    )


DEFAULT_MODEL_LABEL = "jev-latest"


async def judge_15m_market(
    market: KalshiMarket,
    *,
    spot: float,
    annual_vol: float,
    funding_rate: float | None = None,
    recent_closes: list[float] | None = None,
) -> Jev15mJudgment:
    """Call Jev for one KXBTC15M market. Raises JevError on transport/API failure."""
    if not jev_enabled():
        raise JevError("Jev disabled or API key missing")

    state = build_15m_state(
        market,
        spot=spot,
        annual_vol=annual_vol,
        funding_rate=funding_rate,
        recent_closes=recent_closes,
    )
    data = await system_one(state, _questions())
    return _parse_answers(data, state)


def heuristic_15m_judgment(
    market: KalshiMarket,
    *,
    spot: float,
    annual_vol: float,
    funding_rate: float | None = None,
    recent_closes: list[float] | None = None,
) -> Jev15mJudgment:
    """
    Offline fallback when Jev is unavailable (tests / paper without a key).

    Uses a simple GBM-style distance-to-target probability so the 15m path
    still runs and logs; never pretends to be Jev in ``model``.
    """
    state = build_15m_state(
        market,
        spot=spot,
        annual_vol=annual_vol,
        funding_rate=funding_rate,
        recent_closes=recent_closes,
    )
    t_years = max(market.hours_to_expiry, 1 / 60) / 8760.0
    sigma = max(annual_vol, 0.20) * math.sqrt(t_years)
    if market.strike <= 0 or spot <= 0 or sigma < 1e-9:
        p_up = 0.5
    else:
        # P(S_T > K) under zero-drift lognormal using spot as proxy
        from statistics import NormalDist
        d2 = math.log(spot / market.strike) / sigma - 0.5 * sigma
        p_up = NormalDist().cdf(d2)
    p_up = max(0.02, min(0.98, p_up))

    yes_edge = p_up - market.yes_ask
    no_edge = (1.0 - p_up) - market.no_ask
    min_edge = _env_float("KALSHI_15M_MIN_EDGE", 0.08)
    # Side must agree with the probability: never "buy NO" while P(UP)>0.5
    # (that pattern manufactures edge from a miscalibrated book vs belief).
    if p_up >= 0.5 and yes_edge >= min_edge:
        action: ActionChoice = "buy_yes"
    elif p_up <= 0.5 and no_edge >= min_edge:
        action = "buy_no"
    else:
        action = "hold"

    return Jev15mJudgment(
        p_settle_up=p_up,
        action=action,
        action_confidence=abs(p_up - 0.5) * 2.0,
        action_probs={action: 1.0},
        model="heuristic-fallback",
        reasoning=(
            f"heuristic settle_up={p_up:.3f} action={action} "
            f"yes_edge={yes_edge:.3f} no_edge={no_edge:.3f}"
        ),
        state=state,
        raw_answers={},
    )


async def evaluate_15m_market(
    market: KalshiMarket,
    *,
    spot: float,
    annual_vol: float,
    funding_rate: float | None = None,
    recent_closes: list[float] | None = None,
) -> Jev15mJudgment:
    """Prefer live Jev; fall back to heuristic if disabled or the call fails."""
    if jev_enabled():
        try:
            return await judge_15m_market(
                market,
                spot=spot,
                annual_vol=annual_vol,
                funding_rate=funding_rate,
                recent_closes=recent_closes,
            )
        except JevError as exc:
            fb = heuristic_15m_judgment(
                market,
                spot=spot,
                annual_vol=annual_vol,
                funding_rate=funding_rate,
                recent_closes=recent_closes,
            )
            fb.reasoning = f"jev_error={exc}; {fb.reasoning}"
            return fb
    return heuristic_15m_judgment(
        market,
        spot=spot,
        annual_vol=annual_vol,
        funding_rate=funding_rate,
        recent_closes=recent_closes,
    )
