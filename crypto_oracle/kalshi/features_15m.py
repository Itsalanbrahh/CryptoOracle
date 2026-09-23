"""Feature engineering for KXBTC15M self-improvement loop."""
from __future__ import annotations

import math
from typing import Any

from .markets import KalshiMarket

# Stable column order for the residual model (must match train/predict).
FEATURE_KEYS: list[str] = [
    "minutes_left",
    "distance_bps",
    "distance_usd",
    "yes_ask",
    "no_ask",
    "yes_mid",
    "kalshi_spread",
    "annual_vol",
    "sigma_remaining",          # vol * sqrt(t_years) — distance scale
    "z_distance",               # distance / (spot * sigma_remaining)
    "gbm_p_up",
    "jev_p_up",
    "binance_imbalance",
    "binance_trade_imbalance",
    "binance_spread_bps",
    "bybit_imbalance",
    "bybit_spread_bps",
    "venue_mid_dispersion_bps",
    "combined_ofi",
    "funding_rate_8h",
    "path_change_bps",
]


def _gbm_p_up(spot: float, strike: float, hours_to_expiry: float, annual_vol: float) -> float:
    from statistics import NormalDist

    t = max(hours_to_expiry, 1 / 60) / 8760.0
    sigma = max(annual_vol, 0.20) * math.sqrt(t)
    if spot <= 0 or strike <= 0 or sigma < 1e-9:
        return 0.5
    d2 = math.log(spot / strike) / sigma - 0.5 * sigma
    return max(0.02, min(0.98, NormalDist().cdf(d2)))


def build_feature_vector(
    market: KalshiMarket,
    *,
    spot: float,
    annual_vol: float,
    funding_rate: float | None,
    recent_closes: list[float] | None,
    micro: dict[str, Any] | None,
    jev_p_up: float | None,
) -> dict[str, float]:
    """Dense numeric features for logging + residual model."""
    minutes_left = market.hours_to_expiry * 60.0
    target = market.strike
    distance_usd = (spot - target) if spot > 0 and target > 0 else 0.0
    distance_bps = (distance_usd / target * 10_000.0) if target > 0 else 0.0
    t_years = max(market.hours_to_expiry, 1 / 60) / 8760.0
    sigma_remaining = max(annual_vol, 0.20) * math.sqrt(t_years)
    z_distance = (distance_usd / (spot * sigma_remaining)) if spot > 0 and sigma_remaining > 0 else 0.0
    gbm = _gbm_p_up(spot, target, market.hours_to_expiry, annual_vol)

    path_change_bps = 0.0
    if recent_closes and len(recent_closes) >= 2 and recent_closes[0] > 0:
        path_change_bps = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 10_000.0

    micro = micro or {}

    def _f(key: str, default: float = 0.0) -> float:
        v = micro.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    feats = {
        "minutes_left": round(minutes_left, 3),
        "distance_bps": round(distance_bps, 2),
        "distance_usd": round(distance_usd, 2),
        "yes_ask": round(market.yes_ask, 4),
        "no_ask": round(market.no_ask, 4),
        "yes_mid": round(market.mid, 4),
        "kalshi_spread": round(max(0.0, market.yes_ask - market.yes_bid), 4),
        "annual_vol": round(annual_vol, 4),
        "sigma_remaining": round(sigma_remaining, 6),
        "z_distance": round(z_distance, 4),
        "gbm_p_up": round(gbm, 4),
        "jev_p_up": round(float(jev_p_up) if jev_p_up is not None else gbm, 4),
        "binance_imbalance": _f("binance_imbalance"),
        "binance_trade_imbalance": _f("binance_trade_imbalance"),
        "binance_spread_bps": _f("binance_spread_bps"),
        "bybit_imbalance": _f("bybit_imbalance"),
        "bybit_spread_bps": _f("bybit_spread_bps"),
        "venue_mid_dispersion_bps": _f("venue_mid_dispersion_bps"),
        "combined_ofi": _f("combined_ofi"),
        "funding_rate_8h": float(funding_rate or 0.0),
        "path_change_bps": round(path_change_bps, 2),
    }
    # Ensure every key present
    return {k: float(feats.get(k, 0.0)) for k in FEATURE_KEYS}


def vector_as_list(feats: dict[str, float]) -> list[float]:
    return [float(feats[k]) for k in FEATURE_KEYS]
