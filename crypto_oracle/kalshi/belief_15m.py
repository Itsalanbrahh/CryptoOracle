"""Blend Jev + residual model into a single P(UP) for 15m gates."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .features_15m import build_feature_vector
from .markets import KalshiMarket
from .residual_model import ResidualPrediction, predict_p_up


@dataclass
class BlendedBelief:
    p_up: float
    jev_p_up: float
    gbm_p_up: float
    residual: ResidualPrediction
    features: dict
    source: str


def blend_belief(
    market: KalshiMarket,
    *,
    spot: float,
    annual_vol: float,
    funding_rate: float | None,
    recent_closes: list[float] | None,
    micro: dict | None,
    jev_p_up: float,
) -> BlendedBelief:
    feats = build_feature_vector(
        market,
        spot=spot,
        annual_vol=annual_vol,
        funding_rate=funding_rate,
        recent_closes=recent_closes,
        micro=micro,
        jev_p_up=jev_p_up,
    )
    use_residual = os.getenv("KALSHI_15M_USE_RESIDUAL", "1").strip() != "0"
    # Anchor preference: Jev when present, else GBM inside features
    anchor = jev_p_up
    if use_residual:
        residual = predict_p_up(feats, anchor=anchor)
        p_up = residual.p_up
        source = f"residual:{residual.model_version}" if residual.used_model else "jev_anchor"
    else:
        residual = ResidualPrediction(
            p_up=anchor, anchor=anchor, residual=0.0, model_version="disabled", used_model=False
        )
        p_up = anchor
        source = "jev_only"

    return BlendedBelief(
        p_up=p_up,
        jev_p_up=jev_p_up,
        gbm_p_up=float(feats["gbm_p_up"]),
        residual=residual,
        features=feats,
        source=source,
    )
