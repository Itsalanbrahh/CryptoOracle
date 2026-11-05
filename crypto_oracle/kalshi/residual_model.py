"""Residual probability model for KXBTC15M.

Learns a correction on top of the GBM/Jev anchor:
  P_up = clip(anchor + residual(features))

Starts with logistic regression on (label_up - 0.5) style target using
feature residuals; falls back to identity (no correction) until enough
labeled samples exist.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .features_15m import FEATURE_KEYS, vector_as_list

_MODEL_PATH = Path.home() / ".hermes" / "state" / "kalshi_15m_residual_model.json"
_MIN_TRAIN = 40          # need this many labeled rows before fitting
_MIN_CLASS = 8           # at least this many of each class


@dataclass
class ResidualPrediction:
    p_up: float
    anchor: float
    residual: float
    model_version: str
    used_model: bool


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip01(x: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return max(lo, min(hi, x))


def load_model() -> dict[str, Any] | None:
    if not _MODEL_PATH.exists():
        return None
    try:
        return json.loads(_MODEL_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_model(model: dict[str, Any]) -> None:
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MODEL_PATH.write_text(json.dumps(model, indent=2))


def predict_p_up(features: dict[str, float], anchor: float | None = None) -> ResidualPrediction:
    """Apply residual model if present; else return anchor (GBM/Jev)."""
    anchor_v = float(anchor if anchor is not None else features.get("jev_p_up") or features.get("gbm_p_up") or 0.5)
    model = load_model()
    if not model or model.get("type") != "logistic_residual":
        return ResidualPrediction(
            p_up=_clip01(anchor_v),
            anchor=anchor_v,
            residual=0.0,
            model_version="none",
            used_model=False,
        )

    means = model.get("means") or {}
    scales = model.get("scales") or {}
    coefs = model.get("coefs") or {}
    intercept = float(model.get("intercept", 0.0))
    max_residual = float(model.get("max_residual", 0.15))

    z = intercept
    for k in FEATURE_KEYS:
        x = float(features.get(k, 0.0))
        mu = float(means.get(k, 0.0))
        sd = float(scales.get(k, 1.0)) or 1.0
        z += float(coefs.get(k, 0.0)) * ((x - mu) / sd)

    # Model predicts P(up); blend as residual around anchor so we don't discard physics.
    p_model = _sigmoid(z)
    residual = max(-max_residual, min(max_residual, p_model - 0.5))
    # Prefer learning the gap vs anchor: if trained with target=label, residual≈p_model-0.5
    # Alternate blend: p = 0.7*anchor + 0.3*p_model when sample is thin
    blend = float(model.get("anchor_blend", 0.65))
    p_up = blend * anchor_v + (1.0 - blend) * p_model
    p_up = _clip01(p_up)
    return ResidualPrediction(
        p_up=p_up,
        anchor=anchor_v,
        residual=round(p_up - anchor_v, 4),
        model_version=str(model.get("version", "v1")),
        used_model=True,
    )


def train_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Fit logistic regression predicting label_up from features.

    Returns metrics dict; persists model when fit succeeds.
    """
    usable = [
        r for r in rows
        if r.get("labeled") and isinstance(r.get("features"), dict) and r.get("label_up") in (0, 1)
    ]
    result: dict[str, Any] = {
        "n": len(usable),
        "trained": False,
        "reason": None,
        "path": str(_MODEL_PATH),
    }
    if len(usable) < _MIN_TRAIN:
        result["reason"] = f"need>={_MIN_TRAIN} labeled rows, have {len(usable)}"
        return result

    y = [int(r["label_up"]) for r in usable]
    if sum(y) < _MIN_CLASS or (len(y) - sum(y)) < _MIN_CLASS:
        result["reason"] = f"need>={_MIN_CLASS} per class (up={sum(y)} down={len(y)-sum(y)})"
        return result

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss
        from sklearn.model_selection import TimeSeriesSplit
    except ImportError as exc:
        result["reason"] = f"sklearn unavailable: {exc}"
        return result

    X = np.array([vector_as_list(r["features"]) for r in usable], dtype=float)
    y_arr = np.array(y, dtype=int)

    means = X.mean(axis=0)
    scales = X.std(axis=0)
    scales[scales < 1e-8] = 1.0
    Xn = (X - means) / scales

    # Walk-forward-ish: last 20% holdout chronologically
    split = max(_MIN_TRAIN // 2, int(len(Xn) * 0.8))
    X_train, X_test = Xn[:split], Xn[split:]
    y_train, y_test = y_arr[:split], y_arr[split:]
    if len(X_test) < 5 or len(set(y_train.tolist())) < 2:
        X_train, y_train = Xn, y_arr
        X_test, y_test = Xn, y_arr

    clf = LogisticRegression(max_iter=500, C=0.5, solver="lbfgs")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "holdout_n": int(len(y_test)),
        "holdout_brier": float(brier_score_loss(y_test, proba)),
        "holdout_log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
        "holdout_acc": float(((proba >= 0.5).astype(int) == y_test).mean()),
        "train_n": int(len(y_train)),
        "base_rate_up": float(y_arr.mean()),
    }

    # Optional CV for reporting
    try:
        tscv = TimeSeriesSplit(n_splits=min(4, max(2, len(Xn) // 30)))
        cv_briers = []
        for tr, te in tscv.split(Xn):
            if len(set(y_arr[tr].tolist())) < 2 or len(te) < 3:
                continue
            c = LogisticRegression(max_iter=500, C=0.5, solver="lbfgs")
            c.fit(Xn[tr], y_arr[tr])
            p = c.predict_proba(Xn[te])[:, 1]
            cv_briers.append(float(brier_score_loss(y_arr[te], p)))
        if cv_briers:
            metrics["cv_brier_mean"] = float(sum(cv_briers) / len(cv_briers))
    except Exception:
        pass

    coefs = {FEATURE_KEYS[i]: float(clf.coef_[0][i]) for i in range(len(FEATURE_KEYS))}
    model = {
        "type": "logistic_residual",
        "version": "v1",
        "trained_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "n_samples": len(usable),
        "means": {FEATURE_KEYS[i]: float(means[i]) for i in range(len(FEATURE_KEYS))},
        "scales": {FEATURE_KEYS[i]: float(scales[i]) for i in range(len(FEATURE_KEYS))},
        "coefs": coefs,
        "intercept": float(clf.intercept_[0]),
        "anchor_blend": 0.65,
        "max_residual": 0.15,
        "metrics": metrics,
    }
    save_model(model)
    result.update({"trained": True, "metrics": metrics, "top_coefs": sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:8]})
    return result


def model_path() -> Path:
    return _MODEL_PATH
