"""Learning job: label settled features, retrain residual, retune gates, calibrate.

Run:
  python -m crypto_oracle.kalshi.learn_15m
  python -m crypto_oracle.kalshi.learn_15m --spot 84500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

_GATES_PATH = Path.home() / ".hermes" / "state" / "kalshi_15m_gates.json"


def _load_gates() -> dict:
    if _GATES_PATH.exists():
        try:
            return json.loads(_GATES_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "min_edge": 0.08,
        "min_confidence": 0.55,
        "min_minutes_left": 0.5,
        "max_minutes_left": 14.5,
        "max_spread": 0.08,
        "updated_at": None,
        "note": "defaults",
    }


def save_gates(gates: dict) -> None:
    _GATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _GATES_PATH.write_text(json.dumps(gates, indent=2))


def load_gates() -> dict:
    return _load_gates()


def _calibration_table(rows: list[dict]) -> list[dict]:
    """Bucket predicted p_up vs realized UP rate."""
    buckets = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]
    out = []
    for lo, hi in buckets:
        subset = []
        for r in rows:
            p = r.get("p_model")
            if p is None:
                p = (r.get("features") or {}).get("jev_p_up")
            if p is None:
                continue
            if lo <= float(p) < hi:
                subset.append(r)
        if not subset:
            continue
        rate = sum(int(r["label_up"]) for r in subset) / len(subset)
        mean_p = sum(float(r.get("p_model") or (r.get("features") or {}).get("jev_p_up") or 0.5) for r in subset) / len(subset)
        out.append({
            "bucket": f"{lo:.1f}-{hi:.1f}",
            "n": len(subset),
            "mean_predicted": round(mean_p, 3),
            "realized_up_rate": round(rate, 3),
            "calibration_error": round(abs(mean_p - rate), 3),
        })
    return out


def _counterfactual_ev(rows: list[dict], min_edge: float) -> dict:
    """EV of taking a side only when |p - ask| >= min_edge, aligned with p."""
    pnl = 0.0
    n = 0
    wins = 0
    for r in rows:
        feats = r.get("features") or {}
        p = float(r.get("p_model") or feats.get("jev_p_up") or feats.get("gbm_p_up") or 0.5)
        yes_ask = float(feats.get("yes_ask") or 0)
        no_ask = float(feats.get("no_ask") or 0)
        label = int(r["label_up"])
        yes_edge = p - yes_ask
        no_edge = (1.0 - p) - no_ask
        if p >= 0.5 and yes_edge >= min_edge and yes_ask > 0:
            n += 1
            trade_pnl = (1.0 - yes_ask) if label == 1 else (-yes_ask)
            pnl += trade_pnl
            wins += int(label == 1)
        elif p <= 0.5 and no_edge >= min_edge and no_ask > 0:
            n += 1
            trade_pnl = (1.0 - no_ask) if label == 0 else (-no_ask)
            pnl += trade_pnl
            wins += int(label == 0)
    return {
        "min_edge": min_edge,
        "trades": n,
        "total_pnl_1x": round(pnl, 3),
        "avg_pnl": round(pnl / n, 4) if n else None,
        "win_rate": round(wins / n, 3) if n else None,
    }


def retune_gates(rows: list[dict]) -> dict:
    """Pick min_edge that maximizes avg paper EV on labeled history."""
    gates = _load_gates()
    if len(rows) < 30:
        gates["note"] = f"insufficient labels ({len(rows)}) — keeping prior gates"
        save_gates(gates)
        return gates

    candidates = [0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    scored = [_counterfactual_ev(rows, e) for e in candidates]
    # Prefer positive EV with at least 10 trades; else best avg among those with trades
    viable = [s for s in scored if (s["trades"] or 0) >= 10 and (s["avg_pnl"] or -1) > 0]
    if not viable:
        viable = [s for s in scored if (s["trades"] or 0) >= 5]
    if not viable:
        gates["note"] = "no viable edge grid — keeping prior"
        gates["grid"] = scored
        save_gates(gates)
        return gates

    best = max(viable, key=lambda s: (s["avg_pnl"] or -999, s["trades"]))
    gates["min_edge"] = best["min_edge"]
    gates["updated_at"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()
    gates["note"] = f"retuned from {len(rows)} labels"
    gates["grid"] = scored
    gates["selected"] = best
    save_gates(gates)
    return gates


async def _spot() -> float:
    from crypto_oracle.polymarket.agents import fetch_spot_price
    return await fetch_spot_price()


async def run_async(spot: float | None = None) -> dict:
    from crypto_oracle.kalshi import feature_store as fs
    from crypto_oracle.kalshi.residual_model import train_from_rows

    if spot is None:
        spot = await _spot()

    newly = await fs.label_settled_async(settle_spot=float(spot))
    rows = fs.labeled_training_rows()
    train = train_from_rows(rows)
    gates = retune_gates(rows)
    calib = _calibration_table(rows)

    # Shadow model comparison on labeled set
    shadows = defaultdict(lambda: {"n": 0, "brier": 0.0, "correct": 0})
    for r in rows:
        feats = r.get("features") or {}
        y = int(r["label_up"])
        for name, p in (
            ("gbm", feats.get("gbm_p_up")),
            ("jev", feats.get("jev_p_up")),
            ("model", r.get("p_model")),
        ):
            if p is None:
                continue
            p = float(p)
            shadows[name]["n"] += 1
            shadows[name]["brier"] += (p - y) ** 2
            shadows[name]["correct"] += int((p >= 0.5) == bool(y))
    shadow_report = {}
    for name, s in shadows.items():
        if s["n"]:
            shadow_report[name] = {
                "n": s["n"],
                "brier": round(s["brier"] / s["n"], 4),
                "acc": round(s["correct"] / s["n"], 3),
            }

    report = {
        "spot_used_for_labels": spot,
        "newly_labeled": newly,
        "labeled_total": len(rows),
        "label_counts": fs.label_counts(),
        "feature_store": str(fs.store_path()),
        "train": train,
        "gates": gates,
        "calibration": calib,
        "shadow_models": shadow_report,
    }
    out_path = Path.home() / ".hermes" / "state" / "kalshi_15m_learn_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(out_path)
    return report


def run(spot: float | None = None) -> dict:
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    if running:
        raise RuntimeError("learn_15m.run() called inside a running event loop; use await run_async()")
    return asyncio.run(run_async(spot=spot))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot", type=float, default=None)
    args = parser.parse_args()
    report = asyncio.run(run_async(spot=args.spot))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
