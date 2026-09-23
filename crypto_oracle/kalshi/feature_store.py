"""Append-only feature store for 15m decisions + settlement labels."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STORE_PATH = Path.home() / ".hermes" / "state" / "kalshi_15m_features.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_decision(row: dict[str, Any]) -> None:
    """Append one decision snapshot (features + model outputs + action)."""
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload.setdefault("ts", _now())
    payload.setdefault("labeled", False)
    with _STORE_PATH.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def read_all() -> list[dict[str, Any]]:
    if not _STORE_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in _STORE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def rewrite_all(rows: list[dict[str, Any]]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STORE_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def label_settled(settle_spot: float, now: datetime | None = None) -> int:
    """
    Label unlabeled rows whose market close_time has passed.

    settled_up = settle_spot >= strike (spot proxy for BRTI — documented limitation).
    Returns number of newly labeled rows.
    """
    now = now or datetime.now(timezone.utc)
    rows = read_all()
    n = 0
    for row in rows:
        if row.get("labeled"):
            continue
        close_time = row.get("close_time")
        strike = row.get("strike")
        if not close_time or strike is None:
            continue
        try:
            close_dt = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
        except ValueError:
            continue
        if close_dt > now:
            continue
        settled_up = float(settle_spot) >= float(strike)
        row["labeled"] = True
        row["settle_spot"] = float(settle_spot)
        row["settled_up"] = settled_up
        row["label_up"] = 1 if settled_up else 0
        row["labeled_at"] = _now()
        # Realized EV proxy if we had taken the logged side at ask
        action = row.get("action")
        yes_ask = float(row.get("features", {}).get("yes_ask") or row.get("yes_ask") or 0)
        no_ask = float(row.get("features", {}).get("no_ask") or row.get("no_ask") or 0)
        if action == "BUY_YES" and yes_ask > 0:
            row["realized_pnl_1x"] = (1.0 - yes_ask) if settled_up else (-yes_ask)
        elif action == "BUY_NO" and no_ask > 0:
            row["realized_pnl_1x"] = (1.0 - no_ask) if not settled_up else (-no_ask)
        else:
            row["realized_pnl_1x"] = 0.0
        n += 1
    if n:
        rewrite_all(rows)
    return n


def labeled_training_rows(min_rows: int = 1) -> list[dict[str, Any]]:
    rows = [r for r in read_all() if r.get("labeled") and r.get("features")]
    return rows if len(rows) >= min_rows else rows


def store_path() -> Path:
    return _STORE_PATH
