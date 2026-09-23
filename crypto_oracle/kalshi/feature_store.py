"""Append-only feature store for 15m decisions + settlement labels."""
from __future__ import annotations

import asyncio
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


def _apply_label(row: dict[str, Any], *, settled_up: bool, settle_value: float | None, source: str) -> None:
    row["labeled"] = True
    row["settled_up"] = settled_up
    row["label_up"] = 1 if settled_up else 0
    row["label_source"] = source
    row["labeled_at"] = _now()
    if settle_value is not None:
        row["settle_spot"] = float(settle_value)
        row["expiration_value"] = float(settle_value)
    action = row.get("action")
    yes_ask = float(row.get("features", {}).get("yes_ask") or row.get("yes_ask") or 0)
    no_ask = float(row.get("features", {}).get("no_ask") or row.get("no_ask") or 0)
    if action == "BUY_YES" and yes_ask > 0:
        row["realized_pnl_1x"] = (1.0 - yes_ask) if settled_up else (-yes_ask)
    elif action == "BUY_NO" and no_ask > 0:
        row["realized_pnl_1x"] = (1.0 - no_ask) if not settled_up else (-no_ask)
    else:
        row["realized_pnl_1x"] = 0.0


async def label_settled_async(settle_spot: float | None = None, now: datetime | None = None) -> int:
    """
    Label unlabeled rows past close_time.

    Preference order:
      1. Official Kalshi market ``result`` / ``expiration_value`` (BRTI settlement)
      2. Spot proxy fallback (``settle_spot >= strike``) when Kalshi result not ready
    """
    from .settlement_15m import fetch_settlement_label

    now = now or datetime.now(timezone.utc)
    rows = read_all()
    n = 0
    for row in rows:
        if row.get("labeled"):
            continue
        close_time = row.get("close_time")
        strike = row.get("strike")
        ticker = row.get("ticker")
        if not close_time or strike is None or not ticker:
            continue
        try:
            close_dt = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
        except ValueError:
            continue
        # Wait a bit after close for Kalshi to finalize result
        if close_dt > now:
            continue

        label = await fetch_settlement_label(str(ticker), strike=float(strike))
        if label is not None:
            _apply_label(
                row,
                settled_up=label.settled_up,
                settle_value=label.expiration_value,
                source=label.source,
            )
            n += 1
            continue

        # Fallback only if close was >2 minutes ago and still no Kalshi result
        if settle_spot is None:
            continue
        if (now - close_dt).total_seconds() < 120:
            continue
        settled_up = float(settle_spot) >= float(strike)
        _apply_label(
            row,
            settled_up=settled_up,
            settle_value=float(settle_spot),
            source="spot_proxy",
        )
        n += 1

    if n:
        rewrite_all(rows)
    return n


def label_settled(settle_spot: float, now: datetime | None = None) -> int:
    """Sync wrapper for learn job / older callers."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Should not happen from sync context; fall back to proxy-only path
        return _label_proxy_only(settle_spot, now=now)
    return asyncio.run(label_settled_async(settle_spot=settle_spot, now=now))


def _label_proxy_only(settle_spot: float, now: datetime | None = None) -> int:
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
        _apply_label(row, settled_up=settled_up, settle_value=float(settle_spot), source="spot_proxy")
        n += 1
    if n:
        rewrite_all(rows)
    return n


def labeled_training_rows(min_rows: int = 1) -> list[dict[str, Any]]:
    rows = [r for r in read_all() if r.get("labeled") and r.get("features")]
    return rows if len(rows) >= min_rows else rows


def label_counts() -> dict[str, int]:
    rows = read_all()
    labeled = [r for r in rows if r.get("labeled")]
    official = sum(1 for r in labeled if r.get("label_source") == "kalshi_result")
    return {
        "total_rows": len(rows),
        "labeled": len(labeled),
        "kalshi_official": official,
        "spot_proxy": sum(1 for r in labeled if r.get("label_source") == "spot_proxy"),
        "unlabeled": len(rows) - len(labeled),
    }


def store_path() -> Path:
    return _STORE_PATH
