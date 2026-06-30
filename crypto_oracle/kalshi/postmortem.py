"""
Per-trade postmortem logger for Kalshi BTC bot.

Each decision — whether a trade was executed, skipped, or blocked by a gate —
is written as a JSONL entry to `~/.hermes/state/kalshi_postmortem.jsonl`.

Fields include agent signals, market conditions, decision rationale,
execution outcome, and (on later ticks) how the trade eventually resolved.

Entries are immutable once written; resolution is appended to the same
entry on a later tick when Kalshi settles the market.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOG_PATH = Path.home() / ".hermes" / "state" / "kalshi_postmortem.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _agent_signal_key(agent_name: str) -> str:
    """Normalise agent names to snake_case keys."""
    return agent_name[0].lower() + "".join(
        f"_{c.lower()}" if c.isupper() else c for c in agent_name[1:]
    )


def build_entry(
    *,
    # Trade metadata
    ticker: str,
    strike: float,
    is_range: bool,
    side: str | None,          # "yes", "no", or None for HOLD
    action: str,               # "BUY_YES", "BUY_NO", "HOLD"
    count: int,
    entry_price: float | None,  # executed price in dollars, or None
    position_usd: float,
    profit_if_win: float | None,
    order_id: str | None,
    # Agent signals (raw before aggregation)
    agent_signals: dict[str, dict],  # {agent_name: {"score": float, "confidence": float}}
    aggregate: float,
    confidence: float,
    edge: float,
    gbm_baseline: float | None,
    belief_yes: float | None,
    market_yes_price: float,
    # Market conditions
    spot_price: float,
    realized_vol: float,
    funding_rate: float,
    funding_tilt: float,
    hours_to_expiry: float,
    # Gate info
    gate_blocked: str | None,
    daily_deployed_usd: float,
    daily_trades: int,
    daily_cap_usd: float,
    max_trades: int,
    balance_usd: float | None,
    # Execution
    exec_status: str,           # "submitted", "hold", "paper", "error"
    exec_error: str | None,
    reasoning: str,
    # Resolution (filled on later ticks)
    resolved: bool = False,
    resolved_itm: bool | None = None,
    resolved_pnl_usd: float | None = None,
) -> dict:
    """Build a postmortem entry dict. Does NOT write it — call log_entry()."""
    signals_flat: dict[str, float] = {}
    for agent_name, sig in (agent_signals or {}).items():
        key = _agent_signal_key(agent_name)
        signals_flat[f"{key}_score"] = sig.get("score")
        signals_flat[f"{key}_confidence"] = sig.get("confidence")

    return {
        "ts": _now(),
        "ticker": ticker,
        "strike": strike,
        "is_range": is_range,
        "side": side,
        "action": action,
        "count": count,
        "entry_price": entry_price,
        "position_usd": round(position_usd, 2),
        "profit_if_win": round(profit_if_win, 2) if profit_if_win is not None else None,
        # Realized payoff ratio = win/risk on the contract. For a binary paying
        # $1, this is (1 - price)/price regardless of side: NO@0.30 → 2.33:1,
        # NO@0.88 → 0.14:1. Lets the calibration report rank NO price bands by
        # which actually compounded the account.
        "payoff_ratio": round((1.0 - entry_price) / entry_price, 3)
        if entry_price not in (None, 0) else None,
        "order_id": order_id,
        # Agent signals
        "agent_signals": agent_signals,
        "aggregate": round(aggregate, 4),
        "confidence": round(confidence, 4),
        "edge": round(edge, 4),
        "gbm_baseline": round(gbm_baseline, 4) if gbm_baseline is not None else None,
        "belief_yes": round(belief_yes, 4) if belief_yes is not None else None,
        "market_yes_price": round(market_yes_price, 4),
        # Market conditions
        "spot_price": round(spot_price, 2),
        "realized_vol_annual": round(realized_vol, 4),
        "funding_rate_8h": round(funding_rate, 6),
        "funding_tilt": round(funding_tilt, 4),
        "hours_to_expiry": round(hours_to_expiry, 2),
        # Gates
        "gate_blocked": gate_blocked,
        "daily_deployed_usd": round(daily_deployed_usd, 2),
        "daily_trades": daily_trades,
        "daily_cap_usd": daily_cap_usd,
        "max_trades_per_day": max_trades,
        "balance_usd": round(balance_usd, 2) if balance_usd is not None else None,
        # Execution
        "exec_status": exec_status,
        "exec_error": exec_error,
        "reasoning": reasoning,
        # Resolution
        "resolved": resolved,
        "resolved_itm": resolved_itm,
        "resolved_pnl_usd": round(resolved_pnl_usd, 2) if resolved_pnl_usd is not None else None,
    }


def log_entry(entry: dict) -> None:
    """Append a postmortem entry to the JSONL log. Creates dir if needed."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_recent(n: int = 50) -> list[dict]:
    """Return the last N entries from the postmortem log, newest-first."""
    if not _LOG_PATH.exists():
        return []
    with _LOG_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    entries = []
    for line in reversed(lines):
        try:
            entries.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
        if len(entries) >= n:
            break
    return entries


def log_close_event(
    ticker: str,
    count: int,
    side: str,
    close_price: float,
    reason: str,
    entry_price: float | None = None,
    realized_pnl: float | None = None,
) -> None:
    """Log a heartbeat-initiated close event to the postmortem.

    Lightweight entry — no agent signals or market conditions (the heartbeat
    doesn't have access to those). Used by the dashboard generator to show
    real outcomes even for positions closed by the cron heartbeat.
    """
    # Parse strike from ticker: KXBTCD-26JUN2517-T59249.99 → 59249.99
    try:
        strike = float(ticker.split("-")[-1][1:])
    except (ValueError, IndexError):
        strike = 0.0

    entry = {
        "ts": _now(),
        "ticker": ticker,
        "strike": strike,
        "side": side,
        "action": "CLOSE",
        "count": count,
        "close_price": round(close_price, 4),
        "entry_price": round(entry_price, 4) if entry_price is not None else None,
        "realized_pnl": round(realized_pnl, 2) if realized_pnl is not None else None,
        "reason": reason,
    }
    log_entry(entry)
