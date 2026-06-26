"""
Agent performance tracker — learns from resolved trades which agents are reliable.

Each resolved position is compared against the agent signals that were recorded
at entry time. The tracker builds a per-agent track record:
  - Direction accuracy (did the agent's sign match the outcome?)
  - Win rate (what % of trades was this agent on the right side?)
  - Confidence calibration (how confident was it when it was right vs wrong?)

These stats are persisted to `~/.hermes/state/kalshi_agent_stats.json` and
loaded on every trading tick to weight agents dynamically.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_STATS_PATH = Path.home() / ".hermes" / "state" / "kalshi_agent_stats.json"

# ── Weighting function ─────────────────────────────────────────────────────


def _winrate_weight(
    wins: int,
    losses: int,
    right_side: int = 0,
    total_calls: int = 0,
    min_weight: float = 0.3,
    max_weight: float = 1.5,
) -> float:
    """Map direction accuracy to a multiplicative weight for agent scores.

    Uses direction accuracy (right_side_calls / total_calls), NOT trade win rate.
    An agent that's 100% sure but always wrong about direction should be dampened
    regardless of whether the trade happened to win for other reasons.

    accuracy > 60% → weight > 1.0 (amplify)
    accuracy ~50% → weight ~1.0 (neutral)
    accuracy < 40% → weight < 1.0 (dampen)

    Falls back to win rate when direction data isn't available (< 3 calls).
    Ceilings at min_weight / max_weight to prevent runaway effects.
    """
    if total_calls >= 3:
        accuracy = right_side / total_calls
    else:
        total = wins + losses
        if total < 3:
            return 1.0  # no adjustment until 3+ trades
        accuracy = wins / total
    # logistic-ish mapping: 0%→0.3, 50%→1.0, 100%→1.5
    raw = 0.3 + (accuracy * 1.2)
    return max(min_weight, min(max_weight, raw))


# ── Data model ─────────────────────────────────────────────────────────────


def _default_stats() -> dict:
    return {
        "version": 2,
        "last_updated": None,
        "agents": {},
    }


def load_stats() -> dict:
    """Load agent stats from disk. Returns a dict of agent_name -> stats."""
    if _STATS_PATH.exists():
        try:
            return json.loads(_STATS_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return _default_stats()


def save_stats(stats: dict) -> None:
    _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    stats["last_updated"] = datetime.now(timezone.utc).isoformat()
    _STATS_PATH.write_text(json.dumps(stats, indent=2))


def _ensure_agent(stats: dict, agent_name: str) -> dict:
    """Return the stats dict for an agent, creating a default entry if missing."""
    if agent_name not in stats["agents"]:
        stats["agents"][agent_name] = {
            "trades_analyzed": 0,
            "wins": 0,
            "losses": 0,
            "right_side_calls": 0,
            "wrong_side_calls": 0,
            "avg_confidence_when_right": 0.0,
            "avg_confidence_when_wrong": 0.0,
            "total_confidence": 0.0,
            "last_resolved_ticker": None,
        }
    return stats["agents"][agent_name]


# ── Recording resolved trades ──────────────────────────────────────────────


def record_resolution(
    pos: dict,
    agent_signals: dict[str, dict] | None = None,
) -> dict:
    """Record a resolved position against each agent's track record.

    Args:
        pos: A position dict from position_manager (with realized_pnl, ticker, side, strike)
        agent_signals: Optional dict of agent_name -> {"score": float, "confidence": float}
                        recorded at entry time. If None, no per-agent analysis is done.

    Returns:
        The updated stats dict.
    """
    stats = load_stats()
    ticker = pos.get("ticker", "?")
    side = pos.get("side", "")
    pnl = pos.get("realized_pnl")
    strike = pos.get("strike", 0)
    spot = pos.get("spot_at_entry", 0)

    # Determine if the trade was a win
    won = pnl is not None and pnl > 0

    if not agent_signals:
        # No agent-level data — just record the aggregate outcome
        save_stats(stats)
        return stats

    # For each agent that had a signal at entry, record
    for agent_name, sig in agent_signals.items():
        entry = _ensure_agent(stats, agent_name)
        entry["trades_analyzed"] += 1

        score = sig.get("score", 0.0)
        confidence = sig.get("confidence", 0.5)
        entry["total_confidence"] = (entry["total_confidence"] * (entry["trades_analyzed"] - 1) + confidence) / entry["trades_analyzed"]

        if won:
            entry["wins"] += 1
        else:
            entry["losses"] += 1

        # Direction accuracy: did the agent's sign match the correct direction?
        # For a NO position: the "right" direction is bearish (score < 0)
        # For a YES position: the "right" direction is bullish (score > 0)
        correct_direction = (side == "no" and score < 0) or (side == "yes" and score > 0)
        if correct_direction:
            entry["right_side_calls"] += 1
            entry["avg_confidence_when_right"] = (
                (entry["avg_confidence_when_right"] * (entry["right_side_calls"] - 1) + confidence)
                / entry["right_side_calls"]
            )
        else:
            entry["wrong_side_calls"] += 1
            entry["avg_confidence_when_wrong"] = (
                (entry["avg_confidence_when_wrong"] * (entry["wrong_side_calls"] - 1) + confidence)
                / entry["wrong_side_calls"]
            )

        entry["last_resolved_ticker"] = ticker

    save_stats(stats)
    return stats


def resolve_from_postmortem() -> int:
    """Scan resolved positions and match them to postmortem entries by ticker.

    Postmortem entries are written with ``resolved=False`` at trade creation time.
    This function matches them against ``position_manager`` closed positions
    to find resolutions, so per-agent stats can be computed from the agent
    signals that were recorded at entry.

    Uses a persistent ``recorded_tickers`` set in the stats file to avoid
    re-processing the same position on subsequent calls.

    Returns the number of entries processed.
    """
    from . import position_manager as pm

    all_positions = pm._load_all()
    closed_positions = [p for p in all_positions if p.get("closed")]

    # Load persistent recorded-tickers set to avoid re-processing
    stats = load_stats()
    recorded_tickers: set[str] = set(stats.get("_recorded_ticker_set", []))

    from .postmortem import read_recent

    entries = read_recent(n=1000)
    ticker_to_entry: dict[str, dict] = {}
    for e in entries:
        t = e.get("ticker", "")
        if t and t not in ticker_to_entry:
            ticker_to_entry[t] = e  # first entry wins (earliest)

    count = 0
    for pos in closed_positions:
        ticker = pos.get("ticker", "")
        if not ticker or ticker in recorded_tickers:
            continue
        entry = ticker_to_entry.get(ticker)
        if entry is None:
            continue
        agent_signals = entry.get("agent_signals")
        record_resolution(pos, agent_signals)
        recorded_tickers.add(ticker)
        count += 1

    # Reload stats to get what record_resolution saved, then update recorded_tickers
    stats = load_stats()
    stats["_recorded_ticker_set"] = sorted(recorded_tickers)
    save_stats(stats)

    return count


# ── Computing weights ───────────────────────────────────────────────────────


def get_agent_weights(
    raw_aggregate: float = 0.0,
    agent_signals: dict[str, dict] | None = None,
) -> dict[str, float]:
    """Return per-agent score multipliers based on their track records.

    Args:
        raw_aggregate: The current aggregate signal (used for overall tilt)
        agent_signals: Current agent signals dict from _run_agents

    Returns:
        dict of agent_name -> weight (e.g. KnowledgeMarket -> 1.2)
        Includes a special key "divergence_cut" when agents strongly disagree.
    """
    stats = load_stats()
    result: dict[str, float] = {}

    # Per-agent weights based on direction accuracy
    for agent_name, s in stats.get("agents", {}).items():
        wins = s.get("wins", 0)
        losses = s.get("losses", 0)
        right_side = s.get("right_side_calls", 0)
        total_calls = s.get("trades_analyzed", 0)
        result[agent_name] = _winrate_weight(wins, losses, right_side, total_calls)

    # Divergence detection
    if agent_signals:
        technical = agent_signals.get("TechnicalMarket", {}).get("score", 0)
        knowledge = agent_signals.get("KnowledgeMarket", {}).get("score", 0)
        spread = abs(technical - knowledge)
        if spread > 0.5:
            # Strong divergence — cut position sizes
            cut = 1.0 - min(0.5, (spread - 0.5))
            result["divergence_cut"] = max(0.5, cut)
        else:
            result["divergence_cut"] = 1.0
    else:
        result["divergence_cut"] = 1.0

    return result


# ── Summary ─────────────────────────────────────────────────────────────────


def summary_text() -> str:
    """Return a human-readable summary of agent stats."""
    stats = load_stats()
    agents = stats.get("agents", {})
    if not agents:
        return "No agent tracking data yet — waiting for resolved trades."

    lines = ["📊 **Agent Performance Tracker**"]
    for name, s in sorted(agents.items()):
        total = s.get("wins", 0) + s.get("losses", 0)
        wr = s["wins"] / total * 100 if total > 0 else 0
        weight = _winrate_weight(s["wins"], s["losses"])
        lines.append(
            f"  • **{name}**: {s['wins']}W/{s['losses']}L ({wr:.0f}%) "
            f"→ weight {weight:.2f}x "
            f"| right-calls {s.get('right_side_calls', 0)}/{total}"
        )
    return "\n".join(lines)
