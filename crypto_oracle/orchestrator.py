"""CryptoOracle Orchestrator — reflective, self-improving master agent."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import anthropic

from crypto_oracle.agents.kronos import KronosAgent
from crypto_oracle.agents.macro import MacroAgent
from crypto_oracle.agents.micro import MicroAgent
from crypto_oracle.agents.volume import VolumeAgent
from crypto_oracle.agents.onchain import OnChainAgent
from crypto_oracle.agents.sentiment import SentimentAgent
from crypto_oracle.agents.technical import TechnicalAgent
from crypto_oracle.models.signals import AgentSignal, MasterRecommendation
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Synthesis prompt — reflective, self-improving
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = """You are the CryptoOracle master analyst running on a PAPER TRADING account.
There is ZERO real-money risk. This is purely a data-collection and learning exercise.

PRIME DIRECTIVE: Generate trades. HOLD is almost always wrong here.
Every trade — win or lose — gives us performance data we need to improve.
Sitting on cash generates nothing. Default to BUY or SELL; use HOLD only when
the range trap gate explicitly requires it.

You synthesise 7 specialist sub-agents AND reflect on your own track record to improve continuously.
You also have the ability to rewrite any agent's system prompt on the fly when you spot a pattern.

CONTEXT PROVIDED:
1. Agent signals (confidence pre-weighted by recent per-agent accuracy)
2. Past recommendation outcomes — was each BUY/SELL/HOLD correct?
3. Per-agent accuracy scores — who has been most reliable recently
4. Your previous strategy notes — carry forward and refine
5. Any agent prompts you have previously updated (so you can iterate further)

DATA QUALITY GATE:
- If Micro AND OnChain are BOTH offline (conf < 0.10) AND all remaining agents
  have conf < 0.45: only then default to HOLD. If any of the other 5 agents has
  a directional signal above 0.45, trade on it.
- Agents with "data_feed_failure" or "data_anomaly" in DATA_POINTS are absent —
  exclude them from aggregation but do not block trading on the others.

MANDATORY RANGE CONSOLIDATION GATE (only active gate that fully blocks trades):
- If a RANGE TRAP WARNING is present: do NOT SELL. BUY is still allowed if
  bullish signals exist; HOLD only if no directional lean at all.

DECISION RULES:
- 2+ agents agreeing at >45% conf → sufficient to act, lean BUY/SELL
- 1 high-accuracy agent (>70% historical accuracy) at >55% conf → act on it alone
- Trust agents with >60% accuracy 2x more; discount agents below 40%
- Position size 10-20% on moderate signals, 20-30% on strong (3+ agents aligned)
- On any winning streak: lower threshold further, increase size
- On losing streak: adjust weights and try contrarian — do NOT go idle
- SELL into weakness fast; crypto drops fast and we want the data on SELL timing

STRATEGY UPDATE RULES:
- Boost weight of agents correct 2+ moves in a row (up to 1.8x)
- Cut weight of agents wrong 2+ in a row (down to 0.4x)
- Keep confidence_threshold between 0.48–0.65; never raise above 0.65 (we need trades)
- Adjust auto_trade_amount by $50 per 3-trade streak (win or lose)

TOOLS — call all needed tools in a single response (you will not get a follow-up turn):
1. make_trading_decision — REQUIRED every run.
2. update_agent_config — OPTIONAL. Update an agent's system prompt AND/OR its forecast parameters.
   Prompt: fix systematic misreads, wrong thresholds, ignored data fields. Require 2+ runs of evidence.
   Config (Kronos): pred_len (days), sample_count (paths), temperature (1.0=diverse, 0.5=conservative),
   top_p. Increase sample_count when Kronos confidence seems overfit; lower temperature when forecasts
   are too volatile. Adjust pred_len to match the trade horizon you're targeting.
   Always preserve SIGNAL/CONFIDENCE/SUMMARY/DATA_POINTS format in any prompt rewrite.
"""

_TOOLS: list[dict] = [
    {
        "name": "make_trading_decision",
        "description": (
            "Submit the final trading recommendation and full strategy metadata update. "
            "MUST be called exactly once per run."
        ),
        "input_schema": {
            "type": "object",
            "required": [
                "action", "confidence", "reasoning", "catalysts", "risks",
                "position_size", "agent_weights", "strategy_notes",
                "confidence_threshold", "auto_trade_amount",
            ],
            "properties": {
                "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string", "description": "2-3 sentences referencing specific agent data"},
                "catalysts": {"type": "array", "items": {"type": "string"}},
                "risks": {"type": "array", "items": {"type": "string"}},
                "position_size": {"type": "string"},
                "agent_weights": {
                    "type": "object",
                    "properties": {k: {"type": "number", "minimum": 0.3, "maximum": 2.0}
                                   for k in ["Kronos", "Macro", "Micro", "Volume", "OnChain", "Sentiment", "Technical"]},
                },
                "strategy_notes": {"type": "string"},
                "confidence_threshold": {"type": "number", "minimum": 0.50, "maximum": 0.90},
                "auto_trade_amount": {"type": "number", "minimum": 25.0, "maximum": 20000.0},
            },
        },
    },
    {
        "name": "update_agent_config",
        "description": (
            "Update an agent's system prompt and/or forecast parameters. "
            "Call once per agent you want to update. "
            "new_system_prompt: always keep SIGNAL/CONFIDENCE/SUMMARY/DATA_POINTS format. "
            "config: agent-specific parameter overrides (see below). "
            "Kronos config keys: pred_len (int, default 7 — forecast horizon in days), "
            "sample_count (int, default 10 for model / 2000 for GBM — more = better distribution), "
            "temperature (float, default 1.0 — lower = tighter/more conservative forecasts), "
            "top_p (float, default 0.9 — nucleus sampling, lower = more conservative). "
            "Technical config keys: lookback_days (int, default 90), rsi_period (int, default 14), "
            "bollinger_period (int, default 20). "
            "Omit new_system_prompt to update config only; omit config to update prompt only."
        ),
        "input_schema": {
            "type": "object",
            "required": ["agent_name", "reason"],
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": ["Kronos", "Macro", "Micro", "Volume", "OnChain", "Sentiment", "Technical"],
                },
                "new_system_prompt": {
                    "type": "string",
                    "description": "Complete replacement system prompt. Omit to leave unchanged.",
                },
                "config": {
                    "type": "object",
                    "description": "Agent-specific parameter overrides (e.g. Kronos forecast params).",
                },
                "reason": {
                    "type": "string",
                    "description": "Specific evidence from recent runs that motivates this change.",
                },
            },
        },
    },
]


class CryptoOracle:
    """Orchestrates 7 specialist agents and synthesises a self-improving recommendation."""

    _DEFAULT_WEIGHTS: dict[str, float] = {
        "Kronos": 1.0, "Macro": 1.0, "Micro": 1.0, "Volume": 1.0,
        "OnChain": 1.0, "Sentiment": 1.0, "Technical": 1.0,
    }

    def __init__(self) -> None:
        skip_kronos = os.getenv("SKIP_KRONOS", "false").lower() == "true"
        self.agents = [
            KronosAgent() if not skip_kronos else None,
            MacroAgent(), MicroAgent(), VolumeAgent(),
            OnChainAgent(), SentimentAgent(), TechnicalAgent(),
        ]
        self.agents = [a for a in self.agents if a is not None]
        self.client = anthropic.AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_retries=4,
        )
        logger.info("CryptoOracle initialised with %d agents", len(self.agents))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, symbol: str) -> MasterRecommendation:
        symbol = symbol.upper()
        logger.info("Oracle run started for %s", symbol)

        # Capture price before agents run (needed for outcome evaluation later)
        price_at_time = await self._get_price(symbol)

        # Run all agents concurrently
        signals: list[AgentSignal] = await asyncio.gather(
            *[agent.run(symbol) for agent in self.agents], return_exceptions=False
        )
        valid_signals = [s for s in signals if isinstance(s, AgentSignal)]
        logger.info("Collected %d/%d signals for %s", len(valid_signals), len(self.agents), symbol)

        # Load strategy memory
        from crypto_oracle.models.db import (
            get_agent_accuracy,
            get_recommendation_outcomes,
            get_strategy_state,
            save_strategy_state,
        )
        strategy    = await get_strategy_state(symbol)
        outcomes    = await get_recommendation_outcomes(symbol, limit=10)
        agent_perf  = await get_agent_accuracy(symbol, limit=20)

        # Apply learned weights so the synthesis sees performance-adjusted confidence
        weighted_signals = self._apply_weights(valid_signals, strategy["agent_weights"])

        # Synthesise + get strategy updates
        rec, strategy_update = await self._synthesise(
            symbol, weighted_signals, strategy, outcomes, agent_perf, price_at_time
        )
        rec.agent_signals = valid_signals   # store raw (unweighted) signals in the record
        rec.price_at_time = price_at_time

        # Persist updated strategy
        new_weights    = strategy_update.get("agent_weights", strategy["agent_weights"])
        new_notes      = strategy_update.get("strategy_notes", strategy["strategy_notes"])
        new_threshold  = strategy_update.get("confidence_threshold", strategy["confidence_threshold"])
        new_amount     = strategy_update.get("auto_trade_amount", strategy["auto_trade_amount"])
        await save_strategy_state(symbol, new_weights, new_notes, new_threshold, new_amount)

        # Propagate threshold/amount to auto-trader settings if auto-trade is on
        try:
            from crypto_oracle.autotrader import get_auto_trade_settings, update_auto_trade_settings
            at = await get_auto_trade_settings()
            if at["enabled"]:
                await update_auto_trade_settings(True, new_amount, new_threshold)
                logger.info(
                    "Strategy updated for %s: threshold=%.2f amount=$%.0f",
                    symbol, new_threshold, new_amount,
                )
        except Exception as exc:
            logger.warning("Could not propagate strategy update: %s", exc)

        logger.info(
            "Oracle → %s %s %.0f%% | weights=%s",
            symbol, rec.action, rec.confidence * 100,
            {k: round(v, 2) for k, v in new_weights.items()},
        )
        return rec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _get_price(symbol: str) -> Optional[float]:
        if os.getenv("SKIP_ALPACA", "false").lower() == "true":
            return None
        try:
            from crypto_oracle.alpaca.client import get_crypto_price
            return await get_crypto_price(symbol)
        except Exception:
            return None

    @staticmethod
    def _apply_weights(signals: list[AgentSignal], weights: dict[str, float]) -> list[AgentSignal]:
        """Scale each agent's confidence by its learned weight (clamped 0–1)."""
        if not weights:
            return signals
        result = []
        for sig in signals:
            w = weights.get(sig.agent_name, 1.0)
            new_conf = max(0.0, min(1.0, sig.confidence * w))
            result.append(sig.model_copy(update={"confidence": new_conf}))
        return result

    async def _synthesise(
        self,
        symbol: str,
        signals: list[AgentSignal],
        strategy: dict,
        outcomes: list[dict],
        agent_perf: dict,
        price_at_time: Optional[float] = None,
    ) -> tuple[MasterRecommendation, dict]:

        # Build the user message with full context
        signals_json = json.dumps(
            [{"agent": s.agent_name, "signal": s.signal,
              "confidence": s.confidence, "summary": s.summary}
             for s in signals],
            indent=2,
        )

        outcomes_summary = self._format_outcomes(outcomes)
        perf_summary = self._format_perf(agent_perf)

        from crypto_oracle.models.db import get_all_agent_configs, save_agent_config

        range_warning = self._detect_range_trap(outcomes, price_at_time)
        range_section = f"\n=== RANGE TRAP WARNING ===\n{range_warning}\n" if range_warning else ""

        agent_configs = await get_all_agent_configs()
        config_section = ""
        if agent_configs:
            lines = []
            for name, cfg in agent_configs.items():
                prompt_flag = "prompt+config" if cfg.get("system_prompt") and cfg.get("config") else (
                    "prompt" if cfg.get("system_prompt") else "config"
                )
                lines.append(
                    f"  {name} [{prompt_flag}] (updated {cfg.get('updated_at', '')[:10]}): "
                    f"config={cfg.get('config', {})} — {cfg.get('reason', '')[:100]}"
                )
            config_section = "\n=== MASTER-UPDATED AGENT CONFIGS ===\n" + "\n".join(lines) + "\n"

        user_msg = (
            f"Symbol: {symbol}\n\n"
            f"=== CURRENT AGENT SIGNALS (performance-weighted) ===\n{signals_json}\n\n"
            f"=== PAST RECOMMENDATION OUTCOMES (newest first) ===\n{outcomes_summary}\n\n"
            f"=== PER-AGENT ACCURACY (recent {len(outcomes)} evaluated runs) ===\n{perf_summary}\n\n"
            f"=== PREVIOUS STRATEGY NOTES ===\n{strategy['strategy_notes'] or 'No notes yet — first run.'}\n\n"
            f"=== CURRENT SETTINGS ===\n"
            f"Confidence threshold: {strategy['confidence_threshold']}\n"
            f"Auto-trade amount: ${strategy['auto_trade_amount']:.0f}\n"
            f"{range_section}"
            f"{config_section}\n"
            f"Call make_trading_decision and optionally update_agent_config."
        )

        msg = await self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=_SYNTH_SYSTEM,
            tools=_TOOLS,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_msg}],
        )

        decision_input: dict | None = None
        config_updates: list[dict] = []
        for block in msg.content:
            if block.type == "tool_use":
                if block.name == "make_trading_decision":
                    decision_input = block.input
                elif block.name == "update_agent_config":
                    config_updates.append(block.input)

        for upd in config_updates:
            await save_agent_config(
                agent_name=upd["agent_name"],
                system_prompt=upd.get("new_system_prompt", ""),
                config=upd.get("config"),
                reason=upd.get("reason", ""),
                updated_by="master",
            )
            logger.info(
                "Master updated %s — prompt=%s config=%s reason: %s",
                upd["agent_name"],
                bool(upd.get("new_system_prompt")),
                upd.get("config"),
                upd.get("reason", "")[:100],
            )

        if not decision_input:
            logger.warning("Master did not call make_trading_decision — defaulting to HOLD")
            return MasterRecommendation(
                symbol=symbol, action="HOLD", confidence=0.5,
                reasoning="Synthesis failed to produce a decision.", agent_signals=signals,
            ), {}

        action = decision_input.get("action", "HOLD")
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"

        rec = MasterRecommendation(
            symbol=symbol,
            action=action,
            confidence=max(0.0, min(1.0, float(decision_input.get("confidence", 0.5)))),
            reasoning=decision_input.get("reasoning", ""),
            key_catalysts=decision_input.get("catalysts", []),
            key_risks=decision_input.get("risks", []),
            suggested_position_size=decision_input.get("position_size", ""),
            agent_signals=signals,
        )

        strategy_update: dict = {}
        known = {"Kronos", "Macro", "Micro", "Volume", "OnChain", "Sentiment", "Technical"}
        raw_weights = decision_input.get("agent_weights") or {}
        weights = {k: max(0.3, min(2.0, float(v))) for k, v in raw_weights.items() if k in known}
        if weights:
            strategy_update["agent_weights"] = weights
        notes = (decision_input.get("strategy_notes") or "").strip()
        if notes:
            strategy_update["strategy_notes"] = notes[:500]
        ct = decision_input.get("confidence_threshold")
        if ct is not None:
            strategy_update["confidence_threshold"] = max(0.48, min(0.65, float(ct)))
        amt = decision_input.get("auto_trade_amount")
        if amt is not None:
            strategy_update["auto_trade_amount"] = max(25.0, min(20000.0, float(amt)))

        return rec, strategy_update

    @staticmethod
    def _format_outcomes(outcomes: list[dict]) -> str:
        if not outcomes:
            return "No evaluated outcomes yet (need price data from ≥2 runs)."
        lines = []
        for o in outcomes[:8]:
            ts = (o.get("timestamp") or "")[:10]
            lines.append(
                f"{ts} {o['action']:4s} conf={o['confidence']:.0%} "
                f"entry=${o['entry_price']:,.0f} exit=${o['exit_price']:,.0f} "
                f"chg={o['change_pct']:+.2f}% → {o['outcome'].upper()}"
            )
        correct = sum(1 for o in outcomes if o["outcome"] == "correct")
        lines.append(f"\nOverall: {correct}/{len(outcomes)} correct ({correct/len(outcomes)*100:.0f}%)")
        return "\n".join(lines)

    @staticmethod
    def _format_perf(agent_perf: dict) -> str:
        if not agent_perf:
            return "Not enough history to score agents yet."
        lines = []
        for agent, s in sorted(agent_perf.items(), key=lambda x: -x[1].get("accuracy_pct", 50)):
            bar = "█" * int(s["accuracy_pct"] / 10) + "░" * (10 - int(s["accuracy_pct"] / 10))
            lines.append(f"  {agent:12s} {bar} {s['accuracy_pct']:.0f}% ({s['correct']}/{s['total']})")
        return "\n".join(lines)

    # ------------------------------------------------------------------

    @staticmethod
    def _detect_range_trap(outcomes: list[dict], current_price: Optional[float]) -> str:
        """Return a warning string if current price is inside a recent loss cluster."""
        if not current_price or not outcomes:
            return ""
        wrong_sells = [
            o for o in outcomes
            if o.get("outcome") == "incorrect" and o.get("action") == "SELL"
            and o.get("entry_price")
        ]
        if len(wrong_sells) < 2:
            return ""
        prices = [o["entry_price"] for o in wrong_sells]
        p_min, p_max = min(prices), max(prices)
        spread_pct = (p_max - p_min) / p_min * 100 if p_min > 0 else 99.0
        if spread_pct > 2.0:
            return ""
        midpoint = (p_min + p_max) / 2
        dist_pct = abs(current_price - midpoint) / midpoint * 100
        if dist_pct > 1.5:
            return ""
        return (
            f"WARNING: {len(wrong_sells)} recent SELL losses all occurred between "
            f"${p_min:,.0f}–${p_max:,.0f} ({spread_pct:.1f}% spread). "
            f"Current price ${current_price:,.0f} is {dist_pct:.1f}% from the center of this "
            f"consolidation zone. Do NOT SELL here — prior losses confirm this is range chop, "
            f"not a trend break. Output HOLD and wait for a confirmed breakout."
        )

