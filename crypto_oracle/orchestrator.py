"""CryptoOracle Orchestrator — reflective, self-improving master agent."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

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
There is ZERO real-money risk. Your mandate is to maximise returns.

You synthesise 7 specialist sub-agents AND reflect on your own track record to improve continuously.

CONTEXT PROVIDED:
1. Agent signals (confidence pre-weighted by recent per-agent accuracy)
2. Past recommendation outcomes — was each BUY/SELL/HOLD correct?
3. Per-agent accuracy scores — who has been most reliable recently
4. Your previous strategy notes — carry forward and refine

MANDATORY DATA QUALITY GATE (check this first):
- If Micro confidence < 0.10 AND OnChain confidence < 0.10: primary data feeds are offline.
  These are the two highest-accuracy agents. Acting on the remaining noise signals is the
  primary cause of losses. You MUST output HOLD with CONFIDENCE < 0.55. Do not override this.
- Any agent whose DATA_POINTS contains "data_feed_failure" or "data_anomaly" must be treated
  as absent — do not factor that signal into the directional decision.

MANDATORY RANGE CONSOLIDATION GATE (check this second):
- If a RANGE TRAP WARNING is present in the context: price is inside a proven loss zone.
  Do NOT issue a SELL in consolidation. Output HOLD until a confirmed breakout occurs.
  A breakout requires: price close outside the range AND at least one high-accuracy agent
  confirming direction with confidence > 0.65.

DECISION RULES — when data quality is good:
- 4+ agents agreeing at >55% conf → treat as high conviction, lean BUY/SELL
- Trust agents with >60% accuracy 2x more; ignore agents below 40%
- Default position size 15-25% of portfolio on strong signals
- On winning streak (>60% hit rate): increase size and lower threshold
- On losing streak: tighten requirements, do not chase — quality over quantity
- Prefer HOLD when signal quality is poor; one good trade beats three noise-trades
- SELL only when a confirmed directional move is underway, not into range chop

STRATEGY UPDATE RULES:
- Boost weight of agents that called the last 2+ moves correctly (up to 1.8x)
- Cut weight of agents that were wrong 2+ times in a row (down to 0.4x)
- Lower confidence_threshold toward 0.55 when win rate > 65%
- Raise confidence_threshold toward 0.75 when win rate < 40%
- Increase auto_trade_amount by $50 per winning streak of 3, decrease by $50 per losing streak of 3

Respond in this EXACT format (no extra text, no markdown):
ACTION: BUY|SELL|HOLD
CONFIDENCE: 0.XX
REASONING: 2-3 sentences referencing specific agent data and price history
CATALYSTS: item1 | item2 | item3
RISKS: item1 | item2
POSITION_SIZE: e.g. "20% of portfolio" or "full position close"
AGENT_WEIGHTS: {"Kronos":1.0,"Macro":1.0,"Micro":1.0,"Volume":1.0,"OnChain":1.0,"Sentiment":1.0,"Technical":1.0}
STRATEGY_NOTES: 2-3 sentences on what's working, what patterns you see, what you're adjusting
CONFIDENCE_THRESHOLD: 0.XX
AUTO_TRADE_AMOUNT: XXX
"""


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

        range_warning = self._detect_range_trap(outcomes, price_at_time)
        range_section = f"\n=== RANGE TRAP WARNING ===\n{range_warning}\n" if range_warning else ""

        user_msg = (
            f"Symbol: {symbol}\n\n"
            f"=== CURRENT AGENT SIGNALS (performance-weighted) ===\n{signals_json}\n\n"
            f"=== PAST RECOMMENDATION OUTCOMES (newest first) ===\n{outcomes_summary}\n\n"
            f"=== PER-AGENT ACCURACY (recent {len(outcomes)} evaluated runs) ===\n{perf_summary}\n\n"
            f"=== PREVIOUS STRATEGY NOTES ===\n{strategy['strategy_notes'] or 'No notes yet — first run.'}\n\n"
            f"=== CURRENT SETTINGS ===\n"
            f"Confidence threshold: {strategy['confidence_threshold']}\n"
            f"Auto-trade amount: ${strategy['auto_trade_amount']:.0f}\n"
            f"{range_section}\n"
            f"Synthesise a recommendation AND output updated strategy metadata."
        )

        msg = await self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            system=_SYNTH_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = msg.content[0].text
        rec = self._parse_rec(symbol, text, signals)
        strategy_update = self._parse_strategy(text, strategy)
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

    @staticmethod
    def _parse_rec(symbol: str, text: str, signals: list[AgentSignal]) -> MasterRecommendation:
        lines: dict[str, str] = {}
        for line in text.strip().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                lines[k.strip().upper()] = v.strip()

        action = lines.get("ACTION", "HOLD").upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"

        try:
            confidence = max(0.0, min(1.0, float(lines.get("CONFIDENCE", "0.5"))))
        except ValueError:
            confidence = 0.5

        return MasterRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=lines.get("REASONING", "Insufficient data."),
            key_catalysts=[c.strip() for c in lines.get("CATALYSTS", "").split("|") if c.strip()],
            key_risks=[r.strip() for r in lines.get("RISKS", "").split("|") if r.strip()],
            suggested_position_size=lines.get("POSITION_SIZE", ""),
            agent_signals=signals,
        )

    @staticmethod
    def _parse_strategy(text: str, current: dict) -> dict:
        """Extract AGENT_WEIGHTS / STRATEGY_NOTES / CONFIDENCE_THRESHOLD / AUTO_TRADE_AMOUNT."""
        lines: dict[str, str] = {}
        for line in text.strip().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                lines[k.strip().upper()] = v.strip()

        update: dict[str, Any] = {}

        # Agent weights
        raw_w = lines.get("AGENT_WEIGHTS", "")
        try:
            # Handle multi-line JSON by finding the first {...} blob
            import re
            m = re.search(r"\{[^}]+\}", raw_w or text, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                # Validate: all keys must be known agents, values 0.3–2.0
                known = {"Kronos", "Macro", "Micro", "Volume", "OnChain", "Sentiment", "Technical"}
                weights = {}
                for k, v in parsed.items():
                    if k in known:
                        weights[k] = max(0.3, min(2.0, float(v)))
                if weights:
                    update["agent_weights"] = weights
        except Exception:
            pass

        # Strategy notes
        notes = lines.get("STRATEGY_NOTES", "").strip()
        if notes:
            update["strategy_notes"] = notes[:500]

        # Confidence threshold
        try:
            ct = float(lines.get("CONFIDENCE_THRESHOLD", ""))
            update["confidence_threshold"] = max(0.50, min(0.90, ct))
        except (ValueError, KeyError):
            pass

        # Auto-trade amount
        try:
            amt_str = lines.get("AUTO_TRADE_AMOUNT", "").replace("$", "").replace(",", "").strip()
            amt = float(amt_str)
            update["auto_trade_amount"] = max(25.0, min(20000.0, amt))
        except (ValueError, KeyError):
            pass

        return update
