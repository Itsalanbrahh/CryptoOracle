"""StockOracle — reflective master agent for equity long/short trading."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Optional

import anthropic

from crypto_oracle.agents.stock_macro import StockMacroAgent
from crypto_oracle.agents.stock_sentiment import StockSentimentAgent
from crypto_oracle.agents.stock_technical import StockTechnicalAgent
from crypto_oracle.models.signals import AgentSignal, MasterRecommendation
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

_SYNTH_SYSTEM = """You are the StockOracle master analyst on a PAPER TRADING account.
ZERO real-money risk. Maximise returns aggressively — both LONG and SHORT.

TRADE SEMANTICS:
- BUY  = go LONG  (buy shares, profit when price rises)
- SELL = go SHORT (short sell, profit when price FALLS)
- HOLD = stay flat

DECISION RULES:
- 2+ agents agreeing >55% conf → act on it
- BEARISH consensus → SELL (short), NOT just HOLD
- RSI >70 with bearish divergence → strong short signal
- RSI <30 with bullish divergence → strong long signal
- On winning streak (>60% hit rate): increase size, lower threshold
- On losing streak: tighten threshold, cut size

POSITION SIZING:
- Long (BUY): default 15-20% of portfolio
- Short (SELL): default 10-15% (asymmetric risk)
- Increase by $50 per 3-trade winning streak, decrease by $50 per losing streak

AGENT WEIGHTS:
- Boost correct agents up to 1.8x, cut wrong agents to 0.4x

Respond in EXACT format (no markdown, no extra text):
ACTION: BUY|SELL|HOLD
CONFIDENCE: 0.XX
REASONING: 2-3 sentences referencing specific agent data
CATALYSTS: item1 | item2 | item3
RISKS: item1 | item2
POSITION_SIZE: e.g. "15% of portfolio"
AGENT_WEIGHTS: {"Technical":1.0,"Macro":1.0,"Sentiment":1.0}
STRATEGY_NOTES: 2-3 sentences on what is working
CONFIDENCE_THRESHOLD: 0.XX
AUTO_TRADE_AMOUNT: XXX
"""


class StockOracle:
    """Orchestrates 3 specialist agents for equity long/short paper trading."""

    _DEFAULT_WEIGHTS: dict[str, float] = {
        "Technical": 1.0, "Macro": 1.0, "Sentiment": 1.0,
    }

    def __init__(self) -> None:
        self.agents = [StockTechnicalAgent(), StockMacroAgent(), StockSentimentAgent()]
        self.client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        logger.info("StockOracle initialised with %d agents", len(self.agents))

    async def run(self, symbol: str) -> MasterRecommendation:
        symbol = symbol.upper()
        logger.info("StockOracle run started for %s", symbol)

        price_at_time = await self._get_price(symbol)

        signals: list[AgentSignal] = await asyncio.gather(
            *[agent.run(symbol) for agent in self.agents], return_exceptions=False
        )
        valid_signals = [s for s in signals if isinstance(s, AgentSignal)]
        logger.info("Collected %d/%d signals for stock %s", len(valid_signals), len(self.agents), symbol)

        from crypto_oracle.models.db import (
            get_agent_accuracy,
            get_recommendation_outcomes,
            get_strategy_state,
            save_strategy_state,
        )
        strategy = await get_strategy_state(symbol)
        outcomes = await get_recommendation_outcomes(symbol, limit=10)
        agent_perf = await get_agent_accuracy(symbol, limit=20)

        weighted_signals = self._apply_weights(valid_signals, strategy["agent_weights"])
        rec, strategy_update = await self._synthesise(
            symbol, weighted_signals, strategy, outcomes, agent_perf
        )
        rec.agent_signals = valid_signals
        rec.price_at_time = price_at_time

        new_weights = strategy_update.get("agent_weights", strategy["agent_weights"])
        new_notes = strategy_update.get("strategy_notes", strategy["strategy_notes"])
        new_threshold = strategy_update.get("confidence_threshold", strategy["confidence_threshold"])
        new_amount = strategy_update.get("auto_trade_amount", strategy["auto_trade_amount"])
        await save_strategy_state(symbol, new_weights, new_notes, new_threshold, new_amount)

        logger.info(
            "StockOracle → %s %s %.0f%%", symbol, rec.action, rec.confidence * 100
        )
        return rec

    @staticmethod
    async def _get_price(symbol: str) -> Optional[float]:
        if os.getenv("SKIP_ALPACA", "false").lower() == "true":
            return None
        try:
            from crypto_oracle.alpaca.client import get_stock_price
            return await get_stock_price(symbol)
        except Exception:
            return None

    @staticmethod
    def _apply_weights(signals: list[AgentSignal], weights: dict[str, float]) -> list[AgentSignal]:
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
    ) -> tuple[MasterRecommendation, dict]:
        signals_json = json.dumps(
            [{"agent": s.agent_name, "signal": s.signal,
              "confidence": s.confidence, "summary": s.summary}
             for s in signals],
            indent=2,
        )

        if outcomes:
            lines = [
                f"{o.get('timestamp','')[:10]} {o['action']:4s} conf={o['confidence']:.0%} "
                f"entry=${o['entry_price']:,.2f} exit=${o['exit_price']:,.2f} "
                f"chg={o['change_pct']:+.2f}% → {o['outcome'].upper()}"
                for o in outcomes[:8]
            ]
            correct = sum(1 for o in outcomes if o["outcome"] == "correct")
            lines.append(f"\nOverall: {correct}/{len(outcomes)} correct ({correct/len(outcomes)*100:.0f}%)")
            outcomes_summary = "\n".join(lines)
        else:
            outcomes_summary = "No evaluated outcomes yet — first run."

        if agent_perf:
            perf_lines = []
            for agent, s in sorted(agent_perf.items(), key=lambda x: -x[1].get("accuracy_pct", 50)):
                bar = "█" * int(s["accuracy_pct"] / 10) + "░" * (10 - int(s["accuracy_pct"] / 10))
                perf_lines.append(f"  {agent:12s} {bar} {s['accuracy_pct']:.0f}% ({s['correct']}/{s['total']})")
            perf_summary = "\n".join(perf_lines)
        else:
            perf_summary = "No agent accuracy history yet."

        user_msg = (
            f"Stock Symbol: {symbol}\n\n"
            f"=== CURRENT AGENT SIGNALS ===\n{signals_json}\n\n"
            f"=== PAST RECOMMENDATION OUTCOMES ===\n{outcomes_summary}\n\n"
            f"=== PER-AGENT ACCURACY ===\n{perf_summary}\n\n"
            f"=== PREVIOUS STRATEGY NOTES ===\n{strategy['strategy_notes'] or 'No notes yet.'}\n\n"
            f"=== CURRENT SETTINGS ===\n"
            f"Confidence threshold: {strategy['confidence_threshold']}\n"
            f"Auto-trade amount: ${strategy['auto_trade_amount']:.0f}\n\n"
            f"Synthesise a BUY/SELL/HOLD recommendation. SELL = go SHORT."
        )

        msg = await self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=_SYNTH_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = msg.content[0].text
        rec = self._parse_rec(symbol, text, signals)
        strategy_update = self._parse_strategy(text, strategy)
        return rec, strategy_update

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
        lines: dict[str, str] = {}
        for line in text.strip().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                lines[k.strip().upper()] = v.strip()

        update: dict[str, Any] = {}

        raw_w = lines.get("AGENT_WEIGHTS", "")
        try:
            m = re.search(r"\{[^}]+\}", raw_w or text, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                known = {"Technical", "Macro", "Sentiment"}
                weights = {}
                for k, v in parsed.items():
                    if k in known:
                        weights[k] = max(0.3, min(2.0, float(v)))
                if weights:
                    update["agent_weights"] = weights
        except Exception:
            pass

        notes = lines.get("STRATEGY_NOTES", "").strip()
        if notes:
            update["strategy_notes"] = notes[:500]

        try:
            ct = float(lines.get("CONFIDENCE_THRESHOLD", ""))
            update["confidence_threshold"] = max(0.50, min(0.90, ct))
        except (ValueError, KeyError):
            pass

        try:
            amt_str = lines.get("AUTO_TRADE_AMOUNT", "").replace("$", "").replace(",", "").strip()
            update["auto_trade_amount"] = max(25.0, min(20000.0, float(amt_str)))
        except (ValueError, KeyError):
            pass

        return update
