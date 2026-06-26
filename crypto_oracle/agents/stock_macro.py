"""StockMacro — Macro-economic context agent adapted for equity analysis."""

from __future__ import annotations

from crypto_oracle.agents.macro import MacroAgent

_STOCK_SYSTEM = """You are Macro, a macro-economic analysis agent for equity trading.
You receive macro indicators (Fed language, interest rates, DXY, risk-on/off) and
recent market news. Assess how the macro environment affects equities broadly.

BEARISH macro = consider shorting high-beta stocks.

Respond ONLY in this exact format:
SIGNAL: BULLISH|BEARISH|NEUTRAL
CONFIDENCE: 0.XX
SUMMARY: one or two sentences
DATA_POINTS: point1 | point2 | point3"""


class StockMacroAgent(MacroAgent):
    """MacroAgent with equity-focused system prompt."""

    async def analyze(self, symbol: str, data: dict) -> object:
        import json
        prompt = (
            f"Stock symbol context: {symbol}\n\n"
            f"Fear & Greed Index: {json.dumps(data.get('fear_greed', {}), indent=2)}\n\n"
            f"Recent Headlines:\n"
            + "\n".join(f"- {h}" for h in data.get("headlines", []))
            + "\n\nAnalyse macro environment for equity trading."
        )
        text = await self._call_claude(_STOCK_SYSTEM, prompt)
        signal, confidence, summary, data_points = self._parse_signal_from_text(text)
        from crypto_oracle.models.signals import AgentSignal
        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            summary=summary,
            data_points=data_points,
        )
