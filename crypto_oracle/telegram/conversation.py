"""Claude-powered conversational agent for the Telegram bot."""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from crypto_oracle.models.db import (
    append_conversation,
    get_all_latest,
    get_conversation_history,
    get_trade_stats,
    get_watchlist,
    save_recommendation,
)
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You are CryptoOracle Assistant — an autonomous crypto trading intelligence agent.

You have access to live oracle recommendations from 7 specialist agents, live portfolio data,
and trade history. You can also trigger fresh oracle analysis using the run_oracle tool.

The user is Alan — a Sr. Director of Analytics with a data-driven trading style, primarily
trading Bitcoin and ETH on paper. Be direct, concise, and data-driven. Reference actual
signal data and P&L numbers in every answer. Never guarantee outcomes. Note key risks.

Use the run_oracle tool when Alan asks for fresh analysis, current signals, what to do now,
or mentions running/checking/analyzing a symbol.

Current oracle data:
{oracle_data}

Portfolio summary:
{portfolio_data}

Trade performance:
{trade_stats}

Auto-trade settings:
{auto_trade}"""

_TOOLS = [
    {
        "name": "run_oracle",
        "description": (
            "Trigger a fresh multi-agent oracle analysis for one or more crypto symbols. "
            "Use this when the user asks to run analysis, check current signals, get a fresh "
            "recommendation, or asks what they should do right now."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Symbols to analyze, e.g. ['BTC'] or ['BTC','ETH']",
                }
            },
            "required": ["symbols"],
        },
    }
]


async def _get_portfolio_context() -> str:
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        return "Alpaca integration disabled."
    try:
        from crypto_oracle.alpaca.client import get_account_summary, get_crypto_positions
        summary = await get_account_summary()
        positions = await get_crypto_positions()
        return json.dumps({"account": summary, "positions": positions}, indent=2)
    except Exception as exc:
        return f"Portfolio unavailable: {exc}"


async def _run_oracle_for_symbols(symbols: list[str]) -> dict[str, Any]:
    from crypto_oracle.orchestrator import CryptoOracle
    from crypto_oracle.api.websocket import manager
    from crypto_oracle.autotrader import maybe_auto_trade

    oracle = CryptoOracle()
    results: dict[str, Any] = {}
    for symbol in symbols:
        try:
            rec = await oracle.run(symbol)
            rec_id = await save_recommendation(rec)
            rec.id = rec_id
            await manager.broadcast({"type": "recommendation", "data": rec.model_dump(mode="json")})
            trade = await maybe_auto_trade(rec)
            results[symbol] = {
                "action": rec.action,
                "confidence": round(rec.confidence, 3),
                "reasoning": rec.reasoning,
                "key_catalysts": rec.key_catalysts[:3],
                "key_risks": rec.key_risks[:2],
                "suggested_position_size": rec.suggested_position_size,
                "auto_trade_executed": trade,
            }
        except Exception as exc:
            logger.error("Oracle run failed for %s in conversation: %s", symbol, exc)
            results[symbol] = {"error": str(exc)}
    return results


async def handle_free_text(chat_id: str, user_message: str) -> str:
    """Route a free-text message through the Claude conversational agent with tool use."""
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    recs = await get_all_latest()
    oracle_data = (
        json.dumps([r.model_dump(mode="json") for r in recs], indent=2)
        if recs else "No recommendations yet — use run_oracle tool to generate one."
    )
    portfolio_data = await _get_portfolio_context()
    trade_stats = json.dumps(await get_trade_stats(), indent=2)

    from crypto_oracle.autotrader import get_auto_trade_settings
    auto_trade = json.dumps(await get_auto_trade_settings(), indent=2)

    system_prompt = _SYSTEM_TEMPLATE.format(
        oracle_data=oracle_data,
        portfolio_data=portfolio_data,
        trade_stats=trade_stats,
        auto_trade=auto_trade,
    )

    history = await get_conversation_history(chat_id, limit=20)
    messages = history + [{"role": "user", "content": user_message}]

    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=system_prompt,
        tools=_TOOLS,
        messages=messages,
    )

    # Handle tool use (oracle run request)
    if response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "run_oracle":
                symbols = block.input.get("symbols") or await get_watchlist()
                logger.info("Conversational agent triggering oracle run for %s", symbols)
                run_result = await _run_oracle_for_symbols(symbols)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(run_result),
                })

        if tool_results:
            messages_cont = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": tool_results},
            ]
            response = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                system=system_prompt,
                tools=_TOOLS,
                messages=messages_cont,
            )

    reply = next(
        (block.text for block in response.content if hasattr(block, "text")),
        "I couldn't generate a response. Try /status.",
    )

    await append_conversation(chat_id, "user", user_message)
    await append_conversation(chat_id, "assistant", reply)
    return reply
