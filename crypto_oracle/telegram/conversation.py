"""Claude-powered conversational agent for the Telegram bot."""

from __future__ import annotations

import json
import os

import anthropic

from crypto_oracle.models.db import (
    append_conversation,
    get_all_latest,
    get_conversation_history,
)
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """You are CryptoOracle Assistant, a quant crypto trading intelligence agent.

You have access to the latest oracle recommendations, agent signals, and portfolio data.
The user is Alan, a Sr. Director of Analytics with a thesis-driven, catalyst-focused
trading style. He primarily trades Bitcoin on Robinhood but is open to other suggestions.

Be direct, data-driven, and concise. Reference the actual signal data in every answer.
Never give financial advice as a guarantee. Always note risks.

Current oracle data:
{oracle_data}

Current portfolio:
{portfolio_data}"""


async def _get_portfolio_context() -> str:
    skip = os.getenv("SKIP_ROBINHOOD", "false").lower() == "true"
    if skip:
        return "Robinhood integration disabled."
    try:
        from crypto_oracle.robinhood.client import get_crypto_positions
        positions = await get_crypto_positions()
        return json.dumps(positions, indent=2) if positions else "No crypto positions."
    except Exception as exc:
        return f"Portfolio unavailable: {exc}"


async def handle_free_text(chat_id: str, user_message: str) -> str:
    """Route a free-text message through the Claude conversational agent."""
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Build context
    recs = await get_all_latest()
    oracle_data = (
        json.dumps([r.model_dump(mode="json") for r in recs], indent=2)
        if recs
        else "No recommendations available yet."
    )
    portfolio_data = await _get_portfolio_context()

    system_prompt = _SYSTEM_TEMPLATE.format(
        oracle_data=oracle_data,
        portfolio_data=portfolio_data,
    )

    # Fetch conversation history (last 10 turns)
    history = await get_conversation_history(chat_id, limit=20)

    # Append current user message
    messages = history + [{"role": "user", "content": user_message}]

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system_prompt,
            messages=messages,
        )
        reply = response.content[0].text
    except Exception as exc:
        logger.error("Conversational agent failed: %s", exc, exc_info=True)
        reply = "I'm having trouble connecting to my analysis engine right now. Please try again in a moment."

    # Persist conversation
    await append_conversation(chat_id, "user", user_message)
    await append_conversation(chat_id, "assistant", reply)

    return reply
