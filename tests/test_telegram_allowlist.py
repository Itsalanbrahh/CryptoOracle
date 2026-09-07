"""
Test that Telegram mutating command handlers enforce TELEGRAM_ALLOWED_CHAT_IDS.

Behaviour:
- TELEGRAM_ALLOWED_CHAT_IDS unset/blank → mutating commands fail closed (rejected for all).
- chat_id in allowlist → mutating command proceeds.
- chat_id NOT in allowlist → mutating command rejected.
- Read-only commands remain accessible regardless of allowlist state.
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_update(chat_id: int = 12345, text: str = "/buy BTC 100") -> MagicMock:
    """Build a minimal fake telegram Update object."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    update.message.text = text
    return update


def _make_context(args: list[str] | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.args = args or []
    return ctx


async def _call_mutating(chat_id: int, env: dict) -> AsyncMock:
    """Invoke cmd_buy (a mutating command) for the given chat_id under env."""
    for mod in list(sys.modules.keys()):
        if "crypto_oracle.telegram.bot" in mod:
            del sys.modules[mod]

    with patch.dict(os.environ, env, clear=True):
        from crypto_oracle.telegram.bot import cmd_buy
        update = _make_update(chat_id=chat_id)
        ctx = _make_context(args=["BTC", "100"])
        await cmd_buy(update, ctx)
        return update.message.reply_text


async def _call_readonly(chat_id: int, env: dict) -> AsyncMock:
    """Invoke cmd_status (read-only) for the given chat_id under env."""
    for mod in list(sys.modules.keys()):
        if "crypto_oracle.telegram.bot" in mod:
            del sys.modules[mod]

    with patch.dict(os.environ, env, clear=True):
        from crypto_oracle.telegram.bot import cmd_status

        async def _fake_get_watchlist():
            return []

        with patch("crypto_oracle.models.db.get_watchlist", _fake_get_watchlist):
            update = _make_update(chat_id=chat_id)
            ctx = _make_context()
            await cmd_status(update, ctx)
            return update.message.reply_text


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTelegramAllowlist:
    """TELEGRAM_ALLOWED_CHAT_IDS enforcement on mutating Telegram commands."""

    def test_mutating_rejected_when_allowlist_unset(self):
        """cmd_buy must reject all callers when TELEGRAM_ALLOWED_CHAT_IDS is unset."""
        env = {"SKIP_ALPACA": "true"}  # no TELEGRAM_ALLOWED_CHAT_IDS
        reply_mock = asyncio.run(_call_mutating(chat_id=999, env=env))
        # reply_text must have been called with a rejection message
        assert reply_mock.called, "Expected a reply (rejection message)"
        first_call_arg = reply_mock.call_args_list[0][0][0]
        assert "not authorized" in first_call_arg.lower() or "unauthori" in first_call_arg.lower(), (
            f"Expected rejection text, got: {first_call_arg!r}"
        )

    def test_mutating_allowed_when_chat_id_in_list(self):
        """cmd_buy should not reply 'not authorized' when chat_id is in the allowlist."""
        env = {
            "TELEGRAM_ALLOWED_CHAT_IDS": "12345,67890",
            "SKIP_ALPACA": "true",
        }
        reply_mock = asyncio.run(_call_mutating(chat_id=12345, env=env))
        # Rejection must NOT appear — the command will proceed (and hit Alpaca-disabled reply)
        if reply_mock.called:
            for call in reply_mock.call_args_list:
                msg = call[0][0] if call[0] else ""
                assert "not authorized" not in msg.lower() and "unauthori" not in msg.lower(), (
                    f"Allowed chat_id should not receive rejection, got: {msg!r}"
                )

    def test_mutating_rejected_when_chat_id_not_in_list(self):
        """cmd_buy must reject a chat_id that is not in TELEGRAM_ALLOWED_CHAT_IDS."""
        env = {
            "TELEGRAM_ALLOWED_CHAT_IDS": "11111,22222",
            "SKIP_ALPACA": "true",
        }
        reply_mock = asyncio.run(_call_mutating(chat_id=99999, env=env))
        assert reply_mock.called, "Expected a reply (rejection message)"
        first_call_arg = reply_mock.call_args_list[0][0][0]
        assert "not authorized" in first_call_arg.lower() or "unauthori" in first_call_arg.lower(), (
            f"Unlisted chat_id should be rejected, got: {first_call_arg!r}"
        )

    def test_readonly_accessible_without_allowlist(self):
        """Read-only commands (cmd_status) must remain accessible even when no allowlist set."""
        env = {"SKIP_ALPACA": "true"}  # no allowlist
        with patch("crypto_oracle.models.db.get_latest_recommendation", new_callable=AsyncMock, return_value=None), \
             patch("crypto_oracle.models.db.get_watchlist", new_callable=AsyncMock, return_value=[]):
            reply_mock = asyncio.run(_call_readonly(chat_id=12345, env=env))
        # cmd_status uses _guard (permissive) — should not reject
        if reply_mock.called:
            for call in reply_mock.call_args_list:
                msg = call[0][0] if call[0] else ""
                assert "not authorized" not in msg.lower() and "unauthori" not in msg.lower(), (
                    f"Read-only cmd should not reject, got: {msg!r}"
                )
