"""Telegram bot: command handlers + conversational agent routing."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from crypto_oracle.models.db import (
    add_to_watchlist,
    get_all_latest,
    get_latest_recommendation,
    get_recommendation_history,
    get_registered_chats,
    get_watchlist,
    register_chat,
    remove_from_watchlist,
    set_alerts,
)
from crypto_oracle.telegram.conversation import handle_free_text
from crypto_oracle.telegram.notifications import build_status_message
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

_app: Optional[Application] = None
_run_cooldowns: dict[str, float] = {}
_RUN_COOLDOWN = 1800  # 30 minutes


def _check_whitelist(chat_id: str) -> bool:
    allowed_raw = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
    if not allowed_raw.strip():
        return True
    allowed = {s.strip() for s in allowed_raw.split(",")}
    return chat_id in allowed


async def _guard(update: Update) -> bool:
    chat_id = str(update.effective_chat.id)
    if not _check_whitelist(chat_id):
        await update.message.reply_text("Unauthorised.")
        return False
    return True


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    chat_id = str(update.effective_chat.id)
    await register_chat(chat_id)
    await update.message.reply_text(
        "👋 *Welcome to CryptoOracle*\n\n"
        "I'm your autonomous crypto trading intelligence system.\n"
        "Running on Alpaca paper trading.\n\n"
        "Commands:\n"
        "/status — latest oracle recommendation\n"
        "/run — trigger fresh oracle analysis\n"
        "/history — last 5 recommendations\n"
        "/portfolio — Alpaca account summary\n"
        "/watchlist — show watchlist\n"
        "/watch ETH — add symbol\n"
        "/unwatch ETH — remove symbol\n"
        "/alerts on|off — toggle proactive alerts\n"
        "/interval 120 — set oracle interval (min)\n"
        "/help — this message\n\n"
        "Or just ask me anything about the market!",
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    args = context.args
    symbols = [args[0].upper()] if args else await get_watchlist()
    for symbol in symbols:
        rec = await get_latest_recommendation(symbol)
        if rec is None:
            await update.message.reply_text(
                f"No data for {symbol} yet. Use /run to trigger analysis.",
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await update.message.reply_text(
                build_status_message(rec), parse_mode=ParseMode.MARKDOWN
            )


async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    chat_id = str(update.effective_chat.id)
    now = time.time()
    remaining = _RUN_COOLDOWN - (now - _run_cooldowns.get(chat_id, 0))
    if remaining > 0:
        await update.message.reply_text(
            f"⏳ Cooldown active. Wait {int(remaining // 60)}m {int(remaining % 60)}s.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    _run_cooldowns[chat_id] = now
    await update.message.reply_text("🔄 Running oracle analysis... ~30 seconds.")

    from crypto_oracle.orchestrator import CryptoOracle
    from crypto_oracle.models.db import save_recommendation
    from crypto_oracle.api.websocket import manager

    oracle = CryptoOracle()
    for symbol in await get_watchlist():
        try:
            rec = await oracle.run(symbol)
            rec_id = await save_recommendation(rec)
            rec.id = rec_id
            await manager.broadcast(
                {"type": "recommendation", "data": rec.model_dump(mode="json")}
            )
            await update.message.reply_text(
                build_status_message(rec), parse_mode=ParseMode.MARKDOWN
            )
        except Exception as exc:
            logger.error("Telegram /run failed for %s: %s", symbol, exc, exc_info=True)
            await update.message.reply_text(f"❌ Oracle run failed for {symbol}: {exc}")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    for symbol in await get_watchlist():
        recs = await get_recommendation_history(symbol, limit=5)
        if not recs:
            await update.message.reply_text(f"No history for {symbol}.")
            continue
        lines = [f"📜 *{symbol} — Last {len(recs)} recommendations*\n"]
        for rec in recs:
            ts = rec.timestamp.strftime("%m-%d %H:%M")
            lines.append(
                f"`{ts}` {rec.action} ({rec.confidence*100:.0f}%) — "
                f"{rec.reasoning[:60]}..."
            )
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        await update.message.reply_text("Alpaca integration is disabled.")
        return
    try:
        from crypto_oracle.alpaca.client import get_account_summary, get_crypto_positions

        summary = await get_account_summary()
        positions = await get_crypto_positions()
        paper_label = " *(Paper)*" if summary.get("paper_trading") else ""

        lines = [
            f"💰 *Alpaca Account{paper_label}*\n",
            f"Portfolio Value: *${summary.get('portfolio_value', 0):,.2f}*",
            f"Equity: *${summary.get('equity', 0):,.2f}*",
            f"Buying Power: *${summary.get('buying_power', 0):,.2f}*",
            f"Cash: *${summary.get('cash', 0):,.2f}*",
            f"Crypto Value: *${summary.get('crypto_value', 0):,.2f}*",
        ]
        if positions:
            lines += ["", "📊 *Crypto Positions:*"]
            for p in positions:
                sign = "+" if p["unrealized_pl_pct"] >= 0 else ""
                lines.append(
                    f"• {p['symbol']}: {p['quantity']:.6f} "
                    f"@ ${p['current_price']:,.2f} "
                    f"= ${p['market_value']:,.2f} "
                    f"({sign}{p['unrealized_pl_pct']:.1f}%)"
                )
        else:
            lines.append("\n_No open crypto positions._")

        lines.append("\n⚠️ _Paper trading account. Not real money._")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:
        logger.error("Telegram /portfolio failed: %s", exc, exc_info=True)
        await update.message.reply_text(f"❌ Portfolio unavailable: {exc}")


async def cmd_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    symbols = await get_watchlist()
    text = "👁 *Watchlist:* " + ", ".join(symbols) if symbols else "Watchlist is empty."
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def cmd_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /watch ETH")
        return
    symbol = context.args[0].upper()
    await add_to_watchlist(symbol)
    await update.message.reply_text(f"✅ Added {symbol} to watchlist.")


async def cmd_unwatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /unwatch ETH")
        return
    symbol = context.args[0].upper()
    await remove_from_watchlist(symbol)
    await update.message.reply_text(f"✅ Removed {symbol} from watchlist.")


async def cmd_alerts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    chat_id = str(update.effective_chat.id)
    if not context.args or context.args[0].lower() not in ("on", "off"):
        await update.message.reply_text("Usage: /alerts on|off")
        return
    enabled = context.args[0].lower() == "on"
    await set_alerts(chat_id, enabled)
    await update.message.reply_text(
        f"✅ Proactive alerts {'enabled' if enabled else 'disabled'}."
    )


async def cmd_interval(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /interval <minutes> (60-1440)")
        return
    try:
        minutes = max(60, min(1440, int(context.args[0])))
    except ValueError:
        await update.message.reply_text("Please provide a number of minutes.")
        return
    from crypto_oracle.scheduler.heartbeat import get_scheduler
    get_scheduler().reschedule_job("oracle_run", trigger="interval", minutes=minutes)
    await update.message.reply_text(f"✅ Oracle interval updated to {minutes} minutes.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _guard(update):
        return
    chat_id = str(update.effective_chat.id)
    text = (update.message.text or "").strip()
    if not text:
        return
    await update.message.chat.send_action("typing")
    try:
        reply = await handle_free_text(chat_id, text)
        await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)
    except Exception as exc:
        logger.error("handle_message failed: %s", exc, exc_info=True)
        await update.message.reply_text("Something went wrong. Try /status.")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

async def send_message_to_chat(chat_id: str, text: str) -> None:
    global _app
    if _app is None:
        return
    try:
        await _app.bot.send_message(
            chat_id=int(chat_id),
            text=text,
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as exc:
        logger.warning("send_message_to_chat(%s) failed: %s", chat_id, exc)


def build_application() -> Application:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("run", cmd_run))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("portfolio", cmd_portfolio))
    app.add_handler(CommandHandler("watchlist", cmd_watchlist))
    app.add_handler(CommandHandler("watch", cmd_watch))
    app.add_handler(CommandHandler("unwatch", cmd_unwatch))
    app.add_handler(CommandHandler("alerts", cmd_alerts))
    app.add_handler(CommandHandler("interval", cmd_interval))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app


async def start_bot() -> None:
    global _app
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not set — Telegram bot disabled.")
        return
    _app = build_application()
    await _app.initialize()
    await _app.start()
    await _app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot started (polling)")


async def stop_bot() -> None:
    global _app
    if _app is None:
        return
    await _app.updater.stop()
    await _app.stop()
    await _app.shutdown()
    logger.info("Telegram bot stopped")
