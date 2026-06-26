"""APScheduler jobs: oracle runs, heartbeat pushes, market-open brief."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

from crypto_oracle.models.db import (
    get_all_latest,
    get_meta,
    get_registered_chats,
    get_stock_watchlist,
    get_watchlist,
    save_alert,
    save_recommendation,
    set_meta,
)
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

_scheduler: Optional[AsyncIOScheduler] = None


def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        jobstores = {
            "default": SQLAlchemyJobStore(url="sqlite:///crypto_oracle.db")
        }
        _scheduler = AsyncIOScheduler(jobstores=jobstores, timezone="UTC")
    return _scheduler


# ---------------------------------------------------------------------------
# Job: oracle run
# ---------------------------------------------------------------------------

_running_symbols: set[str] = set()   # debounce — prevent overlapping runs per symbol


async def _run_symbol(symbol: str, triggered_by: str = "scheduled") -> None:
    """Run oracle for one symbol, broadcast, auto-trade, alert. Shared by all trigger paths."""
    from crypto_oracle.orchestrator import CryptoOracle
    from crypto_oracle.api.websocket import manager
    from crypto_oracle.autotrader import maybe_auto_trade

    if symbol in _running_symbols:
        logger.debug("Oracle for %s already running, skipping duplicate trigger", symbol)
        return
    _running_symbols.add(symbol)
    try:
        alert_threshold = float(os.getenv("ALERT_THRESHOLD", "0.60"))
        oracle = CryptoOracle()
        logger.info("Oracle run [%s]: %s", triggered_by, symbol)
        rec = await oracle.run(symbol)
        rec_id = await save_recommendation(rec)
        rec.id = rec_id
        await set_meta(f"last_run_{symbol}", datetime.utcnow().isoformat())
        await set_meta(f"last_price_{symbol}", str(rec.price_at_time or ""))

        await manager.broadcast({"type": "recommendation", "data": rec.model_dump(mode="json")})

        if rec.action in ("BUY", "SELL") and rec.confidence >= alert_threshold:
            await _send_proactive_alert(rec)

        trade = await maybe_auto_trade(rec)
        if trade:
            await _send_trade_alert(trade, symbol)

    except Exception as exc:
        logger.error("_run_symbol failed for %s: %s", symbol, exc, exc_info=True)
    finally:
        _running_symbols.discard(symbol)


async def oracle_run_job() -> None:
    """Scheduled full oracle run for all watchlist symbols."""
    symbols = await get_watchlist()
    for symbol in symbols:
        await _run_symbol(symbol, triggered_by="scheduled")
    await _send_scheduled_report()


# ---------------------------------------------------------------------------
# Stock oracle jobs
# ---------------------------------------------------------------------------

_running_stock_symbols: set[str] = set()


async def _run_stock_symbol(symbol: str, triggered_by: str = "scheduled") -> None:
    """Run stock oracle for one symbol, broadcast, auto-trade, alert."""
    from crypto_oracle.stock_oracle import StockOracle
    from crypto_oracle.stock_autotrader import maybe_stock_auto_trade
    from crypto_oracle.api.websocket import manager

    if symbol in _running_stock_symbols:
        logger.debug("Stock oracle for %s already running, skipping", symbol)
        return
    _running_stock_symbols.add(symbol)
    try:
        alert_threshold = float(os.getenv("ALERT_THRESHOLD", "0.60"))
        oracle = StockOracle()
        logger.info("Stock oracle run [%s]: %s", triggered_by, symbol)
        rec = await oracle.run(symbol)
        rec_id = await save_recommendation(rec)
        rec.id = rec_id
        await set_meta(f"last_run_stock_{symbol}", datetime.utcnow().isoformat())
        await set_meta(f"last_price_stock_{symbol}", str(rec.price_at_time or ""))

        await manager.broadcast({"type": "stock_recommendation", "data": rec.model_dump(mode="json")})

        if rec.action in ("BUY", "SELL") and rec.confidence >= alert_threshold:
            await _send_stock_alert(rec)

        trade = await maybe_stock_auto_trade(rec)
        if trade:
            await _send_stock_trade_alert(trade, symbol)

    except Exception as exc:
        logger.error("_run_stock_symbol failed for %s: %s", symbol, exc, exc_info=True)
    finally:
        _running_stock_symbols.discard(symbol)


async def stock_oracle_run_job() -> None:
    """Scheduled stock oracle run — only fires during market hours."""
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        return
    try:
        from crypto_oracle.alpaca.client import is_market_open
        if not await is_market_open():
            logger.debug("stock_oracle_run_job: market closed, skipping")
            return
    except Exception:
        pass

    symbols = await get_stock_watchlist()
    for symbol in symbols:
        await _run_stock_symbol(symbol, triggered_by="scheduled")


async def _send_stock_alert(rec) -> None:
    from crypto_oracle.telegram.bot import send_message_to_chat
    chats = await get_registered_chats()
    direction = "📈 LONG" if rec.action == "BUY" else "📉 SHORT"
    price_str = f"${rec.price_at_time:,.2f}" if rec.price_at_time else "N/A"
    msg = (
        f"🔔 *Stock Signal — {rec.symbol}*\n"
        f"{direction} @ {price_str} | Confidence: {rec.confidence*100:.0f}%\n\n"
        f"*Reasoning:* {rec.reasoning}\n\n"
        f"*Catalysts:* {' | '.join(rec.key_catalysts[:3])}"
    )
    for chat in chats:
        if not chat.get("alerts_on", 1):
            continue
        try:
            await send_message_to_chat(chat["chat_id"], msg)
        except Exception as exc:
            logger.warning("Failed to send stock alert to %s: %s", chat["chat_id"], exc)


async def _send_stock_trade_alert(trade: dict, symbol: str) -> None:
    from crypto_oracle.telegram.bot import send_message_to_chat
    chats = await get_registered_chats()
    action = trade.get("action", "")
    trade_type = trade.get("trade_type", "")
    if action in ("BUY", "SELL"):
        emoji = "🟢" if action == "BUY" else "🔴"
        label = "LONG" if trade_type == "long" else "SHORT"
        msg = (
            f"{emoji} *Stock Auto-Trade — {symbol}*\n"
            f"{label} ${trade.get('amount_usd', 0):.0f} @ ${trade.get('entry_price', 0):,.2f}\n"
            f"Qty: {trade.get('quantity', 0):.4f} shares"
        )
    else:
        pnl = trade.get("realized_pnl", 0)
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"⚪ *Stock Auto-Trade CLOSED — {symbol}*\n"
            f"Exit @ ${trade.get('exit_price', 0):,.2f}\n"
            f"Realized P&L: *{sign}${pnl:.2f}*"
        )
    for chat in chats:
        try:
            await send_message_to_chat(chat["chat_id"], msg)
        except Exception:
            pass


async def price_trigger_job() -> None:
    """Fire an immediate oracle run when price moves > ORACLE_PRICE_TRIGGER_PCT since last run."""
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        return

    threshold_pct = float(os.getenv("ORACLE_PRICE_TRIGGER_PCT", "1.5"))
    symbols = await get_watchlist()

    from crypto_oracle.alpaca.client import get_crypto_price
    for symbol in symbols:
        try:
            last_price_str = await get_meta(f"last_price_{symbol}")
            if not last_price_str:
                continue
            last_price = float(last_price_str)
            if last_price <= 0:
                continue
            current = await get_crypto_price(symbol)
            change_pct = abs((current - last_price) / last_price * 100)
            if change_pct >= threshold_pct:
                direction = "📈" if current > last_price else "📉"
                logger.info(
                    "Price trigger fired: %s moved %.2f%% (%s→%s) %s",
                    symbol, change_pct, last_price, current, direction,
                )
                from crypto_oracle.telegram.bot import send_message_to_chat
                chats = await get_registered_chats()
                msg = (
                    f"{direction} *Price alert — {symbol}*\n"
                    f"Moved *{change_pct:+.2f}%* since last oracle run "
                    f"(${last_price:,.0f} → ${current:,.0f})\n"
                    f"Running fresh analysis…"
                )
                for chat in chats:
                    try:
                        await send_message_to_chat(chat["chat_id"], msg)
                    except Exception:
                        pass
                await _run_symbol(symbol, triggered_by=f"price_trigger({change_pct:+.1f}%)")
        except Exception as exc:
            logger.warning("price_trigger_job error for %s: %s", symbol, exc)


async def _retry_oracle(symbol: str) -> None:
    await _run_symbol(symbol, triggered_by="retry")


async def _send_trade_alert(trade: dict, symbol: str) -> None:
    """Push an auto-trade execution notification to all Telegram chats."""
    from crypto_oracle.telegram.bot import send_message_to_chat
    chats = await get_registered_chats()
    action = trade.get("action", "")
    emoji = "🟢" if action == "BUY" else "🔴"
    if action == "BUY":
        msg = (
            f"{emoji} *Auto-Trade EXECUTED — {symbol}*\n"
            f"BUY ${trade.get('amount_usd', 0):.0f} @ ${trade.get('entry_price', 0):,.2f}\n"
            f"Qty: {trade.get('quantity', 0):.6f} coins"
        )
    else:
        pnl = trade.get("realized_pnl", 0)
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"{emoji} *Auto-Trade EXECUTED — {symbol}*\n"
            f"SELL @ ${trade.get('exit_price', 0):,.2f}\n"
            f"Realized P&L: *{sign}${pnl:.2f}*"
        )
    for chat in chats:
        try:
            await send_message_to_chat(chat["chat_id"], msg)
        except Exception:
            pass


async def _send_scheduled_report() -> None:
    """Generate and send a PDF dashboard report to all registered Telegram chats."""
    from crypto_oracle.telegram.bot import send_message_to_chat
    from crypto_oracle.telegram.pdf_report import build_report_pdf

    chats = await get_registered_chats()
    if not chats:
        return

    try:
        pdf_buf = await build_report_pdf()
    except Exception as exc:
        logger.error("PDF report generation failed: %s", exc, exc_info=True)
        return

    now_str = datetime.utcnow().strftime("%Y%m%d_%H%M")
    filename = f"crypto_oracle_{now_str}.pdf"

    import os as _os
    from telegram import Bot
    token = _os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return

    bot = Bot(token=token)
    for chat in chats:
        try:
            pdf_buf.seek(0)
            await bot.send_document(
                chat_id=int(chat["chat_id"]),
                document=pdf_buf,
                filename=filename,
                caption=f"📊 CryptoOracle Scheduled Report — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            )
            logger.info("Sent PDF report to chat %s", chat["chat_id"])
        except Exception as exc:
            logger.warning("Failed to send PDF to chat %s: %s", chat["chat_id"], exc)


async def _send_proactive_alert(rec) -> None:  # noqa: ANN001
    from crypto_oracle.telegram.notifications import build_alert_message
    from crypto_oracle.telegram.bot import send_message_to_chat

    chats = await get_registered_chats()
    msg = build_alert_message(rec)

    for chat in chats:
        if not chat.get("alerts_on", 1):
            continue
        try:
            await send_message_to_chat(chat["chat_id"], msg)
            await save_alert(chat["chat_id"], msg, "threshold")
        except Exception as exc:
            logger.warning("Failed to send alert to %s: %s", chat["chat_id"], exc)


# ---------------------------------------------------------------------------
# Job: heartbeat
# ---------------------------------------------------------------------------

async def heartbeat_job() -> None:
    """Send a status heartbeat to all registered Telegram chats."""
    from crypto_oracle.telegram.bot import send_message_to_chat

    chats = await get_registered_chats()
    if not chats:
        logger.debug("heartbeat_job: no registered chats")
        return

    recs = await get_all_latest()
    interval = int(os.getenv("ORACLE_INTERVAL_MINUTES", "240"))

    for rec in recs:
        last_run_str = await get_meta(f"last_run_{rec.symbol}")
        if last_run_str:
            last_run = datetime.fromisoformat(last_run_str)
            now = datetime.utcnow()
            age_min = int((now - last_run).total_seconds() / 60)
            time_ago = f"{age_min}m ago"
            next_run_min = interval - age_min
            next_run_str = f"{max(next_run_min, 0)} min"
        else:
            time_ago = "unknown"
            next_run_str = f"{interval} min"

        msg = (
            f"💓 *CryptoOracle Heartbeat* — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"{rec.symbol}: *{rec.action}* ({rec.confidence*100:.0f}%) | Last run: {time_ago}\n"
            f"Next oracle run in: {next_run_str}\n\n"
            f"Reply /status for full report or /run to trigger fresh analysis."
        )

        for chat in chats:
            try:
                await send_message_to_chat(chat["chat_id"], msg)
                await save_alert(chat["chat_id"], msg, "heartbeat")
            except Exception as exc:
                logger.warning("heartbeat send failed for %s: %s", chat["chat_id"], exc)


# ---------------------------------------------------------------------------
# Job: market-open morning brief
# ---------------------------------------------------------------------------

async def market_open_job() -> None:
    """Send a morning brief to all registered chats at market open."""
    from crypto_oracle.telegram.bot import send_message_to_chat
    from datetime import date

    chats = await get_registered_chats()
    if not chats:
        return

    recs = await get_all_latest()
    today = date.today().strftime("%A, %B %d %Y")

    for rec in recs:
        macro_summary = next(
            (s.summary for s in rec.agent_signals if s.agent_name == "Macro"),
            "Macro data unavailable.",
        )
        catalysts = "\n".join(f"• {c}" for c in rec.key_catalysts[:3]) or "None noted."

        msg = (
            f"🌅 *CryptoOracle Morning Brief — {today}*\n\n"
            f"Current {rec.symbol} signal: *{rec.action}* {rec.confidence*100:.0f}%\n\n"
            f"📊 *Macro context:*\n{macro_summary}\n\n"
            f"👀 *Watch today:*\n{catalysts}"
        )

        for chat in chats:
            try:
                await send_message_to_chat(chat["chat_id"], msg)
                await save_alert(chat["chat_id"], msg, "market_open")
            except Exception as exc:
                logger.warning("morning brief failed for %s: %s", chat["chat_id"], exc)


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def setup_scheduler() -> AsyncIOScheduler:
    scheduler = get_scheduler()

    oracle_interval = int(os.getenv("ORACLE_INTERVAL_MINUTES", "240"))
    heartbeat_interval = int(os.getenv("HEARTBEAT_INTERVAL_MINUTES", "360"))

    scheduler.add_job(
        oracle_run_job,
        "interval",
        minutes=oracle_interval,
        id="oracle_run",
        replace_existing=True,
        misfire_grace_time=300,
    )

    scheduler.add_job(
        heartbeat_job,
        "interval",
        minutes=heartbeat_interval,
        id="heartbeat",
        replace_existing=True,
        misfire_grace_time=300,
    )

    scheduler.add_job(
        market_open_job,
        "cron",
        hour=14,   # 14:30 UTC = 9:30 AM ET
        minute=30,
        id="market_open",
        replace_existing=True,
        misfire_grace_time=600,
    )

    # Price-movement trigger (disabled by default to control token usage)
    if os.getenv("ORACLE_PRICE_TRIGGER_ENABLED", "false").lower() == "true":
        scheduler.add_job(
            price_trigger_job,
            "interval",
            minutes=5,
            id="price_trigger",
            replace_existing=True,
            misfire_grace_time=60,
        )

    # Stock oracle: every 30 minutes (job itself checks market hours)
    stock_interval = int(os.getenv("STOCK_INTERVAL_MINUTES", "30"))
    scheduler.add_job(
        stock_oracle_run_job,
        "interval",
        minutes=stock_interval,
        id="stock_oracle_run",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # Ping WebSocket clients every 30 seconds
    scheduler.add_job(
        _ws_ping_job,
        "interval",
        seconds=30,
        id="ws_ping",
        replace_existing=True,
    )

    logger.info(
        "Scheduler configured. Oracle every %dm, stock oracle every %dm, heartbeat every %dm, price-trigger every 5m.",
        oracle_interval,
        stock_interval,
        heartbeat_interval,
    )
    return scheduler


async def _ws_ping_job() -> None:
    from crypto_oracle.api.websocket import manager
    await manager.ping_all()
