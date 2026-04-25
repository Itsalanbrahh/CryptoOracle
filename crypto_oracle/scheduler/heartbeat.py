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

async def oracle_run_job() -> None:
    """Run the oracle for every symbol in the watchlist and broadcast results."""
    from crypto_oracle.orchestrator import CryptoOracle
    from crypto_oracle.api.websocket import manager

    symbols = await get_watchlist()
    oracle = CryptoOracle()
    alert_threshold = float(os.getenv("ALERT_THRESHOLD", "0.70"))

    for symbol in symbols:
        try:
            logger.info("Scheduled oracle run: %s", symbol)
            rec = await oracle.run(symbol)
            rec_id = await save_recommendation(rec)
            rec.id = rec_id
            await set_meta(f"last_run_{symbol}", datetime.utcnow().isoformat())

            await manager.broadcast(
                {"type": "recommendation", "data": rec.model_dump(mode="json")}
            )

            if rec.action in ("BUY", "SELL") and rec.confidence >= alert_threshold:
                await _send_proactive_alert(rec)

        except Exception as exc:
            logger.error("oracle_run_job failed for %s: %s", symbol, exc, exc_info=True)
            # Retry once after 5 minutes
            scheduler = get_scheduler()
            scheduler.add_job(
                _retry_oracle,
                "date",
                run_date=datetime.utcnow().replace(second=0).replace(
                    minute=datetime.utcnow().minute + 5
                    if datetime.utcnow().minute < 55
                    else 59
                ),
                args=[symbol],
                id=f"retry_{symbol}_{datetime.utcnow().timestamp():.0f}",
                misfire_grace_time=120,
            )


async def _retry_oracle(symbol: str) -> None:
    from crypto_oracle.orchestrator import CryptoOracle
    from crypto_oracle.api.websocket import manager

    try:
        oracle = CryptoOracle()
        rec = await oracle.run(symbol)
        rec_id = await save_recommendation(rec)
        rec.id = rec_id
        await manager.broadcast(
            {"type": "recommendation", "data": rec.model_dump(mode="json")}
        )
        logger.info("Retry oracle run succeeded for %s", symbol)
    except Exception as exc:
        logger.error("Retry oracle run also failed for %s: %s", symbol, exc)


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

    # Ping WebSocket clients every 30 seconds
    scheduler.add_job(
        _ws_ping_job,
        "interval",
        seconds=30,
        id="ws_ping",
        replace_existing=True,
    )

    logger.info(
        "Scheduler configured. Oracle every %dm, heartbeat every %dm.",
        oracle_interval,
        heartbeat_interval,
    )
    return scheduler


async def _ws_ping_job() -> None:
    from crypto_oracle.api.websocket import manager
    await manager.ping_all()
