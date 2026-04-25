"""Async SQLite layer using aiosqlite."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from crypto_oracle.models.signals import AgentSignal, MasterRecommendation
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path("crypto_oracle.db")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    action       TEXT NOT NULL,
    confidence   REAL NOT NULL,
    reasoning    TEXT NOT NULL,
    raw_json     TEXT NOT NULL,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_signals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id INTEGER REFERENCES recommendations(id),
    agent_name        TEXT,
    signal            TEXT,
    confidence        REAL,
    summary           TEXT,
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telegram_alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      TEXT NOT NULL,
    message      TEXT NOT NULL,
    triggered_by TEXT,
    sent_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telegram_chats (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT UNIQUE NOT NULL,
    alerts_on  INTEGER DEFAULT 1,
    registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telegram_conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS robinhood_orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    side         TEXT NOT NULL,
    amount_usd   REAL NOT NULL,
    status       TEXT NOT NULL,
    order_id     TEXT,
    response_json TEXT,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist (
    symbol TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS scheduler_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

INSERT OR IGNORE INTO watchlist (symbol) VALUES ('BTC');
"""


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await db.commit()
    logger.info("Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

async def save_recommendation(rec: MasterRecommendation) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """
            INSERT INTO recommendations (symbol, action, confidence, reasoning, raw_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                rec.symbol,
                rec.action,
                rec.confidence,
                rec.reasoning,
                rec.model_dump_json(),
            ),
        )
        rec_id = cur.lastrowid

        for sig in rec.agent_signals:
            await db.execute(
                """
                INSERT INTO agent_signals
                    (recommendation_id, agent_name, signal, confidence, summary)
                VALUES (?, ?, ?, ?, ?)
                """,
                (rec_id, sig.agent_name, sig.signal, sig.confidence, sig.summary),
            )

        await db.commit()
    logger.debug("Saved recommendation id=%s for %s", rec_id, rec.symbol)
    return rec_id


async def get_latest_recommendation(symbol: str) -> Optional[MasterRecommendation]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT raw_json FROM recommendations WHERE symbol=? ORDER BY id DESC LIMIT 1",
            (symbol.upper(),),
        ) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    return MasterRecommendation.model_validate_json(row["raw_json"])


async def get_recommendation_history(
    symbol: str, limit: int = 50
) -> list[MasterRecommendation]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT raw_json FROM recommendations
            WHERE symbol=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (symbol.upper(), limit),
        ) as cur:
            rows = await cur.fetchall()
    return [MasterRecommendation.model_validate_json(r["raw_json"]) for r in rows]


async def get_all_latest() -> list[MasterRecommendation]:
    """Return the single most-recent recommendation per symbol."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT r.raw_json
            FROM recommendations r
            INNER JOIN (
                SELECT symbol, MAX(id) AS max_id
                FROM recommendations
                GROUP BY symbol
            ) latest ON r.id = latest.max_id
            """
        ) as cur:
            rows = await cur.fetchall()
    return [MasterRecommendation.model_validate_json(r["raw_json"]) for r in rows]


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

async def save_alert(chat_id: str, message: str, triggered_by: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO telegram_alerts (chat_id, message, triggered_by) VALUES (?,?,?)",
            (chat_id, message, triggered_by),
        )
        await db.commit()


async def count_alerts_today() -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM telegram_alerts WHERE DATE(sent_at) = DATE('now')"
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# Telegram chats
# ---------------------------------------------------------------------------

async def register_chat(chat_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO telegram_chats (chat_id) VALUES (?)", (chat_id,)
        )
        await db.commit()


async def get_registered_chats() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT chat_id, alerts_on FROM telegram_chats") as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def set_alerts(chat_id: str, enabled: bool) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE telegram_chats SET alerts_on=? WHERE chat_id=?",
            (1 if enabled else 0, chat_id),
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

async def append_conversation(chat_id: str, role: str, content: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO telegram_conversations (chat_id, role, content) VALUES (?,?,?)",
            (chat_id, role, content),
        )
        await db.commit()


async def get_conversation_history(chat_id: str, limit: int = 10) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT role, content FROM (
                SELECT role, content, id
                FROM telegram_conversations
                WHERE chat_id=?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (chat_id, limit),
        ) as cur:
            rows = await cur.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


async def clear_conversation(chat_id: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM telegram_conversations WHERE chat_id=?", (chat_id,)
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

async def get_watchlist() -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT symbol FROM watchlist") as cur:
            rows = await cur.fetchall()
    return [r[0] for r in rows]


async def add_to_watchlist(symbol: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)", (symbol.upper(),)
        )
        await db.commit()


async def remove_from_watchlist(symbol: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM watchlist WHERE symbol=?", (symbol.upper(),)
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Robinhood orders
# ---------------------------------------------------------------------------

async def log_order(
    symbol: str,
    side: str,
    amount_usd: float,
    status: str,
    order_id: Optional[str] = None,
    response_json: Optional[str] = None,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO robinhood_orders
                (symbol, side, amount_usd, status, order_id, response_json)
            VALUES (?,?,?,?,?,?)
            """,
            (symbol, side, amount_usd, status, order_id, response_json),
        )
        row_id = cur.lastrowid
        await db.commit()
    return row_id


# ---------------------------------------------------------------------------
# Scheduler meta
# ---------------------------------------------------------------------------

async def set_meta(key: str, value: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO scheduler_meta (key, value) VALUES (?,?)",
            (key, value),
        )
        await db.commit()


async def get_meta(key: str) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT value FROM scheduler_meta WHERE key=?", (key,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else None
