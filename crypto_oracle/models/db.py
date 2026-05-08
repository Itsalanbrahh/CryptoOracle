"""Async SQLite layer using aiosqlite."""

from __future__ import annotations

import json
import os
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

CREATE TABLE IF NOT EXISTS strategy_state (
    symbol                TEXT PRIMARY KEY,
    agent_weights         TEXT NOT NULL DEFAULT '{}',
    strategy_notes        TEXT DEFAULT '',
    confidence_threshold  REAL DEFAULT 0.65,
    auto_trade_amount     REAL DEFAULT 100.0,
    updated_at            DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_config (
    agent_name   TEXT PRIMARY KEY,
    system_prompt TEXT NOT NULL,
    reason        TEXT DEFAULT '',
    updated_by    TEXT DEFAULT 'master',
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    amount_usd      REAL NOT NULL,
    quantity        REAL,
    entry_price     REAL,
    exit_price      REAL,
    status          TEXT DEFAULT 'open',
    realized_pnl    REAL,
    alpaca_order_id TEXT,
    triggered_by    TEXT DEFAULT 'manual',
    confidence      REAL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    closed_at       DATETIME
);

INSERT OR IGNORE INTO watchlist (symbol) VALUES ('BTC');
INSERT OR IGNORE INTO watchlist (symbol) VALUES ('ETH');

CREATE TABLE IF NOT EXISTS stock_watchlist (
    symbol TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS stock_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    trade_type      TEXT NOT NULL DEFAULT 'long',
    amount_usd      REAL NOT NULL,
    quantity        REAL,
    entry_price     REAL,
    exit_price      REAL,
    status          TEXT DEFAULT 'open',
    realized_pnl    REAL,
    alpaca_order_id TEXT,
    triggered_by    TEXT DEFAULT 'manual',
    confidence      REAL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    closed_at       DATETIME
);

INSERT OR IGNORE INTO stock_watchlist (symbol) VALUES ('NVDA');
INSERT OR IGNORE INTO stock_watchlist (symbol) VALUES ('TSLA');
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


async def get_strategy_state(symbol: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM strategy_state WHERE symbol=?", (symbol.upper(),)
        ) as cur:
            row = await cur.fetchone()
    if row is None:
        default_amount = float(os.getenv("AUTO_TRADE_AMOUNT_USD", "200"))
        return {
            "symbol": symbol.upper(),
            "agent_weights": {},
            "strategy_notes": "",
            "confidence_threshold": 0.60,
            "auto_trade_amount": default_amount,
        }
    return {
        "symbol": row["symbol"],
        "agent_weights": json.loads(row["agent_weights"]) if row["agent_weights"] else {},
        "strategy_notes": row["strategy_notes"] or "",
        "confidence_threshold": float(row["confidence_threshold"] or 0.65),
        "auto_trade_amount": float(row["auto_trade_amount"] or 100.0),
    }


async def save_strategy_state(
    symbol: str,
    agent_weights: dict,
    strategy_notes: str,
    confidence_threshold: float,
    auto_trade_amount: float,
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO strategy_state
                (symbol, agent_weights, strategy_notes, confidence_threshold, auto_trade_amount, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                symbol.upper(),
                json.dumps(agent_weights),
                strategy_notes,
                max(0.50, min(0.90, confidence_threshold)),
                max(25.0, min(20000.0, auto_trade_amount)),
            ),
        )
        await db.commit()


async def get_recommendation_outcomes(symbol: str, limit: int = 15) -> list[dict]:
    """Pair consecutive recommendations to determine if each was correct."""
    recs = await get_recommendation_history(symbol, limit=limit + 1)
    recs = list(reversed(recs))  # oldest first

    outcomes = []
    for i in range(len(recs) - 1):
        rec = recs[i]
        nxt = recs[i + 1]
        entry = rec.price_at_time
        exit_ = nxt.price_at_time
        if not entry or not exit_ or entry <= 0:
            continue
        change_pct = (exit_ - entry) / entry * 100
        if rec.action == "BUY":
            correct = change_pct > 0.5
        elif rec.action == "SELL":
            correct = change_pct < -0.5
        else:
            correct = abs(change_pct) < 2.0
        outcomes.append({
            "action": rec.action,
            "confidence": rec.confidence,
            "entry_price": entry,
            "exit_price": exit_,
            "change_pct": round(change_pct, 3),
            "outcome": "correct" if correct else "incorrect",
            "agent_signals": [
                {"agent": s.agent_name, "signal": s.signal, "confidence": s.confidence}
                for s in rec.agent_signals
            ],
            "timestamp": rec.timestamp.isoformat() if rec.timestamp else None,
        })
    return list(reversed(outcomes))  # newest first


async def get_agent_accuracy(symbol: str, limit: int = 20) -> dict:
    """Per-agent hit-rate over recent outcomes."""
    outcomes = await get_recommendation_outcomes(symbol, limit=limit)
    stats: dict[str, dict] = {}
    for o in outcomes:
        direction = "up" if o["change_pct"] > 0.5 else "down" if o["change_pct"] < -0.5 else "flat"
        for sig in o.get("agent_signals", []):
            a = sig["agent"]
            if a not in stats:
                stats[a] = {"correct": 0, "total": 0}
            hit = (
                (sig["signal"] == "BULLISH" and direction == "up") or
                (sig["signal"] == "BEARISH" and direction == "down") or
                (sig["signal"] == "NEUTRAL" and direction == "flat")
            )
            stats[a]["total"] += 1
            if hit:
                stats[a]["correct"] += 1
    for a, s in stats.items():
        s["accuracy_pct"] = round(s["correct"] / s["total"] * 100, 1) if s["total"] else 50.0
    return stats


async def get_meta(key: str) -> Optional[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT value FROM scheduler_meta WHERE key=?", (key,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# Trades (auto + manual)
# ---------------------------------------------------------------------------

async def log_trade(
    symbol: str,
    amount_usd: float,
    entry_price: float,
    quantity: float,
    alpaca_order_id: Optional[str] = None,
    triggered_by: str = "manual",
    confidence: Optional[float] = None,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO trades
                (symbol, amount_usd, entry_price, quantity, alpaca_order_id, triggered_by, confidence, status)
            VALUES (?,?,?,?,?,?,?,'open')
            """,
            (symbol.upper(), amount_usd, entry_price, quantity, alpaca_order_id, triggered_by, confidence),
        )
        row_id = cur.lastrowid
        await db.commit()
    return row_id


async def close_trade(trade_id: int, exit_price: float, realized_pnl: float) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE trades
            SET status='closed', exit_price=?, realized_pnl=?, closed_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (exit_price, realized_pnl, trade_id),
        )
        await db.commit()


async def get_open_trades(symbol: Optional[str] = None) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if symbol:
            async with db.execute(
                "SELECT * FROM trades WHERE status='open' AND symbol=? ORDER BY id",
                (symbol.upper(),),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY id"
            ) as cur:
                rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_trade_history(symbol: Optional[str] = None, limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if symbol:
            async with db.execute(
                "SELECT * FROM trades WHERE symbol=? ORDER BY id DESC LIMIT ?",
                (symbol.upper(), limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ) as cur:
                rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_trade_stats(symbol: Optional[str] = None) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        where = "WHERE symbol=?" if symbol else ""
        params = (symbol.upper(),) if symbol else ()
        async with db.execute(
            f"""
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END), 0) as closed,
                COALESCE(SUM(CASE WHEN status='open'   THEN 1 ELSE 0 END), 0) as open_count,
                COALESCE(SUM(CASE WHEN status='closed' AND realized_pnl > 0 THEN 1 ELSE 0 END), 0) as winners,
                COALESCE(SUM(CASE WHEN status='closed' THEN realized_pnl ELSE 0 END), 0) as total_pnl,
                COALESCE(AVG(CASE WHEN status='closed' THEN realized_pnl END), 0) as avg_pnl,
                COALESCE(MAX(CASE WHEN status='closed' THEN realized_pnl END), 0) as best_trade,
                COALESCE(MIN(CASE WHEN status='closed' THEN realized_pnl END), 0) as worst_trade
            FROM trades {where}
            """,
            params,
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return {"total": 0, "closed": 0, "open_count": 0, "winners": 0,
                "total_pnl": 0, "avg_pnl": 0, "best_trade": 0, "worst_trade": 0, "win_rate": 0}
    d = dict(zip(
        ["total", "closed", "open_count", "winners", "total_pnl", "avg_pnl", "best_trade", "worst_trade"],
        row,
    ))
    d["win_rate"] = round(d["winners"] / d["closed"] * 100, 1) if d["closed"] else 0
    return d


# ---------------------------------------------------------------------------
# Stock watchlist
# ---------------------------------------------------------------------------

async def get_stock_watchlist() -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT symbol FROM stock_watchlist") as cur:
            rows = await cur.fetchall()
    return [r[0] for r in rows]


async def add_stock_symbol(symbol: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO stock_watchlist (symbol) VALUES (?)", (symbol.upper(),)
        )
        await db.commit()


async def remove_stock_symbol(symbol: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "DELETE FROM stock_watchlist WHERE symbol=?", (symbol.upper(),)
        )
        await db.commit()


# ---------------------------------------------------------------------------
# Stock trades
# ---------------------------------------------------------------------------

async def log_stock_trade(
    symbol: str,
    trade_type: str,
    amount_usd: float,
    entry_price: float,
    quantity: float,
    alpaca_order_id: Optional[str] = None,
    triggered_by: str = "manual",
    confidence: Optional[float] = None,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """
            INSERT INTO stock_trades
                (symbol, trade_type, amount_usd, entry_price, quantity,
                 alpaca_order_id, triggered_by, confidence, status)
            VALUES (?,?,?,?,?,?,?,?,'open')
            """,
            (symbol.upper(), trade_type, amount_usd, entry_price, quantity,
             alpaca_order_id, triggered_by, confidence),
        )
        row_id = cur.lastrowid
        await db.commit()
    return row_id


async def close_stock_trade(trade_id: int, exit_price: float, realized_pnl: float) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE stock_trades
            SET status='closed', exit_price=?, realized_pnl=?, closed_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (exit_price, realized_pnl, trade_id),
        )
        await db.commit()


async def get_open_stock_trades(symbol: Optional[str] = None) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if symbol:
            async with db.execute(
                "SELECT * FROM stock_trades WHERE status='open' AND symbol=? ORDER BY id",
                (symbol.upper(),),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM stock_trades WHERE status='open' ORDER BY id"
            ) as cur:
                rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_stock_trade_history(symbol: Optional[str] = None, limit: int = 50) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        if symbol:
            async with db.execute(
                "SELECT * FROM stock_trades WHERE symbol=? ORDER BY id DESC LIMIT ?",
                (symbol.upper(), limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with db.execute(
                "SELECT * FROM stock_trades ORDER BY id DESC LIMIT ?", (limit,)
            ) as cur:
                rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Agent config (master-editable system prompts)
# ---------------------------------------------------------------------------

async def get_agent_config(agent_name: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM agent_config WHERE agent_name=?", (agent_name,)
        ) as cur:
            row = await cur.fetchone()
    return dict(row) if row else None


async def save_agent_config(
    agent_name: str,
    system_prompt: str,
    reason: str = "",
    updated_by: str = "master",
) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO agent_config
                (agent_name, system_prompt, reason, updated_by, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (agent_name, system_prompt, reason, updated_by),
        )
        await db.commit()
    logger.info("Agent config saved for %s (by=%s): %s", agent_name, updated_by, reason[:80])


async def get_all_agent_configs() -> dict[str, dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM agent_config") as cur:
            rows = await cur.fetchall()
    return {row["agent_name"]: dict(row) for row in rows}


async def get_stock_trade_stats(symbol: Optional[str] = None) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        where = "WHERE symbol=?" if symbol else ""
        params = (symbol.upper(),) if symbol else ()
        async with db.execute(
            f"""
            SELECT
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END), 0) as closed,
                COALESCE(SUM(CASE WHEN status='open'   THEN 1 ELSE 0 END), 0) as open_count,
                COALESCE(SUM(CASE WHEN status='closed' AND realized_pnl > 0 THEN 1 ELSE 0 END), 0) as winners,
                COALESCE(SUM(CASE WHEN status='closed' THEN realized_pnl ELSE 0 END), 0) as total_pnl,
                COALESCE(AVG(CASE WHEN status='closed' THEN realized_pnl END), 0) as avg_pnl
            FROM stock_trades {where}
            """,
            params,
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return {"total": 0, "closed": 0, "open_count": 0, "winners": 0, "total_pnl": 0, "avg_pnl": 0, "win_rate": 0}
    d = dict(zip(["total", "closed", "open_count", "winners", "total_pnl", "avg_pnl"], row))
    d["win_rate"] = round(d["winners"] / d["closed"] * 100, 1) if d["closed"] else 0
    return d
