"""FastAPI router — REST endpoints for CryptoOracle."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from crypto_oracle.api.websocket import manager
from crypto_oracle.models.db import (
    get_all_latest,
    get_latest_recommendation,
    get_recommendation_history,
    get_watchlist,
    log_order,
    save_recommendation,
)
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

_START_TIME = time.time()

# Pending orders awaiting second confirmation
_pending_orders: dict[str, dict[str, Any]] = {}

# Cooldown tracking for user-triggered oracle runs (30 min)
_last_run: dict[str, float] = {}
_RUN_COOLDOWN_SECONDS = 1800


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _START_TIME),
        "ws_connections": manager.connection_count,
    }


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

@router.get("/api/recommendations")
async def get_recommendations() -> list[dict]:
    recs = await get_all_latest()
    return [r.model_dump(mode="json") for r in recs]


@router.get("/api/recommendations/{symbol}")
async def get_recommendation(symbol: str) -> dict:
    rec = await get_latest_recommendation(symbol.upper())
    if rec is None:
        raise HTTPException(status_code=404, detail=f"No recommendation for {symbol}")
    return rec.model_dump(mode="json")


@router.get("/api/recommendations/{symbol}/history")
async def get_history(
    symbol: str, limit: int = Query(default=50, ge=1, le=200)
) -> list[dict]:
    recs = await get_recommendation_history(symbol.upper(), limit)
    return [r.model_dump(mode="json") for r in recs]


# ---------------------------------------------------------------------------
# Oracle run trigger
# ---------------------------------------------------------------------------

@router.post("/api/run/{symbol}")
async def trigger_run(symbol: str, background_tasks: BackgroundTasks) -> dict:
    sym = symbol.upper()
    now = time.time()
    remaining = _RUN_COOLDOWN_SECONDS - (now - _last_run.get(sym, 0))
    if remaining > 0:
        raise HTTPException(
            status_code=429,
            detail=f"Cooldown active. Try again in {int(remaining)} seconds.",
        )
    _last_run[sym] = now
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(_run_oracle_and_save, sym, job_id)
    return {"job_id": job_id, "symbol": sym, "status": "queued"}


async def _run_oracle_and_save(symbol: str, job_id: str) -> None:
    try:
        from crypto_oracle.orchestrator import CryptoOracle
        oracle = CryptoOracle()
        rec = await oracle.run(symbol)
        rec_id = await save_recommendation(rec)
        rec.id = rec_id
        await manager.broadcast(
            {"type": "recommendation", "data": rec.model_dump(mode="json")}
        )
        logger.info("Job %s completed: %s %s", job_id, symbol, rec.action)
    except Exception as exc:
        logger.error("Job %s failed: %s", job_id, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Portfolio — Alpaca paper account
# ---------------------------------------------------------------------------

def _alpaca_disabled() -> bool:
    return os.getenv("SKIP_ALPACA", "false").lower() == "true"


@router.get("/api/portfolio")
async def get_portfolio() -> dict:
    if _alpaca_disabled():
        return {"error": "Alpaca integration disabled (SKIP_ALPACA=true)"}
    try:
        from crypto_oracle.alpaca.client import get_account_summary
        return await get_account_summary()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/api/portfolio/crypto")
async def get_crypto_positions() -> list[dict]:
    if _alpaca_disabled():
        return []
    try:
        from crypto_oracle.alpaca.client import get_crypto_positions
        return await get_crypto_positions()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/api/portfolio/orders")
async def get_open_orders() -> list[dict]:
    if _alpaca_disabled():
        return []
    try:
        from crypto_oracle.alpaca.client import get_open_orders
        return await get_open_orders()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# Orders (two-step confirmation)
# ---------------------------------------------------------------------------

class OrderRequest(BaseModel):
    symbol: str
    side: str        # buy | sell
    amount_usd: float


@router.post("/api/order")
async def create_order(req: OrderRequest) -> dict:
    if _alpaca_disabled():
        raise HTTPException(status_code=503, detail="Alpaca integration disabled")

    max_usd = float(os.getenv("ALPACA_MAX_ORDER_USD", "500"))
    if req.amount_usd > max_usd:
        await log_order(req.symbol, req.side, req.amount_usd, "rejected_over_limit")
        raise HTTPException(
            status_code=400,
            detail=f"Order amount exceeds ALPACA_MAX_ORDER_USD limit of ${max_usd:.2f}",
        )

    order_id = str(uuid.uuid4())
    _pending_orders[order_id] = req.model_dump()
    await log_order(req.symbol, req.side, req.amount_usd, "pending", order_id=order_id)
    return {
        "order_id": order_id,
        "status": "pending_confirmation",
        "message": f"POST /api/order/confirm/{order_id} to execute.",
    }


@router.post("/api/order/confirm/{order_id}")
async def confirm_order(order_id: str) -> dict:
    if order_id not in _pending_orders:
        raise HTTPException(status_code=404, detail="Order not found or already processed")

    req_data = _pending_orders.pop(order_id)
    try:
        from crypto_oracle.alpaca.client import place_crypto_order
        result = await place_crypto_order(
            req_data["symbol"], req_data["side"], req_data["amount_usd"]
        )
        await log_order(
            req_data["symbol"],
            req_data["side"],
            req_data["amount_usd"],
            "submitted",
            order_id=order_id,
            response_json=str(result),
        )
        return {"order_id": order_id, "status": "submitted", "result": result}
    except Exception as exc:
        await log_order(
            req_data["symbol"],
            req_data["side"],
            req_data["amount_usd"],
            "error",
            order_id=order_id,
            response_json=str(exc),
        )
        raise HTTPException(status_code=502, detail=str(exc))


@router.delete("/api/order/{order_id}")
async def cancel_order(order_id: str) -> dict:
    if _alpaca_disabled():
        raise HTTPException(status_code=503, detail="Alpaca integration disabled")
    try:
        from crypto_oracle.alpaca.client import cancel_order as _cancel
        await _cancel(order_id)
        return {"order_id": order_id, "status": "cancelled"}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

@router.get("/api/watchlist")
async def get_watchlist_endpoint() -> list[str]:
    return await get_watchlist()
