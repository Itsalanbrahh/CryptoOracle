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
    add_stock_symbol,
    close_trade,
    get_agent_accuracy,
    get_all_latest,
    get_latest_recommendation,
    get_open_stock_trades,
    get_open_trades,
    get_recommendation_history,
    get_recommendation_outcomes,
    get_stock_trade_history,
    get_stock_trade_stats,
    get_stock_watchlist,
    get_strategy_state,
    get_trade_history,
    get_trade_stats,
    get_watchlist,
    log_order,
    log_stock_trade,
    log_trade,
    remove_stock_symbol,
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
_RUN_COOLDOWN_SECONDS = 300   # 5 minutes


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
        from crypto_oracle.autotrader import maybe_auto_trade
        oracle = CryptoOracle()
        rec = await oracle.run(symbol)
        rec_id = await save_recommendation(rec)
        rec.id = rec_id
        await manager.broadcast(
            {"type": "recommendation", "data": rec.model_dump(mode="json")}
        )
        await maybe_auto_trade(rec)
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

    max_usd = float(os.getenv("ALPACA_MAX_ORDER_USD", "20000"))
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
        from crypto_oracle.alpaca.client import (
            close_crypto_position,
            get_crypto_price,
            place_crypto_order,
        )

        symbol = req_data["symbol"]
        side = req_data["side"]
        amount_usd = req_data["amount_usd"]

        if side == "sell":
            result = await close_crypto_position(symbol)
        else:
            result = await place_crypto_order(symbol, side, amount_usd)

        await log_order(symbol, side, amount_usd, "submitted", order_id=order_id, response_json=str(result))

        # Track in trades table
        try:
            price = await get_crypto_price(symbol)
            if side == "buy":
                await log_trade(
                    symbol=symbol, amount_usd=amount_usd,
                    entry_price=price, quantity=amount_usd / price if price > 0 else 0,
                    alpaca_order_id=order_id, triggered_by="manual",
                )
            else:
                open_trades = await get_open_trades(symbol)
                for t in open_trades:
                    qty = t["quantity"] or 0
                    entry = t["entry_price"] or price
                    await close_trade(t["id"], price, round((price - entry) * qty, 4))
        except Exception as track_exc:
            logger.warning("Trade tracking failed (order still submitted): %s", track_exc)

        await manager.broadcast({"type": "trade", "data": {
            "action": side.upper(), "symbol": symbol,
            "amount_usd": amount_usd, "triggered_by": "manual",
        }})
        return {"order_id": order_id, "status": "submitted", "result": result}
    except Exception as exc:
        await log_order(
            req_data["symbol"], req_data["side"], req_data["amount_usd"],
            "error", order_id=order_id, response_json=str(exc),
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
# Trades
# ---------------------------------------------------------------------------

@router.get("/api/trades")
async def get_trades_endpoint(
    symbol: Optional[str] = None, limit: int = Query(default=50, ge=1, le=200)
) -> list[dict]:
    return await get_trade_history(symbol=symbol.upper() if symbol else None, limit=limit)


@router.get("/api/trades/stats")
async def get_trades_stats_endpoint(symbol: Optional[str] = None) -> dict:
    return await get_trade_stats(symbol=symbol.upper() if symbol else None)


# ---------------------------------------------------------------------------
# Auto-trade settings
# ---------------------------------------------------------------------------

class AutoTradeSettings(BaseModel):
    enabled: bool
    amount_usd: float
    confidence_threshold: float


@router.get("/api/settings/auto-trade")
async def get_auto_trade() -> dict:
    from crypto_oracle.autotrader import get_auto_trade_settings
    return await get_auto_trade_settings()


@router.post("/api/settings/auto-trade")
async def set_auto_trade(settings: AutoTradeSettings) -> dict:
    from crypto_oracle.autotrader import update_auto_trade_settings
    await update_auto_trade_settings(
        settings.enabled, settings.amount_usd, settings.confidence_threshold
    )
    return {"status": "ok", **settings.model_dump()}


# ---------------------------------------------------------------------------
# Strategy state
# ---------------------------------------------------------------------------

@router.get("/api/strategy/{symbol}")
async def get_strategy(symbol: str) -> dict:
    state   = await get_strategy_state(symbol.upper())
    perf    = await get_agent_accuracy(symbol.upper(), limit=20)
    outcomes = await get_recommendation_outcomes(symbol.upper(), limit=10)
    correct = sum(1 for o in outcomes if o["outcome"] == "correct")
    return {
        **state,
        "agent_accuracy": perf,
        "evaluated_runs": len(outcomes),
        "overall_accuracy_pct": round(correct / len(outcomes) * 100, 1) if outcomes else None,
    }


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

@router.get("/api/watchlist")
async def get_watchlist_endpoint() -> list[str]:
    return await get_watchlist()


# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------

@router.get("/api/stocks/watchlist")
async def get_stock_watchlist_endpoint() -> list[str]:
    return await get_stock_watchlist()


@router.post("/api/stocks/watchlist/{symbol}")
async def add_stock_endpoint(symbol: str) -> dict:
    await add_stock_symbol(symbol.upper())
    return {"symbol": symbol.upper(), "status": "added"}


@router.delete("/api/stocks/watchlist/{symbol}")
async def remove_stock_endpoint(symbol: str) -> dict:
    await remove_stock_symbol(symbol.upper())
    return {"symbol": symbol.upper(), "status": "removed"}


@router.get("/api/stocks/positions")
async def get_stock_positions_endpoint() -> list[dict]:
    if _alpaca_disabled():
        return []
    try:
        from crypto_oracle.alpaca.client import get_stock_positions
        return await get_stock_positions()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/api/stocks/market-open")
async def get_market_open() -> dict:
    if _alpaca_disabled():
        return {"is_open": None, "error": "Alpaca disabled"}
    try:
        from crypto_oracle.alpaca.client import is_market_open
        open_ = await is_market_open()
        return {"is_open": open_}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/api/stocks/run/{symbol}")
async def trigger_stock_run(symbol: str, background_tasks: BackgroundTasks) -> dict:
    sym = symbol.upper()
    now = time.time()
    remaining = _RUN_COOLDOWN_SECONDS - (now - _last_run.get(f"stock_{sym}", 0))
    if remaining > 0:
        raise HTTPException(
            status_code=429,
            detail=f"Cooldown active. Try again in {int(remaining)} seconds.",
        )
    _last_run[f"stock_{sym}"] = now
    background_tasks.add_task(_run_stock_oracle_and_save, sym)
    return {"symbol": sym, "status": "queued"}


async def _run_stock_oracle_and_save(symbol: str) -> None:
    try:
        from crypto_oracle.stock_oracle import StockOracle
        from crypto_oracle.stock_autotrader import maybe_stock_auto_trade
        oracle = StockOracle()
        rec = await oracle.run(symbol)
        rec_id = await save_recommendation(rec)
        rec.id = rec_id
        await manager.broadcast(
            {"type": "stock_recommendation", "data": rec.model_dump(mode="json")}
        )
        await maybe_stock_auto_trade(rec)
        logger.info("Stock oracle job completed: %s %s", symbol, rec.action)
    except Exception as exc:
        logger.error("Stock oracle job failed for %s: %s", symbol, exc, exc_info=True)


class StockOrderRequest(BaseModel):
    symbol: str
    side: str    # buy (long) | sell (short)
    amount_usd: float


@router.post("/api/stocks/order")
async def create_stock_order(req: StockOrderRequest) -> dict:
    if _alpaca_disabled():
        raise HTTPException(status_code=503, detail="Alpaca integration disabled")

    max_usd = float(os.getenv("ALPACA_MAX_ORDER_USD", "20000"))
    if req.amount_usd > max_usd:
        raise HTTPException(
            status_code=400,
            detail=f"Amount exceeds max ${max_usd:.2f}",
        )

    try:
        from crypto_oracle.alpaca.client import (
            get_stock_price,
            is_market_open,
            place_stock_order,
        )
        if not await is_market_open():
            raise HTTPException(status_code=400, detail="Market is closed")

        symbol = req.symbol.upper()
        price = await get_stock_price(symbol)
        result = await place_stock_order(symbol, req.side, req.amount_usd)
        qty = req.amount_usd / price if price > 0 else 0
        trade_type = "long" if req.side.lower() == "buy" else "short"
        await log_stock_trade(
            symbol=symbol, trade_type=trade_type, amount_usd=req.amount_usd,
            entry_price=price, quantity=qty,
            alpaca_order_id=result["order_id"], triggered_by="manual",
        )
        await manager.broadcast({"type": "stock_trade", "data": {
            "action": req.side.upper(), "symbol": symbol,
            "trade_type": trade_type, "amount_usd": req.amount_usd,
            "entry_price": price, "triggered_by": "manual",
        }})
        return {"status": "submitted", "result": result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.post("/api/stocks/close/{symbol}")
async def close_stock_order(symbol: str) -> dict:
    if _alpaca_disabled():
        raise HTTPException(status_code=503, detail="Alpaca integration disabled")
    try:
        from crypto_oracle.alpaca.client import close_stock_position, get_stock_price
        from crypto_oracle.models.db import close_stock_trade as _close_st

        sym = symbol.upper()
        price = await get_stock_price(sym)
        await close_stock_position(sym)

        open_trades = await get_open_stock_trades(sym)
        total_pnl = 0.0
        for t in open_trades:
            qty = t.get("quantity") or 0
            entry = t.get("entry_price") or price
            trade_type = t.get("trade_type", "long")
            pnl = (price - entry) * qty if trade_type == "long" else (entry - price) * qty
            total_pnl += pnl
            await _close_st(t["id"], price, round(pnl, 4))

        return {"symbol": sym, "exit_price": price, "realized_pnl": round(total_pnl, 4)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/api/stocks/trades")
async def get_stock_trades_endpoint(
    symbol: Optional[str] = None, limit: int = Query(default=50, ge=1, le=200)
) -> list[dict]:
    return await get_stock_trade_history(
        symbol=symbol.upper() if symbol else None, limit=limit
    )


@router.get("/api/stocks/trades/stats")
async def get_stock_trades_stats_endpoint(symbol: Optional[str] = None) -> dict:
    return await get_stock_trade_stats(symbol=symbol.upper() if symbol else None)
