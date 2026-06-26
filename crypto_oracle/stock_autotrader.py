"""Stock auto-trading engine: executes long/short orders on oracle recommendations."""

from __future__ import annotations

import os
from typing import Any, Optional

from crypto_oracle.models.db import (
    close_stock_trade,
    get_open_stock_trades,
    get_strategy_state,
    log_stock_trade,
)
from crypto_oracle.models.signals import MasterRecommendation
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)


async def maybe_stock_auto_trade(rec: MasterRecommendation) -> Optional[dict[str, Any]]:
    """Execute a long or short stock trade if conditions are met."""
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        return None

    from crypto_oracle.autotrader import get_auto_trade_settings
    settings = await get_auto_trade_settings()
    if not settings["enabled"]:
        return None

    strategy = await get_strategy_state(rec.symbol)
    threshold = strategy["confidence_threshold"]
    amount_usd = strategy["auto_trade_amount"]

    if rec.confidence < threshold:
        logger.debug(
            "Stock auto-trade: %s conf %.0f%% < threshold %.0f%%",
            rec.symbol, rec.confidence * 100, threshold * 100,
        )
        return None

    if rec.action == "BUY":
        return await _auto_long(rec.symbol, amount_usd, rec.confidence)
    if rec.action == "SELL":
        return await _auto_short_or_close(rec.symbol, amount_usd, rec.confidence)
    return None


async def _auto_long(symbol: str, amount_usd: float, confidence: float) -> Optional[dict]:
    """Go long — buy shares."""
    from crypto_oracle.alpaca.client import get_stock_price, place_stock_order, is_market_open
    from crypto_oracle.api.websocket import manager

    if not await is_market_open():
        logger.info("Stock auto-trade: market closed, skipping long %s", symbol)
        return None

    open_trades = await get_open_stock_trades(symbol)
    long_trades = [t for t in open_trades if t["trade_type"] == "long"]
    if long_trades:
        logger.info("Stock auto-trade: already long %s, skipping", symbol)
        return None

    # Close any short positions first before going long
    short_trades = [t for t in open_trades if t["trade_type"] == "short"]
    if short_trades:
        logger.info("Stock auto-trade: covering short before going long on %s", symbol)
        await _cover_position(symbol, short_trades)

    try:
        price = await get_stock_price(symbol)
        result = await place_stock_order(symbol, "buy", amount_usd)
        qty = amount_usd / price if price > 0 else 0
        trade_id = await log_stock_trade(
            symbol=symbol, trade_type="long", amount_usd=amount_usd,
            entry_price=price, quantity=qty,
            alpaca_order_id=result["order_id"], triggered_by="auto", confidence=confidence,
        )
        logger.info("Stock auto-trade LONG: %s $%.2f @ $%.4f (id=%d)", symbol, amount_usd, price, trade_id)
        payload = {
            "action": "BUY", "symbol": symbol, "trade_type": "long",
            "amount_usd": amount_usd, "entry_price": price, "quantity": round(qty, 6),
            "trade_id": trade_id, "triggered_by": "auto",
        }
        await manager.broadcast({"type": "stock_trade", "data": payload})
        return payload
    except Exception as exc:
        logger.error("Stock auto-trade LONG failed for %s: %s", symbol, exc, exc_info=True)
        return None


async def _auto_short_or_close(symbol: str, amount_usd: float, confidence: float) -> Optional[dict]:
    """Sell signal: close long if open, then go short."""
    from crypto_oracle.alpaca.client import get_stock_price, place_stock_order, is_market_open
    from crypto_oracle.api.websocket import manager

    if not await is_market_open():
        logger.info("Stock auto-trade: market closed, skipping short %s", symbol)
        return None

    open_trades = await get_open_stock_trades(symbol)
    long_trades = [t for t in open_trades if t["trade_type"] == "long"]
    short_trades = [t for t in open_trades if t["trade_type"] == "short"]

    # Close any existing long position first
    if long_trades:
        logger.info("Stock auto-trade: closing long before shorting %s", symbol)
        result = await _cover_position(symbol, long_trades)
        if result:
            return result  # Return the close trade info

    # Skip if already short
    if short_trades:
        logger.info("Stock auto-trade: already short %s, skipping", symbol)
        return None

    # Go short
    try:
        price = await get_stock_price(symbol)
        result = await place_stock_order(symbol, "sell", amount_usd)
        qty = amount_usd / price if price > 0 else 0
        trade_id = await log_stock_trade(
            symbol=symbol, trade_type="short", amount_usd=amount_usd,
            entry_price=price, quantity=qty,
            alpaca_order_id=result["order_id"], triggered_by="auto", confidence=confidence,
        )
        logger.info("Stock auto-trade SHORT: %s $%.2f @ $%.4f (id=%d)", symbol, amount_usd, price, trade_id)
        payload = {
            "action": "SELL", "symbol": symbol, "trade_type": "short",
            "amount_usd": amount_usd, "entry_price": price, "quantity": round(qty, 6),
            "trade_id": trade_id, "triggered_by": "auto",
        }
        await manager.broadcast({"type": "stock_trade", "data": payload})
        return payload
    except Exception as exc:
        logger.error("Stock auto-trade SHORT failed for %s: %s", symbol, exc, exc_info=True)
        return None


async def _cover_position(symbol: str, trades: list[dict]) -> Optional[dict]:
    """Close/cover an existing position (long or short)."""
    from crypto_oracle.alpaca.client import close_stock_position, get_stock_price
    from crypto_oracle.api.websocket import manager

    try:
        price = await get_stock_price(symbol)
        try:
            await close_stock_position(symbol)
        except Exception as exc:
            logger.warning("close_stock_position failed for %s (may already be flat): %s", symbol, exc)

        total_pnl = 0.0
        for trade in trades:
            qty = trade.get("quantity") or 0
            entry = trade.get("entry_price") or price
            trade_type = trade.get("trade_type", "long")
            if trade_type == "long":
                pnl = (price - entry) * qty
            else:
                pnl = (entry - price) * qty
            total_pnl += pnl
            await close_stock_trade(trade["id"], price, round(pnl, 4))

        total_pnl = round(total_pnl, 4)
        trade_type = trades[0].get("trade_type", "long") if trades else "long"
        logger.info(
            "Stock closed %s %s @ $%.4f | P&L $%.2f",
            trade_type, symbol, price, total_pnl,
        )
        payload = {
            "action": "CLOSE", "symbol": symbol, "trade_type": trade_type,
            "exit_price": price, "realized_pnl": total_pnl,
            "trades_closed": len(trades), "triggered_by": "auto",
        }
        await manager.broadcast({"type": "stock_trade", "data": payload})
        return payload
    except Exception as exc:
        logger.error("_cover_position failed for %s: %s", symbol, exc, exc_info=True)
        return None
