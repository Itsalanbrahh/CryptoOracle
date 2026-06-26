"""Auto-trading engine: executes orders based on oracle recommendations."""

import os
from typing import Any, Optional

from crypto_oracle.models.db import (
    close_trade,
    get_meta,
    get_open_trades,
    log_trade,
    set_meta,
)
from crypto_oracle.models.signals import MasterRecommendation
from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

_KEY_ENABLED = "auto_trade_enabled"
_KEY_AMOUNT = "auto_trade_amount_usd"
_KEY_THRESHOLD = "auto_trade_confidence_threshold"


async def get_auto_trade_settings() -> dict[str, Any]:
    enabled_raw = await get_meta(_KEY_ENABLED)
    amount_raw = await get_meta(_KEY_AMOUNT)
    threshold_raw = await get_meta(_KEY_THRESHOLD)
    default_threshold = float(os.getenv("ALERT_THRESHOLD", "0.70"))
    default_amount = float(os.getenv("AUTO_TRADE_AMOUNT_USD", "100"))
    return {
        "enabled": enabled_raw == "true" if enabled_raw is not None else False,
        "amount_usd": float(amount_raw) if amount_raw else default_amount,
        "confidence_threshold": float(threshold_raw) if threshold_raw else default_threshold,
    }


async def update_auto_trade_settings(
    enabled: bool, amount_usd: float, confidence_threshold: float
) -> None:
    await set_meta(_KEY_ENABLED, "true" if enabled else "false")
    await set_meta(_KEY_AMOUNT, str(amount_usd))
    await set_meta(_KEY_THRESHOLD, str(confidence_threshold))


async def maybe_auto_trade(rec: MasterRecommendation) -> Optional[dict[str, Any]]:
    """Check recommendation and auto-execute a trade if conditions are met."""
    if os.getenv("SKIP_ALPACA", "false").lower() == "true":
        return None

    settings = await get_auto_trade_settings()
    if not settings["enabled"]:
        return None

    # Use per-symbol threshold/amount from strategy state (master may have tuned it)
    from crypto_oracle.models.db import get_strategy_state
    strategy = await get_strategy_state(rec.symbol)
    threshold  = strategy["confidence_threshold"]
    amount_usd = strategy["auto_trade_amount"]

    if rec.confidence < threshold:
        logger.debug(
            "Auto-trade: %s conf %.0f%% < threshold %.0f%%",
            rec.symbol, rec.confidence * 100, threshold * 100,
        )
        return None

    if rec.action == "BUY":
        return await _auto_buy(rec.symbol, amount_usd, rec.confidence)
    if rec.action == "SELL":
        return await _auto_sell(rec.symbol, rec.confidence)
    return None


async def _auto_buy(symbol: str, amount_usd: float, confidence: float) -> Optional[dict]:
    from crypto_oracle.alpaca.client import get_crypto_price, place_crypto_order
    from crypto_oracle.api.websocket import manager

    open_trades = await get_open_trades(symbol)
    if open_trades:
        logger.info("Auto-trade: skipping BUY %s — already holding %d position(s)", symbol, len(open_trades))
        return None

    try:
        price = await get_crypto_price(symbol)
        result = await place_crypto_order(symbol, "buy", amount_usd)
        qty = amount_usd / price if price > 0 else 0
        trade_id = await log_trade(
            symbol=symbol,
            amount_usd=amount_usd,
            entry_price=price,
            quantity=qty,
            alpaca_order_id=result["order_id"],
            triggered_by="auto",
            confidence=confidence,
        )
        logger.info("Auto-trade BUY: %s $%.2f @ $%.4f (trade_id=%d)", symbol, amount_usd, price, trade_id)
        payload = {
            "action": "BUY", "symbol": symbol, "amount_usd": amount_usd,
            "entry_price": price, "quantity": round(qty, 8),
            "trade_id": trade_id, "triggered_by": "auto",
            "alpaca_order_id": result["order_id"],
        }
        await manager.broadcast({"type": "trade", "data": payload})
        return payload
    except Exception as exc:
        logger.error("Auto-trade BUY failed for %s: %s", symbol, exc, exc_info=True)
        return None


async def _auto_sell(symbol: str, confidence: float) -> Optional[dict]:
    from crypto_oracle.alpaca.client import close_crypto_position, get_crypto_price
    from crypto_oracle.api.websocket import manager

    open_trades = await get_open_trades(symbol)
    if not open_trades:
        logger.info("Auto-trade: skipping SELL %s — no open positions in DB", symbol)
        return None

    try:
        price = await get_crypto_price(symbol)
        try:
            await close_crypto_position(symbol)
        except Exception as exc:
            logger.warning("Auto-trade: Alpaca close_position failed for %s (may be already flat): %s", symbol, exc)

        total_pnl = 0.0
        for trade in open_trades:
            qty = trade["quantity"] or 0
            entry = trade["entry_price"] or price
            pnl = (price - entry) * qty
            total_pnl += pnl
            await close_trade(trade["id"], price, round(pnl, 4))

        total_pnl = round(total_pnl, 4)
        logger.info("Auto-trade SELL: %s @ $%.4f | P&L $%.2f (%d trades closed)", symbol, price, total_pnl, len(open_trades))
        payload = {
            "action": "SELL", "symbol": symbol, "exit_price": price,
            "realized_pnl": total_pnl, "trades_closed": len(open_trades),
            "triggered_by": "auto",
        }
        await manager.broadcast({"type": "trade", "data": payload})
        return payload
    except Exception as exc:
        logger.error("Auto-trade SELL failed for %s: %s", symbol, exc, exc_info=True)
        return None
