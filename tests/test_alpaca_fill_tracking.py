"""
Security fix tests:
  (1) Alpaca auto-trader must only log from confirmed fills — never from quote estimates.
  (2) If close (SELL) fails or returns unfilled status, DB trades must remain open.
"""
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from crypto_oracle import autotrader


# ---------------------------------------------------------------------------
# BUY: must read filled_qty / filled_avg_price, not call get_crypto_price
# ---------------------------------------------------------------------------

def test_auto_buy_uses_fill_qty_and_price_not_quote():
    """Entry is logged using exchange-confirmed fill, never a quote estimate."""
    filled = {
        "order_id": "order-1",
        "status": "filled",
        "filled_qty": 0.25,
        "filled_avg_price": 401.5,
        "fees": 1.25,
    }

    with (
        patch.object(autotrader, "get_open_trades", new=AsyncMock(return_value=[])),
        patch.object(autotrader, "log_trade", new=AsyncMock(return_value=7)) as log_trade,
        patch("crypto_oracle.alpaca.client.place_crypto_order", new=AsyncMock(return_value=filled)),
        patch("crypto_oracle.api.websocket.manager.broadcast", new=AsyncMock()),
    ):
        # get_crypto_price must NOT be imported/called — we deleted it from autotrader
        result = asyncio.run(autotrader._auto_buy("BTC", 100.0, 0.8))

    assert result is not None
    assert result["entry_price"] == 401.5
    assert result["quantity"] == 0.25
    assert abs(result["amount_usd"] - 100.375) < 0.01  # qty * price
    assert result["fees"] == 1.25

    call_kwargs = log_trade.call_args.kwargs
    assert call_kwargs["entry_price"] == 401.5
    assert call_kwargs["quantity"] == 0.25
    assert call_kwargs.get("entry_fees") == 1.25


def test_auto_buy_returns_none_when_not_filled():
    """If order status is not 'filled', no DB record is created and None is returned."""
    pending = {
        "order_id": "order-2",
        "status": "pending_new",
    }

    with (
        patch.object(autotrader, "get_open_trades", new=AsyncMock(return_value=[])),
        patch.object(autotrader, "log_trade", new=AsyncMock()) as log_trade,
        patch("crypto_oracle.alpaca.client.place_crypto_order", new=AsyncMock(return_value=pending)),
        patch("crypto_oracle.api.websocket.manager.broadcast", new=AsyncMock()),
    ):
        result = asyncio.run(autotrader._auto_buy("BTC", 100.0, 0.8))

    assert result is None
    log_trade.assert_not_awaited()


def test_auto_buy_returns_none_when_fill_price_missing():
    """If filled_avg_price is 0/None, treat as bad fill and return None."""
    bad_fill = {
        "order_id": "order-3",
        "status": "filled",
        "filled_qty": 0.1,
        "filled_avg_price": 0,
    }

    with (
        patch.object(autotrader, "get_open_trades", new=AsyncMock(return_value=[])),
        patch.object(autotrader, "log_trade", new=AsyncMock()) as log_trade,
        patch("crypto_oracle.alpaca.client.place_crypto_order", new=AsyncMock(return_value=bad_fill)),
        patch("crypto_oracle.api.websocket.manager.broadcast", new=AsyncMock()),
    ):
        result = asyncio.run(autotrader._auto_buy("BTC", 100.0, 0.8))

    assert result is None
    log_trade.assert_not_awaited()


# ---------------------------------------------------------------------------
# SELL: if close fails/unfilled, DB trades must stay open
# ---------------------------------------------------------------------------

def test_auto_sell_leaves_db_trades_open_when_close_fails():
    """If close_crypto_position raises, DB trades are NOT closed."""
    open_trade = {"id": 5, "quantity": 0.25, "entry_price": 400.0}

    with (
        patch.object(autotrader, "get_open_trades", new=AsyncMock(return_value=[open_trade])),
        patch.object(autotrader, "close_trade", new=AsyncMock()) as close_trade,
        patch("crypto_oracle.alpaca.client.close_crypto_position",
              new=AsyncMock(side_effect=RuntimeError("connection error"))),
        patch("crypto_oracle.api.websocket.manager.broadcast", new=AsyncMock()),
    ):
        result = asyncio.run(autotrader._auto_sell("BTC", 0.8))

    assert result is None
    close_trade.assert_not_awaited()


def test_auto_sell_leaves_db_trades_open_when_not_filled():
    """If close result is unfilled status, DB trades are NOT closed."""
    open_trade = {"id": 6, "quantity": 0.25, "entry_price": 400.0}
    unfilled = {"order_id": "order-9", "status": "rejected", "qty": 0.25}

    with (
        patch.object(autotrader, "get_open_trades", new=AsyncMock(return_value=[open_trade])),
        patch.object(autotrader, "close_trade", new=AsyncMock()) as close_trade,
        patch("crypto_oracle.alpaca.client.close_crypto_position", new=AsyncMock(return_value=unfilled)),
        patch("crypto_oracle.api.websocket.manager.broadcast", new=AsyncMock()),
    ):
        result = asyncio.run(autotrader._auto_sell("BTC", 0.8))

    assert result is None
    close_trade.assert_not_awaited()


def test_auto_sell_closes_db_trades_on_confirmed_fill():
    """On a confirmed fill, DB trades are closed with exchange exit price."""
    open_trade = {"id": 7, "quantity": 0.25, "entry_price": 400.0}
    filled = {
        "order_id": "order-10",
        "status": "filled",
        "qty": 0.25,
        "filled_avg_price": 420.0,
        "fees": 0.50,
    }

    with (
        patch.object(autotrader, "get_open_trades", new=AsyncMock(return_value=[open_trade])),
        patch.object(autotrader, "close_trade", new=AsyncMock()) as close_trade,
        patch("crypto_oracle.alpaca.client.close_crypto_position", new=AsyncMock(return_value=filled)),
        patch("crypto_oracle.api.websocket.manager.broadcast", new=AsyncMock()),
    ):
        result = asyncio.run(autotrader._auto_sell("BTC", 0.8))

    assert result is not None
    assert result["exit_price"] == 420.0
    # DB close must use exchange price, not a guess
    close_trade.assert_awaited_once()
    call_args = close_trade.call_args
    assert call_args.args[1] == 420.0   # exit_price
