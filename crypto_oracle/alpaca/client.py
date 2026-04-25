"""Alpaca paper-trading client using the official alpaca-py SDK.

Paper trading base URL: https://paper-api.alpaca.markets
Set ALPACA_PAPER=true (default) to use paper, false for live.
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoLatestQuoteRequest

from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)

# Alpaca crypto pairs use "BTC/USD" format
_PAIR_MAP = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
    "DOGE": "DOGE/USD",
    "AVAX": "AVAX/USD",
    "LTC": "LTC/USD",
    "BCH": "BCH/USD",
    "LINK": "LINK/USD",
    "UNI": "UNI/USD",
    "AAVE": "AAVE/USD",
}


def _get_pair(symbol: str) -> str:
    return _PAIR_MAP.get(symbol.upper(), f"{symbol.upper()}/USD")


def _make_trading_client() -> TradingClient:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in environment")
    paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    return TradingClient(api_key=api_key, secret_key=secret, paper=paper)


def _make_data_client() -> CryptoHistoricalDataClient:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    # Data client works without keys for free crypto data, but use them if available
    if api_key and secret:
        return CryptoHistoricalDataClient(api_key=api_key, secret_key=secret)
    return CryptoHistoricalDataClient()


async def get_account_summary() -> dict[str, Any]:
    """Return account equity, buying power, and crypto value."""
    import asyncio

    def _sync() -> dict:
        client = _make_trading_client()
        account = client.get_account()

        equity = float(account.equity or 0)
        buying_power = float(account.buying_power or 0)
        cash = float(account.cash or 0)
        portfolio_value = float(account.portfolio_value or 0)

        # Sum crypto positions
        positions = client.get_all_positions()
        crypto_value = sum(
            float(p.market_value or 0)
            for p in positions
            if "/" in p.symbol or p.asset_class == "crypto"
        )

        paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
        return {
            "account_id": str(account.id),
            "paper_trading": paper,
            "equity": round(equity, 2),
            "portfolio_value": round(portfolio_value, 2),
            "buying_power": round(buying_power, 2),
            "cash": round(cash, 2),
            "crypto_value": round(crypto_value, 2),
        }

    return await asyncio.to_thread(_sync)


async def get_crypto_positions() -> list[dict[str, Any]]:
    """Return all open crypto positions with P&L."""
    import asyncio

    def _sync() -> list:
        client = _make_trading_client()
        positions = client.get_all_positions()
        result = []
        for p in positions:
            # Alpaca crypto symbols look like "BTCUSD" or "BTC/USD"
            sym = p.symbol.replace("/", "").replace("USD", "")
            qty = float(p.qty or 0)
            avg_entry = float(p.avg_entry_price or 0)
            current_price = float(p.current_price or 0)
            market_value = float(p.market_value or 0)
            unrealized_pl = float(p.unrealized_pl or 0)
            unrealized_plpc = float(p.unrealized_plpc or 0) * 100

            result.append({
                "symbol": sym,
                "quantity": round(qty, 8),
                "average_entry_price": round(avg_entry, 4),
                "current_price": round(current_price, 4),
                "market_value": round(market_value, 2),
                "unrealized_pl": round(unrealized_pl, 2),
                "unrealized_pl_pct": round(unrealized_plpc, 2),
                "side": str(p.side.value) if p.side else "long",
            })
        return result

    return await asyncio.to_thread(_sync)


async def get_crypto_price(symbol: str) -> float:
    """Fetch the latest quote mid-price for a crypto symbol."""
    import asyncio

    def _sync() -> float:
        client = _make_data_client()
        pair = _get_pair(symbol)
        req = CryptoLatestQuoteRequest(symbol_or_symbols=pair)
        quotes = client.get_crypto_latest_quote(req)
        quote = quotes.get(pair)
        if quote is None:
            return 0.0
        ask = float(quote.ask_price or 0)
        bid = float(quote.bid_price or 0)
        return round((ask + bid) / 2, 4) if ask and bid else round(ask or bid, 4)

    return await asyncio.to_thread(_sync)


async def place_crypto_order(
    symbol: str,
    side: str,
    amount_usd: float,
) -> dict[str, Any]:
    """Place a notional (dollar-amount) market order on the paper account.

    Args:
        symbol: e.g. "BTC"
        side: "buy" or "sell"
        amount_usd: dollar amount to trade
    """
    import asyncio

    max_usd = float(os.getenv("ALPACA_MAX_ORDER_USD", "500"))
    if amount_usd > max_usd:
        raise ValueError(
            f"Order amount ${amount_usd:.2f} exceeds ALPACA_MAX_ORDER_USD=${max_usd:.2f}"
        )

    def _sync() -> dict:
        client = _make_trading_client()
        pair = _get_pair(symbol)
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        req = MarketOrderRequest(
            symbol=pair,
            notional=round(amount_usd, 2),
            side=order_side,
            time_in_force=TimeInForce.IOC,  # Immediate-or-cancel for crypto
        )
        order = client.submit_order(req)
        return {
            "order_id": str(order.id),
            "client_order_id": str(order.client_order_id),
            "symbol": str(order.symbol),
            "side": str(order.side.value),
            "notional": float(order.notional or 0),
            "status": str(order.status.value),
            "created_at": str(order.created_at),
        }

    return await asyncio.to_thread(_sync)


async def cancel_order(order_id: str) -> bool:
    """Cancel an open order by ID."""
    import asyncio

    def _sync() -> bool:
        client = _make_trading_client()
        client.cancel_order_by_id(order_id)
        return True

    return await asyncio.to_thread(_sync)


async def get_open_orders() -> list[dict[str, Any]]:
    """Return all open orders."""
    import asyncio

    def _sync() -> list:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = _make_trading_client()
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = client.get_orders(filter=req)
        return [
            {
                "order_id": str(o.id),
                "symbol": str(o.symbol),
                "side": str(o.side.value),
                "notional": float(o.notional or 0),
                "qty": float(o.qty or 0),
                "status": str(o.status.value),
                "created_at": str(o.created_at),
            }
            for o in orders
        ]

    return await asyncio.to_thread(_sync)
