from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

import aiohttp

from crypto_oracle.polymarket.clob import get_midpoint, get_orderbook, get_price_history, get_spread
from crypto_oracle.polymarket.models import ParsedMarketQuestion, PolymarketMarket

_PRICE_RE = re.compile(r'\$([0-9][0-9,]*(?:\.[0-9]+)?)(?:\s*([kKmM]))?')


def _price_from_match(match: re.Match[str]) -> float:
    raw = match.group(1).replace(',', '')
    suffix = (match.group(2) or '').lower()
    value = float(raw)
    if suffix == 'k':
        value *= 1000
    elif suffix == 'm':
        value *= 1_000_000
    return value


def parse_market_question(market: PolymarketMarket, spot_price: float | None = None) -> ParsedMarketQuestion:
    text = market.question.lower()
    comparator = 'below' if any(term in text for term in [' below ', ' under ', ' less than ']) else 'above'
    prices = [_price_from_match(m) for m in _PRICE_RE.finditer(market.question)]
    threshold = prices[0] if prices else (spot_price or 0.0)
    days = None
    if market.end_date_iso:
        try:
            expiry = datetime.fromisoformat(market.end_date_iso.replace('Z', '+00:00'))
            days = max(0.0, (expiry - datetime.now(timezone.utc)).total_seconds() / 86400)
        except Exception:
            days = None
    if spot_price is not None and threshold > 0:
        distance = ((spot_price - threshold) / threshold) * 100 if comparator == 'above' else ((threshold - spot_price) / threshold) * 100
    else:
        distance = None
    yes_condition = f'BTC {comparator} ${threshold:,.0f}'
    return ParsedMarketQuestion(
        comparator=comparator,
        threshold_price=threshold,
        reference_price=spot_price,
        expiry_iso=market.end_date_iso,
        days_to_expiry=days,
        distance_to_threshold_pct=distance,
        yes_condition=yes_condition,
    )


async def fetch_spot_price() -> float:
    urls = [
        'https://api.coinbase.com/v2/prices/BTC-USD/spot',
        'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd',
    ]
    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                if 'coinbase' in url:
                    return float(data['data']['amount'])
                return float(data['bitcoin']['usd'])
            except Exception:
                continue
    raise RuntimeError('could not fetch BTC spot price')


async def fetch_spot_history(days: int = 90) -> list[float]:
    async with aiohttp.ClientSession() as session:
        # Kraken public OHLC — reliable, no geo-block, no key needed
        try:
            url = 'https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1440'
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                resp.raise_for_status()
                data = await resp.json()
            rows = data.get('result', {}).get('XXBTZUSD', [])
            closes = [float(row[4]) for row in rows]
            return closes[-days:] if len(closes) > days else closes
        except Exception:
            pass
        # Fallback: Coinbase Exchange
        try:
            url = 'https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400'
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                resp.raise_for_status()
                data = await resp.json()
            # data is [[time, low, high, open, close, volume], ...] newest first
            closes = [float(row[4]) for row in reversed(data)]
            return closes[-days:] if len(closes) > days else closes
        except Exception:
            pass
        # Last resort: CoinGecko
        try:
            url = f'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily'
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return [float(row[1]) for row in data.get('prices', [])]
        except Exception:
            pass
    raise RuntimeError('could not fetch BTC spot history from any source')


def realized_volatility(prices: list[float], lookback: int = 30) -> float:
    if len(prices) < lookback + 1:
        return 0.0
    window = prices[-(lookback + 1):]
    returns = []
    for prev, curr in zip(window, window[1:]):
        if prev > 0 and curr > 0:
            returns.append(math.log(curr / prev))
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(max(0.0, variance))


async def fetch_market_microstructure(market: PolymarketMarket) -> dict[str, Any]:
    yes_token = market.yes_outcome.token_id or ''
    midpoint = await get_midpoint(yes_token) if yes_token else None
    spread = await get_spread(yes_token) if yes_token else None
    book = await get_orderbook(yes_token) if yes_token else {}
    history = await get_price_history(market.condition_id, interval='1w', fidelity=60) if market.condition_id else []
    bids = book.get('bids', []) if isinstance(book, dict) else []
    asks = book.get('asks', []) if isinstance(book, dict) else []

    def _sum_side(rows: list[dict]) -> float:
        total = 0.0
        for row in rows[:8]:
            try:
                price = float(row.get('price') or row.get('p') or 0)
                size = float(row.get('size') or row.get('s') or 0)
                total += price * size
            except Exception:
                continue
        return total

    bid_notional = _sum_side(bids)
    ask_notional = _sum_side(asks)
    if bid_notional + ask_notional > 0:
        imbalance = (bid_notional - ask_notional) / (bid_notional + ask_notional)
    else:
        imbalance = 0.0
    hist_prices = []
    for row in history:
        try:
            hist_prices.append(float(row.get('p') or row.get('price') or row.get('y')))
        except Exception:
            continue
    momentum = 0.0
    if len(hist_prices) >= 2 and hist_prices[0] > 0:
        momentum = (hist_prices[-1] - hist_prices[0]) / hist_prices[0]
    return {
        'midpoint_yes': midpoint,
        'spread_yes': spread,
        'bid_notional': round(bid_notional, 2),
        'ask_notional': round(ask_notional, 2),
        'imbalance': round(imbalance, 4),
        'history_points': len(hist_prices),
        'history_momentum': round(momentum, 4),
    }
