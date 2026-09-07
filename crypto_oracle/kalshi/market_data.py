"""Live BTC market data — LSE-first, public API fallback."""
from __future__ import annotations

import asyncio
import math
import statistics

import aiohttp

# LSE provider (wraps lse-data with caching + graceful fallback)
from .lse_provider import get_provider as _get_lse

# Sane bounds: don't let noisy short windows feed absurd vol into GBM.
_VOL_FLOOR = 0.30   # 30% annualized
_VOL_CAP = 2.50     # 250% annualized
_VOL_DEFAULT = 0.65 # BTC long-run average — used on any fetch failure


def _hourly_vol_from_closes(closes: list[float]) -> float:
    if len(closes) < 4:
        return _VOL_DEFAULT
    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(log_returns) < 3:
        return _VOL_DEFAULT
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    hourly_std = math.sqrt(max(0.0, variance))
    annual_vol = hourly_std * math.sqrt(8760)  # crypto 24/7 → 8760h/yr
    return max(_VOL_FLOOR, min(_VOL_CAP, annual_vol))


async def fetch_realized_vol(hours: int = 24) -> float:
    """
    Realized volatility from hourly BTC candles, annualized.
    Uses LSE provider (lse-data) → Kraken → Coinbase fallback.
    """
    try:
        prov = _get_lse()
        return await prov.realized_vol(hours=hours)
    except Exception:
        pass

    limit = hours + 1
    async with aiohttp.ClientSession() as session:
        # Primary: Kraken public OHLC (interval=60 = 1h, no geo-block)
        try:
            url = "https://api.kraken.com/0/public/OHLC"
            async with session.get(url, params={"pair": "XBTUSD", "interval": 60},
                                   timeout=aiohttp.ClientTimeout(total=12)) as resp:
                resp.raise_for_status()
                data = await resp.json()
            rows = data.get("result", {}).get("XXBTZUSD", [])
            closes = [float(row[4]) for row in rows[-limit:]]
            return _hourly_vol_from_closes(closes)
        except Exception:
            pass

        # Fallback: Coinbase Exchange 1h candles
        try:
            url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
            async with session.get(url, params={"granularity": 3600},
                                   timeout=aiohttp.ClientTimeout(total=12)) as resp:
                resp.raise_for_status()
                rows = await resp.json()
            # rows: [[time, low, high, open, close, vol], ...] newest-first
            closes = [float(row[4]) for row in reversed(rows[-limit:])]
            return _hourly_vol_from_closes(closes)
        except Exception:
            pass

    return _VOL_DEFAULT


async def fetch_funding_rate() -> float:
    """
    Latest BTC perpetual funding rate from Bybit (BTCUSDT linear).
    Positive = longs paying shorts (crowded long = mild bearish lean).
    Negative = shorts paying longs (crowded short = mild bullish lean).
    Returns 0.0 on any failure (neutral — no tilt applied).
    """
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params={"category": "linear", "symbol": "BTCUSDT"},
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
        items = data.get("result", {}).get("list", [])
        if items:
            return float(items[0].get("fundingRate", 0.0))
        return 0.0
    except Exception:
        return 0.0


def funding_tilt(funding_rate_8h: float) -> float:
    """
    Convert 8h funding rate to a directional tilt in [-0.10, +0.10].

    Normal funding is ~0.01% per 8h (market noise). We ignore that and only
    respond to excess crowding:
      - Strong positive funding (longs very crowded) → bearish lean → negative tilt
      - Strong negative funding (shorts very crowded) → bullish lean → positive tilt

    Uses tanh for smooth scaling so extreme prints don't hard-flip the signal.
    The 0.0003 denominator means ±0.03% excess = ±0.84 tanh input ≈ ±0.083 tilt.
    """
    NEUTRAL = 0.0001   # 0.01% per 8h — ignore this as noise
    excess = funding_rate_8h - NEUTRAL
    return -math.tanh(excess / 0.0003) * 0.10


async def fetch_funding_rate_multi() -> float:
    """
    Multi-exchange BTC perpetual funding rate average.
    Fetches from Bybit, Binance, and OKX simultaneously,
    rejects outliers >2x the median, averages the rest.
    Returns 0.0 on failure (neutral).
    """
    async def _fetch(url: str, parse_fn) -> float | None:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            return parse_fn(data)
        except Exception:
            return None

    sources = [
        ("bybit", "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT",
         lambda d: float(d.get("result", {}).get("list", [{}])[0].get("fundingRate", 0))),
        ("binance", "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
         lambda d: float(d.get("lastFundingRate", 0))),
        ("okx", "https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP",
         lambda d: float(d.get("data", [{}])[0].get("fundingRate", 0))),
    ]

    rates: list[float] = []
    tasks = [asyncio.ensure_future(_fetch(url, fn)) for _, url, fn in sources]
    for label, task in zip(["bybit", "binance", "okx"], tasks):
        r = await task
        if r is not None:
            rates.append(r)

    if len(rates) < 2:
        return rates[0] if rates else 0.0

    med = statistics.median(rates)
    filtered = [r for r in rates if abs(r) <= abs(med) * 2 + 0.0005]
    return sum(filtered) / len(filtered) if filtered else med


async def fetch_order_book_depth() -> dict:
    """
    BTC spot order book depth from Kraken (US-accessible).
    Returns bid/ask imbalance ratio, total bid volume, and total ask volume.
    Useful for detecting thin liquidity before snap moves.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.kraken.com/0/public/Depth?pair=XBTUSD&count=50",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception:
        return {"bid_vol": 0, "ask_vol": 0, "imbalance": 0.0, "bid_ask_ratio": 1.0}

    pair = data.get("result", {}).get("XXBTZUSD", {})
    bids = pair.get("bids", [])
    asks = pair.get("asks", [])
    bid_vol = sum(float(b[1]) for b in bids[:50])
    ask_vol = sum(float(a[1]) for a in asks[:50])
    total = bid_vol + ask_vol
    imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
    ratio = bid_vol / ask_vol if ask_vol > 0 else 1.0
    return {
        "bid_vol": round(bid_vol, 4),
        "ask_vol": round(ask_vol, 4),
        "imbalance": round(imbalance, 4),
        "bid_ask_ratio": round(ratio, 4),
    }


async def fetch_basis_signal() -> dict:
    """
    Cross-exchange BTC basis: how far each exchange's spot price is from the
    multi-exchange median. Detects exchange drift that could signal arb activity
    or a real move starting on one venue first.
    Returns {exchange: basis_bps, ...} and a summary.
    """
    import re

    async def _fetch_spot(url: str, parse_fn) -> float | None:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            return parse_fn(data)
        except Exception:
            return None

    sources: list[tuple[str, str, callable]] = [
        ("coinbase", "https://api.coinbase.com/v2/prices/BTC-USD/spot",
         lambda d: float(d.get("data", {}).get("amount", 0))),
        ("kraken", "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
         lambda d: float(d.get("result", {}).get("XXBTZUSD", {}).get("c", [0])[0])),
        ("binance", "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
         lambda d: float(d.get("price", 0))),
        ("bybit", "https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT",
         lambda d: float(d.get("result", {}).get("list", [{}])[0].get("lastPrice", 0))),
    ]

    prices: dict[str, float] = {}
    tasks = [asyncio.ensure_future(_fetch_spot(url, fn)) for _, url, fn in sources]
    for label, task in zip([s[0] for s in sources], tasks):
        price = await task
        if price and price > 1000:
            prices[label] = price

    if len(prices) < 2:
        return {"prices": prices, "median": 0, "max_spread_bps": 0, "exchanges": {}}

    med = statistics.median(prices.values())
    max_spread = max(prices.values()) - min(prices.values())
    max_spread_bps = round((max_spread / med) * 10000, 1)

    basis = {}
    for label, p in prices.items():
        bps = round((p / med - 1.0) * 10000, 1)
        basis[label] = bps

    return {
        "prices": {k: round(v, 1) for k, v in prices.items()},
        "median": round(med, 1),
        "max_spread_bps": max_spread_bps,
        "exchanges": basis,
    }
