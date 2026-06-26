"""Live BTC market data from public APIs — no API key required, US-accessible."""
from __future__ import annotations

import math

import aiohttp

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
    Uses hourly log-returns so it reflects the current intraday vol regime.
    Sources: Kraken → Coinbase Exchange fallback.
    Falls back to 0.65 on any failure.
    """
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
