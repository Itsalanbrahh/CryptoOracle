"""LSE market data provider — single source for BTC spot, candles, vol, macro.

Reads API key from ~/.hermes/x_credentials.json (``lse_api_key`` field).
All functions fall back to existing public API calls when LSE is unavailable,
so the system never blocks on LSE being down.

Usage:
    from crypto_oracle.kalshi.lse_provider import LSEProvider
    provider = LSEProvider()
    spot = await provider.spot_price()
    candles = await provider.candles("BTC/USD", "1h", limit=100)
    vol = await provider.realized_vol()
    macro = await provider.macro_context()
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any

# ── LSE client (lazy import, may not be installed) ──────────────────────────
try:
    from lse import LSE as _LSE, LSEError as _LSEError
    _HAS_LSE = True
except ImportError:
    _HAS_LSE = False

# ── Public API fallbacks (same as existing market_data.py / base.py) ─────────
import aiohttp
import statistics

# ── Sane bounds ──────────────────────────────────────────────────────────────
_VOL_FLOOR = 0.30
_VOL_CAP = 2.50
_VOL_DEFAULT = 0.65
_CACHE_TTL_SPOT = 15       # seconds before refreshing spot
_CACHE_TTL_CANDLES = 60    # seconds before refreshing candles
_CACHE_TTL_VOL = 120       # seconds before refreshing vol
_CACHE_TTL_MACRO = 300     # seconds before refreshing macro

# ── Credentials ──────────────────────────────────────────────────────────────
_CREDS_PATH = Path.home() / ".hermes" / "x_credentials.json"


def _read_api_key() -> str | None:
    """Read LSE API key from credentials file."""
    if not _CREDS_PATH.exists():
        return None
    try:
        data = json.loads(_CREDS_PATH.read_text())
        return data.get("lse_api_key")
    except Exception:
        return None


def _hourly_vol_from_closes(closes: list[float]) -> float:
    """Annualized vol from hourly close prices."""
    if len(closes) < 4:
        return _VOL_DEFAULT
    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
                   if closes[i - 1] > 0]
    if len(log_returns) < 3:
        return _VOL_DEFAULT
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    hourly_std = math.sqrt(max(0.0, variance))
    annual_vol = hourly_std * math.sqrt(8760)
    return max(_VOL_FLOOR, min(_VOL_CAP, annual_vol))


# ── Public API fallbacks (same logic as existing, but as methods) ─────────────

async def _fallback_spot_price() -> float:
    """Multi-exchange BTC spot (same as existing fetch_spot_price)."""
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
        ('https://api.coinbase.com/v2/prices/BTC-USD/spot',
         lambda d: float(d.get('data', {}).get('amount', 0))),
        ('https://api.kraken.com/0/public/Ticker?pair=XBTUSD',
         lambda d: float(d.get('result', {}).get('XXBTZUSD', {}).get('c', [0])[0])),
        ('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT',
         lambda d: float(d.get('price', 0))),
        ('https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT',
         lambda d: float(d.get('result', {}).get('list', [{}])[0].get('lastPrice', 0))),
    ]

    prices = []
    tasks = [asyncio.ensure_future(_fetch(url, fn)) for url, fn in sources]
    for task in tasks:
        p = await task
        if p and p > 1000:
            prices.append(p)

    if len(prices) < 2:
        for url, fn in [
            ('https://api.coinbase.com/v2/prices/BTC-USD/spot',
             lambda d: float(d.get('data', {}).get('amount', 0))),
            ('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd',
             lambda d: float(d.get('bitcoin', {}).get('usd', 0))),
        ]:
            p = await _fetch(url, fn)
            if p and p > 1000:
                return p
        raise RuntimeError("could not fetch BTC spot price")

    median = statistics.median(prices)
    filtered = [p for p in prices if abs(p / median - 1.0) <= 0.005]
    if len(filtered) < 2:
        filtered = prices
    return statistics.mean(filtered)


async def _fallback_candles(timeframe: str = "1h", limit: int = 100) -> list[dict]:
    """BTC candles from existing public API fallback."""
    import time as _time
    interval_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
    granularity = interval_map.get(timeframe, 3600)

    # Coinbase exchange candles
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
            async with session.get(url, params={"granularity": granularity},
                                   timeout=aiohttp.ClientTimeout(total=12)) as resp:
                resp.raise_for_status()
                rows = await resp.json()
        # rows: [[time, low, high, open, close, vol], ...] newest-first
        result = []
        for row in reversed(rows[-limit:]):
            result.append({
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(row[0])),
                "open": float(row[3]),
                "high": float(row[2]),
                "low": float(row[1]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })
        return result
    except Exception:
        pass

    # Kraken fallback
    interval = interval_map.get(timeframe, 3600)
    kraken_interval = interval // 60  # Kraken uses minutes
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.kraken.com/0/public/OHLC"
            async with session.get(url, params={"pair": "XBTUSD", "interval": kraken_interval},
                                   timeout=aiohttp.ClientTimeout(total=12)) as resp:
                resp.raise_for_status()
                data = await resp.json()
        rows = data.get("result", {}).get("XXBTZUSD", [])
        result = []
        for row in rows[-limit:]:
            result.append({
                "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(int(row[0]))),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[6]),
            })
        return result
    except Exception:
        return []


async def _fallback_history(days: int = 90) -> list[float]:
    """Daily close prices from existing public APIs."""
    try:
        candles = await _fallback_candles("1d", limit=days)
        return [c["close"] for c in candles]
    except Exception:
        return []


async def _fallback_realized_vol(hours: int = 24) -> float:
    """Vol from public API candles."""
    try:
        candles = await _fallback_candles("1h", limit=hours + 1)
        closes = [c["close"] for c in candles]
        return _hourly_vol_from_closes(closes)
    except Exception:
        return _VOL_DEFAULT


# ── LSE Provider ─────────────────────────────────────────────────────────────

_WARNED_NO_LSE = False


class LSEProvider:
    """LSE-backed market data with caching and public-API fallback.

    All methods cache results; use ``force=True`` to bypass cache.
    """

    def __init__(self) -> None:
        global _WARNED_NO_LSE
        self._client: Any = None
        self._api_key: str | None = _read_api_key()

        if not _HAS_LSE:
            if not _WARNED_NO_LSE:
                print("[LSE] lse-data not installed — using public API fallback only")
                _WARNED_NO_LSE = True
        elif not self._api_key:
            if not _WARNED_NO_LSE:
                print("[LSE] No API key in ~/.hermes/x_credentials.json — using fallback")
                _WARNED_NO_LSE = True
        else:
            try:
                self._client = _LSE(api_key=self._api_key)
            except Exception as e:
                print(f"[LSE] Failed to init client: {e} — using fallback")
                self._client = None

        # Cache
        self._spot: float | None = None
        self._spot_time: float = 0
        self._candles_cache: dict[str, tuple[float, list[dict]]] = {}
        self._vol: float | None = None
        self._vol_time: float = 0
        self._macro: dict | None = None
        self._macro_time: float = 0

    @property
    def available(self) -> bool:
        return self._client is not None

    # ── Spot Price ────────────────────────────────────────────────────────────

    async def spot_price(self, force: bool = False) -> float:
        """BTC spot price. LSE → public API fallback."""
        now = time.time()
        if not force and self._spot is not None and (now - self._spot_time) < _CACHE_TTL_SPOT:
            return self._spot

        if self._client:
            try:
                # Quick REST query instead of stream (stream needs to disconnect)
                candles = self._client.candles("BTC/USD", "1m", limit=1, order="desc")
                if candles:
                    self._spot = candles[0]["close"]
                    self._spot_time = now
                    return self._spot
            except Exception:
                pass

        self._spot = await _fallback_spot_price()
        self._spot_time = now
        return self._spot

    # ── Candles ───────────────────────────────────────────────────────────────

    async def candles(self, symbol: str = "BTC/USD", timeframe: str = "1h",
                      limit: int = 100, force: bool = False) -> list[dict]:
        """OHLCV candles. LSE → public API fallback."""
        cache_key = f"{symbol}:{timeframe}"
        now = time.time()
        if not force and cache_key in self._candles_cache:
            cached_time, cached_data = self._candles_cache[cache_key]
            if (now - cached_time) < _CACHE_TTL_CANDLES and len(cached_data) >= limit:
                return cached_data[-limit:]

        if self._client:
            try:
                data = self._client.candles(symbol, timeframe, limit=limit, order="desc")
                # data comes newest-first from LSE; reverse to chronological
                for row in data:
                    if "timestamp" not in row and "date" in row:
                        row["timestamp"] = row["date"]
                self._candles_cache[cache_key] = (now, data)
                return data
            except Exception:
                pass

        result = await _fallback_candles(timeframe, limit)
        if result:
            self._candles_cache[cache_key] = (now, result)
        return result

    # ── Historical Close Prices ───────────────────────────────────────────────

    async def historical_prices(self, days: int = 90, force: bool = False) -> list[float]:
        """Daily close prices for backtesting/strategy."""
        if self._client:
            try:
                data = self._client.candles("BTC/USD", "1d", limit=days, order="desc")
                if data:
                    return [c["close"] for c in reversed(data)]
            except Exception:
                pass
        return await _fallback_history(days)

    # ── Realized Volatility ───────────────────────────────────────────────────

    async def realized_vol(self, hours: int = 24, force: bool = False) -> float:
        """Annualized realized vol from hourly candles. LSE → fallback."""
        now = time.time()
        if not force and self._vol is not None and (now - self._vol_time) < _CACHE_TTL_VOL:
            return self._vol

        if self._client:
            try:
                data = self._client.candles("BTC/USD", "1h", limit=hours + 1, order="desc")
                if data and len(data) >= 4:
                    closes = [c["close"] for c in reversed(data)]
                    self._vol = _hourly_vol_from_closes(closes)
                    self._vol_time = now
                    return self._vol
            except Exception:
                pass

        self._vol = await _fallback_realized_vol(hours)
        self._vol_time = now
        return self._vol

    # ── Funding Rate (still from Bybit — LSE doesn't offer this) ──────────────

    async def funding_rate(self) -> float:
        """BTC perpetual funding rate from Bybit (not available on LSE)."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.bybit.com/v5/market/tickers",
                    params={"category": "linear", "symbol": "BTCUSDT"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            items = data.get("result", {}).get("list", [])
            if items:
                return float(items[0].get("fundingRate", 0.0))
        except Exception:
            pass
        return 0.0

    # ── Macro Context ─────────────────────────────────────────────────────────

    async def macro_context(self, force: bool = False) -> dict:
        """Fed Funds, US10Y, SPY, Gold — feeds KronosMarket conviction."""
        now = time.time()
        if not force and self._macro is not None and (now - self._macro_time) < _CACHE_TTL_MACRO:
            return self._macro

        ctx: dict = {
            "fed_rate": None,
            "us10y": None,
            "spy_1d_pct": None,
            "gold_1d_pct": None,
            "btc_1d_pct": None,
        }

        if self._client:
            try:
                # Fed Funds rate
                ffr = self._client.economics("fdtr")
                if ffr:
                    ctx["fed_rate"] = float(ffr[-1].get("value", 0))

                # SPY 1d change
                spy = self._client.candles("SPY", "1d", limit=2, order="desc")
                if spy and len(spy) >= 2:
                    ctx["spy_1d_pct"] = round((spy[0]["close"] / spy[1]["close"] - 1) * 100, 2)

                # Gold 1d change
                gld = self._client.candles("GLD", "1d", limit=2, order="desc")
                if gld and len(gld) >= 2:
                    ctx["gold_1d_pct"] = round((gld[0]["close"] / gld[1]["close"] - 1) * 100, 2)

                # BTC 1d change
                btc = self._client.candles("BTC/USD", "1d", limit=2, order="desc")
                if btc and len(btc) >= 2:
                    ctx["btc_1d_pct"] = round((btc[0]["close"] / btc[1]["close"] - 1) * 100, 2)

                # US10Y yield
                try:
                    us10y = self._client.bond_yields("US10Y", limit=1)
                    if us10y:
                        ctx["us10y"] = float(us10y[-1].get("close", 0))
                except Exception:
                    pass

            except Exception as e:
                print(f"[LSE/MACRO] Error: {e}")

        self._macro = ctx
        self._macro_time = now
        return ctx

    # ── Multi-asset snapshot ──────────────────────────────────────────────────

    async def asset_snapshot(self) -> dict:
        """BTC, ETH, SPY, QQQ, GLD prices with 1d changes."""
        result = {}
        assets = [
            ("BTC/USD", "crypto"),
            ("ETH/USD", "crypto"),
            ("SPY", "equity"),
            ("QQQ", "equity"),
            ("GLD", "commodity"),
        ]

        for symbol, category in assets:
            try:
                if self._client:
                    data = self._client.candles(symbol, "1d", limit=2, order="desc")
                    if data and len(data) >= 2:
                        current = data[0]["close"]
                        prev = data[1]["close"]
                        chg_pct = round((current / prev - 1) * 100, 2)
                        result[symbol] = {"price": current, "change_pct": chg_pct}
                    elif data:
                        result[symbol] = {"price": data[0]["close"], "change_pct": 0}
            except Exception:
                pass

        return result


# ── Singleton for convenience ─────────────────────────────────────────────────
_provider: LSEProvider | None = None


def get_provider() -> LSEProvider:
    global _provider
    if _provider is None:
        _provider = LSEProvider()
    return _provider
