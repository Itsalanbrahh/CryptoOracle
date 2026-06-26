"""
KronosMarketAgent — deep learning BTC price forecaster.

Loads the Kronos foundation model (AAAI 2026), fetches ~60 days of hourly
BTC OHLCV data, forecasts the next 24 hours, and scores Kalshi markets
by comparing the forecasted price path to the market's strike price.

The forecast is then adjusted by real-time market microstructure signals
(funding rate, open interest trend, volume anomaly) so Kronos has context
beyond just price history.

Key capabilities:
  - Forecast BTC price for the next N hours (up to 24)
  - Score a market: does the forecast agree with the YES or NO outcome?
  - Confidence: based on forecast consistency (low direction changes = high confidence)
  - Microstructure adjustment: funding rate, OI trend, volume anomaly modify score & confidence
  - Caches forecasts to avoid re-running on every market in a single scan
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add Kronos to path
_KRONOS_PATH = Path("/Users/alanruelas/vendor/kronos")
if _KRONOS_PATH.exists():
    sys.path.insert(0, str(_KRONOS_PATH))

# ── Cache ──────────────────────────────────────────────────────────────────

_FORECAST_CACHE_PATH = Path.home() / ".hermes" / "state" / "kronos_forecast_cache.json"
_CACHE_TTL_SECONDS = 1800  # Re-forecast every 30 min (matches scan cadence)


def _load_cached_forecast() -> dict | None:
    """Load cached forecast if it's still fresh."""
    if _FORECAST_CACHE_PATH.exists():
        try:
            data = json.loads(_FORECAST_CACHE_PATH.read_text())
            generated = datetime.fromisoformat(data.get("generated_at", "2000-01-01"))
            age = (datetime.now(timezone.utc) - generated).total_seconds()
            if age < _CACHE_TTL_SECONDS:
                return data
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    return None


def _save_forecast_cache(data: dict) -> None:
    _FORECAST_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    _FORECAST_CACHE_PATH.write_text(json.dumps(data, indent=2, default=str))


# ── Model loading (lazy singleton) ────────────────────────────────────────

_model = None
_tokenizer = None
_predictor = None
_device = None


def _ensure_model():
    """Lazy-load Kronos model once per process."""
    global _model, _tokenizer, _predictor, _device

    if _predictor is not None:
        return _predictor

    import torch
    from model import get_model_class, KronosPredictor

    _device = "mps" if torch.backends.mps.is_available() else "cpu"

    _tokenizer = get_model_class("kronos_tokenizer").from_pretrained(
        "NeoQuasar/Kronos-Tokenizer-base"
    ).to(_device).eval()

    _model = get_model_class("kronos").from_pretrained(
        "NeoQuasar/Kronos-small"
    ).to(_device).eval()

    _predictor = KronosPredictor(_model, _tokenizer, device=_device)
    return _predictor


# ── Forecast ──────────────────────────────────────────────────────────────


async def _fetch_btc_data(days: int = 60) -> pd.DataFrame:
    """Fetch hourly BTC candles and return a DataFrame for Kronos.

    Uses 60 days of data (up from 21) for better model context.
    """
    # Import here to avoid circular imports at module level
    from crypto_oracle.kalshi.backtest import fetch_historical_btc

    candles = await fetch_historical_btc(days=days)
    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["ts"])
    df = df.set_index("timestamp")
    feed = df[["open", "high", "low", "close", "volume"]].copy()
    feed.columns = ["open", "high", "low", "close", "volume"]
    return feed


async def run_forecast(pred_len: int = 24) -> dict:
    """Run Kronos forecast for the next N hours.

    Returns dict with:
      - hourly_prices: list of {hour, close, change_pct}
      - direction: "BULLISH", "BEARISH", or "NEUTRAL"
      - conviction: float 0-1 (how consistent the forecast is)
      - generated_at: ISO timestamp
    """
    # Check cache first
    cached = _load_cached_forecast()
    if cached and cached.get("pred_len") == pred_len:
        return cached

    predictor = _ensure_model()
    feed = await _fetch_btc_data(days=21)
    last_close = float(feed["close"].iloc[-1])

    x_ts = feed.index.to_series()
    last_ts = feed.index[-1]
    y_ts = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=pred_len, freq="h")
    y_ts = pd.Series(y_ts)

    result = predictor.predict(feed, x_ts, y_ts, pred_len=pred_len, sample_count=10)

    # Build output
    hourly = []
    closes = result["close"].values
    for i in range(pred_len):
        change_pct = float((closes[i] - last_close) / last_close * 100)
        hourly.append({
            "hour": i + 1,
            "close": round(float(closes[i]), 2),
            "change_pct": round(change_pct, 2),
        })

    # Direction and conviction
    up_count = sum(1 for h in hourly if h["change_pct"] > 0)
    down_count = pred_len - up_count

    if up_count > down_count * 2:
        direction = "BULLISH"
    elif down_count > up_count * 2:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    # Conviction: lower direction changes = higher conviction
    dir_changes = 0
    prev_dir = "UP" if closes[0] > last_close else "DOWN"
    for c in closes[1:]:
        d = "UP" if c > last_close else "DOWN"
        if d != prev_dir:
            dir_changes += 1
        prev_dir = d

    # Max possible changes = pred_len - 1. Conviction = 1 - (changes / max_changes)
    max_changes = pred_len - 1
    conviction = 1.0 - (dir_changes / max_changes) if max_changes > 0 else 0.5

    # Also compute average direction magnitude
    avg_change = sum(abs(h["change_pct"]) for h in hourly) / len(hourly)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "last_close": last_close,
        "pred_len": pred_len,
        "direction": direction,
        "conviction": round(conviction, 3),
        "avg_magnitude_pct": round(avg_change, 2),
        "up_hours": up_count,
        "down_hours": down_count,
        "direction_changes": dir_changes,
        "hourly_forecast": hourly,
    }

    _save_forecast_cache(output)
    return output


# ── Market scoring ─────────────────────────────────────────────────────────


# ── Microstructure data (funding rate, OI, volume anomaly) ──────────────────


async def _fetch_funding_rate() -> float:
    """Latest BTC perpetual funding rate from Bybit (public, no key).

    Positive = longs paying shorts (crowded long = mild bearish lean).
    Negative = shorts paying longs (crowded short = mild bullish lean).
    Returns 0.0 on any failure (neutral).
    """
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            url = "https://api.bybit.com/v5/market/tickers"
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


async def _fetch_open_interest_trend() -> dict:
    """Fetch BTC OI history (48h) from Bybit public API.

    Returns dict with:
      - current_oi_usd: latest OI in USD
      - oi_change_pct: % change over last 24h
      - oi_with_price: "up", "down", "mixed", or "divergent" — OI vs price direction
      - trend_24h: "rising", "falling", "stable"
    Returns neutral values on any failure.
    """
    default = {
        "current_oi_usd": 0.0,
        "oi_change_pct": 0.0,
        "oi_with_price": "mixed",
        "trend_24h": "stable",
    }
    try:
        import aiohttp
        import math
        async with aiohttp.ClientSession() as session:
            url = "https://api.bybit.com/v5/market/open-interest"
            async with session.get(url, params={
                "category": "linear", "symbol": "BTCUSDT",
                "intervalTime": "1h", "limit": 48,
            }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
        items = data.get("result", {}).get("list", [])
        if not items or len(items) < 2:
            return default

        # Items newest-first: [{timestamp, openInterest}, ...]
        oi_values = [float(item["openInterest"]) for item in items]
        current_oi = oi_values[0]
        oi_24h_ago = oi_values[min(23, len(oi_values) - 1)]
        oi_change_pct = ((current_oi - oi_24h_ago) / oi_24h_ago * 100) if oi_24h_ago > 0 else 0.0

        # Rough price proxy from first/last OI (OI data doesn't carry price)
        # We use: OI increasing = money coming in, OI decreasing = money leaving
        if abs(oi_change_pct) < 2.0:
            trend = "stable"
        elif oi_change_pct > 0:
            trend = "rising"
        else:
            trend = "falling"

        return {
            "current_oi_usd": round(current_oi, 2),
            "oi_change_pct": round(oi_change_pct, 2),
            "oi_with_price": "mixed",  # we can't compare to price without fetching price here
            "trend_24h": trend,
        }
    except Exception:
        return default


async def _fetch_volume_anomaly() -> dict:
    """Check if current hourly volume is anomalous vs the 7-day rolling average.

    Uses the same OHLCV data the Kronos model fetches (Kraken hourly).
    Returns dict with:
      - recent_vol: last hour's volume
      - avg_vol: 7-day average hourly volume
      - ratio: recent_vol / avg_vol
      - anomaly: "high", "normal", "low", or "unknown"
    """
    default = {"recent_vol": 0.0, "avg_vol": 0.0, "ratio": 1.0, "anomaly": "unknown"}
    try:
        from crypto_oracle.kalshi.backtest import fetch_historical_btc
        candles = await fetch_historical_btc(days=7)
        volumes = [c["volume"] for c in candles if c.get("volume", 0) > 0]
        if len(volumes) < 24:
            return default
        recent = volumes[-1]
        avg = sum(volumes[:-1]) / (len(volumes) - 1) if len(volumes) > 1 else recent
        ratio = recent / avg if avg > 0 else 1.0
        if ratio > 1.5:
            anomaly = "high"
        elif ratio < 0.5:
            anomaly = "low"
        else:
            anomaly = "normal"
        return {
            "recent_vol": round(recent, 2),
            "avg_vol": round(avg, 2),
            "ratio": round(ratio, 2),
            "anomaly": anomaly,
        }
    except Exception:
        return default


async def fetch_market_microstructure() -> dict:
    """Fetch all microstructure signals in parallel.

    Returns dict with funding_rate, oi, volume, and ready-to-use modifiers.
    Cached at module level for the duration of one scan.
    """
    fund, oi, vol = await asyncio.gather(
        _fetch_funding_rate(),
        _fetch_open_interest_trend(),
        _fetch_volume_anomaly(),
    )
    return {
        "funding_rate": fund,
        "open_interest": oi,
        "volume_anomaly": vol,
    }


def _calculate_micro_adjustments(micro: dict, forecast_direction: str) -> dict:
    """Convert microstructure signals into score and confidence modifiers.

    Returns:
      score_adj: ±0.0 to ±0.20 additive adjustment to the raw score
      conf_mult: 0.50 to 1.30 multiplier on confidence
      signals: list of strings explaining what was detected
    """
    score_adj = 0.0
    conf_mult = 1.0
    signals = []

    # ── 1. Funding rate adjustment ──────────────────────────────────────
    fund = micro.get("funding_rate", 0.0)
    if isinstance(fund, (int, float)):
        fr = float(fund)
    elif isinstance(fund, dict):
        fr = fund.get("funding_rate", 0.0) if "funding_rate" in fund else 0.0
    else:
        fr = 0.0

    # Normal funding ~0.01% per 8h. Only react to excess.
    NEUTRAL_FUNDING = 0.0001
    excess = fr - NEUTRAL_FUNDING
    if abs(excess) > 0.0002:  # >0.02% excess = notable
        # Positive excess (longs crowded) → bearish pressure on score
        # Negative excess (shorts crowded) → bullish pressure on score
        funding_signal = excess / 0.0005  # ±0.05% excess → ±0.10 score adj
        funding_signal = max(-0.15, min(0.15, funding_signal))

        if forecast_direction == "BULLISH" and funding_signal < -0.05:
            # Forecast says up but market is crowded long → reduce conviction
            conf_mult *= 0.75
            signals.append(f"funding={fr:.5f} (crowded long, -25% conf)")
        elif forecast_direction == "BEARISH" and funding_signal > 0.05:
            # Forecast says down but shorts are crowded → reduce conviction
            conf_mult *= 0.75
            signals.append(f"funding={fr:.5f} (crowded short, -25% conf)")
        else:
            # Funding supports the direction → add to score
            score_adj += funding_signal * 0.3
            conf_mult *= 1.10
            side_label = "bearish" if funding_signal < 0 else "bullish"
            signals.append(f"funding={fr:.5f} ({side_label} tilt, +10% conf)")

    # ── 2. Open interest trend ─────────────────────────────────────────-
    oi = micro.get("open_interest", {})
    if isinstance(oi, dict):
        oi_trend = oi.get("trend_24h", "stable")
        oi_change = oi.get("oi_change_pct", 0.0)

        if oi_trend == "rising" and abs(oi_change) > 5.0:
            # OI surging → real money entering → confirmed trend
            conf_mult *= 1.15
            signals.append(f"OI={oi_change:+.1f}% (surging, +15% conf)")
        elif oi_trend == "falling" and abs(oi_change) > 5.0:
            # OI collapsing → money leaving → weak conviction
            conf_mult *= 0.80
            signals.append(f"OI={oi_change:+.1f}% (collapsing, -20% conf)")

    # ── 3. Volume anomaly ────────────────────────────────────────────────
    vol = micro.get("volume_anomaly", {})
    if isinstance(vol, dict):
        anomaly = vol.get("anomaly", "normal")
        ratio = vol.get("ratio", 1.0)

        if anomaly == "high" and ratio > 2.0:
            # Extremely high volume → real institutional move → strong conviction
            conf_mult *= 1.20
            signals.append(f"volume={ratio:.1f}x avg (spike, +20% conf)")
        elif anomaly == "high":
            conf_mult *= 1.10
            signals.append(f"volume={ratio:.1f}x avg (elevated, +10% conf)")
        elif anomaly == "low":
            conf_mult *= 0.70
            signals.append(f"volume={ratio:.1f}x avg (thin, -30% conf)")

    return {
        "score_adj": round(score_adj, 4),
        "conf_mult": round(conf_mult, 4),
        "signals": signals,
    }


async def score_market(
    strike: float,
    hours_to_expiry: float,
    spot: float,
    forecast: dict | None = None,
    microstructure: dict | None = None,
) -> dict:
    """Score a Kalshi binary market using the Kronos forecast + microstructure.

    Considers:
      1. Kronos OHLCV price forecast (the base prediction)
      2. Funding rate (crowded long/short → dampen or boost conviction)
      3. Open interest trend (surging OI = real money, collapsing OI = weak conviction)
      4. Volume anomaly (high volume = real, low volume = thin / fake out)

    Returns:
      - score: -1 (strong NO) to +1 (strong YES)
      - confidence: 0-1 (adjusted by microstructure)
      - reasoning: str (includes microstructure signals when active)
    """
    if forecast is None:
        forecast = await run_forecast(pred_len=24)

    direction = forecast.get("direction", "NEUTRAL")
    conviction = forecast.get("conviction", 0.0)
    hourly = forecast.get("hourly_forecast", [])

    # Fetch microstructure if not provided
    if microstructure is None:
        microstructure = await fetch_market_microstructure()

    # Find the forecasted price at the market's expiry hour
    expiry_hour = int(min(hours_to_expiry, 24))
    forecast_at_expiry = None
    for h in hourly:
        if h["hour"] == expiry_hour:
            forecast_at_expiry = h
            break
    if forecast_at_expiry is None and hourly:
        forecast_at_expiry = hourly[-1]  # use furthest available

    if forecast_at_expiry is None:
        return {"score": 0.0, "confidence": 0.0, "reasoning": "No forecast available"}

    forecast_price = forecast_at_expiry["close"]

    # Does the forecasted price support YES or NO?
    if forecast_price > strike:
        raw_score = min(1.0, (forecast_price - strike) / strike * 10)
        score = raw_score
        side = "YES"
    elif forecast_price < strike:
        raw_score = min(1.0, (strike - forecast_price) / strike * 10)
        score = -raw_score
        side = "NO"
    else:
        score = 0.0
        side = "NEUTRAL"

    # Confidence: conviction * distance from strike
    strike_dist = abs(forecast_price - strike) / strike * 100
    confidence = conviction * min(1.0, strike_dist * 2)
    confidence = max(0.0, min(1.0, confidence))

    # ── Apply microstructure adjustments ──────────────────────────────────────
    adjustments = _calculate_micro_adjustments(microstructure, direction)
    score = max(-1.0, min(1.0, score + adjustments["score_adj"]))
    confidence = max(0.0, min(1.0, confidence * adjustments["conf_mult"]))
    micro_signals = adjustments["signals"]

    reasoning = (
        f"Kronos forecasts BTC→${forecast_price:,.0f} in {expiry_hour}h "
        f"({side}, strike=${strike:,.0f}, dist={strike_dist:.2f}%, "
        f"conviction={conviction:.2f})"
    )
    if micro_signals:
        reasoning += " | " + ", ".join(micro_signals)

    return {
        "score": round(score, 4),
        "confidence": round(confidence, 4),
        "reasoning": reasoning,
        "forecast_price": forecast_price,
        "forecast_direction": direction,
        "forecast_conviction": conviction,
        "microstructure": {
            "funding_rate": microstructure.get("funding_rate", 0.0),
            "open_interest": microstructure.get("open_interest", {}),
            "volume_anomaly": microstructure.get("volume_anomaly", {}),
            "adjustments": adjustments,
        },
    }


# ── KronosMarketAgent ─────────────────────────────────────────────────────


class KronosMarketAgent:
    """Agent that uses Kronos foundation model for BTC price forecasting."""

    name = "KronosMarket"

    def __init__(self):
        self._forecast_cache: dict | None = None

    async def run(self, market, **kwargs) -> dict:
        """Run Kronos on a single market. Returns {score, confidence, reasoning} dict.

        Args:
            market: A KalshiMarket or compatible object
            kwargs: May contain kalshi=Ctx(strike, hours_to_expiry, spot_price)
        """
        # Extract Kalshi context from kwargs or market object
        kalshi = kwargs.get("kalshi")
        if kalshi is not None:
            strike = kalshi.strike
            hours_to_expiry = kalshi.hours_to_expiry
            spot = kalshi.spot_price
        else:
            strike = getattr(market, "strike", 0)
            hours_to_expiry = getattr(market, "hours_to_expiry", 24)
            spot = getattr(market, "spot_price", 0)

        if strike <= 0:
            return {"score": 0.0, "confidence": 0.0, "reasoning": "Invalid strike"}

        # Get or generate forecast
        if self._forecast_cache is None:
            self._forecast_cache = await run_forecast(pred_len=24)

        result = await score_market(strike, hours_to_expiry, spot, self._forecast_cache)
        return result

    async def get_signal(self, market) -> dict:
        """Compatibility wrapper: returns {score, confidence, agent_name, summary}."""
        result = await self.run(market)
        # Polymarket-style signal
        return type("Signal", (), {
            "agent_name": self.name,
            "score": result["score"],
            "confidence": result["confidence"],
            "summary": result["reasoning"],
        })()


# ── Standalone test ────────────────────────────────────────────────────────


if __name__ == "__main__":
    async def test():
        print("Testing KronosMarketAgent...")
        agent = KronosMarketAgent()
        forecast = await run_forecast(pred_len=24)
        print(f"\nForecast: {forecast['direction']} (conviction={forecast['conviction']})")
        print(f"  {forecast['up_hours']}h up / {forecast['down_hours']}h down / {forecast['direction_changes']} changes")
        print(f"  Avg magnitude: {forecast['avg_magnitude_pct']}%")
        print(f"  Last close: ${forecast['last_close']:,.2f}")

        # Test scoring
        test_markets = [
            {"strike": 59500, "hours_to_expiry": 12, "spot": 60975},
            {"strike": 61000, "hours_to_expiry": 6, "spot": 60975},
            {"strike": 62000, "hours_to_expiry": 24, "spot": 60975},
            {"strike": 57000, "hours_to_expiry": 48, "spot": 60975},
        ]
        print(f"\nMarket scoring:")
        for m in test_markets:
            s = await score_market(**m, forecast=forecast)
            micro_info = ""
            if "microstructure" in s and s["microstructure"].get("adjustments", {}).get("signals"):
                micro_info = " [ms: " + ", ".join(s["microstructure"]["adjustments"]["signals"]) + "]"
            print(f"  strike=${m['strike']:>6,} tte={m['hours_to_expiry']:2d}h → score={s['score']:+.4f} conf={s['confidence']:.3f}{micro_info}")

    asyncio.run(test())
