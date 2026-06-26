"""
FibonacciRetracementAgent — identifies key Fibonacci retracement levels
from recent BTC swing highs/lows and scores Kalshi binary markets based
on how price behaves at these levels.

Theory:
  - After a strong directional move, BTC often retraces to a Fibonacci level
    (0.382, 0.5, 0.618, 0.786) before continuing or reversing.
  - When price is AT a Fib level with bullish momentum → likely bounce → YES
  - When price is breaking THROUGH a Fib level with conviction → likely continue → follow
  - Confluence of Fib level + other S/R increases confidence

For Kalshi binary options (price above/below strike in X hours):
  - Checks if strike is at or near a key Fib level relative to recent swing
  - Scores based on which side of the Fib level the forecast/trend supports
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import aiohttp

# ── Fibonacci levels ──────────────────────────────────────────────────────

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
# Retracement levels (between 0 and 1): these act as support/resistance
FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
# Extension levels (above 1): continuation targets
FIB_EXTENSION = [1.0, 1.272, 1.618]


async def _fetch_hourly_candles(hours: int = 168) -> list[dict]:
    """Fetch hourly BTC candles from Kraken for swing analysis.

    Returns list of {timestamp, open, high, low, close, volume}.
    168 hours = 7 days — enough for short-mid term swing detection.
    """
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, params={"pair": "XBTUSD", "interval": 60},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        rows = data.get("result", {}).get("XXBTZUSD", [])
        candles = [
            {
                "timestamp": datetime.fromtimestamp(int(r[0]), tz=timezone.utc),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[6]),
            }
            for r in rows[-hours:]
        ]
        return candles
    except Exception:
        return []


def _find_swings(candles: list[dict], lookback: int = 96) -> tuple[float | None, float | None]:
    """Find the most recent significant swing high and swing low.

    Uses a simple peak/trough detector:
    - Swing high: a bar whose high is higher than the 5 bars before AND after
    - Swing low: a bar whose low is lower than the 5 bars before AND after

    Returns (swing_high, swing_low) — the most recent pair.
    Both are None if insufficient data.
    """
    if len(candles) < 20:
        return None, None

    highs = [c["high"] for c in candles[-lookback:]]
    lows = [c["low"] for c in candles[-lookback:]]
    n = len(highs)

    peaks: list[tuple[int, float]] = []
    troughs: list[tuple[int, float]] = []

    for i in range(5, n - 5):
        # Swing high
        if highs[i] == max(highs[i - 5:i + 6]):
            peaks.append((i, highs[i]))
        # Swing low
        if lows[i] == min(lows[i - 5:i + 6]):
            troughs.append((i, lows[i]))

    if not peaks and not troughs:
        return None, None

    # Take the most recent swing high above the current price (resistance)
    # and the most recent swing low below (support)
    current = candles[-1]["close"]
    recent_high = None
    recent_low = None

    # Find swing levels that are relevant for the current price context
    for _, val in reversed(peaks):
        if val > current:
            recent_high = val
            break
    if recent_high is None and peaks:
        # No resistance above — take highest peak as reference
        recent_high = max(v for _, v in peaks)

    for _, val in reversed(troughs):
        if val < current:
            recent_low = val
            break
    if recent_low is None and troughs:
        # No support below — take lowest trough as reference
        recent_low = min(v for _, v in troughs)

    return recent_high, recent_low


def _calculate_fib_levels(swing_high: float, swing_low: float) -> dict[float, float]:
    """Calculate Fibonacci retracement and extension levels.

    Returns dict mapping Fib ratio → price level.
    Positive ratio = price above swing_low (uptrend framework).
    """
    range_size = swing_high - swing_low
    if range_size <= 0:
        return {}

    levels: dict[float, float] = {}
    for fib in FIB_LEVELS:
        levels[fib] = swing_high - range_size * fib
    return levels


def _closest_fib_level(price: float, fib_levels: dict[float, float], threshold_pct: float = 0.5) -> tuple[float | None, float | None]:
    """Find the nearest Fib level to the given price within threshold.

    Returns (fib_ratio, price_at_level) or (None, None) if none within range.
    """
    closest_ratio = None
    closest_dist = float("inf")
    closest_price = None

    for fib_ratio, level_price in fib_levels.items():
        if level_price <= 0:
            continue
        dist_pct = abs(price - level_price) / level_price * 100
        if dist_pct < closest_dist:
            closest_dist = dist_pct
            closest_ratio = fib_ratio
            closest_price = level_price

    if closest_ratio is not None and closest_dist <= threshold_pct:
        return closest_ratio, closest_price
    return None, None


async def run_fibonacci_analysis() -> dict:
    """Run full Fibonacci analysis and return current levels."""
    candles = await _fetch_hourly_candles(hours=168)
    if len(candles) < 20:
        return {"swing_high": None, "swing_low": None, "fib_levels": {}, "spot": None, "nearest_level": None}

    spot = candles[-1]["close"]
    swing_high, swing_low = _find_swings(candles)

    if swing_high is None or swing_low is None or swing_high <= swing_low:
        return {"swing_high": swing_high, "swing_low": swing_low, "fib_levels": {}, "spot": spot, "nearest_level": None}

    fib_levels = _calculate_fib_levels(swing_high, swing_low)
    nearest_ratio, nearest_price = _closest_fib_level(spot, fib_levels)

    return {
        "spot": spot,
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "swing_range": round(swing_high - swing_low, 2),
        "swing_range_pct": round((swing_high - swing_low) / swing_low * 100, 2),
        "fib_levels": {str(k): round(v, 2) for k, v in sorted(fib_levels.items())},
        "nearest_level": {"fib": nearest_ratio, "price": round(nearest_price, 2)} if nearest_ratio is not None else None,
        "spot_vs_swing_low_pct": round((spot - swing_low) / swing_low * 100, 2),
        "trend": "BULLISH" if spot > swing_low + (swing_high - swing_low) * 0.618 else "BEARISH" if spot < swing_low + (swing_high - swing_low) * 0.382 else "NEUTRAL",
    }


class FibonacciRetracementAgent:
    """Agent that uses Fibonacci retracement levels to score Kalshi BTC markets."""

    name = "FibonacciRetracement"
    _analysis_cache: dict | None = None

    async def _get_analysis(self) -> dict:
        """Get or cache Fibonacci analysis."""
        if self._analysis_cache is None:
            self._analysis_cache = await run_fibonacci_analysis()
        return self._analysis_cache

    async def run(self, market, **kwargs) -> dict:
        """Score a market using Fibonacci retracement levels.

        Core logic:
          1. Map strike and spot to Fib ratios within the swing range
          2. Key Fib levels (0.382, 0.5, 0.618) act as magnets — price gravitates there
          3. Score = directional bias from spot position + convergence bonus at key levels
          4. Fib confirms trend: if spot is at 0.618 and bouncing = trend continuation

        Returns:
            {score, confidence, reasoning} dict compatible with Kalshi agent interface.
        """
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

        analysis = await self._get_analysis()
        fib_levels = analysis.get("fib_levels", {})
        swing_high = analysis.get("swing_high")
        swing_low = analysis.get("swing_low")
        nearest = analysis.get("nearest_level")

        if not fib_levels or swing_high is None or swing_low is None or swing_high <= swing_low:
            return {"score": 0.0, "confidence": 0.0, "reasoning": "Insufficient swing data for levels"}

        range_size = swing_high - swing_low
        if range_size <= 0:
            return {"score": 0.0, "confidence": 0.0, "reasoning": "Invalid swing range"}

        # ── Map strike and spot to Fib ratios ─────────────────────────────────
        # Fib ratio 0.0 = swing_high (resistance), 1.0 = swing_low (support)
        # Lower ratio = closer to resistance, higher = closer to support
        def _ratio_from_price(price: float) -> float:
            r = (swing_high - price) / range_size
            return max(0.0, min(1.0, r))

        strike_ratio = _ratio_from_price(strike)
        spot_ratio = _ratio_from_price(spot)

        # ── Determine which major Fib level strike is near ────────────────────
        # Major Fibs: 0.236, 0.382, 0.5, 0.618, 0.786
        # Each level can be either support or resistance
        MAJOR_LEVELS = {0.236: 0.6, 0.382: 0.85, 0.5: 1.0, 0.618: 0.9, 0.786: 0.7}
        closest_major = min(MAJOR_LEVELS, key=lambda f: abs(f - strike_ratio))
        dist_to_major = abs(strike_ratio - closest_major)
        level_significance = MAJOR_LEVELS[closest_major] * max(0.0, 1.0 - dist_to_major / 0.08)

        # ── Is strike within the swing range? ─────────────────────────────────
        in_range = swing_low <= strike <= swing_high

        # ── Score calculation ─────────────────────────────────────────────────
        score = 0.0
        reasons = []
        hours_factor = min(1.0, 24.0 / max(hours_to_expiry, 1.0))

        # Factor 1: Trend from spot position in the range
        # If spot is in the lower half (<0.5 ratio, i.e. near resistance) = bearish pressure
        # If spot is in the upper half (>0.5 ratio, i.e. near support) = bullish pressure
        spot_trend = (spot_ratio - 0.5) * 2  # -1 (near high) to +1 (near low)
        score += spot_trend * 0.15 * hours_factor
        if spot_trend > 0.3:
            reasons.append(f"spot near support (ratio={spot_ratio:.2f})")
        elif spot_trend < -0.3:
            reasons.append(f"spot near resistance (ratio={spot_ratio:.2f})")

        # Factor 2: Fib level magnet — strike at a key level attracts price
        if dist_to_major < 0.05 and level_significance > 0.5 and in_range:
            # Strike is at a major Fib level — magnet effect
            # Direction: where is spot relative to strike?
            if hours_to_expiry < 48:  # Fib magnetism is strongest on shorter timeframes
                if spot < strike:
                    # Price below the level → uptrend momentum needed to reach it
                    # Most magnetic when spot is within the same Fib zone
                    if spot_ratio - strike_ratio < 0.3:  # same approximate zone
                        score += level_significance * 0.20 * hours_factor
                        reasons.append(f"strike at Fib {closest_major}, spot below (magnet)")
                    else:
                        score += level_significance * 0.10 * hours_factor
                        reasons.append(f"strike at Fib {closest_major}, spot below (weak magnet)")
                elif spot > strike + range_size * 0.02:  # spot clearly above strike
                    # Price above the level → pulling down risk
                    if strike_ratio - spot_ratio < 0.3:
                        score -= level_significance * 0.20 * hours_factor
                        reasons.append(f"strike at Fib {closest_major}, spot above (resistance)")
                    else:
                        score -= level_significance * 0.10 * hours_factor
                        reasons.append(f"strike at Fib {closest_major}, spot above (weak resistance)")
                else:
                    # Price right at the level → decision point
                    dir_score = spot_trend * level_significance * 0.25 * hours_factor
                    score += dir_score
                    reasons.append(f"price at Fib {closest_major} decision point (trend={spot_trend:+.2f})")
            else:
                reasons.append(f"strike at Fib {closest_major} but TTE too long for magnet")
        elif in_range:
            # Strike inside range but not at a major Fib
            reasons.append(f"strike inside range (ratio={strike_ratio:.2f}), between Fib levels")

            # Factor 3: Trend alignment — does the trend support hitting this strike?
            trend_dir = "up" if spot_trend > 0 else "down"
            strike_dir = "up" if strike_ratio < spot_ratio else "down"
            # If trend and strike direction match → more likely to get there
            if (trend_dir == "up" and strike_dir == "up") or (trend_dir == "down" and strike_dir == "down"):
                # Moving toward the strike
                dist_factor = abs(strike_ratio - spot_ratio)
                if dist_factor < 0.2:
                    score += 0.08 * hours_factor
                    reasons.append(f"trend aligns ({trend_dir}), strike within range")
                elif dist_factor < 0.4:
                    score += 0.04 * hours_factor
                    reasons.append(f"trend aligns ({trend_dir}), strike farther out")
        else:
            # Strike outside the swing range — extension or breakdown
            swing_mid = (swing_high + swing_low) / 2
            if strike > swing_high and spot_ratio < 0.4:
                # Above range, bearish pressure would push back down
                score -= 0.05
                reasons.append("strike above range, trend needs breakdown")
            elif strike < swing_low and spot_ratio > 0.6:
                # Below range, bullish pressure would push back up
                score += 0.05
                reasons.append("strike below range, trend needs rally")

        # Clamp
        score = max(-1.0, min(1.0, score))

        # ── Confidence ────────────────────────────────────────────────────────
        if dist_to_major < 0.03 and level_significance > 0.7:
            # Strong confluence at a major Fib level
            confidence = 0.55 + level_significance * 0.25
        elif abs(score) > 0.2:
            confidence = 0.40 + abs(score) * 0.35
        else:
            confidence = 0.30 + abs(score) * 0.5

        # Time adjustment
        if hours_to_expiry < 2:
            confidence *= 0.70
        elif hours_to_expiry > 72:
            confidence *= 0.85  # Fib is less reliable on long timeframes

        # Range size bonus: wider swings = more reliable levels
        range_pct = analysis.get("swing_range_pct", 0)
        if range_pct > 5.0:
            confidence = min(0.95, confidence * 1.10)
        elif range_pct < 2.0:
            confidence *= 0.85  # tight range = levels less meaningful

        confidence = max(0.05, min(0.95, confidence))

        reasons_str = " | ".join(reasons) if reasons else "No strong Fib confluence"
        reasoning = f"{reasons_str}  [swing=${swing_high:,.0f}/{swing_low:,.0f}, range={range_pct:.1f}%]"

        return {
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "reasoning": reasoning,
        }

    async def get_signal(self, market) -> dict:
        """Compatibility wrapper."""
        result = await self.run(market)
        return type("Signal", (), {
            "agent_name": self.name,
            "score": result["score"],
            "confidence": result["confidence"],
            "summary": result["reasoning"],
        })()
