"""
Technical analysis agents for Kalshi BTC trading.

Implements:
  - CandlestickPatternAgent: detects doji, engulfing, hammer, shooting star
  - SupportResistanceAgent: identifies key S/R levels from recent price action
  - DynamicSRAgent: moving average based trend detection
  - FairValueGapAgent: detects FVG patterns for price attraction
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

# ── Shared helpers ─────────────────────────────────────────────────────────


async def _get_recent_prices(hours: int = 48) -> list[dict]:
    """Fetch recent hourly BTC candles. Returns list of {open, high, low, close, volume}."""
    from crypto_oracle.kalshi.backtest import fetch_historical_btc

    candles = await fetch_historical_btc(days=max(3, hours // 24 + 1))
    return candles[-hours:] if len(candles) > hours else candles


# ── Candlestick Pattern Agent ─────────────────────────────────────────────


class CandlestickPatternAgent:
    """Detects reversal candlestick patterns in recent BTC price action.

    Patterns:
      - DOJI: open ≈ close (indecision, potential reversal)
      - HAMMER: long lower wick, small body at top (bullish reversal)
      - SHOOTING_STAR: long upper wick, small body at bottom (bearish reversal)
      - BULLISH_ENGULFING: green candle fully engulfs previous red (bullish)
      - BEARISH_ENGULFING: red candle fully engulfs previous green (bearish)
    """
    name = "CandlestickPatterns"

    def _analyze_candle(self, candle: dict) -> dict:
        """Analyze a single candle for pattern features."""
        o, h, l, c = candle["open"], candle["high"], candle["low"], candle["close"]
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        total_range = h - l
        if total_range <= 0:
            return {"body_pct": 0, "upper_wick_pct": 0, "lower_wick_pct": 0, "is_bullish": c > o}

        return {
            "body_pct": body / total_range,
            "upper_wick_pct": upper_wick / total_range,
            "lower_wick_pct": lower_wick / total_range,
            "is_bullish": c > o,
        }

    def _detect_patterns(self, candles: list[dict]) -> list[dict]:
        """Detect patterns in the last 2 candles. Returns list of pattern dicts."""
        if len(candles) < 2:
            return []

        last = candles[-1]
        prev = candles[-2]
        la = self._analyze_candle(last)
        pa = self._analyze_candle(prev)
        patterns = []

        # Doji: tiny body (≤10% of range)
        if la["body_pct"] <= 0.10 and la["upper_wick_pct"] > 0.3 and la["lower_wick_pct"] > 0.3:
            patterns.append({"pattern": "DOJI", "bullish": None, "strength": 0.3})

        # Hammer: small body at top, long lower wick (≥2x body), little upper wick
        if (la["body_pct"] <= 0.35 and la["lower_wick_pct"] >= 0.50
                and la["upper_wick_pct"] <= 0.15 and la["is_bullish"]):
            patterns.append({"pattern": "HAMMER", "bullish": True, "strength": 0.5})

        # Shooting Star: small body at bottom, long upper wick (≥2x body)
        if (la["body_pct"] <= 0.35 and la["upper_wick_pct"] >= 0.50
                and la["lower_wick_pct"] <= 0.15 and not la["is_bullish"]):
            patterns.append({"pattern": "SHOOTING_STAR", "bullish": False, "strength": 0.5})

        # Bullish Engulfing: green candle fully engulfs previous red
        if (la["is_bullish"] and not pa["is_bullish"]
                and last["close"] > prev["high"] and last["open"] < prev["low"]):
            patterns.append({"pattern": "BULLISH_ENGULFING", "bullish": True, "strength": 0.7})

        # Bearish Engulfing: red candle fully engulfs previous green
        if (not la["is_bullish"] and pa["is_bullish"]
                and last["low"] < prev["open"] and last["high"] > prev["close"]):
            patterns.append({"pattern": "BEARISH_ENGULFING", "bullish": False, "strength": 0.7})

        return patterns

    async def run(self, market, **kwargs) -> dict:
        """Score a market based on candlestick patterns.

        Returns: {score, confidence, reasoning}
        """
        candles = await _get_recent_prices(hours=48)
        if len(candles) < 3:
            return {"score": 0.0, "confidence": 0.0, "reasoning": "Insufficient data"}

        patterns = self._detect_patterns(candles)

        if not patterns:
            return {"score": 0.0, "confidence": 0.0, "reasoning": "No clear pattern detected"}

        # Aggregate pattern signals
        total_strength = sum(p["strength"] for p in patterns)
        bullish_strength = sum(p["strength"] for p in patterns if p.get("bullish") is True)
        bearish_strength = sum(p["strength"] for p in patterns if p.get("bullish") is False)

        if total_strength > 0:
            net = (bullish_strength - bearish_strength) / total_strength
            confidence = min(1.0, total_strength)
        else:
            net = 0.0
            confidence = 0.0

        pattern_names = [p["pattern"] for p in patterns]
        reasoning = f"Candlestick patterns: {', '.join(pattern_names)}"

        return {"score": round(net, 4), "confidence": round(confidence, 4), "reasoning": reasoning}


# ── Support & Resistance Agent ─────────────────────────────────────────────


class SupportResistanceAgent:
    """Identifies key Support and Resistance levels from recent price action.

    A level is "significant" if the price has bounced off it multiple times.
    The agent scores a market based on how close the strike is to S/R levels.
    """
    name = "SupportResistance"

    def _find_levels(self, candles: list[dict], lookback: int = 48) -> dict:
        """Find support and resistance levels.

        Returns: {support: float, resistance: float, strength: float}
        """
        if len(candles) < lookback:
            lookback = len(candles)

        recent = candles[-lookback:]

        # Find local minima and maxima
        lows = []
        highs = []
        for i in range(2, len(recent) - 2):
            r = recent
            if r[i]["low"] < r[i - 1]["low"] and r[i]["low"] < r[i + 1]["low"]:
                lows.append(r[i]["low"])
            if r[i]["high"] > r[i - 1]["high"] and r[i]["high"] > r[i + 1]["high"]:
                highs.append(r[i]["high"])

        if not lows or not highs:
            last = recent[-1]
            return {"support": last["low"], "resistance": last["high"], "strength": 0.0}

        # Cluster levels: group nearby levels
        support = sum(lows) / len(lows)
        resistance = sum(highs) / len(highs)

        # Strength: how many touches
        support_touches = sum(1 for l in lows if abs(l - support) / support < 0.005)
        resistance_touches = sum(1 for h in highs if abs(h - resistance) / resistance < 0.005)

        strength = min(1.0, (support_touches + resistance_touches) / 10.0)

        return {
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "strength": strength,
        }

    async def run(self, market, **kwargs) -> dict:
        """Score a market: YES if strike near support, NO if near resistance."""
        candles = await _get_recent_prices(hours=72)
        if len(candles) < 10:
            return {"score": 0.0, "confidence": 0.0, "reasoning": "Insufficient data"}

        spot = candles[-1]["close"]
        # Get strike from kwargs (Kalshi context) or market object
        kalshi = kwargs.get("kalshi")
        strike = kalshi.strike if kalshi is not None else (market.strike if hasattr(market, "strike") else spot)

        levels = self._find_levels(candles)
        support = levels["support"]
        resistance = levels["resistance"]
        level_strength = levels["strength"]

        # How close is the strike to support or resistance?
        dist_to_support = abs(strike - support) / support * 100 if support > 0 else 999
        dist_to_resistance = abs(strike - resistance) / resistance * 100 if resistance > 0 else 999

        score = 0.0
        reasoning_parts = []

        if dist_to_support < 1.0:
            # Strike is near support → bullish (price likely to bounce up from support)
            score = 0.6 * level_strength
            reasoning_parts.append(f"strike near support ${support:,.0f} ({dist_to_support:.1f}%)")
        elif dist_to_resistance < 1.0:
            # Strike is near resistance → bearish (price likely to bounce down from resistance)
            score = -0.6 * level_strength
            reasoning_parts.append(f"strike near resistance ${resistance:,.0f} ({dist_to_resistance:.1f}%)")
        else:
            # Neither — neutral, but bias toward the trend direction
            if spot > (support + resistance) / 2:
                score = 0.2  # mildly bullish (above middle)
                reasoning_parts.append("above mid-range")
            else:
                score = -0.2  # mildly bearish
                reasoning_parts.append("below mid-range")

        reasoning = f"S&R: {', '.join(reasoning_parts)}"
        return {"score": round(score, 4), "confidence": round(level_strength, 4), "reasoning": reasoning}


# ── Dynamic S&R (Moving Average) Agent ─────────────────────────────────────


class DynamicSRAgent:
    """Uses EMA crossovers and price vs MA position for trend detection.

    Bullish signals:
      - Price above EMA(20) and EMA(50)
      - EMA(20) crossed above EMA(50) (golden cross)
      - Price pulled back to EMA(20) and bounced

    Bearish signals:
      - Price below EMA(20) and EMA(50)
      - EMA(20) crossed below EMA(50) (death cross)
    """
    name = "DynamicSR"

    def _ema(self, data: list[float], period: int) -> list[float]:
        """Calculate exponential moving average."""
        if len(data) < period:
            return [sum(data) / len(data)] * len(data)
        multiplier = 2.0 / (period + 1)
        ema = [sum(data[:period]) / period]
        for i in range(period, len(data)):
            ema.append((data[i] - ema[-1]) * multiplier + ema[-1])
        return [ema[0]] * (period - 1) + ema

    async def run(self, market, **kwargs) -> dict:
        """Score market based on MA position."""
        candles = await _get_recent_prices(hours=72)
        if len(candles) < 55:
            return {"score": 0.0, "confidence": 0.0, "reasoning": "Insufficient data"}

        closes = [c["close"] for c in candles]
        spot = closes[-1]
        prev_spot = closes[-2] if len(closes) > 1 else spot

        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)

        if len(ema20) < 2 or len(ema50) < 2:
            return {"score": 0.0, "confidence": 0.0, "reasoning": "Not enough MA data"}

        current_ema20 = ema20[-1]
        current_ema50 = ema50[-1]
        prev_ema20 = ema20[-2]
        prev_ema50 = ema50[-2]

        score = 0.0
        signals = []

        # Price vs EMA position
        if spot > current_ema20 > current_ema50:
            score += 0.3
            signals.append("above EMA20/50")
        elif spot > current_ema20:
            score += 0.15
            signals.append("above EMA20")
        elif spot < current_ema20 and spot < current_ema50:
            score -= 0.3
            signals.append("below EMA20/50")
        elif spot < current_ema20:
            score -= 0.15
            signals.append("below EMA20")

        # EMA crossover
        if prev_ema20 <= prev_ema50 and current_ema20 > current_ema50:
            score += 0.35
            signals.append("GOLDEN_CROSS")
        elif prev_ema20 >= prev_ema50 and current_ema20 < current_ema50:
            score -= 0.35
            signals.append("DEATH_CROSS")

        # Pullback to EMA20
        ema20_3h_ago = ema20[-3] if len(ema20) >= 3 else current_ema20
        if spot < current_ema20 * 1.002 and spot > current_ema20 * 0.998 and prev_spot < ema20_3h_ago:
            score += 0.2  # bounced off EMA20
            signals.append("EMA20 bounce")

        score = max(-1.0, min(1.0, score))
        confidence = min(1.0, abs(score) * 0.7 + 0.3)

        reasoning = f"DynamicS&R: {', '.join(signals)}" if signals else "DynamicS&R: neutral"
        return {"score": round(score, 4), "confidence": round(confidence, 4), "reasoning": reasoning}


# ── Fair Value Gap Agent ────────────────────────────────────────────────────


class FairValueGapAgent:
    """Detects Fair Value Gaps (FVG) in hourly BTC data.

    Bullish FVG: gap between candle N-2's high and candle N's low, with
    the gap above current price → price expected to fill gap upward.

    Bearish FVG: gap between candle N-2's low and candle N's high, with
    the gap below current price → price expected to fill gap downward.
    """
    name = "FairValueGap"

    async def run(self, market, **kwargs) -> dict:
        candles = await _get_recent_prices(hours=48)
        if len(candles) < 5:
            return {"score": 0.0, "confidence": 0.0, "reasoning": "Insufficient data"}

        spot = candles[-1]["close"]
        bullish_gaps = []
        bearish_gaps = []

        for i in range(2, len(candles)):
            c1, c2, c3 = candles[i - 2], candles[i - 1], candles[i]
            # Bullish FVG: c2's low > c1's high (gap up)
            if c2["low"] > c1["high"]:
                gap_low = c1["high"]
                gap_high = c2["low"]
                gap_mid = (gap_low + gap_high) / 2
                if gap_mid > spot:
                    bullish_gaps.append({"gap_mid": gap_mid, "size_pct": (gap_high - gap_low) / spot * 100})
            # Bearish FVG: c2's high < c1's low (gap down)
            if c2["high"] < c1["low"]:
                gap_low = c2["high"]
                gap_high = c1["low"]
                gap_mid = (gap_low + gap_high) / 2
                if gap_mid < spot:
                    bearish_gaps.append({"gap_mid": gap_mid, "size_pct": (gap_high - gap_low) / spot * 100})

        score = 0.0
        confidence = 0.0

        if bullish_gaps:
            nearest = min(bullish_gaps, key=lambda g: g["gap_mid"])
            dist = (nearest["gap_mid"] - spot) / spot * 100
            score = min(0.6, 0.3 + dist * 0.1)
            confidence = min(0.5, nearest["size_pct"] * 2)
            reasoning = f"Bullish FVG ${nearest['gap_mid']:,.0f} ({dist:.2f}% above spot)"
        elif bearish_gaps:
            nearest = min(bearish_gaps, key=lambda g: -(g["gap_mid"]))
            dist = (spot - nearest["gap_mid"]) / spot * 100
            score = max(-0.6, -0.3 - dist * 0.1)
            confidence = min(0.5, nearest["size_pct"] * 2)
            reasoning = f"Bearish FVG ${nearest['gap_mid']:,.0f} ({dist:.2f}% below spot)"
        else:
            reasoning = "No FVG detected"

        return {"score": round(score, 4), "confidence": round(confidence, 4), "reasoning": reasoning}
