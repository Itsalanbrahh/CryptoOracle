"""Kalshi BTC market fetcher and selection logic."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .client import KalshiClient


@dataclass
class KalshiMarket:
    ticker: str
    strike: float           # BTC price threshold (floor_strike for directional; bin floor for range)
    yes_bid: float          # YES best bid in dollars [0,1]
    yes_ask: float          # YES best ask in dollars [0,1]
    no_bid: float           # NO best bid in dollars
    no_ask: float           # NO best ask in dollars
    mid: float              # mid-market YES price
    volume: float           # total contracts traded
    close_time: str | None  # ISO8601 when trading closes
    cap_strike: float | None = None  # upper bound for KXBTC range bin markets; None = directional

    @property
    def is_range(self) -> bool:
        return self.cap_strike is not None

    @property
    def bin_center(self) -> float:
        """Mid-point of the range bin, or strike for directional markets."""
        if self.cap_strike is not None:
            return (self.strike + self.cap_strike) / 2
        return self.strike

    @property
    def mid_cents(self) -> int:
        return int(round(self.mid * 100))

    @property
    def hours_to_expiry(self) -> float:
        if not self.close_time:
            return 24.0
        dt = datetime.fromisoformat(self.close_time.replace("Z", "+00:00"))
        diff = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, diff / 3600)

    @property
    def is_liquid(self) -> bool:
        return self.yes_bid > 0.01 and self.yes_ask < 0.99 and self.volume > 100


def _parse_market(raw: dict) -> KalshiMarket | None:
    try:
        yes_bid = float(raw.get("yes_bid_dollars") or 0)
        yes_ask = float(raw.get("yes_ask_dollars") or 1)
        no_bid = float(raw.get("no_bid_dollars") or 0)
        no_ask = float(raw.get("no_ask_dollars") or 1)
        mid = (yes_bid + yes_ask) / 2
        floor = raw.get("floor_strike")
        cap = raw.get("cap_strike")
        strike = float(floor or cap or 0)
        cap_strike = float(cap) if (cap and floor and float(cap) != float(floor)) else None
        return KalshiMarket(
            ticker=raw["ticker"],
            strike=strike,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            mid=mid,
            volume=float(raw.get("volume_fp") or 0),
            close_time=raw.get("close_time"),
            cap_strike=cap_strike,
        )
    except (KeyError, TypeError, ValueError):
        return None


async def fetch_btc_markets(min_volume: float = 100.0) -> list[KalshiMarket]:
    """Fetch active Kalshi directional (KXBTCD) markets."""
    client = KalshiClient()
    raw_markets = await client.get_markets(series_ticker="KXBTCD", status="open", limit=200)
    markets = [m for raw in raw_markets if (m := _parse_market(raw)) and m.is_liquid and m.volume >= min_volume]
    return sorted(markets, key=lambda m: m.volume, reverse=True)


async def fetch_btc_range_markets(min_volume: float = 50.0) -> list[KalshiMarket]:
    """Fetch active Kalshi range bin (KXBTC) markets."""
    client = KalshiClient()
    raw_markets = await client.get_markets(series_ticker="KXBTC", status="open", limit=200)
    markets = [m for raw in raw_markets if (m := _parse_market(raw)) and m.is_liquid and m.volume >= min_volume]
    return sorted(markets, key=lambda m: m.volume, reverse=True)


async def fetch_btc_15m_markets(min_volume: float = 0.0) -> list[KalshiMarket]:
    """Fetch active Kalshi 15-minute BTC UP/DOWN markets (KXBTC15M).

    These settle on CF Benchmarks BRTI open/close averages for a 15-minute
    window. ``floor_strike`` is the opening target. Liquidity filters are looser
    than hourly markets because only one contract is typically open at a time
    and volume accrues fast inside the window.
    """
    client = KalshiClient()
    raw_markets = await client.get_markets(series_ticker="KXBTC15M", status="open", limit=50)
    markets: list[KalshiMarket] = []
    for raw in raw_markets:
        m = _parse_market(raw)
        if not m:
            continue
        # 15m books can be one-sided near expiry; require a real two-sided quote
        # but do not demand the hourly ``is_liquid`` volume floor.
        if m.yes_bid <= 0.0 or m.yes_ask >= 1.0:
            continue
        if m.volume < min_volume:
            continue
        # Skip already-expired / sub-second leftovers
        if m.hours_to_expiry <= 0.0:
            continue
        markets.append(m)
    return sorted(markets, key=lambda m: m.hours_to_expiry)


def select_target_markets(markets: list[KalshiMarket], spot_price: float, top_n: int = 8) -> list[KalshiMarket]:
    """
    Rank markets for edge potential.

    Directional markets: scored by mid-price uncertainty (near 0.50 = most edge)
    plus a lane for high-confidence contracts near 0.85-0.97.

    Range bin markets: scored by how close the bin center is to spot — the bins
    around spot are the most actionable (GBM probability is most meaningful there).
    """
    def _score(m: KalshiMarket) -> float:
        p = m.mid
        if p < 0.02 or p > 0.99:
            return 999.0
        if m.is_range:
            # Prefer bins whose center is near spot; normalize by a $500 window
            return abs(m.bin_center - spot_price) / 500.0
        uncertainty_score = abs(p - 0.5)
        high_conf_score = abs(p - 0.87) * 0.3
        return min(uncertainty_score, high_conf_score)

    return sorted(markets, key=_score)[:top_n]
