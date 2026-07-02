"""
Deribit options-implied probability anchor for Kalshi BTC binaries.

The professionals trading Kalshi's BTC ladder price it off the options market
(IBIT/Deribit), which publishes an independently calibrated implied volatility
surface for the same underlying. Anchoring belief_yes to options-implied
P(BTC > strike) — instead of homegrown realized-vol GBM — means the bot only
perceives edge when Kalshi's quote genuinely diverges from professional
pricing, which is the documented profitable pattern in these markets.

Method:
  1. Fetch Deribit's BTC option chain (public API, no auth).
  2. Pick the listed expiry closest to the Kalshi contract's expiry.
  3. Linearly interpolate mark IV across strikes at the Kalshi strike.
  4. P(BTC > K) = N(d2) with the interpolated IV over the KALSHI horizon
     (risk-neutral, no drift — this is the probability the pros trade off).

Caveats (accepted):
  - Deribit expiries settle 08:00 UTC; for very short Kalshi horizons the
    nearest listed expiry can be many hours away. We use its IV (term
    mismatch) over the exact Kalshi horizon — still far better calibrated
    than 24h realized vol.
  - All failures return None; callers fall back to the GBM anchor.

The chain is cached for 5 minutes, so a full scan costs one HTTP call.
"""
from __future__ import annotations

import math
import re
import time
from datetime import datetime, timezone
from statistics import NormalDist as _ND

import aiohttp

_CHAIN_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
_CACHE: dict = {"fetched_at": 0.0, "chain": None}
_CACHE_TTL = 300  # seconds

_MONTH_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}
_NAME_RE = re.compile(r"^BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])$")


def _parse_instrument(name: str) -> tuple[datetime, float, str] | None:
    """BTC-4JUL26-108000-C → (expiry 08:00 UTC, strike, 'C')."""
    m = _NAME_RE.match(name or "")
    if not m:
        return None
    day, mon, yy, strike, typ = m.groups()
    mon_num = _MONTH_NUM.get(mon)
    if mon_num is None:
        return None
    try:
        expiry = datetime(2000 + int(yy), mon_num, int(day), 8, 0, tzinfo=timezone.utc)
        return expiry, float(strike), typ
    except ValueError:
        return None


async def _fetch_chain() -> list[dict] | None:
    """Fetch (and cache) Deribit's full BTC option book summary."""
    now = time.time()
    if _CACHE["chain"] is not None and now - _CACHE["fetched_at"] < _CACHE_TTL:
        return _CACHE["chain"]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                _CHAIN_URL,
                params={"currency": "BTC", "kind": "option"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        chain = data.get("result") or []
        if chain:
            _CACHE["chain"] = chain
            _CACHE["fetched_at"] = now
        return chain or None
    except Exception:
        return None  # caller falls back to GBM


def _implied_prob_from_chain(
    chain: list[dict],
    strike: float,
    hours_to_expiry: float,
    spot: float,
    now: datetime | None = None,
) -> float | None:
    """Pure computation — split out so it can be tested with synthetic chains."""
    if spot <= 0 or strike <= 0 or hours_to_expiry <= 0:
        return None
    now = now or datetime.now(timezone.utc)

    # Group usable IV points by expiry: {expiry: {strike: iv}}
    by_expiry: dict[datetime, dict[float, float]] = {}
    for row in chain:
        iv = row.get("mark_iv")
        if not iv or iv <= 0:
            continue
        parsed = _parse_instrument(row.get("instrument_name", ""))
        if parsed is None:
            continue
        expiry, k, typ = parsed
        if expiry <= now:
            continue
        # Calls and puts carry the same mark IV; prefer calls, keep either
        by_expiry.setdefault(expiry, {})
        if k not in by_expiry[expiry] or typ == "C":
            by_expiry[expiry][k] = float(iv)

    if not by_expiry:
        return None

    # Nearest listed expiry to the Kalshi horizon (need ≥2 strikes to interpolate)
    target_h = hours_to_expiry
    candidates = [
        (abs((exp - now).total_seconds() / 3600.0 - target_h), exp)
        for exp, ks in by_expiry.items()
        if len(ks) >= 2
    ]
    if not candidates:
        return None
    _, best_expiry = min(candidates)
    smile = sorted(by_expiry[best_expiry].items())  # [(strike, iv%), ...]

    # Linear IV interpolation at the Kalshi strike; clamp outside the range
    strikes = [k for k, _ in smile]
    ivs = [v for _, v in smile]
    if strike <= strikes[0]:
        iv_pct = ivs[0]
    elif strike >= strikes[-1]:
        iv_pct = ivs[-1]
    else:
        for i in range(1, len(strikes)):
            if strike <= strikes[i]:
                k0, k1 = strikes[i - 1], strikes[i]
                w = (strike - k0) / (k1 - k0)
                iv_pct = ivs[i - 1] * (1 - w) + ivs[i] * w
                break

    sigma = iv_pct / 100.0
    if sigma <= 0:
        return None

    # Risk-neutral digital: no drift — this is the market's probability
    t = max(hours_to_expiry, 0.05) / 8760.0
    sigma_t = sigma * math.sqrt(t)
    if sigma_t < 1e-9:
        return 1.0 if spot >= strike else 0.0
    d2 = (math.log(spot / strike) - 0.5 * sigma * sigma * t) / sigma_t
    return _ND().cdf(d2)


async def implied_prob_above(strike: float, hours_to_expiry: float, spot: float) -> float | None:
    """Options-implied P(BTC > strike at the Kalshi expiry), or None on any failure."""
    chain = await _fetch_chain()
    if not chain:
        return None
    try:
        return _implied_prob_from_chain(chain, strike, hours_to_expiry, spot)
    except Exception:
        return None
