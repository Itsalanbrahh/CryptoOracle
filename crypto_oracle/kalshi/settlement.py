"""Pure settlement helpers shared by paper and live calibration paths."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation


_FINAL_MARKET_STATUSES = {"determined", "amended", "finalized"}


def official_market_result(market: dict) -> str | None:
    """Return an official binary result only after Kalshi determines it."""
    result = str(market.get("result") or "").lower()
    status = str(market.get("status") or "").lower()
    if status in _FINAL_MARKET_STATUSES and result in {"yes", "no"}:
        return result
    return None


def apply_portfolio_settlement(position: dict, settlement: dict) -> dict:
    """Return a position resolved from Kalshi's authenticated Settlement row.

    The Settlement schema is the account-level source of truth: revenue is in
    cents, while YES/NO total costs and fee_cost are fixed-point dollar strings.
    Invalid, scalar, ticker-mismatched, or zero-fill rows leave the position
    unresolved rather than manufacturing a loss.
    """
    updated = dict(position)
    result = str(settlement.get("market_result") or "").lower()
    if result not in {"yes", "no"} or settlement.get("ticker") != position.get("ticker"):
        return updated
    try:
        yes_count = Decimal(str(settlement["yes_count_fp"]))
        no_count = Decimal(str(settlement["no_count_fp"]))
        yes_cost = Decimal(str(settlement["yes_total_cost_dollars"]))
        no_cost = Decimal(str(settlement["no_total_cost_dollars"]))
        fees = Decimal(str(settlement["fee_cost"]))
        revenue = Decimal(int(settlement["revenue"])) / Decimal(100)
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return updated
    if yes_count + no_count <= 0:
        return updated

    total_cost = yes_cost + no_cost
    updated.update(
        {
            "closed": True,
            "closed_at": settlement.get("settled_time"),
            "close_reason": "official_kalshi_settlement",
            "close_price": 1.0 if position.get("side") == result else 0.0,
            "official_result": result,
            "official_settled_time": settlement.get("settled_time"),
            "official_resolution_id": f"{position.get('ticker')}:{settlement.get('settled_time')}",
            "settlement_source": "kalshi_portfolio_settlements",
            "fill_confirmed": True,
            "confirmed_yes_count": float(yes_count),
            "confirmed_no_count": float(no_count),
            "confirmed_fill_cost_usd": float(total_cost),
            "confirmed_fees_usd": float(fees),
            "settlement_revenue_usd": float(revenue),
            "realized_pnl": float(revenue - total_cost - fees),
        }
    )
    return updated


_MIN_LIVE_RESOLUTIONS = 100


def live_entry_gate(
    settlements: list[dict],
    *,
    require_live: bool = True,
    min_resolutions: int = _MIN_LIVE_RESOLUTIONS,
) -> tuple[bool, str]:
    """Return (allowed, reason) for placing live orders.

    Paper scans (require_live=False) are always permitted.
    Live orders require at least `min_resolutions` uniquely attributed official
    settlements with valid market_result ∈ {yes, no} to calibrate on.

    Deduplication by ticker ensures a single market counted multiple times
    (e.g., from API pagination) doesn't inflate the readiness count.
    """
    if not require_live:
        return True, ""

    valid_results = {"yes", "no"}
    seen_tickers: set[str] = set()
    for s in settlements:
        result = str(s.get("market_result") or "").lower()
        ticker = str(s.get("ticker") or "")
        if result in valid_results and ticker and ticker not in seen_tickers:
            seen_tickers.add(ticker)

    count = len(seen_tickers)
    if count < min_resolutions:
        return (
            False,
            f"live_entry_gate: {count} unique official resolutions < {min_resolutions} required; "
            "use paper mode until the account has seen ≥100 settled markets",
        )
    return True, ""


def yes_outcome_at_settlement(
    settle_price: float, strike: float, cap_strike: float | None = None
) -> bool:
    """Return whether YES wins for a directional threshold or range contract."""
    if cap_strike is not None:
        return strike <= settle_price < cap_strike
    return settle_price >= strike