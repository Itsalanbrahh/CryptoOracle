"""
Position manager for Kalshi BTC bot — tracks open positions, stop-loss,
take-profit, and rebalancing.

All positions are persisted to a JSON file so they survive restarts.
Each tick the manager checks open positions against live market mid-prices
and recommends closes when thresholds are triggered.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from .client import KalshiClient

_POSITIONS_PATH = Path.home() / ".hermes" / "state" / "kalshi_positions.json"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw and raw.strip():
        try:
            return float(raw.strip())
        except ValueError:
            pass
    return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Position data model ──────────────────────────────────────────────────────


def make_position(
    ticker: str,
    side: str,            # "yes" or "no"
    count: int,
    entry_price: float,   # per-contract cost in dollars
    strike: float,
    event_ticker: str,
    order_id: str | None,
    edge: float,
    confidence: float,
    spot_at_entry: float,
) -> dict:
    """Create a new position dict for persistence."""
    return {
        "ticker": ticker,
        "side": side,
        "count": count,
        "entry_price": round(entry_price, 4),
        "strike": strike,
        "event_ticker": event_ticker,
        "order_id": order_id or "",
        "edge": round(edge, 4),
        "confidence": round(confidence, 4),
        "spot_at_entry": round(spot_at_entry, 2),
        "entered_at": _now_iso(),
        "closed": False,
        "closed_at": None,
        "close_reason": None,
        "close_price": None,
        "realized_pnl": None,
    }


# ── Persistence ──────────────────────────────────────────────────────────────


def _load_all() -> list[dict]:
    if not _POSITIONS_PATH.exists():
        return []
    try:
        return json.loads(_POSITIONS_PATH.read_text())
    except (json.JSONDecodeError, ValueError):
        return []


def _save_all(positions: list[dict]) -> None:
    _POSITIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _POSITIONS_PATH.write_text(json.dumps(positions, indent=2))


def get_open_positions() -> list[dict]:
    """Return all positions that are not yet closed."""
    return [p for p in _load_all() if not p.get("closed")]


def get_open_count() -> int:
    return len(get_open_positions())


def get_closed_today() -> list[dict]:
    """Return positions closed today (for daily cap accounting)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return [
        p for p in _load_all()
        if p.get("closed") and p.get("closed_at", "").startswith(today)
    ]


def _never_filled(p: dict) -> bool:
    """True if this entry's order never executed — it was never a real position."""
    return (p.get("close_reason") or "").startswith("entry_never_filled")


def get_entry_count_today() -> int:
    """Number of bot-placed positions opened today (for entry cap).

    Excludes manual trades and never-filled entries — an order that rested
    and was canceled shouldn't consume the daily entry budget.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return sum(
        1 for p in _load_all()
        if (
            p.get("entered_at", "").startswith(today)
            and not _never_filled(p)
            and p.get("confidence", 0) > 0  # bot-placed only
            and p.get("edge", 0) > 0         # bot-placed only
        )
    )


def get_today_deployed_usd() -> float:
    """Sum of entry costs (in dollars) for bot-placed positions opened today.
    Excludes manual trades and never-filled orders.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total_cents = sum(
        p["entry_price"] * p["count"]
        for p in _load_all()
        if (
            p.get("entered_at", "").startswith(today)
            and not _never_filled(p)
            and p.get("confidence", 0) > 0  # bot-placed only
            and p.get("edge", 0) > 0         # bot-placed only
        )
    )
    return round(total_cents / 100.0, 2)


def save_new_position(pos: dict) -> None:
    positions = _load_all()
    positions.append(pos)
    _save_all(positions)


async def sync_from_kalshi(client: KalshiClient | None = None) -> int:
    """Fetch real open positions from Kalshi API and merge into positions.json.

    Reconciles the local positions file with what Kalshi actually holds.
    Positions found on API but not locally: added with estimated entry prices.
    Positions in local file but settled on API: marked closed.
    Returns the number of open positions after sync.
    """
    try:
        c = client or KalshiClient()
        resp = await c._get("/portfolio/positions", params={"limit": 100}, auth=True)
        api_markets = resp.get("market_positions", [])
    except Exception as e:
        print(f"[Kalshi/SYNC] Failed to fetch positions from API: {e}")
        return len(get_open_positions())

    # Build ticker → API position map
    api_by_ticker: dict[str, dict] = {}
    for mp in api_markets:
        pf = float(mp.get("position_fp", 0))
        if pf == 0:
            continue  # fully closed
        side = "no" if pf < 0 else "yes"
        count = int(abs(pf))
        ticker = mp["ticker"]
        # Parse ticker → event_ticker + strike
        parts = ticker.split("-")
        event_ticker = "-".join(parts[:2])
        try:
            strike = float(parts[-1][1:])  # strip T/B prefix
        except (ValueError, IndexError):
            strike = 0.0
        traded = float(mp.get("total_traded_dollars", 0))
        realized = float(mp.get("realized_pnl_dollars", 0))
        exposure = float(mp.get("market_exposure_dollars", 0))
        if realized > 0:
            # Partially filled — total_traded includes sale proceeds,
            # so remaining_cost doesn't equal entry_price × count.
            # Use current exposure value as a proxy for estimated entry cost.
            entry_price = round(exposure / count, 4) if count > 0 else 0.0
        else:
            # Fully open — total_traded IS the entry cost
            entry_price = round(traded / count, 4) if count > 0 else 0.0
        api_by_ticker[ticker] = {
            "ticker": ticker,
            "side": side,
            "count": count,
            "entry_price": entry_price,
            "strike": strike,
            "event_ticker": event_ticker,
        }

    local_positions = _load_all()
    api_tickers = set(api_by_ticker.keys())
    local_open = {p["ticker"] for p in local_positions if not p.get("closed")}
    local_all = {p["ticker"] for p in local_positions}

    added = 0
    removed = 0
    updated = 0
    reopened = 0
    new_positions: list[dict] = []

    for p in local_positions:
        ticker = p["ticker"]
        if not p.get("closed") and ticker not in api_tickers:
            p["closed"] = True
            p["closed_at"] = _now_iso()
            if p.get("order_pending"):
                # Entry order never executed (resting maker order that was
                # canceled or expired). NOT a settled position — must not be
                # PnL-logged by the heartbeat or counted by the calibration.
                # If the order actually fills later, the reopen branch below
                # restores it when it appears on the API.
                p["close_reason"] = "entry_never_filled (order did not execute)"
                p["realized_pnl"] = None
            else:
                # Filled position no longer on API → settled or fully closed.
                # Leave realized_pnl as None — the heartbeat resolves it with
                # the current BTC price proxy.
                p["close_reason"] = "settled (no longer on API)"
                p["realized_pnl"] = None
            removed += 1
        elif p.get("closed") and ticker in api_tickers:
            # Position was incorrectly marked closed but is still alive on API → re-open
            ap = api_by_ticker[ticker]
            p["closed"] = False
            p["closed_at"] = None
            p["close_reason"] = None
            p["close_price"] = None
            p["realized_pnl"] = None
            p["count"] = ap["count"]
            p["order_pending"] = False  # it's on the API — the fill happened
            if p["entry_price"] == 0 and ap["entry_price"] > 0:
                p["entry_price"] = ap["entry_price"]
            reopened += 1
        elif not p.get("closed") and ticker in api_tickers:
            # Update count/entry from API in case of partial fills
            ap = api_by_ticker[ticker]
            p["count"] = ap["count"]
            p["order_pending"] = False  # confirmed filled
            if p["entry_price"] == 0 and ap["entry_price"] > 0:
                p["entry_price"] = ap["entry_price"]
            updated += 1
        new_positions.append(p)

    # Add API positions not in local file — try to recover entry data from postmortem
    POSTMORTEM_PATH = Path.home() / ".hermes" / "state" / "kalshi_postmortem.jsonl"
    postmortem_entries: list[dict] = []
    if POSTMORTEM_PATH.exists():
        try:
            for line in POSTMORTEM_PATH.read_text().splitlines():
                if line.strip():
                    postmortem_entries.append(json.loads(line))
        except Exception:
            pass
    for ticker, ap in api_by_ticker.items():
        if ticker not in local_all:
            # Search postmortem for this ticker's BUY entry
            spot_at_entry = 0.0
            entry_edge = 0.0
            entry_conf = 0.0
            for e in postmortem_entries:
                if e.get("ticker") == ticker and e.get("action", "").startswith("BUY"):
                    spot_at_entry = e.get("spot_price", 0) or 0
                    entry_edge = e.get("edge", 0) or 0
                    entry_conf = e.get("confidence", 0) or 0
                    break
            new_pos = make_position(
                ticker=ap["ticker"],
                side=ap["side"],
                count=ap["count"],
                entry_price=ap["entry_price"],
                strike=ap["strike"],
                event_ticker=ap["event_ticker"],
                order_id=None,
                edge=entry_edge,
                confidence=entry_conf,
                spot_at_entry=spot_at_entry,
            )
            new_positions.append(new_pos)
            added += 1

    _save_all(new_positions)
    open_count = len(get_open_positions())

    print(
        f"[Kalshi/SYNC] API returned {len(api_markets)} markets, "
        f"{len(api_by_ticker)} with open positions. "
        f"Added={added} Reopened={reopened} Removed(settled)={removed} Updated={updated} "
        f"Open={open_count}"
    )
    return open_count


def close_position(ticker: str, reason: str, close_price: float) -> dict | None:
    """Mark a position as closed with reason and price. Returns the closed pos or None."""
    positions = _load_all()
    for p in positions:
        if p["ticker"] == ticker and not p.get("closed"):
            p["closed"] = True
            p["closed_at"] = _now_iso()
            p["close_reason"] = reason
            p["close_price"] = round(close_price, 4)
            entry_total = p["entry_price"] * p["count"]
            close_total = close_price * p["count"]
            if p["side"] == "yes":
                p["realized_pnl"] = round(close_total - entry_total, 2)
            else:
                # For NO positions: bought at entry_price, sold at close_price
                # PnL = (sale_price - purchase_price) * count
                p["realized_pnl"] = round(close_total - entry_total, 2)
            _save_all(positions)
            return p
    return None


# ── Stop-loss / take-profit logic ────────────────────────────────────────────


def _position_value_now(pos: dict, current_mid: float) -> float:
    """Current market value of a position at the given mid price."""
    if pos["side"] == "yes":
        return round(current_mid * pos["count"], 2)
    else:
        return round((1.0 - current_mid) * pos["count"], 2)


def _position_cost(pos: dict) -> float:
    return round(pos["entry_price"] * pos["count"], 2)


def _pnl_pct(pos: dict, current_mid: float) -> float:
    """Return P&L as fraction of entry cost. Negative = loss, positive = gain."""
    cost = _position_cost(pos)
    if cost <= 0:
        return 0.0
    value = _position_value_now(pos, current_mid)
    return (value - cost) / cost


async def check_stop_loss(
    position: dict,
    current_mid: float,
    stop_loss_pct: float = 0.50,
) -> tuple[bool, str]:
    """Check if a position should be stopped out.

    stop_loss_pct: fraction of entry cost lost before triggering (default 50%).
    Returns (should_close, reason_string).
    """
    pnl = _pnl_pct(position, current_mid)
    if pnl <= -stop_loss_pct:
        cost = _position_cost(position)
        value = _position_value_now(position, current_mid)
        return True, (
            f"stop_loss: pnl={pnl:.1%} loss=${cost - value:.2f} "
            f"entry={position['entry_price']:.4f} mid={current_mid:.4f}"
        )
    return False, ""


async def check_take_profit(
    position: dict,
    current_mid: float,
    take_profit_pct: float = 0.70,
) -> tuple[bool, str]:
    """Check if a position should take profit.

    take_profit_pct: fraction of max theoretical gain captured (default 70%).
    For YES: max gain = (1 - entry_price) * count; captured when mid rises.
    For NO:  max gain = (1 - entry_price) * count; captured when mid falls.
    Returns (should_close, reason_string).
    """
    cost = _position_cost(position)
    value = _position_value_now(position, current_mid)
    if cost <= 0:
        return False, ""

    if position["side"] == "yes":
        max_gain = (1.0 - position["entry_price"]) * position["count"]
        captured = (value - cost) / max_gain if max_gain > 0 else 0.0
    else:
        # NO bought at entry_price; wins $1 if BTC stays below strike.
        # Max gain = (1 - entry_price) * count, same structure as YES.
        max_gain = (1.0 - position["entry_price"]) * position["count"]
        captured = (value - cost) / max_gain if max_gain > 0 else 0.0

    if captured >= take_profit_pct:
        return True, (
            f"take_profit: captured={captured:.1%} of max gain "
            f"pnl=${value - cost:.2f} entry={position['entry_price']:.4f} mid={current_mid:.4f}"
        )
    return False, ""


async def check_rebalance(
    position: dict,
    current_mid: float,
    new_opportunity_edge: float,
    min_hold_minutes: float = 30.0,
    rebalance_edge_threshold: float = 1.5,
) -> tuple[bool, str]:
    """Check if a position should be closed to free capital for a better opportunity.

    Closes the lowest-edge open position if the new opportunity has
    rebalance_edge_threshold × higher edge.

    min_hold_minutes: don't rebalance positions held less than this long.
    Returns (should_close, reason_string).
    """
    cost = _position_cost(position)
    value = _position_value_now(position, current_mid)

    # Don't rebalance if held less than min hold time
    try:
        entered = datetime.fromisoformat(position["entered_at"])
        held_min = (datetime.now(timezone.utc) - entered).total_seconds() / 60
    except (ValueError, TypeError):
        held_min = min_hold_minutes + 1  # skip time check on parse failure

    if held_min < min_hold_minutes:
        return False, ""

    if value <= 0:
        return True, "rebalance: position has no remaining value"

    pnl = (value - cost) / cost
    # Only rebalance losing or barely-winning positions
    if pnl > 0.15:
        return False, ""  # don't rebalance strong winners

    if position["edge"] > 0 and new_opportunity_edge > position["edge"] * rebalance_edge_threshold:
        return True, (
            f"rebalance: new_edge={new_opportunity_edge:.3f} > "
            f"current_edge={position['edge']:.3f} × {rebalance_edge_threshold:.1f} "
            f"pnl={pnl:.1%}"
        )
    return False, ""


async def _fetch_btc_spot() -> float:
    """Fetch current BTC spot price from Kraken."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                data = await resp.json()
                return float(data["result"]["XXBTZUSD"]["c"][0])
    except Exception:
        return 0.0


async def check_strike_distance_stop_loss(
    position: dict,
    current_spot: float,
    strike_distance_pct: float = 0.50,
) -> tuple[bool, str]:
    """Exit early if BTC has moved strike_distance_pct of the way from entry to strike.

    For a NO position (betting BTC stays BELOW strike): the risk is BTC RISING toward
    the strike. Exit when BTC has risen halfway from entry toward strike.

    For a YES position (betting BTC stays ABOVE strike): the risk is BTC FALLING away
    from the strike. Exit when BTC has fallen halfway from entry down from strike.
    """
    spot_at_entry = position.get("spot_at_entry", 0)
    strike = position.get("strike", 0)
    if spot_at_entry <= 0 or strike <= 0 or current_spot <= 0:
        return False, ""

    if position["side"] == "no":
        # NO: bought because BTC is below strike. Risk = BTC rises toward strike.
        # Threshold: spot_at_entry + (strike - spot_at_entry) * pct
        dist = abs(strike - spot_at_entry)
        if dist < 10:
            return False, ""
        threshold = spot_at_entry + dist * strike_distance_pct
        if current_spot >= threshold:
            return True, (
                f"strike_dist_stop: BTC ${current_spot:.0f} rose "
                f"{strike_distance_pct:.0%} of distance toward strike ${strike:,.0f} "
                f"(entry=${spot_at_entry:,.0f}, dist=${dist:,.0f}, threshold=${threshold:,.0f})"
            )
    elif position["side"] == "yes":
        # YES: bought because BTC is near/above strike. Risk = BTC falls away from strike.
        # Threshold: spot_at_entry - (spot_at_entry - strike) * pct (if spot > strike)
        #            or: spot_at_entry - dist * pct if spot is below strike at entry
        dist = abs(spot_at_entry - strike)
        if dist < 10:
            return False, ""
        threshold = spot_at_entry - dist * strike_distance_pct
        if current_spot <= threshold:
            return True, (
                f"strike_dist_stop: BTC ${current_spot:.0f} fell "
                f"{strike_distance_pct:.0%} of distance away from strike ${strike:,.0f} "
                f"(entry=${spot_at_entry:,.0f}, dist=${dist:,.0f}, threshold=${threshold:,.0f})"
            )
    return False, ""


async def scan_positions(
    stop_loss_pct: float = 0.50,
    take_profit_pct: float = 0.70,
    strike_dist_stop_pct: float = 0.50,
    rebalance_min_hold: float = 30.0,
    rebalance_edge_mult: float = 1.5,
    best_new_edge: float = 0.0,
) -> list[dict]:
    """Run stop-loss, take-profit, and rebalance checks on all open positions.

    Returns a list of close actions to execute: [{"ticker", "reason", "close_price"}, ...]
    """
    client = KalshiClient()
    positions = get_open_positions()
    if not positions:
        return []

    actions: list[dict] = []

    # Fetch BTC spot once for strike-distance stop-loss checks
    btc_spot = await _fetch_btc_spot()

    for pos in positions:
        # Fetch current market mid for this position's ticker
        try:
            resp = await client._get(f"/markets/{pos['ticker']}")
        except Exception:
            continue  # skip if market data unavailable

        m = resp.get("market", {})
        yes_bid = float(m.get("yes_bid_dollars") or 0)
        yes_ask = float(m.get("yes_ask_dollars") or 1)
        current_mid = (yes_bid + yes_ask) / 2

        if current_mid <= 0 or current_mid >= 1:
            continue

        # 1. Stop-loss check
        should_close, reason = await check_stop_loss(pos, current_mid, stop_loss_pct)
        if should_close:
            # Determine close price: for YES positions sell at bid, for NO sell at (1 - ask)
            close_price = yes_bid if pos["side"] == "yes" else (1.0 - yes_ask)
            actions.append({"pos": pos, "reason": reason, "close_price": close_price})
            continue

        # 2. Take-profit check
        should_close, reason = await check_take_profit(pos, current_mid, take_profit_pct)
        if should_close:
            close_price = yes_bid if pos["side"] == "yes" else (1.0 - yes_ask)
            actions.append({"pos": pos, "reason": reason, "close_price": close_price})
            continue

        # 3. Strike-distance stop-loss: exit if BTC crossed 50% toward strike
        should_close, reason = await check_strike_distance_stop_loss(pos, btc_spot, strike_dist_stop_pct)
        if should_close:
            close_price = yes_bid if pos["side"] == "yes" else (1.0 - yes_ask)
            actions.append({"pos": pos, "reason": reason, "close_price": close_price})
            continue

        # 4. Rebalancing check (only if there's a better opportunity)
        if best_new_edge > 0:
            should_close, reason = await check_rebalance(
                pos, current_mid, best_new_edge, rebalance_min_hold, rebalance_edge_mult
            )
            if should_close:
                close_price = yes_bid if pos["side"] == "yes" else (1.0 - yes_ask)
                actions.append({"pos": pos, "reason": reason, "close_price": close_price})

    return actions
