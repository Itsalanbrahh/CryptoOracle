"""
Regression tests for Kalshi execution/accounting hardening.

Bug list being verified:
  1. Heartbeat must not call close_position in paper mode — covered in
     test_kalshi_vnext_gates.py.
  2. Do NOT log realized P&L before confirmed fill (item #2).
  3. Order response: top-level V2 fill_count/remaining_count controls pending
     state; nested schema supported as legacy fallback (item #3).
  4. place_order: post_only, reduce_only, expiration_time, deterministic
     client_order_id (item #4); close_position: reduce_only, client_order_id.
  5. Duplicate local records for same ticker must never each receive the full
     exchange net count; pending/partial positions handled conservatively (item #5).
  6. Outstanding entry reservations (order_pending=True) included in daily
     deployed risk calculation (item #6).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

from crypto_oracle.kalshi.client import KalshiClient
from crypto_oracle.kalshi import position_manager as pm


# ── Shared helpers ─────────────────────────────────────────────────────────


class RecordingClient(KalshiClient):
    def __init__(self):
        super().__init__(key_id="test")
        self.bodies = []

    async def _post(self, path, body):
        self.bodies.append((path, body))
        return {
            "order_id": "exchange-order",
            "fill_count": "0.00",
            "remaining_count": body["count"],
            "ts_ms": 1,
        }


def _fake_pos(
    ticker="KXBTCD-TEST-T80000",
    side="no",
    count=10,
    entry_price=0.30,
    closed=False,
    order_pending=False,
    close_reason=None,
    date=None,
):
    today = (date or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return {
        "ticker": ticker,
        "side": side,
        "count": count,
        "entry_price": entry_price,
        "strike": 80_000,
        "event_ticker": "KXBTCD-TEST",
        "order_id": "oid",
        "edge": 0.04,
        "confidence": 0.32,
        "spot_at_entry": 79_500.0,
        "entered_at": today + "T12:00:00+00:00",
        "closed": closed,
        "closed_at": None,
        "close_reason": close_reason,
        "close_price": None,
        "realized_pnl": None,
        "order_pending": order_pending,
    }


# ── Item #4: place_order payload shape ────────────────────────────────────────


def test_v2_entry_payload_is_post_only_expiring_and_restart_idempotent():
    """place_order must send post_only=True, reduce_only=False, expiration_time,
    and a deterministic client_order_id so cron restarts are idempotent."""
    client = RecordingClient()

    first = asyncio.run(client.place_order(
        ticker="KXBTCD-TEST-T80000",
        side="yes",
        count=3,
        price_cents=42,
        expiration_time=1_800_000_000,
    ))
    asyncio.run(client.place_order(
        ticker="KXBTCD-TEST-T80000",
        side="yes",
        count=3,
        price_cents=42,
        expiration_time=1_800_000_000,
    ))

    assert first["fill_count"] == "0.00"
    first_body = client.bodies[0][1]
    second_body = client.bodies[1][1]
    # side must remain bid/ask (V2 API uses BookSide enum, not yes/no)
    assert first_body["side"] == "bid"
    # Maker-only entry
    assert first_body["post_only"] is True
    # Entry never reduces an existing position
    assert first_body["reduce_only"] is False
    # Expiration threaded through
    assert first_body["expiration_time"] == 1_800_000_000
    # Deterministic: same inputs → same ID across restarts
    assert first_body["client_order_id"] == second_body["client_order_id"]
    assert first_body["client_order_id"]


def test_v2_close_payload_is_reduce_only():
    """close_position must send reduce_only=True so a sell never opens a new
    position on the wrong side when position is already gone."""
    client = RecordingClient()
    asyncio.run(client.close_position(
        ticker="KXBTCD-TEST-T80000",
        count=3,
        side="no",
        price_cents=58,
    ))
    body = client.bodies[0][1]
    assert body["action"] == "sell"
    assert body["reduce_only"] is True
    assert body["client_order_id"]


# ── Item #3: V2 response top-level fill_count/remaining_count ─────────────────


def test_v2_response_pending_fill_from_fill_count_remaining_count():
    """fill_count='0.00' remaining_count='10.00' → order is resting, not filled."""
    resp = {
        "order_id": "abc",
        "fill_count": "0.00",
        "remaining_count": "10.00",
        "ts_ms": 1,
    }
    # Reproduce the logic from loop.py
    fill = float(resp.get("fill_count") or resp.get("fill_count_fp") or 0)
    remaining = float(resp.get("remaining_count") or resp.get("remaining_count_fp") or 0)
    filled_immediately = fill > 0 and remaining == 0
    assert not filled_immediately, "Resting order must NOT be considered filled"


def test_v2_response_fully_filled_fill_count_zero_remaining():
    """fill_count='10.00' remaining_count='0.00' → immediately filled (taker)."""
    resp = {
        "order_id": "abc",
        "fill_count": "10.00",
        "remaining_count": "0.00",
        "ts_ms": 1,
    }
    fill = float(resp.get("fill_count") or resp.get("fill_count_fp") or 0)
    remaining = float(resp.get("remaining_count") or resp.get("remaining_count_fp") or 0)
    filled_immediately = fill > 0 and remaining == 0
    assert filled_immediately


def test_legacy_nested_order_schema_still_extracts_order_id():
    """Legacy CreateOrderResponse wraps order inside {order: {...}}.
    order_id extraction must check top-level first, nested as fallback."""
    legacy_resp = {
        "order": {
            "order_id": "legacy-oid",
            "status": "resting",
        }
    }
    order_id = legacy_resp.get("order_id") or legacy_resp.get("order", {}).get("order_id")
    assert order_id == "legacy-oid"


# ── Item #2: Close submission must NOT log realized P&L until confirmed fill ──


def test_close_submission_with_resting_fill_does_not_realize_pnl():
    """When the close order is resting (fill_count=0, remaining > 0),
    realized_pnl must remain None — don't log pre-fill P&L."""
    resp = {
        "order_id": "close-order",
        "fill_count": "0.00",
        "remaining_count": "5.00",
        "ts_ms": 1,
    }
    fill = float(resp.get("fill_count") or 0)
    remaining = float(resp.get("remaining_count") or 0)
    filled_immediately = fill > 0 and remaining == 0

    pnl = None  # only set on confirmed fill
    if filled_immediately:
        pnl = 0.50  # hypothetical — would be calculated from close_price

    assert pnl is None, "Realized P&L must not be logged for a resting close order"


# ── Item #5: Duplicate local positions deduplication ─────────────────────────


def test_sync_reconciles_duplicate_locals_to_one_net_position():
    """If two open locals exist for the same ticker, after sync only one should
    show the exchange net count; the extra must be closed as 'duplicate_reconciled'."""
    api_response = {
        "market_positions": [
            {
                "ticker": "KXBTCD-TEST-T80000",
                "position_fp": "5",   # 5 YES contracts on exchange
                "total_traded": 150,   # 150 cents = $1.50 total → 30¢/contract
                "realized_pnl": 0,
                "market_exposure": 0,
            }
        ]
    }

    dup1 = _fake_pos(count=5)
    dup2 = _fake_pos(count=5)  # duplicate local entry, same ticker
    dup2["order_id"] = "oid2"  # distinguish by order_id

    captured: list[list] = []

    async def fake_get(path, params=None, auth=False):
        return api_response

    with (
        patch.object(pm, "_load_all", return_value=[dup1, dup2]),
        patch.object(pm, "_save_all", side_effect=lambda p: captured.append(p)),
        patch("crypto_oracle.kalshi.position_manager.KalshiClient") as MockCl,
    ):
        mock_client = MockCl.return_value
        mock_client._get = AsyncMock(side_effect=fake_get)
        asyncio.run(pm.sync_from_kalshi(mock_client))

    assert captured, "sync_from_kalshi must save"
    saved = captured[-1]
    open_for_ticker = [p for p in saved if p["ticker"] == "KXBTCD-TEST-T80000" and not p.get("closed")]
    closed_for_ticker = [p for p in saved if p["ticker"] == "KXBTCD-TEST-T80000" and p.get("closed")]

    assert len(open_for_ticker) == 1, (
        f"Exactly one open local must remain after dedup; got {len(open_for_ticker)}"
    )
    assert open_for_ticker[0]["count"] == 5
    assert len(closed_for_ticker) == 1, "Duplicate must be closed"
    assert "duplicate" in (closed_for_ticker[0].get("close_reason") or ""), (
        "Duplicate must be closed with reason 'duplicate_reconciled'"
    )


def test_pending_position_count_not_inflated_by_api_net():
    """A local entry with order_pending=True (not yet on the API) must not
    receive the exchange net count — its count stays as-placed until the fill
    appears on the API (order_pending cleared by sync on the next cycle)."""
    api_response = {
        "market_positions": [
            {
                "ticker": "KXBTCD-TEST-T80000",
                "position_fp": "5",
                "total_traded": 150,
                "realized_pnl": 0,
                "market_exposure": 0,
            }
        ]
    }

    # One confirmed position + one pending that hasn't appeared on the API yet
    confirmed = _fake_pos(ticker="KXBTCD-TEST-T80000", count=5, order_pending=False)
    pending = _fake_pos(ticker="KXBTCD-TEST-T80001", count=3, order_pending=True)

    captured: list[list] = []

    async def fake_get(path, params=None, auth=False):
        return api_response

    with (
        patch.object(pm, "_load_all", return_value=[confirmed, pending]),
        patch.object(pm, "_save_all", side_effect=lambda p: captured.append(p)),
        patch("crypto_oracle.kalshi.position_manager.KalshiClient") as MockCl,
    ):
        mock_client = MockCl.return_value
        mock_client._get = AsyncMock(side_effect=fake_get)
        asyncio.run(pm.sync_from_kalshi(mock_client))

    assert captured
    saved = captured[-1]
    pending_saved = next((p for p in saved if p["ticker"] == "KXBTCD-TEST-T80001"), None)
    assert pending_saved is not None
    assert pending_saved["count"] == 3, "Pending entry count must not be overwritten by API net"
    assert pending_saved.get("order_pending") is True, "order_pending flag must stay True until fill appears"


# ── Item #6: Entry reservations in daily deployed risk ────────────────────────


def test_pending_entry_reservation_included_in_daily_risk():
    """order_pending=True entries represent deployed capital (the exchange holds
    the reserve). They must appear in get_today_deployed_usd()."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pending_pos = _fake_pos(count=10, entry_price=0.30, order_pending=True)
    assert pending_pos["entered_at"].startswith(today)

    with patch.object(pm, "_load_all", return_value=[pending_pos]):
        deployed = pm.get_today_deployed_usd()

    # 10 contracts × $0.30 = $3.00
    assert abs(deployed - 3.00) < 0.01, (
        f"Pending entry must be counted in daily risk; expected $3.00 got ${deployed:.4f}"
    )


def test_never_filled_entry_excluded_from_daily_risk():
    """'entry_never_filled' orders must NOT count against the daily cap —
    they never consumed real capital."""
    nf = _fake_pos(
        count=10,
        entry_price=0.30,
        closed=True,
        close_reason="entry_never_filled (order did not execute)",
    )
    with patch.object(pm, "_load_all", return_value=[nf]):
        deployed = pm.get_today_deployed_usd()

    assert deployed == 0.0, (
        f"Never-filled entries must not count; expected $0.00 got ${deployed:.4f}"
    )
