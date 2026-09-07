from crypto_oracle.kalshi.settlement import (
    apply_portfolio_settlement,
    official_market_result,
)


def test_unknown_official_result_stays_unknown():
    assert official_market_result({"status": "closed", "result": ""}) is None
    assert official_market_result({"status": "settled"}) is None
    assert official_market_result({"status": "open", "result": "yes"}) is None


def test_range_uses_official_result_and_portfolio_economics_not_spot_proxy():
    position = {
        "ticker": "KXBTC-RANGE-B80000",
        "side": "no",
        "count": 2,
        "entry_price": 0.25,
        "contract_type": "range",
        "floor_strike": 79_500.0,
        "cap_strike": 80_000.0,
        "close_time": "2026-09-07T17:00:00Z",
        "closed": False,
        "realized_pnl": None,
    }
    settlement = {
        "ticker": position["ticker"],
        "market_result": "no",
        "yes_count_fp": "0.00",
        "yes_total_cost_dollars": "0.0000",
        "no_count_fp": "2.00",
        "no_total_cost_dollars": "0.5000",
        "revenue": 200,
        "fee_cost": "0.0600",
        "settled_time": "2026-09-07T17:03:00Z",
    }

    settled = apply_portfolio_settlement(position, settlement)

    assert settled["official_result"] == "no"
    assert settled["realized_pnl"] == 1.44
    assert settled["floor_strike"] == 79_500.0
    assert settled["cap_strike"] == 80_000.0


def test_position_preserves_contract_metadata():
    from crypto_oracle.kalshi.position_manager import make_position

    position = make_position(
        ticker="KXBTC-RANGE-B80000",
        side="yes",
        count=1,
        entry_price=0.40,
        strike=79_500,
        event_ticker="KXBTC-RANGE",
        order_id="order-1",
        edge=0.08,
        confidence=0.70,
        spot_at_entry=79_750,
        contract_type="range",
        floor_strike=79_500,
        cap_strike=80_000,
        close_time="2026-09-07T17:00:00Z",
    )

    assert position["contract_type"] == "range"
    assert position["floor_strike"] == 79_500
    assert position["cap_strike"] == 80_000
    assert position["close_time"] == "2026-09-07T17:00:00Z"


def test_live_entry_gate_blocks_when_fewer_than_100_unique_official_resolutions():
    from crypto_oracle.kalshi.settlement import live_entry_gate

    fake_settlements = [
        {
            "ticker": f"KXBTCD-FAKE-T{60000 + i}",
            "market_result": "no",
            "revenue": 100,
            "settled_time": f"2026-09-0{(i % 9) + 1}T17:00:00Z",
            "fee_cost": "0.0200",
            "yes_count_fp": "0.00",
            "yes_total_cost_dollars": "0.0000",
            "no_count_fp": "1.00",
            "no_total_cost_dollars": "0.5000",
        }
        for i in range(99)
    ]

    allowed, reason = live_entry_gate(fake_settlements)
    assert not allowed
    assert "100" in reason


def test_live_entry_gate_allows_when_100_unique_official_resolutions():
    from crypto_oracle.kalshi.settlement import live_entry_gate

    fake_settlements = [
        {
            "ticker": f"KXBTCD-FAKE-T{60000 + i}",
            "market_result": "no" if i % 2 else "yes",
            "revenue": 100,
            "settled_time": f"2026-08-01T17:00:00Z",
            "fee_cost": "0.0200",
            "yes_count_fp": "0.00",
            "yes_total_cost_dollars": "0.0000",
            "no_count_fp": "1.00",
            "no_total_cost_dollars": "0.5000",
        }
        for i in range(100)
    ]

    allowed, reason = live_entry_gate(fake_settlements)
    assert allowed
    assert reason == ""


def test_paper_scan_always_allowed():
    from crypto_oracle.kalshi.settlement import live_entry_gate

    allowed, _ = live_entry_gate([], require_live=False)
    assert allowed
