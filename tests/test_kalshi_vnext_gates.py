from crypto_oracle.kalshi.markets import KalshiMarket
from crypto_oracle.kalshi.postmortem import _strategy_version
from crypto_oracle.kalshi.settlement import yes_outcome_at_settlement
from crypto_oracle.kalshi.strategy import decide_kalshi_trade, directional_consensus


def _market(*, yes_bid=0.18, yes_ask=0.20, no_bid=0.79, no_ask=0.81):
    return KalshiMarket(
        ticker="KXBTCD-TEST-T80000",
        strike=80_000,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        mid=(yes_bid + yes_ask) / 2,
        volume=1_000,
        close_time=None,
    )


def test_requires_options_implied_anchor_when_enabled():
    decision = decide_kalshi_trade(
        _market(),
        aggregate=0.8,
        confidence=0.70,
        spot=79_500,
        min_edge=0.03,
        min_confidence=0.30,
        require_implied_prob=True,
    )

    assert decision.action == "HOLD"
    assert "options-implied anchor unavailable" in decision.reasoning


def test_rejects_cheap_no_contract_below_configured_floor():
    decision = decide_kalshi_trade(
        _market(yes_bid=0.88, yes_ask=0.90, no_bid=0.10, no_ask=0.12),
        aggregate=0.0,
        confidence=0.70,
        spot=79_500,
        implied_prob=0.60,
        min_edge=0.03,
        min_confidence=0.30,
        min_no_price=0.25,
    )

    assert decision.action == "HOLD"
    assert "NO execution price" in decision.reasoning


def test_rejects_implausibly_large_model_edge():
    decision = decide_kalshi_trade(
        _market(),
        aggregate=0.0,
        confidence=0.70,
        spot=79_500,
        implied_prob=0.60,
        min_edge=0.03,
        min_confidence=0.30,
        max_edge=0.20,
    )

    assert decision.action == "HOLD"
    assert "exceeds calibrated maximum" in decision.reasoning


def test_directional_consensus_requires_three_strong_agents():
    signals = {
        "KnowledgeMarket": {"score": -0.8},
        "KronosMarket": {"score": -0.4},
        "DynamicSR": {"score": -0.2},
        "MomentumContinuation": {"score": 0.6},
    }

    assert directional_consensus(signals, "no", min_agree=3) == (True, 3)
    assert directional_consensus(signals, "yes", min_agree=3) == (False, 1)


def test_postmortem_strategy_version_is_explicit(monkeypatch):
    monkeypatch.setenv("KALSHI_STRATEGY_VERSION", "vnext-paper-1")
    assert _strategy_version() == "vnext-paper-1"


def test_range_contract_settles_only_inside_floor_and_cap():
    assert yes_outcome_at_settlement(79_750, 79_500, 80_000) is True
    assert yes_outcome_at_settlement(80_100, 79_500, 80_000) is False
    assert yes_outcome_at_settlement(79_400, 79_500, 80_000) is False


# ─── Bug-regression tests (verified reproductions, must stay red until fix) ───

def test_heartbeat_does_not_submit_close_orders_in_paper_mode():
    """Bug: close_position() was called unconditionally; ignores live=False."""
    import ast
    from pathlib import Path

    src = Path("/Users/alanruelas/.hermes/scripts/kalshi_position_heartbeat.py").read_text()
    tree = ast.parse(src)

    class CloseSiteVisitor(ast.NodeVisitor):
        def __init__(self):
            self.guarded = True  # assume guarded until we find a bad call

        def visit_If(self, node):
            # Is this `if live:` or `if not live:`?
            live_guard = (
                isinstance(node.test, ast.Name) and node.test.id == "live"
            ) or (
                isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == "live"
            )
            if not live_guard:
                # Check if close_position is called inside an un-guarded block
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute) and func.attr == "close_position":
                            self.guarded = False
            self.generic_visit(node)

    visitor = CloseSiteVisitor()
    visitor.visit(tree)
    assert visitor.guarded, (
        "close_position() must be inside an 'if live:' guard in the heartbeat"
    )


def test_sync_from_kalshi_uses_cents_field_for_entry_price():
    """Bug: sync_from_kalshi read 'total_traded_dollars' (non-existent field).

    Kalshi returns 'total_traded' in cents.  The code must convert it:
      entry_price = total_traded_cents / 100 / count
    """
    from crypto_oracle.kalshi import position_manager as pm
    from unittest.mock import patch, AsyncMock
    import asyncio

    api_response = {
        "market_positions": [
            {
                "ticker": "KXBTCD-TEST-T80000",
                "position_fp": "10",
                "total_traded": 300,  # 300 cents = $3.00 total; 30¢/contract
                "realized_pnl": 0,
                "market_exposure": 0,
            }
        ]
    }

    async def fake_get(path, params=None, auth=False):
        return api_response

    captured: list[list] = []

    def fake_save(positions):
        captured.append(positions)

    with (
        patch.object(pm, "_load_all", return_value=[]),
        patch.object(pm, "_save_all", side_effect=fake_save),
        patch("crypto_oracle.kalshi.position_manager.KalshiClient") as MockClient,
    ):
        mock_client = MockClient.return_value
        mock_client._get = AsyncMock(side_effect=fake_get)
        asyncio.run(pm.sync_from_kalshi(mock_client))

    assert captured, "sync_from_kalshi never called _save_all"
    saved_positions = captured[-1]
    assert saved_positions, "No positions were saved"
    ep = saved_positions[-1]["entry_price"]
    assert 0.25 < ep < 0.35, (
        f"entry_price should be ~$0.30 (300 cents / 100 / 10 contracts) but got {ep}"
    )


def test_daily_deployed_correctly_counts_synced_position():
    """Bug: if entry_price is 0 (from broken sync), deployed_today returns $0
    and the daily risk cap never fires."""
    from datetime import datetime, timezone
    from unittest.mock import patch
    from crypto_oracle.kalshi import position_manager as pm

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    fake_pos = {
        "ticker": "KXBTCD-TEST-T80000",
        "side": "no",
        "count": 10,
        "entry_price": 0.30,
        "strike": 80_000,
        "event_ticker": "KXBTCD-TEST",
        "order_id": "",
        "edge": 0.04,
        "confidence": 0.32,
        "spot_at_entry": 79_500.0,
        "entered_at": today + "T12:00:00+00:00",
        "closed": False,
        "closed_at": None,
        "close_reason": None,
        "close_price": None,
        "realized_pnl": None,
    }
    with patch.object(pm, "_load_all", return_value=[fake_pos]):
        deployed = pm.get_today_deployed_usd()

    assert abs(deployed - 3.00) < 0.01, f"Expected $3.00 deployed, got ${deployed:.4f}"