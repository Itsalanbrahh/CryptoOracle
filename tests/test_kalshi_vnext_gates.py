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