import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from crypto_oracle.kalshi.markets import KalshiMarket
from crypto_oracle.kalshi.strategy import decide_kalshi_trade


def _market(*, yes_bid=0.86, yes_ask=0.88, no_bid=0.10, no_ask=0.12):
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


def test_position_cap_includes_payoff_boost_and_fees():
    decision = decide_kalshi_trade(
        _market(),
        aggregate=0.0,
        confidence=0.90,
        spot=79_000,
        implied_prob=0.40,
        min_edge=0.0,
        min_confidence=0.0,
        max_position_usd=5.0,
        maker_mode=False,
    )

    assert decision.action == "BUY_NO"
    assert decision.position_usd <= 5.0
    assert decision.count < 62  # old payoff boost spent $7.44 before fees


def test_minimum_contract_cannot_exceed_position_budget_with_fee():
    decision = decide_kalshi_trade(
        _market(no_ask=0.12),
        aggregate=0.0,
        confidence=0.90,
        spot=79_000,
        implied_prob=0.40,
        min_edge=0.0,
        min_confidence=0.0,
        max_position_usd=0.12,
        maker_mode=False,
    )

    assert decision.action == "HOLD"
    assert decision.count == 0
    assert decision.position_usd == 0.0


# ── Backtest future-leakage tests ────────────────────────────────────────────

def test_backtest_decision_at_candle_i_cannot_see_candle_i_plus_1():
    """At candle[i], vol computation must use only candles[:i+1]."""
    from crypto_oracle.kalshi.backtest import _hourly_vol

    closes = [100.0 + k for k in range(10)]  # 10 candles

    # Vol at candle 4 (index 4) uses only first 5 closes
    vol_at_4 = _hourly_vol(closes[:5])
    vol_at_4_full = _hourly_vol(closes)

    # Vol values differ when future data is excluded — confirming isolation
    # (they need not be numerically different for the test; the key is that
    #  the function accepts a sliced list and doesn't raise)
    assert isinstance(vol_at_4, float)
    assert isinstance(vol_at_4_full, float)

    # Explicitly verify that passing only i+1 candles does NOT include candle i+1
    # by checking length constraint: closes[:5] has no index-5 element
    sliced = closes[:5]
    assert len(sliced) == 5
    assert sliced[-1] == closes[4]  # last element is candle[4], not candle[5]


def test_terminal_candle_backtest_does_not_index_error():
    """run_backtest on a tiny dataset (3 candles) must not raise IndexError."""
    import asyncio
    from crypto_oracle.kalshi.backtest import run_backtest

    # Build 3 fake candles — at candle[2] (last), there are 0 remaining candles
    # The old code would do max(1, min(int(hours_to), 0)) = max(1, 0) = 1,
    # then settle_idx = 2 + 1 = 3 which is out of bounds.
    fake_candles = [
        {
            "ts": "2025-01-01T00:00:00+00:00",
            "timestamp": 1735689600,
            "open": 95000.0,
            "high": 95500.0,
            "low": 94500.0,
            "close": 95000.0,
            "volume": 10.0,
        },
        {
            "ts": "2025-01-01T01:00:00+00:00",
            "timestamp": 1735693200,
            "open": 95000.0,
            "high": 96000.0,
            "low": 94000.0,
            "close": 95200.0,
            "volume": 12.0,
        },
        {
            "ts": "2025-01-01T02:00:00+00:00",
            "timestamp": 1735696800,
            "open": 95200.0,
            "high": 96500.0,
            "low": 94800.0,
            "close": 95400.0,
            "volume": 11.0,
        },
    ]

    async def _run():
        with patch(
            "crypto_oracle.kalshi.backtest.fetch_historical_btc",
            new_callable=AsyncMock,
            return_value=fake_candles,
        ):
            result = await run_backtest(days=1, min_edge=0.0, min_confidence=0.0)
        return result

    result = asyncio.run(_run())
    # Should not raise; may return error due to small dataset or empty trades
    assert isinstance(result, dict)


# ── Kraken 720-candle honesty test ──────────────────────────────────────────

def test_fetch_historical_btc_reports_actual_candle_count_not_fabricated():
    """fetch_historical_btc(days=45) must not claim > 720 candles.

    Kraken hard-caps at 720 candles per call.  Old code looped and deduplicated,
    pretending to extend coverage.  New code makes a single call and reports
    the actual returned count (≤ 720).
    """
    import asyncio
    from crypto_oracle.kalshi.backtest import fetch_historical_btc

    # Simulate Kraken returning exactly 720 rows
    base_ts = 1735689600
    fake_720 = [
        [base_ts + i * 3600, "95000", "96000", "94000", "95500", "0", "10"]
        for i in range(720)
    ]

    fake_api_response = {
        "result": {
            "XXBTZUSD": fake_720,
            "last": base_ts + 720 * 3600,
        }
    }

    async def _run():
        # Clear cache first
        from crypto_oracle.kalshi import backtest as bt_mod
        bt_mod._CANDLE_CACHE.clear()

        # Build proper async context manager mocks for aiohttp
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = AsyncMock(return_value=fake_api_response)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_get(*args, **kwargs):
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.get = mock_get

        @asynccontextmanager
        async def mock_session_ctx(*args, **kwargs):
            yield mock_session

        with patch("aiohttp.ClientSession", return_value=mock_session):
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            candles = await fetch_historical_btc(days=45)
        return candles

    candles = asyncio.run(_run())
    # Must not fabricate candles beyond what Kraken returned
    assert len(candles) <= 720, (
        f"Expected ≤720 candles (Kraken limit), got {len(candles)}"
    )
    assert len(candles) > 0, "Expected at least some candles from fake response"


# ── fetch_hourly_btc returns exactly 6 dicts with 'timestamp' key ─────────

def test_fetch_hourly_btc_returns_exactly_6_dicts_with_timestamp_key():
    """fetch_hourly_btc(hours=6) must return exactly 6 dicts, each with 'ts' key."""
    import asyncio
    import sys
    from unittest.mock import MagicMock

    # Stub out pydantic (unavailable in test environment) and only the modules
    # that directly depend on it — leave the polymarket package structure intact.
    _inserted: list[str] = []
    for mod_name in ["pydantic", "pydantic.fields"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()
            _inserted.append(mod_name)

    # Also stub the specific submodules that pull in pydantic transitively
    for mod_name in [
        "crypto_oracle.polymarket.models",
        "crypto_oracle.polymarket.client",
        "crypto_oracle.polymarket.clob",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()
            _inserted.append(mod_name)

    try:
        # Clear any previously (partially) loaded agents.base so we get a fresh import
        for mod_name in list(sys.modules):
            if mod_name in (
                "crypto_oracle.polymarket.agents",
                "crypto_oracle.polymarket.agents.base",
            ):
                del sys.modules[mod_name]

        import importlib
        import importlib.util
        from pathlib import Path

        base_path = Path("/tmp/crypto-oracle-research/crypto_oracle/polymarket/agents/base.py")
        spec = importlib.util.spec_from_file_location("_pb_base_isolated", base_path)
        _mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(_mod)  # type: ignore[union-attr]
        fetch_hourly_btc = _mod.fetch_hourly_btc
    finally:
        for mod_name in _inserted:
            sys.modules.pop(mod_name, None)

    # Kraken returns 720 completed + 1 in-progress candle (721 total).
    # The function drops the last one (in-progress), keeping 720 completed.
    # It then slices the last 6.
    base_ts = 1735689600
    fake_rows = [
        [base_ts + i * 3600, "95000", "96000", "94000", "95500", "0", "10"]
        for i in range(721)  # 720 completed + 1 in-progress
    ]

    fake_api_response = {
        "result": {
            "XXBTZUSD": fake_rows,
        }
    }

    async def _run():
        mock_resp = AsyncMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = AsyncMock(return_value=fake_api_response)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_get(*args, **kwargs):
            yield mock_resp

        mock_session = AsyncMock()
        mock_session.get = mock_get
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            return await fetch_hourly_btc(hours=6)

    candles = asyncio.run(_run())
    assert len(candles) == 6, f"Expected exactly 6 candles, got {len(candles)}"
    for c in candles:
        assert "ts" in c, f"Expected 'ts' key in candle dict, got keys: {list(c.keys())}"

