"""Regression: fetch_historical_btc via LSE emits {'ts': str, no 'timestamp' key}.
resolve_paper_trades reads c["timestamp"] unconditionally → KeyError.
Fix must handle both 'timestamp' (int, Kraken path) and 'ts' (str, LSE path).
"""
from crypto_oracle.kalshi.resolve_paper_trades import _candle_ts


def test_candle_ts_returns_int_from_timestamp_key():
    candle = {"timestamp": 1720000000, "ts": "2024-07-03T09:46:40+00:00", "close": 60000.0}
    assert _candle_ts(candle) == 1720000000


def test_candle_ts_falls_back_to_parsing_ts_string_when_no_timestamp_key():
    candle = {"ts": "2024-07-03T12:00:00+00:00", "close": 60000.0}
    # 2024-07-03T12:00:00Z in unix seconds (no drift via python's calendar)
    from datetime import datetime, timezone
    expected = int(datetime(2024, 7, 3, 12, 0, 0, tzinfo=timezone.utc).timestamp())
    assert _candle_ts(candle) == expected


def test_candle_ts_returns_zero_for_malformed_candle():
    assert _candle_ts({}) == 0
    assert _candle_ts({"ts": "not-a-date"}) == 0
