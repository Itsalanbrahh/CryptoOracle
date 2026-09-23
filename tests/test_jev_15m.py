"""Unit tests for Jev 15m judgment parsing and heuristic fallback."""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from crypto_oracle.kalshi.jev_15m import (
    Jev15mJudgment,
    _parse_answers,
    build_15m_state,
    evaluate_15m_market,
    heuristic_15m_judgment,
)
from crypto_oracle.kalshi.markets import KalshiMarket


def _mkt(
    *,
    yes_ask: float = 0.55,
    no_ask: float = 0.46,
    strike: float = 84_000.0,
    hours: float = 0.2,
) -> KalshiMarket:
    yes_bid = max(0.01, yes_ask - 0.01)
    no_bid = max(0.01, no_ask - 0.01)
    from datetime import datetime, timedelta, timezone

    close = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat().replace("+00:00", "Z")
    return KalshiMarket(
        ticker="KXBTC15M-TEST-00",
        strike=strike,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        mid=(yes_bid + yes_ask) / 2,
        volume=10_000,
        close_time=close,
    )


class TestJev15m(unittest.TestCase):
    def test_build_state_includes_session_and_distance(self):
        m = _mkt(strike=100.0)
        state = build_15m_state(m, spot=101.0, annual_vol=0.6)
        self.assertEqual(state["session"], "crypto_24_7")
        self.assertEqual(state["opening_brti_target_usd"], 100.0)
        self.assertAlmostEqual(state["distance_usd"], 1.0)
        self.assertIn("yes_ask", state["book"])

    def test_parse_answers(self):
        m = _mkt()
        state = build_15m_state(m, spot=84_100.0, annual_vol=0.7)
        data = {
            "model": "jev-1.13.0",
            "answers": {
                "settle_up": {"type": "noul", "noul": 0.72},
                "action": {
                    "type": "choice",
                    "choice": "buy_yes",
                    "confidence": 0.81,
                    "probabilities": {"buy_yes": 0.7, "buy_no": 0.1, "hold": 0.2},
                },
            },
        }
        j = _parse_answers(data, state)
        self.assertIsInstance(j, Jev15mJudgment)
        self.assertAlmostEqual(j.p_settle_up, 0.72)
        self.assertEqual(j.action, "buy_yes")
        self.assertAlmostEqual(j.action_confidence, 0.81)
        self.assertEqual(j.model, "jev-1.13.0")
        self.assertGreater(j.confidence, 0.5)

    def test_heuristic_prefers_hold_when_priced_in(self):
        # Spot far above target → high p_up, but yes_ask already 0.95 → no 8pp edge
        m = _mkt(yes_ask=0.95, no_ask=0.06, strike=80_000.0)
        j = heuristic_15m_judgment(m, spot=85_000.0, annual_vol=0.5)
        self.assertEqual(j.model, "heuristic-fallback")
        self.assertEqual(j.action, "hold")

    def test_heuristic_buy_no_when_spot_below_and_no_cheap(self):
        os.environ["KALSHI_15M_MIN_EDGE"] = "0.05"
        try:
            m = _mkt(yes_ask=0.70, no_ask=0.32, strike=90_000.0)
            j = heuristic_15m_judgment(m, spot=80_000.0, annual_vol=0.4)
            self.assertEqual(j.action, "buy_no")
            self.assertLess(j.p_settle_up, 0.5)
        finally:
            os.environ.pop("KALSHI_15M_MIN_EDGE", None)

    def test_heuristic_never_buys_no_while_p_up_high(self):
        # Spot above target → p_up > 0.5; cheap NO must still be hold, not buy_no.
        m = _mkt(yes_ask=0.95, no_ask=0.02, strike=80_000.0)
        j = heuristic_15m_judgment(m, spot=85_000.0, annual_vol=0.5)
        self.assertGreaterEqual(j.p_settle_up, 0.5)
        self.assertNotEqual(j.action, "buy_no")

    def test_evaluate_falls_back_without_key(self):
        os.environ.pop("TYPESAFE_API_KEY", None)
        os.environ.pop("JEV_API_KEY", None)
        m = _mkt()
        j = asyncio.run(evaluate_15m_market(m, spot=84_000.0, annual_vol=0.6))
        self.assertEqual(j.model, "heuristic-fallback")

    def test_evaluate_uses_jev_when_mocked(self):
        m = _mkt()
        fake = {
            "model": "jev-latest",
            "answers": {
                "settle_up": {"noul": 0.40},
                "action": {"choice": "buy_no", "confidence": 0.66, "probabilities": {}},
            },
        }
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "sk-test", "KALSHI_JEV_15M": "1"}):
            with patch(
                "crypto_oracle.kalshi.jev_15m.system_one",
                new=AsyncMock(return_value=fake),
            ):
                j = asyncio.run(evaluate_15m_market(m, spot=84_000.0, annual_vol=0.6))
        self.assertEqual(j.action, "buy_no")
        self.assertAlmostEqual(j.p_settle_up, 0.40)
        self.assertEqual(j.model, "jev-latest")


if __name__ == "__main__":
    unittest.main()
