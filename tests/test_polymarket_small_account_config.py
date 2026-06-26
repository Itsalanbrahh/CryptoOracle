import os
import unittest
from unittest.mock import patch

from crypto_oracle.polymarket.client import parse_gamma_market
from crypto_oracle.polymarket.paper_loop import filter_markets_by_expiry
from crypto_oracle.polymarket.risk import load_risk_policy


class PolymarketSmallAccountConfigTests(unittest.TestCase):
    def test_load_risk_policy_small_account_defaults(self):
        with patch.dict(os.environ, {'POLYMARKET_ACCOUNT_SIZE_USD': '70'}, clear=False):
            policy = load_risk_policy()
        self.assertEqual(policy.max_open_markets, 1)
        self.assertAlmostEqual(policy.max_position_usd, 8.4, places=2)
        self.assertAlmostEqual(policy.max_daily_risk_usd, 12.0, places=2)
        self.assertAlmostEqual(policy.min_confidence, 0.68, places=2)
        self.assertAlmostEqual(policy.min_edge, 0.08, places=2)

    def test_filter_markets_by_expiry_window(self):
        near = parse_gamma_market({
            'id': 'm1',
            'question': 'Will Bitcoin be above $60,000 in 15 minutes?',
            'slug': 'btc-15m',
            'endDate': '2026-06-22T02:20:00Z',
            'liquidity': 5000,
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.55", "0.45"]',
            'clobTokenIds': '["y", "n"]',
        })
        far = parse_gamma_market({
            'id': 'm2',
            'question': 'Will Bitcoin be above $60,000 tomorrow?',
            'slug': 'btc-1d',
            'endDate': '2026-06-23T02:00:00Z',
            'liquidity': 5000,
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.55", "0.45"]',
            'clobTokenIds': '["y", "n"]',
        })
        with patch.dict(os.environ, {
            'POLYMARKET_MIN_MINUTES_TO_EXPIRY': '5',
            'POLYMARKET_MAX_MINUTES_TO_EXPIRY': '30',
        }, clear=False), patch('crypto_oracle.polymarket.paper_loop.datetime') as mock_dt:
            from datetime import datetime, timezone
            mock_dt.now.return_value = datetime(2026, 6, 22, 2, 5, tzinfo=timezone.utc)
            mock_dt.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
            filtered = filter_markets_by_expiry([near, far])
        self.assertEqual([m.market_id for m in filtered], ['m1'])


if __name__ == '__main__':
    unittest.main()
