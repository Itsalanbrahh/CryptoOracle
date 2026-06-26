import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from crypto_oracle.polymarket.agents.base import parse_market_question
from crypto_oracle.polymarket.agents.knowledge_market import KnowledgeMarketAgent
from crypto_oracle.polymarket.agents.linear_regression_market import LinearRegressionMarketAgent
from crypto_oracle.polymarket.agents.technical_market import TechnicalMarketAgent
from crypto_oracle.polymarket.client import parse_gamma_market


class PolymarketAgentTests(unittest.TestCase):
    def test_parse_market_question_extracts_threshold(self):
        market = parse_gamma_market({
            'id': 'btc-1',
            'question': 'Will Bitcoin be above $125k by Dec 31?',
            'slug': 'bitcoin-above-125k',
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.30", "0.70"]',
            'clobTokenIds': '["y", "n"]',
        })
        parsed = parse_market_question(market, 118000)
        self.assertEqual(parsed.comparator, 'above')
        self.assertEqual(parsed.threshold_price, 125000)
        self.assertLess(parsed.distance_to_threshold_pct, 0)

    def test_parse_market_question_handles_below(self):
        market = parse_gamma_market({
            'id': 'btc-2',
            'question': 'Will Bitcoin be below $90,000 on August 1?',
            'slug': 'bitcoin-below-90k',
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.35", "0.65"]',
            'clobTokenIds': '["y", "n"]',
        })
        parsed = parse_market_question(market, 85000)
        self.assertEqual(parsed.comparator, 'below')
        self.assertGreater(parsed.distance_to_threshold_pct, 0)

    def test_knowledge_agent_returns_structured_signal(self):
        market = parse_gamma_market({
            'id': 'btc-3',
            'question': 'Will Bitcoin be above $100,000 by Dec 31?',
            'slug': 'bitcoin-above-100k',
            'endDate': '2026-12-31T23:59:59Z',
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.62", "0.38"]',
            'clobTokenIds': '["y", "n"]',
        })
        agent = KnowledgeMarketAgent()
        with patch('crypto_oracle.polymarket.agents.knowledge_market.fetch_spot_price', new=AsyncMock(return_value=110000.0)), patch('crypto_oracle.polymarket.agents.knowledge_market.fetch_spot_history', new=AsyncMock(return_value=[80000 + i * 400 for i in range(120)])):
            signal = asyncio.run(agent.run(market))
        self.assertIn(signal.stance, {'BULLISH', 'BEARISH', 'NEUTRAL'})
        self.assertTrue(-1.0 <= signal.score <= 1.0)
        self.assertTrue(0.0 <= signal.confidence <= 1.0)

    def test_technical_agent_returns_structured_signal(self):
        market = parse_gamma_market({
            'id': 'btc-4',
            'question': 'Will Bitcoin be above $95,000 by Dec 31?',
            'slug': 'bitcoin-above-95k',
            'endDate': '2026-12-31T23:59:59Z',
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.71", "0.29"]',
            'clobTokenIds': '["y", "n"]',
        })
        prices = [70000 + i * 300 for i in range(100)]
        agent = TechnicalMarketAgent()
        with patch('crypto_oracle.polymarket.agents.technical_market.fetch_spot_price', new=AsyncMock(return_value=98000.0)), patch('crypto_oracle.polymarket.agents.technical_market.fetch_spot_history', new=AsyncMock(return_value=prices)):
            signal = asyncio.run(agent.run(market))
        self.assertIn(signal.stance, {'BULLISH', 'BEARISH', 'NEUTRAL'})
        self.assertTrue(-1.0 <= signal.score <= 1.0)
        self.assertTrue(0.0 <= signal.confidence <= 1.0)

    def test_linear_regression_agent_returns_bullish_signal_for_uptrend(self):
        market = parse_gamma_market({
            'id': 'btc-5',
            'question': 'Will Bitcoin be above $95,000 in 15 minutes?',
            'slug': 'bitcoin-above-95k-15m',
            'endDate': '2026-06-22T02:20:00Z',
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.54", "0.46"]',
            'clobTokenIds': '["y", "n"]',
        })
        prices = [90000 + i * 80 for i in range(120)]
        agent = LinearRegressionMarketAgent()
        with patch('crypto_oracle.polymarket.agents.linear_regression_market.fetch_spot_price', new=AsyncMock(return_value=99550.0)), patch('crypto_oracle.polymarket.agents.linear_regression_market.fetch_spot_history', new=AsyncMock(return_value=prices)):
            signal = asyncio.run(agent.run(market))
        self.assertEqual(signal.agent_name, 'LinearRegressionMarket')
        self.assertEqual(signal.stance, 'BULLISH')
        self.assertGreater(signal.score, 0.08)
        self.assertTrue(0.0 <= signal.confidence <= 1.0)
        self.assertIn('forecast', signal.summary.lower())


if __name__ == '__main__':
    unittest.main()
