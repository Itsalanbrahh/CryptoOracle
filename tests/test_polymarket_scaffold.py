import unittest

from crypto_oracle.polymarket.client import parse_gamma_market, select_btc_markets
from crypto_oracle.polymarket.models import PaperPortfolio
from crypto_oracle.polymarket.paper_trader import apply_paper_decision
from crypto_oracle.polymarket.strategy import decide_trade


class PolymarketScaffoldTests(unittest.TestCase):
    def test_parse_gamma_market_decodes_double_encoded_fields(self):
        row = {
            'id': '123',
            'question': 'Will Bitcoin be above $120k on Dec 31, 2026?',
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.42", "0.58"]',
            'clobTokenIds': '["yes-token", "no-token"]',
            'liquidity': 25000,
            'volume': 100000,
        }
        market = parse_gamma_market(row)
        self.assertEqual(market.market_id, '123')
        self.assertEqual(market.yes_outcome.token_id, 'yes-token')
        self.assertAlmostEqual(market.no_outcome.price, 0.58)

    def test_select_btc_markets_filters_non_btc_rows(self):
        rows = [
            {
                'id': 'btc-1', 'question': 'Will Bitcoin close above $110k this month?',
                'outcomes': '["Yes", "No"]', 'outcomePrices': '["0.40", "0.60"]',
                'liquidity': 5000, 'volume': 8000,
            },
            {
                'id': 'sports-1', 'question': 'Will the Lakers win tomorrow?',
                'outcomes': '["Yes", "No"]', 'outcomePrices': '["0.55", "0.45"]',
                'liquidity': 99999, 'volume': 99999,
            },
        ]
        markets = select_btc_markets(rows)
        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0].market_id, 'btc-1')

    def test_decide_trade_buys_yes_when_bullish_edge_is_real(self):
        market = parse_gamma_market({
            'id': 'btc-2', 'question': 'Will Bitcoin be above $115k by July 31?',
            'outcomes': '["Yes", "No"]', 'outcomePrices': '["0.35", "0.65"]',
            'liquidity': 10000, 'volume': 10000,
        })
        decision = decide_trade(market, micro_score=0.9, macro_score=0.8, knowledge_score=0.7)
        self.assertEqual(decision.action, 'BUY_YES')
        self.assertGreater(decision.position_size_usd, 0)
        self.assertEqual(decision.target_outcome, 'Yes')

    def test_decide_trade_holds_when_confidence_is_too_low(self):
        market = parse_gamma_market({
            'id': 'btc-3', 'question': 'Will Bitcoin finish green this week?',
            'outcomes': '["Yes", "No"]', 'outcomePrices': '["0.49", "0.51"]',
            'liquidity': 10000, 'volume': 10000,
        })
        decision = decide_trade(market, micro_score=0.2, macro_score=0.1, knowledge_score=0.0)
        self.assertEqual(decision.action, 'HOLD')
        self.assertEqual(decision.position_size_usd, 0)

    def test_apply_paper_decision_updates_cash_and_positions(self):
        market = parse_gamma_market({
            'id': 'btc-4', 'question': 'Will Bitcoin be above $130k at year end?',
            'outcomes': '["Yes", "No"]', 'outcomePrices': '["0.30", "0.70"]',
            'liquidity': 10000, 'volume': 10000,
        })
        decision = decide_trade(market, micro_score=0.95, macro_score=0.8, knowledge_score=0.8)
        portfolio = PaperPortfolio(cash_usd=100.0)
        updated = apply_paper_decision(portfolio, decision)
        self.assertLess(updated.cash_usd, 100.0)
        self.assertGreater(updated.yes_positions.get('btc-4', 0.0), 0.0)
        self.assertEqual(updated.trade_log[-1]['action'], 'BUY_YES')


if __name__ == '__main__':
    unittest.main()
