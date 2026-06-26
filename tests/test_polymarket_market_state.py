import asyncio
import tempfile
import unittest
from pathlib import Path

from crypto_oracle.models import db as dbmod
from crypto_oracle.polymarket.client import parse_gamma_market


class PolymarketMarketStateTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_db_path = dbmod.DB_PATH
        dbmod.DB_PATH = Path(self.tmpdir.name) / 'test_polymarket.db'
        asyncio.run(dbmod.init_db())

    def tearDown(self):
        dbmod.DB_PATH = self.old_db_path
        self.tmpdir.cleanup()

    def test_save_and_read_market_snapshot(self):
        market = parse_gamma_market({
            'id': 'btc-100',
            'question': 'Will Bitcoin be above $125k by Dec 31?',
            'slug': 'bitcoin-above-125k-dec-31',
            'conditionId': '0xabc',
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.41", "0.59"]',
            'clobTokenIds': '["yes-1", "no-1"]',
            'liquidity': 20000,
            'volume': 50000,
            'endDate': '2026-12-31T23:59:59Z',
        })
        payload = {
            'market_id': market.market_id,
            'condition_id': market.condition_id,
            'question': market.question,
            'slug': market.slug,
            'yes_outcome': market.yes_outcome.model_dump(),
            'no_outcome': market.no_outcome.model_dump(),
            'volume': market.volume,
            'liquidity': market.liquidity,
            'end_date_iso': market.end_date_iso,
        }
        row_id = asyncio.run(dbmod.save_polymarket_market_snapshot(payload))
        rows = asyncio.run(dbmod.get_latest_polymarket_snapshots(limit=5, market_id='btc-100'))
        self.assertGreater(row_id, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['condition_id'], '0xabc')
        self.assertAlmostEqual(rows[0]['yes_price'], 0.41)

    def test_watchlist_and_strategy_state_round_trip(self):
        asyncio.run(dbmod.add_polymarket_watchlist('btc-200', 'Will BTC hit $150k?', 'btc-150k'))
        items = asyncio.run(dbmod.get_polymarket_watchlist())
        self.assertEqual(items[0]['market_id'], 'btc-200')
        asyncio.run(dbmod.save_polymarket_strategy_state('btc-200', 'Will BTC hit $150k?', 'BUY_YES', 0.77, 75.0, 60.0))
        state = asyncio.run(dbmod.get_polymarket_strategy_state('btc-200'))
        self.assertEqual(state['last_action'], 'BUY_YES')
        self.assertAlmostEqual(state['max_position_usd'], 60.0)

    def test_log_and_fetch_paper_trades(self):
        trade_id = asyncio.run(dbmod.log_polymarket_paper_trade(
            market_id='btc-300',
            question='Will BTC close above $100k this month?',
            action='BUY_YES',
            outcome='Yes',
            price=0.44,
            position_size_usd=25.0,
            shares=56.8,
            confidence=0.81,
            max_loss_usd=0.5,
            notes='paper trade only',
        ))
        trades = asyncio.run(dbmod.get_polymarket_paper_trades(limit=5, market_id='btc-300'))
        self.assertGreater(trade_id, 0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['action'], 'BUY_YES')
        self.assertAlmostEqual(trades[0]['shares'], 56.8)


if __name__ == '__main__':
    unittest.main()
