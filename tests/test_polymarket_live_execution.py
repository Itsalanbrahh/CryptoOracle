import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crypto_oracle.polymarket.client import parse_gamma_market
from crypto_oracle.polymarket.execution import ExecutionConfig, decision_to_live_order, execute_live_order, load_execution_config
from crypto_oracle.polymarket.models import PolymarketMasterDecision


class PolymarketLiveExecutionTests(unittest.TestCase):
    def setUp(self):
        self.market = parse_gamma_market({
            'id': 'btc-live-1',
            'question': 'Will Bitcoin be above $100,000 by Dec 31?',
            'slug': 'bitcoin-above-100k',
            'endDate': '2026-12-31T23:59:59Z',
            'liquidity': 30000,
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.40", "0.60"]',
            'clobTokenIds': '["y", "n"]',
        })
        self.decision = PolymarketMasterDecision(
            market_id='btc-live-1', question=self.market.question, action='BUY_YES', confidence=0.7,
            target_outcome='Yes', price=0.40, trend_bias='BULLISH', reasoning='test',
            catalysts=[], risks=[], max_loss_usd=50.0, position_size_usd=20.0, expected_edge=0.1,
        )

    def test_load_execution_config_defaults_to_paper(self):
        with patch.dict(os.environ, {}, clear=False):
            cfg = load_execution_config()
        self.assertEqual(cfg.mode, 'paper')
        self.assertTrue(cfg.dry_run)

    def test_decision_to_live_order_maps_buy_yes(self):
        payload = decision_to_live_order(self.market, self.decision)
        self.assertEqual(payload['marketSlug'], 'bitcoin-above-100k')
        self.assertEqual(payload['intent'], 'ORDER_INTENT_BUY_LONG')
        self.assertEqual(payload['quantity'], 50)

    def test_execute_live_order_returns_dry_run_result(self):
        with tempfile.TemporaryDirectory() as td:
            bridge = Path(td) / 'bridge.mjs'
            bridge.write_text("process.stdout.write(JSON.stringify({ok:true,response:{id:'abc123'}}));", encoding='utf-8')
            with patch('crypto_oracle.polymarket.execution.BRIDGE_SCRIPT', bridge),                  patch('crypto_oracle.polymarket.execution.NODE_PACKAGE_ROOT', Path(td)):
                result = asyncio.run(execute_live_order(self.market, self.decision, ExecutionConfig(mode='live', dry_run=True)))
        self.assertEqual(result.status, 'dry_run_validated')
        self.assertEqual(result.external_order_id, 'abc123')


if __name__ == '__main__':
    unittest.main()
