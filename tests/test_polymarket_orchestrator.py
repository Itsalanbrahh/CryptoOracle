import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from crypto_oracle.models import db as dbmod
from crypto_oracle.polymarket.client import parse_gamma_market
from crypto_oracle.polymarket.models import PolymarketSpecialistSignal
from crypto_oracle.polymarket.orchestrator import PolymarketOrchestrator
from crypto_oracle.polymarket.risk import RiskPolicy


class PolymarketOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_db_path = dbmod.DB_PATH
        dbmod.DB_PATH = Path(self.tmpdir.name) / 'test_orchestrator.db'
        asyncio.run(dbmod.init_db())

    def tearDown(self):
        dbmod.DB_PATH = self.old_db_path
        self.tmpdir.cleanup()

    def _signal(self, name: str, score: float, confidence: float, stance: str = 'BULLISH') -> PolymarketSpecialistSignal:
        return PolymarketSpecialistSignal(
            agent_name=name,
            stance=stance,
            score=score,
            confidence=confidence,
            summary='test signal',
            data_points=[],
            evidence={},
        )

    def test_orchestrator_blocks_low_edge_trade(self):
        market = parse_gamma_market({
            'id': 'btc-5',
            'question': 'Will Bitcoin be above $100,000 by Dec 31?',
            'slug': 'bitcoin-above-100k',
            'endDate': '2026-12-31T23:59:59Z',
            'liquidity': 20000,
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.95", "0.05"]',
            'clobTokenIds': '["y", "n"]',
        })
        orch = PolymarketOrchestrator(policy=RiskPolicy(min_confidence=0.55, min_edge=0.20, max_position_usd=50))
        with patch('crypto_oracle.polymarket.orchestrator.fetch_spot_price', new=AsyncMock(return_value=101000.0)), \
             patch.object(orch.agents[0], 'run', new=AsyncMock(return_value=self._signal('MicroMarket', 0.8, 0.8))), \
             patch.object(orch.agents[1], 'run', new=AsyncMock(return_value=self._signal('MacroMarket', 0.8, 0.8))), \
             patch.object(orch.agents[2], 'run', new=AsyncMock(return_value=self._signal('KnowledgeMarket', 0.8, 0.8))), \
             patch.object(orch.agents[3], 'run', new=AsyncMock(return_value=self._signal('TechnicalMarket', 0.8, 0.8))), \
             patch.object(orch.agents[4], 'run', new=AsyncMock(return_value=self._signal('LinearRegressionMarket', 0.8, 0.8))):
            result = asyncio.run(orch.run_market(market))
        self.assertEqual(result.decision.action, 'HOLD')
        self.assertIn('does not offer enough edge', result.decision.reasoning)

    def test_orchestrator_allows_high_conviction_trade(self):
        market = parse_gamma_market({
            'id': 'btc-6',
            'question': 'Will Bitcoin be above $100,000 by Dec 31?',
            'slug': 'bitcoin-above-100k',
            'endDate': '2026-12-31T23:59:59Z',
            'liquidity': 30000,
            'outcomes': '["Yes", "No"]',
            'outcomePrices': '["0.30", "0.70"]',
            'clobTokenIds': '["y", "n"]',
        })
        orch = PolymarketOrchestrator(policy=RiskPolicy(min_confidence=0.55, min_edge=0.05, max_position_usd=50))
        with patch('crypto_oracle.polymarket.orchestrator.fetch_spot_price', new=AsyncMock(return_value=112000.0)), \
             patch.object(orch.agents[0], 'run', new=AsyncMock(return_value=self._signal('MicroMarket', 0.9, 0.8))), \
             patch.object(orch.agents[1], 'run', new=AsyncMock(return_value=self._signal('MacroMarket', 0.8, 0.75))), \
             patch.object(orch.agents[2], 'run', new=AsyncMock(return_value=self._signal('KnowledgeMarket', 0.85, 0.78))), \
             patch.object(orch.agents[3], 'run', new=AsyncMock(return_value=self._signal('TechnicalMarket', 0.82, 0.77))), \
             patch.object(orch.agents[4], 'run', new=AsyncMock(return_value=self._signal('LinearRegressionMarket', 0.86, 0.79))):
            result = asyncio.run(orch.run_market(market))
        self.assertEqual(result.decision.action, 'BUY_YES')
        self.assertGreater(result.decision.expected_edge, 0.05)


if __name__ == '__main__':
    unittest.main()
