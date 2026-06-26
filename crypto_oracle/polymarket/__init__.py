"""Polymarket BTC paper/live trading scaffold for CryptoOracle."""

from .client import parse_gamma_market, select_btc_markets
from .gamma import fetch_btc_markets, fetch_event_markets, fetch_markets
from .clob import get_midpoint, get_orderbook, get_price_history, get_spread
from .execution import ExecutionConfig, ExecutionResult, decision_to_live_order, execute_live_order, load_execution_config
from .models import (
    PaperPortfolio,
    ParsedMarketQuestion,
    PolymarketMarket,
    PolymarketMasterDecision,
    PolymarketOutcome,
    PolymarketRunResult,
    PolymarketSpecialistSignal,
)
from .orchestrator import PolymarketOrchestrator
from .paper_loop import run_scan
from .paper_trader import apply_paper_decision
from .risk import RiskPolicy, evaluate_risk
from .strategy import decide_trade

__all__ = [
    "parse_gamma_market",
    "select_btc_markets",
    "fetch_btc_markets",
    "fetch_event_markets",
    "fetch_markets",
    "get_midpoint",
    "get_orderbook",
    "get_price_history",
    "get_spread",
    "ExecutionConfig",
    "ExecutionResult",
    "load_execution_config",
    "decision_to_live_order",
    "execute_live_order",
    "PolymarketMarket",
    "PolymarketOutcome",
    "ParsedMarketQuestion",
    "PolymarketSpecialistSignal",
    "PolymarketMasterDecision",
    "PolymarketRunResult",
    "PaperPortfolio",
    "PolymarketOrchestrator",
    "RiskPolicy",
    "evaluate_risk",
    "apply_paper_decision",
    "decide_trade",
    "run_scan",
]
