from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os

from crypto_oracle.models.db import get_polymarket_paper_trades, get_polymarket_strategy_state
from crypto_oracle.polymarket.models import PolymarketMarket, PolymarketMasterDecision, PolymarketSpecialistSignal


@dataclass
class RiskPolicy:
    min_liquidity: float = 1000.0
    min_confidence: float = 0.62
    min_edge: float = 0.05
    max_position_usd: float = 50.0
    max_daily_risk_usd: float = 50.0
    max_open_markets: int = 5
    cooldown_hours: float = 6.0


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_risk_policy() -> RiskPolicy:
    account_size = _env_float('POLYMARKET_ACCOUNT_SIZE_USD', 70.0)
    default_position = min(10.0, max(1.0, round(account_size * 0.12, 2)))
    default_daily_risk = min(12.0, max(2.0, round(account_size * 0.18, 2)))
    return RiskPolicy(
        min_liquidity=_env_float('POLYMARKET_MIN_LIQUIDITY', 1000.0),
        min_confidence=_env_float('POLYMARKET_MIN_CONFIDENCE', 0.68),
        min_edge=_env_float('POLYMARKET_MIN_EDGE', 0.08),
        max_position_usd=_env_float('POLYMARKET_MAX_POSITION_USD', default_position),
        max_daily_risk_usd=_env_float('POLYMARKET_MAX_DAILY_RISK_USD', default_daily_risk),
        max_open_markets=_env_int('POLYMARKET_MAX_OPEN_MARKETS', 1),
        cooldown_hours=_env_float('POLYMARKET_COOLDOWN_HOURS', 1.0),
    )


async def evaluate_risk(
    market: PolymarketMarket,
    decision: PolymarketMasterDecision,
    specialist_signals: list[PolymarketSpecialistSignal],
    policy: RiskPolicy | None = None,
) -> tuple[bool, list[str]]:
    policy = policy or load_risk_policy()
    reasons: list[str] = []
    if market.liquidity < policy.min_liquidity:
        reasons.append(f'liquidity below threshold ({market.liquidity:.2f} < {policy.min_liquidity:.2f})')
    if decision.confidence < policy.min_confidence:
        reasons.append(f'confidence below threshold ({decision.confidence:.2f} < {policy.min_confidence:.2f})')
    if decision.action != 'HOLD' and decision.expected_edge < policy.min_edge:
        reasons.append(f'expected edge below threshold ({decision.expected_edge:.3f} < {policy.min_edge:.3f})')
    state = await get_polymarket_strategy_state(market.market_id)
    trades = await get_polymarket_paper_trades(limit=200)
    today = datetime.now(timezone.utc).date().isoformat()
    today_risk = 0.0
    open_markets: set[str] = set()
    latest_created = None
    for trade in trades:
        created = str(trade.get('created_at') or '')
        if created.startswith(today):
            today_risk += float(trade.get('max_loss_usd') or 0.0)
        if trade.get('status') == 'open':
            open_markets.add(str(trade.get('market_id')))
        if trade.get('market_id') == market.market_id and not latest_created:
            latest_created = created
    if today_risk + decision.max_loss_usd > policy.max_daily_risk_usd:
        reasons.append(f'daily paper risk budget exceeded ({today_risk + decision.max_loss_usd:.2f} > {policy.max_daily_risk_usd:.2f})')
    if len(open_markets) >= policy.max_open_markets and market.market_id not in open_markets:
        reasons.append(f'open market cap reached ({len(open_markets)} >= {policy.max_open_markets})')
    if latest_created and ' ' in latest_created:
        try:
            ts = datetime.fromisoformat(latest_created.replace(' ', 'T'))
            age_hours = (datetime.now(ts.tzinfo or timezone.utc) - ts).total_seconds() / 3600
            if age_hours < policy.cooldown_hours:
                reasons.append(f'market cooldown active ({age_hours:.1f}h < {policy.cooldown_hours:.1f}h)')
        except Exception:
            pass
    strong_disagreement = sum(1 for s in specialist_signals if s.stance == 'BULLISH') and sum(1 for s in specialist_signals if s.stance == 'BEARISH')
    if strong_disagreement and decision.confidence < 0.72:
        reasons.append('specialist disagreement too high for current conviction')
    return (len(reasons) == 0, reasons)
