from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os

from crypto_oracle.models.db import (
    add_polymarket_watchlist,
    get_latest_polymarket_snapshots,
    get_polymarket_execution_orders,
    get_polymarket_paper_trades,
    log_polymarket_execution_order,
    log_polymarket_paper_trade,
    save_polymarket_market_snapshot,
    save_polymarket_strategy_state,
)
from crypto_oracle.polymarket.execution import execute_live_order, load_execution_config, decision_to_live_order
from crypto_oracle.polymarket.gamma import fetch_btc_markets
from crypto_oracle.polymarket.orchestrator import PolymarketOrchestrator
from crypto_oracle.polymarket.paper_trader import apply_paper_decision
from crypto_oracle.polymarket.models import PaperPortfolio, PolymarketMarket

PORTFOLIO_PATH = Path('/Users/alanruelas/crypto_oracle/data/polymarket_btc_paper_portfolio.json')


def _load_portfolio() -> PaperPortfolio:
    if PORTFOLIO_PATH.exists():
        try:
            return PaperPortfolio.model_validate_json(PORTFOLIO_PATH.read_text())
        except Exception:
            pass
    return PaperPortfolio(cash_usd=float(os.getenv('POLYMARKET_ACCOUNT_SIZE_USD', '70') or '70'))


def _save_portfolio(portfolio: PaperPortfolio) -> None:
    PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_PATH.write_text(portfolio.model_dump_json(indent=2))


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def minutes_to_expiry(market: PolymarketMarket, now: datetime | None = None) -> float | None:
    if not market.end_date_iso:
        return None
    try:
        end = datetime.fromisoformat(str(market.end_date_iso).replace('Z', '+00:00'))
    except ValueError:
        return None
    ref = now or datetime.now(timezone.utc)
    return (end - ref).total_seconds() / 60.0


def filter_markets_by_expiry(markets: list[PolymarketMarket]) -> list[PolymarketMarket]:
    min_minutes = _env_int('POLYMARKET_MIN_MINUTES_TO_EXPIRY')
    max_minutes = _env_int('POLYMARKET_MAX_MINUTES_TO_EXPIRY')
    if min_minutes is None and max_minutes is None:
        return markets
    filtered: list[PolymarketMarket] = []
    for market in markets:
        remaining = minutes_to_expiry(market)
        if remaining is None:
            continue
        if min_minutes is not None and remaining < min_minutes:
            continue
        if max_minutes is not None and remaining > max_minutes:
            continue
        filtered.append(market)
    return filtered


def _rank_markets_by_uncertainty(markets: list[PolymarketMarket]) -> list[PolymarketMarket]:
    """Rank markets by edge opportunity: favors both near-50/50 uncertainty AND high-confidence YES (0.80-0.97)."""
    def _score(m: PolymarketMarket) -> float:
        p = m.yes_outcome.price
        # Exclude truly settled markets (>99% or <1% certain — no real edge left)
        if p < 0.01 or p > 0.99:
            return 999.0
        # Uncertainty track: near 50/50 has max mispricing potential
        uncertainty_score = abs(p - 0.5)
        # High-confidence YES track: 0.80-0.97 range where safe profit is possible
        # Centered at 0.87 — sweet spot for $5+ profit on a $20 position
        high_conf_score = abs(p - 0.87) * 0.3
        return min(uncertainty_score, high_conf_score)
    return sorted(markets, key=_score)


async def run_scan(limit: int = 5) -> dict:
    orchestrator = PolymarketOrchestrator()
    execution_cfg = load_execution_config()
    portfolio = _load_portfolio()
    # Fetch a large pool, filter by expiry, rank by uncertainty, take top candidates
    markets = await fetch_btc_markets(limit=200, min_liquidity=orchestrator.policy.min_liquidity)
    markets = filter_markets_by_expiry(markets)
    markets = _rank_markets_by_uncertainty(markets)
    selected = markets[:limit]
    results = []
    for market in selected:
        result = await orchestrator.run_market(market)
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
        await save_polymarket_market_snapshot(payload)
        await add_polymarket_watchlist(market.market_id, market.question, market.slug)

        execution_info: dict | None = None
        if result.decision.action != 'HOLD':
            if execution_cfg.mode == 'paper':
                portfolio = apply_paper_decision(portfolio, result.decision)
                latest_trade = portfolio.trade_log[-1] if portfolio.trade_log else {}
                await log_polymarket_paper_trade(
                    market_id=market.market_id,
                    question=market.question,
                    action=result.decision.action,
                    outcome=result.decision.target_outcome,
                    price=result.decision.price,
                    position_size_usd=float(latest_trade.get('spent_usd') or result.decision.position_size_usd),
                    shares=float(latest_trade.get('shares') or 0.0),
                    confidence=result.decision.confidence,
                    max_loss_usd=result.decision.max_loss_usd,
                    notes=result.decision.reasoning,
                )
                paper_quantity = int(round(float(latest_trade.get('shares') or 0.0))) or None
                await log_polymarket_execution_order(
                    market_id=market.market_id,
                    question=market.question,
                    action=result.decision.action,
                    execution_mode='paper',
                    dry_run=False,
                    status='paper_filled',
                    outcome=result.decision.target_outcome,
                    price=result.decision.price,
                    quantity=paper_quantity,
                    position_size_usd=float(latest_trade.get('spent_usd') or result.decision.position_size_usd),
                    confidence=result.decision.confidence,
                    response={'paper_trade': latest_trade, 'reasoning': result.decision.reasoning},
                )
                execution_info = {'status': 'paper_filled', 'mode': 'paper', 'dry_run': False}
            else:
                live_execution = await execute_live_order(market, result.decision, execution_cfg)
                live_payload = decision_to_live_order(market, result.decision)
                await log_polymarket_execution_order(
                    market_id=market.market_id,
                    question=market.question,
                    action=result.decision.action,
                    execution_mode=live_execution.mode,
                    dry_run=live_execution.dry_run,
                    status=live_execution.status,
                    external_order_id=live_execution.external_order_id,
                    outcome=result.decision.target_outcome,
                    price=result.decision.price,
                    quantity=int(live_payload['quantity']),
                    position_size_usd=result.decision.position_size_usd,
                    confidence=result.decision.confidence,
                    response=live_execution.response,
                    error_text=live_execution.error or '',
                )
                execution_info = live_execution.model_dump()

        await save_polymarket_strategy_state(
            market.market_id,
            market.question,
            result.decision.action,
            result.decision.confidence,
            orchestrator.policy.max_daily_risk_usd,
            orchestrator.policy.max_position_usd,
        )
        r = result.model_dump(mode='json')
        if execution_info:
            r['execution'] = execution_info
        results.append(r)
    _save_portfolio(portfolio)
    trades = await get_polymarket_paper_trades(limit=20)
    snapshots = await get_latest_polymarket_snapshots(limit=20)
    executions = await get_polymarket_execution_orders(limit=20)
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'execution_mode': execution_cfg.mode,
        'live_dry_run': execution_cfg.dry_run,
        'markets_scanned': len(selected),
        'paper_cash_usd': portfolio.cash_usd,
        'open_yes_positions': len(portfolio.yes_positions),
        'open_no_positions': len(portfolio.no_positions),
        'recent_trade_count': len(trades),
        'recent_snapshot_count': len(snapshots),
        'recent_execution_count': len(executions),
        'results': results,
    }


if __name__ == '__main__':
    import asyncio
    print(json.dumps(asyncio.run(run_scan()), indent=2))
