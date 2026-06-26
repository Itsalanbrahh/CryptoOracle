from __future__ import annotations

from .models import PaperPortfolio, PolymarketMasterDecision


def apply_paper_decision(portfolio: PaperPortfolio, decision: PolymarketMasterDecision) -> PaperPortfolio:
    if decision.action == 'HOLD' or not decision.position_size_usd or not decision.price:
        portfolio.trade_log.append({'action': 'HOLD', 'market_id': decision.market_id, 'reason': decision.reasoning})
        return portfolio

    spend = min(portfolio.cash_usd, decision.position_size_usd)
    if spend <= 0:
        portfolio.trade_log.append({'action': 'SKIP', 'market_id': decision.market_id, 'reason': 'no cash'})
        return portfolio

    shares = round(spend / decision.price, 6)
    portfolio.cash_usd = round(portfolio.cash_usd - spend, 2)
    if decision.action == 'BUY_YES':
        portfolio.yes_positions[decision.market_id] = round(portfolio.yes_positions.get(decision.market_id, 0.0) + shares, 6)
    elif decision.action == 'BUY_NO':
        portfolio.no_positions[decision.market_id] = round(portfolio.no_positions.get(decision.market_id, 0.0) + shares, 6)
    portfolio.trade_log.append({
        'action': decision.action,
        'market_id': decision.market_id,
        'price': decision.price,
        'shares': shares,
        'spent_usd': spend,
        'confidence': decision.confidence,
    })
    return portfolio
