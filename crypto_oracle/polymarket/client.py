from __future__ import annotations

import json
import re
from typing import Any

from .models import PolymarketMarket, PolymarketOutcome

_BTC_RE = re.compile(r'\b(bitcoin|btc|xbt)\b', re.I)


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            decoded = json.loads(value)
            if isinstance(decoded, list):
                return [str(x) for x in decoded]
        except json.JSONDecodeError:
            pass
        return [value]
    return [str(value)]


def parse_gamma_market(row: dict[str, Any]) -> PolymarketMarket:
    outcome_names = _json_list(row.get('outcomes'))
    outcome_prices = [float(x) for x in _json_list(row.get('outcomePrices'))]
    token_ids = _json_list(row.get('clobTokenIds'))
    outcomes = []
    for idx, name in enumerate(outcome_names):
        price = outcome_prices[idx] if idx < len(outcome_prices) else 0.0
        token_id = token_ids[idx] if idx < len(token_ids) else None
        outcomes.append(PolymarketOutcome(name=name, price=price, token_id=token_id))
    if not outcomes:
        raise ValueError('No outcomes found in market row')
    return PolymarketMarket(
        market_id=str(row.get('id') or row.get('marketId') or ''),
        event_id=str(row.get('eventId')) if row.get('eventId') is not None else None,
        question=str(row.get('question') or row.get('title') or ''),
        slug=row.get('slug'),
        end_date_iso=row.get('endDate') or row.get('endDateIso'),
        volume=float(row.get('volume') or 0.0),
        liquidity=float(row.get('liquidity') or 0.0),
        outcomes=outcomes,
        condition_id=row.get('conditionId'),
        raw=row,
    )


def is_btc_market(market: PolymarketMarket) -> bool:
    event_titles = ' '.join((event.get('title') or '') for event in (market.raw.get('events') or []) if isinstance(event, dict)) if isinstance(market.raw, dict) else ''
    series_tokens = ''
    if isinstance(market.raw, dict):
        for event in market.raw.get('events') or []:
            if isinstance(event, dict):
                for series in event.get('series') or []:
                    if isinstance(series, dict):
                        series_tokens += ' ' + str(series.get('title') or '') + ' ' + str(series.get('ticker') or '')
    text = f"{market.question} {market.slug or ''} {event_titles} {series_tokens}".lower()
    return bool(_BTC_RE.search(text))


def select_btc_markets(rows: list[dict[str, Any]], min_liquidity: float = 1000.0) -> list[PolymarketMarket]:
    markets = []
    for row in rows:
        market = parse_gamma_market(row)
        if not is_btc_market(market):
            continue
        if market.liquidity < min_liquidity and market.volume < min_liquidity:
            continue
        if len(market.outcomes) < 2:
            continue
        outcome_names = {outcome.name.strip().lower() for outcome in market.outcomes}
        if not {'yes', 'no'}.issubset(outcome_names):
            continue
        markets.append(market)
    return sorted(markets, key=lambda m: (m.liquidity, m.volume), reverse=True)
