from __future__ import annotations

import re

import aiohttp

from crypto_oracle.polymarket.agents.base import fetch_spot_price, parse_market_question
from crypto_oracle.polymarket.models import PolymarketMarket, PolymarketSpecialistSignal

_POSITIVE = {'etf inflows', 'soft landing', 'rate cut', 'dovish', 'risk-on', 'institutional demand', 'all-time high', 'accumulation'}
_NEGATIVE = {'tariff', 'war', 'liquidation', 'hawkish', 'recession', 'risk-off', 'crackdown', 'outflow', 'selloff'}


async def _fetch_headlines() -> list[str]:
    urls = [
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://feeds.feedburner.com/CoinDesk',
    ]
    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                    text = await resp.text()
                titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', text) or re.findall(r'<title>(.*?)</title>', text)
                cleaned = [t.strip() for t in titles if t and 'coindesk' not in t.lower()]
                if cleaned:
                    return cleaned[:8]
            except Exception:
                continue
    return []


async def _fetch_fear_greed() -> int | None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.alternative.me/fng/?limit=1', timeout=aiohttp.ClientTimeout(total=12)) as resp:
                data = await resp.json()
        current = data.get('data', [{}])[0].get('value')
        return int(current) if current is not None else None
    except Exception:
        return None


class MacroMarketAgent:
    name = 'MacroMarket'

    async def run(self, market: PolymarketMarket) -> PolymarketSpecialistSignal:
        spot = await fetch_spot_price()
        parsed = parse_market_question(market, spot)
        headlines = await _fetch_headlines()
        fear_greed = await _fetch_fear_greed()
        text = ' '.join(headlines).lower()
        pos_hits = sum(1 for kw in _POSITIVE if kw in text)
        neg_hits = sum(1 for kw in _NEGATIVE if kw in text)
        sentiment = (pos_hits - neg_hits) / max(1, pos_hits + neg_hits, 3)
        fg_component = 0.0 if fear_greed is None else (fear_greed - 50) / 50
        distance_component = 0.0
        if parsed.distance_to_threshold_pct is not None:
            distance_component = max(-1.0, min(1.0, parsed.distance_to_threshold_pct / 12.0))
        urgency_penalty = 0.0 if not parsed.days_to_expiry or parsed.days_to_expiry > 10 else min(0.35, 2.5 / max(parsed.days_to_expiry, 0.5))
        score = max(-1.0, min(1.0, 0.45 * sentiment + 0.35 * fg_component + 0.35 * distance_component))
        score = score - urgency_penalty if score > 0 else score + urgency_penalty if score < 0 else score
        confidence = max(0.0, min(1.0, 0.42 + abs(score) * 0.38 + (0.06 if fear_greed is not None else 0.0)))
        stance = 'BULLISH' if score > 0.08 else 'BEARISH' if score < -0.08 else 'NEUTRAL'
        summary = (
            f"Macro tape scored {pos_hits} positive vs {neg_hits} negative BTC/risk headlines with Fear & Greed at {fear_greed if fear_greed is not None else 'n/a'}; "
            f"macro context {'supports' if stance == 'BULLISH' else 'leans against' if stance == 'BEARISH' else 'does not clearly support'} {parsed.yes_condition}."
        )
        return PolymarketSpecialistSignal(
            agent_name='MacroMarket',
            stance=stance,
            score=round(score, 4),
            confidence=round(confidence, 4),
            summary=summary,
            data_points=[f'fng={fear_greed}', f'positive_hits={pos_hits}', f'negative_hits={neg_hits}'],
            evidence={'spot_price': spot, 'parsed_question': parsed.model_dump(), 'headlines': headlines[:6], 'fear_greed': fear_greed},
        )
