#!/usr/bin/env python3
"""Kalshi BTC live trading script — no Claude agent call, runs directly."""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

_env = Path('/Users/alanruelas/crypto_oracle/.env')
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith('#') or '=' not in _line:
            continue
        _k, _, _v = _line.partition('=')
        _k = _k.strip()
        _v = _v.split('#')[0].strip().strip('"').strip("'")
        if _k:
            os.environ.setdefault(_k, _v)

sys.path.insert(0, '/Users/alanruelas/crypto_oracle')


async def main() -> None:
    from crypto_oracle.models.db import init_db
    from crypto_oracle.kalshi.loop import run_kalshi_scan

    await init_db()
    live = os.getenv("KALSHI_LIVE_ENABLED", "0").strip() == "1"
    result = await run_kalshi_scan(limit=8, live=live)

    mode = result['mode'].upper()
    spot = result['spot_price']
    scanned = result['markets_scanned']
    executed = result['trades_executed']
    deployed = result['total_deployed_usd']

    lines = [f'[Kalshi/{mode}] BTC=${spot:,.0f} | {scanned} markets scanned']
    if executed:
        lines.append(f'  {executed} trade(s) executed — ${deployed:.2f} deployed')

    for r in result.get('results', []):
        action = r['action']
        ticker = r['ticker']
        short = ticker.split('-T')[-1] if '-T' in ticker else ticker
        conf = r['confidence']
        edge = r['edge']
        pos = r['position_usd']
        profit = r['profit_if_win']
        status = r['exec_status']
        err = r.get('exec_error') or ''

        if action != 'HOLD':
            lines.append(f'>>> {action}: BTC above ${r["strike"]:,.0f}')
            lines.append(f'    conf={conf:.2f} edge={edge:.3f} pos=${pos:.2f} profit_if_win=${profit:.2f}')
            if err:
                lines.append(f'    ERROR: {err[:80]}')
            else:
                oid = r.get('order_id') or ''
                lines.append(f'    status={status}' + (f' id={oid}' if oid else ''))
        else:
            reason = r.get('reasoning', '')[:60]
            lines.append(f'HOLD | ${r["strike"]:,.0f} | mid={r["market_mid"]:.2f} | {reason}')

    if not result.get('results'):
        lines.append('No liquid BTC markets found.')
    elif executed == 0 and mode == 'LIVE':
        lines.append('No trades executed this tick.')

    print('\n'.join(lines))


asyncio.run(main())
