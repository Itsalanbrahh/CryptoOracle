from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from crypto_oracle.polymarket.models import PolymarketMarket, PolymarketMasterDecision

ExecutionMode = Literal['paper', 'live']

BRIDGE_SCRIPT = Path('/Users/alanruelas/crypto_oracle/crypto_oracle/polymarket/live_bridge.mjs')
NODE_PACKAGE_ROOT = Path('/Users/alanruelas/crypto_oracle')


class ExecutionConfig(BaseModel):
    mode: ExecutionMode = 'paper'
    dry_run: bool = True
    node_bin: str = 'node'


class ExecutionResult(BaseModel):
    mode: ExecutionMode
    status: str
    dry_run: bool = True
    external_order_id: str | None = None
    response: dict = Field(default_factory=dict)
    error: str | None = None


def load_execution_config() -> ExecutionConfig:
    mode = str(os.getenv('POLYMARKET_EXECUTION_MODE', 'paper')).strip().lower()
    if mode not in {'paper', 'live'}:
        mode = 'paper'
    dry_run = str(os.getenv('POLYMARKET_LIVE_DRY_RUN', '1')).strip().lower() not in {'0', 'false', 'no'}
    return ExecutionConfig(mode=mode, dry_run=dry_run)


def decision_to_live_order(market: PolymarketMarket, decision: PolymarketMasterDecision) -> dict:
    if decision.action not in {'BUY_YES', 'BUY_NO'}:
        raise ValueError('decision is not executable')
    price = decision.price or (market.yes_outcome.price if decision.action == 'BUY_YES' else market.no_outcome.price)
    if price is None or price <= 0:
        raise ValueError('missing executable price')
    quantity = max(1, int(math.floor((decision.position_size_usd or 0.0) / price)))
    return {
        'marketSlug': market.slug,
        'intent': 'ORDER_INTENT_BUY_LONG' if decision.action == 'BUY_YES' else 'ORDER_INTENT_BUY_SHORT',
        'type': 'ORDER_TYPE_LIMIT',
        'price': round(float(price), 4),
        'quantity': quantity,
        'tif': 'TIME_IN_FORCE_GOOD_TILL_CANCEL',
    }


async def execute_live_order(market: PolymarketMarket, decision: PolymarketMasterDecision, config: ExecutionConfig | None = None) -> ExecutionResult:
    cfg = config or load_execution_config()
    if cfg.mode != 'live':
        return ExecutionResult(mode=cfg.mode, status='skipped_non_live_mode', dry_run=cfg.dry_run)
    if decision.action == 'HOLD':
        return ExecutionResult(mode=cfg.mode, status='skipped_hold', dry_run=cfg.dry_run)
    if not market.slug:
        return ExecutionResult(mode=cfg.mode, status='error', dry_run=cfg.dry_run, error='market slug is required for live execution')

    payload = decision_to_live_order(market, decision)
    payload['dryRun'] = cfg.dry_run
    proc = await asyncio.create_subprocess_exec(
        cfg.node_bin,
        str(BRIDGE_SCRIPT),
        json.dumps(payload),
        cwd=str(NODE_PACKAGE_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )
    stdout, stderr = await proc.communicate()
    text_out = stdout.decode().strip()
    text_err = stderr.decode().strip()
    if proc.returncode != 0:
        return ExecutionResult(
            mode=cfg.mode,
            status='error',
            dry_run=cfg.dry_run,
            error=text_out or text_err or f'node bridge failed with exit {proc.returncode}',
        )
    try:
        data = json.loads(text_out) if text_out else {}
    except json.JSONDecodeError:
        return ExecutionResult(mode=cfg.mode, status='error', dry_run=cfg.dry_run, error=f'invalid bridge JSON: {text_out[:500]}')

    external_order_id = None
    response = data.get('response') or {}
    for key in ('id', 'orderId', 'order_id'):
        if isinstance(response, dict) and response.get(key):
            external_order_id = str(response[key])
            break
    status = 'dry_run_validated' if cfg.dry_run else 'submitted'
    return ExecutionResult(
        mode=cfg.mode,
        status=status,
        dry_run=cfg.dry_run,
        external_order_id=external_order_id,
        response=data,
    )
