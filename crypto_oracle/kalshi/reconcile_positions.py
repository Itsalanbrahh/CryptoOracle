#!/usr/bin/env python3
"""
Reconcile closed Kalshi positions against the postmortem log.

For each closed position in position_manager, finds the matching postmortem
BUY entry by ticker and writes back resolved=True, resolved_itm, resolved_pnl_usd.
Also runs agent_tracker.resolve_from_postmortem() to update per-agent direction stats.

Run this before confidence_validate.py so calibration data is current.
Typically called by kalshi_confidence_calibration.sh after market close.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

_env = Path("/Users/alanruelas/crypto_oracle/.env")
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k = _k.strip()
        _v = _v.split("#")[0].strip().strip('"').strip("'")
        if _k:
            os.environ.setdefault(_k, _v)

sys.path.insert(0, "/Users/alanruelas/crypto_oracle")


async def main() -> None:
    from crypto_oracle.kalshi import position_manager as pm
    from crypto_oracle.kalshi import agent_tracker as at
    from crypto_oracle.kalshi.postmortem import _LOG_PATH

    # ── Step 1: Sync open/closed positions from Kalshi API ─────────────────
    print("[RECONCILE] Syncing positions from Kalshi API...")
    try:
        n_open = await pm.sync_from_kalshi()
        print(f"[RECONCILE] {n_open} open position(s) after sync")
    except Exception as exc:
        print(f"[RECONCILE] API sync failed ({exc}); using local state only")

    # ── Step 2: Load closed positions that have realized PnL ───────────────
    all_positions = pm._load_all()
    closed_with_pnl = {
        p["ticker"]: p
        for p in all_positions
        if p.get("closed") and p.get("realized_pnl") is not None
    }
    print(f"[RECONCILE] {len(closed_with_pnl)} closed position(s) with known PnL")

    if not _LOG_PATH.exists():
        print("[RECONCILE] No postmortem log found — nothing to reconcile.")
        return

    # ── Step 3: Update postmortem entries with resolution data ─────────────
    raw_lines = _LOG_PATH.read_text().splitlines()
    updated_lines: list[str] = []
    n_updated = 0

    for raw in raw_lines:
        if not raw.strip():
            updated_lines.append(raw)
            continue
        try:
            entry = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            updated_lines.append(raw)
            continue

        # Only update BUY entries that haven't been resolved yet
        if (
            not entry.get("resolved")
            and entry.get("action") in ("BUY_YES", "BUY_NO")
            and entry.get("exec_status") in ("submitted", "paper")
        ):
            ticker = entry.get("ticker", "")
            pos = closed_with_pnl.get(ticker)
            if pos is not None:
                pnl = pos["realized_pnl"]
                entry["resolved"] = True
                entry["resolved_itm"] = pnl > 0
                entry["resolved_pnl_usd"] = round(pnl, 2)
                n_updated += 1
                raw = json.dumps(entry, default=str)

        updated_lines.append(raw)

    if n_updated:
        _LOG_PATH.write_text("\n".join(updated_lines).strip() + "\n")
        print(f"[RECONCILE] Wrote resolution data to {n_updated} postmortem entry/entries")
    else:
        print("[RECONCILE] No new resolutions to record in postmortem")

    # ── Step 4: Update per-agent direction stats ────────────────────────────
    n_agent = at.resolve_from_postmortem()
    print(f"[RECONCILE] Agent tracker processed {n_agent} position(s)")

    # ── Step 5: Summary ─────────────────────────────────────────────────────
    print()
    print(at.summary_text())


asyncio.run(main())
