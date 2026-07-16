#!/usr/bin/env python3
"""
Resolve paper-mode and filtered Kalshi decisions against actual BTC settlement.

Live trades resolve through the Kalshi API (reconcile_positions.py). But paper
decisions and filtered trades never become positions, so they produced zero
calibration data — even though every scan logs them with real Kalshi quotes.
This script settles them against the actual BTC price at their expiry:

  - exec_status == "paper":    the trade the bot would have made in live mode
  - exec_status == "filtered": the trade a signal filter blocked — resolving
    these gives a COUNTERFACTUAL record of whether each filter actually helps

For every directional entry (including HOLDs) it also records
``resolved_yes_outcome`` — did BTC finish above the strike? — which feeds the
GBM-calibration report with a large sample regardless of whether we traded.

Settlement source: Kraken hourly candles (cached in backtest.fetch_historical_btc).
Entries whose expiry candle isn't available (too old / too recent) are skipped
and retried on the next run. Range markets are skipped (cap strike not logged).

Runs from kalshi_confidence_calibration.sh between reconcile and validate.
"""
from __future__ import annotations

import asyncio
import bisect
import json
import os
import sys
from datetime import datetime, timedelta, timezone
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

# Max distance between expiry time and the settlement candle we accept.
_SETTLE_TOLERANCE_SEC = 45 * 60


def _parse_ts(raw: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


async def main() -> None:
    from crypto_oracle.kalshi.backtest import fetch_historical_btc
    from crypto_oracle.kalshi.postmortem import _LOG_PATH
    from crypto_oracle.kalshi import agent_tracker as at

    if not _LOG_PATH.exists():
        print("[PAPER-RESOLVE] No postmortem log found — nothing to resolve.")
        return

    raw_lines = _LOG_PATH.read_text().splitlines()
    now = datetime.now(timezone.utc)

    # ── Fetch settlement candles (last ~30 days of hourly closes) ───────────
    candles = await fetch_historical_btc(days=30)
    ts_list = [c["timestamp"] for c in candles]
    close_list = [c["close"] for c in candles]

    def settle_price_at(expiry: datetime) -> float | None:
        """Hourly close nearest to expiry, or None if no candle is close enough."""
        if not ts_list:
            return None
        target = expiry.timestamp()
        i = bisect.bisect_left(ts_list, target)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ts_list):
                dist = abs(ts_list[j] - target)
                if dist <= _SETTLE_TOLERANCE_SEC and (best is None or dist < best[0]):
                    best = (dist, close_list[j])
        return best[1] if best else None

    resolved_trades = 0
    resolved_outcomes = 0
    skipped_range = 0
    skipped_no_data = 0
    wins = 0
    out_lines: list[str] = []

    for raw in raw_lines:
        if not raw.strip():
            out_lines.append(raw)
            continue
        try:
            entry = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            out_lines.append(raw)
            continue

        action = entry.get("action", "")
        strike = entry.get("strike") or 0
        tte = entry.get("hours_to_expiry")
        entered = _parse_ts(entry.get("ts", ""))
        changed = False

        # Only scan-decision entries with enough data to settle
        if (
            entered is not None
            and strike > 0
            and tte is not None
            and action in ("BUY_YES", "BUY_NO", "HOLD")
        ):
            if entry.get("is_range"):
                if not entry.get("resolved") and action != "HOLD":
                    skipped_range += 1
            else:
                expiry = entered + timedelta(hours=float(tte))
                # Wait a full hour past expiry so the settlement candle exists
                if now >= expiry + timedelta(hours=1):
                    settle = settle_price_at(expiry)
                    if settle is None:
                        if not entry.get("resolved"):
                            skipped_no_data += 1
                    else:
                        # GBM-calibration outcome for EVERY entry (incl. HOLD):
                        # did BTC finish above the strike?
                        if entry.get("resolved_yes_outcome") is None:
                            entry["resolved_yes_outcome"] = bool(settle >= strike)
                            entry["resolved_settle_price"] = round(settle, 2)
                            resolved_outcomes += 1
                            changed = True

                        # Full trade resolution for unresolved paper/filtered BUYs
                        if (
                            not entry.get("resolved")
                            and action in ("BUY_YES", "BUY_NO")
                            and entry.get("exec_status") in ("paper", "filtered")
                            and entry.get("side") in ("yes", "no")
                        ):
                            side = entry["side"]
                            won = settle >= strike if side == "yes" else settle < strike
                            pnl = (
                                entry.get("profit_if_win") or 0.0
                                if won
                                else -(entry.get("position_usd") or 0.0)
                            )
                            entry["resolved"] = True
                            entry["resolved_itm"] = bool(won)
                            entry["resolved_pnl_usd"] = round(pnl, 2)
                            entry["resolved_by"] = "paper_resolver"
                            resolved_trades += 1
                            if won:
                                wins += 1
                            changed = True

        out_lines.append(json.dumps(entry, default=str) if changed else raw)

    if resolved_trades or resolved_outcomes:
        _LOG_PATH.write_text("\n".join(out_lines).strip() + "\n")

    # ── Rebuild agent tracker stats from the full (now-updated) log ─────────
    # A full rebuild every run keeps stats consistent with the current
    # direction metric over ALL history (and cleans out anything accumulated
    # under an older, buggier metric). Cheap: one pass over the JSONL.
    tracked = at.rebuild_from_postmortem()

    print(
        f"[PAPER-RESOLVE] trades resolved={resolved_trades} ({wins} wins) | "
        f"yes-outcomes recorded={resolved_outcomes} | tracker rebuilt from {tracked} | "
        f"skipped: range={skipped_range} no-candle={skipped_no_data}"
    )


asyncio.run(main())
