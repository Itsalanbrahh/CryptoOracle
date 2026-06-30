#!/usr/bin/env python3
"""Kalshi position heartbeat — checks open positions for stop-loss, take-profit,
and rebalancing without scanning for new entries. Runs frequently to protect capital."""
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
    from crypto_oracle.kalshi import position_manager as pm
    from crypto_oracle.kalshi.client import KalshiClient
    from crypto_oracle.kalshi.postmortem import log_close_event

    live = os.getenv("KALSHI_LIVE_ENABLED", "0").strip() == "1"

    # ── Step 1: Sync from Kalshi API so we see real positions ──────────────
    await pm.sync_from_kalshi()

    # ── Step 1b: Fetch pending close orders to avoid duplicates ────────────
    client = KalshiClient(key_id=os.getenv("KALSHI_API_KEY_ID", ""))
    try:
        orders_resp = await client._get("/portfolio/orders", params={"limit": 100}, auth=True)
        pending_close = set()
        for o in orders_resp.get("orders", []):
            if o.get("status") == "resting":
                pending_close.add(o["ticker"])
        if pending_close:
            print(f"[Kalshi/HEARTBEAT] Skipping {len(pending_close)} tickers with pending close orders")
    except Exception:
        pending_close = set()

    # ── Step 1c: Mark positions expired locally (API-agnostic) ──────────
    # If the Kalshi API is unreachable, sync_from_kalshi() can't detect
    # expired positions. We extract event date from the ticker itself.
    import re
    from datetime import datetime, timezone
    utc_now = datetime.now(timezone.utc)
    all_pos = pm._load_all()
    marked_local = 0
    # Extract expiry date from ticker and compare against now.
    # Kalshi KXBTCD tickers encode: KXBTCD-{YY}{MON}{DD}{HH}
    # e.g. KXBTCD-26JUN2517 → expires 2026-JUN-25 at 17:00 UTC
    # Bug was: "== today_mmdoy" never fires for positions that expired yesterday
    # (the script may have been down, or ran before market close). Fixed by
    # computing the actual expiry datetime and checking utc_now >= expiry_dt.
    month_abbrs = "JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC"
    month_num = {m: i+1 for i, m in enumerate(
        ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    )}
    for p in all_pos:
        if p.get("closed"):
            continue
        event = p.get("event_ticker", "")
        # Try to parse full expiry datetime from ticker suffix: MON + DD + HH
        m = re.search(f'(\\d{{2}})({month_abbrs})(\\d{{2}})(\\d{{2}})$', event)
        if m:
            yy, mon_str, dd_str, hh_str = m.group(1), m.group(2), m.group(3), m.group(4)
            try:
                expiry_dt = datetime(
                    2000 + int(yy), month_num[mon_str], int(dd_str),
                    int(hh_str), 0, 0, tzinfo=timezone.utc
                )
                if utc_now < expiry_dt:
                    continue
            except (ValueError, KeyError):
                continue
        else:
            # Fallback: match just MON+DD, assume 17:00 UTC expiry
            m2 = re.search(f'({month_abbrs})(\\d{{2}})', event)
            if not m2:
                continue
            mon_str, dd_str = m2.group(1), m2.group(2)
            try:
                expiry_dt = datetime(
                    utc_now.year, month_num[mon_str], int(dd_str),
                    17, 0, 0, tzinfo=timezone.utc
                )
                if utc_now < expiry_dt:
                    continue
            except (ValueError, KeyError):
                continue
        p["closed"] = True
        p["closed_at"] = utc_now.isoformat()
        p["close_reason"] = "settled (expired locally)"
        marked_local += 1
        print(f"[Kalshi/HEARTBEAT] LOCAL EXPIRE {p['ticker']} — marked as settled")
    if marked_local:
        pm._save_all(all_pos)
        print(f"[Kalshi/HEARTBEAT] Marked {marked_local} positions as expired locally")
    
    newly_expired = [p for p in all_pos 
                     if p.get("closed") and p.get("close_reason") 
                     and "settled" in p.get("close_reason", "")
                     and not p.get("postmortem_logged")]
    if newly_expired:
        # Fetch BTC price to determine if positions expired ITM vs OTM
        import aiohttp
        btc_now = 0.0
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    btc_data = await r.json()
                    btc_now = btc_data.get("bitcoin", {}).get("usd", 0)
        except Exception:
            pass

        for p in newly_expired:
            cost = (p.get("entry_price", 0) or 0) * (p.get("count", 0) or 0)
            strike = p.get("strike", 0)
            side = p.get("side", "no")

            # Determine ITM/OTM using current BTC vs strike (best proxy without API)
            if btc_now > 0 and strike > 0:
                if side == "no":
                    # NO = bet BTC stays BELOW strike
                    itm = btc_now <= strike
                else:
                    # YES = bet BTC goes ABOVE strike
                    itm = btc_now >= strike
                if itm:
                    realized_pnl = round((1.0 - (p.get("entry_price", 0) or 0)) * p.get("count", 0), 2)
                    reason_str = "expired_itm"
                else:
                    realized_pnl = -round(cost, 2)
                    reason_str = "expired_otm"
            else:
                realized_pnl = -round(cost, 2)
                reason_str = "expired_otm (unknown settlement)"

            p["realized_pnl"] = realized_pnl
            p["close_price"] = 1.0 if "itm" in reason_str else 0.0

            log_close_event(
                ticker=p["ticker"],
                count=p.get("count", 0),
                side=side,
                close_price=p["close_price"],
                reason=f"{reason_str} ({p.get('close_reason', 'settled')})",
                entry_price=p.get("entry_price", 0) or 0,
                realized_pnl=realized_pnl,
            )
            p["postmortem_logged"] = True
            print(f"[Kalshi/HEARTBEAT] LOGGED EXPIRY {p['ticker']} — pnl=${realized_pnl:.2f} ({reason_str})")
        pm._save_all(all_pos)

    open_positions = pm.get_open_positions()
    if not open_positions:
        if not newly_expired:
            print(f"[Kalshi/HEARTBEAT] No open positions — nothing to monitor.")
        return

    stop_loss_pct = float(os.getenv("KALSHI_STOP_LOSS_PCT", "0.50"))
    take_profit_pct = float(os.getenv("KALSHI_TAKE_PROFIT_PCT", "0.70"))
    strike_dist_stop_pct = float(os.getenv("KALSHI_STRIKE_DISTANCE_STOP_PCT", "0.50"))
    rebalance_enabled = os.getenv("KALSHI_REBALANCE_ENABLED", "1").strip() == "1"
    rebalance_edge_mult = float(os.getenv("KALSHI_REBALANCE_EDGE_MULTIPLIER", "1.5"))
    rebalance_min_hold = float(os.getenv("KALSHI_REBALANCE_MIN_HOLD_MINUTES", "30.0"))

    actions = await pm.scan_positions(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        strike_dist_stop_pct=strike_dist_stop_pct,
        rebalance_min_hold=rebalance_min_hold,
        rebalance_edge_mult=rebalance_edge_mult if rebalance_enabled else 999.0,
        best_new_edge=0.0,  # no rebalancing from heartbeat (entry scan handles that)
    )

    if not actions:
        print(f"[Kalshi/HEARTBEAT] {len(open_positions)} open — all within thresholds. No actions.")
        return

    # Filter out tickers with pending close orders
    fresh_actions = [a for a in actions if a["pos"]["ticker"] not in pending_close]
    skipped = len(actions) - len(fresh_actions)
    if skipped:
        print(f"[Kalshi/HEARTBEAT] Skipped {skipped} actions with pending close orders.")

    if not fresh_actions:
        print(f"[Kalshi/HEARTBEAT] All actions skipped — orders already pending.")
        return

    for action in fresh_actions:
        pos = action["pos"]
        try:
            # ── Verify actual position from API to prevent overshoot ──
            try:
                verify = await client._get(f"/markets/{pos['ticker']}")
                # Get actual position from portfolio
                pf_resp = await client._get("/portfolio/positions", params={"limit": 200}, auth=True)
                actual_pf = 0.0
                for mp in pf_resp.get("market_positions", []):
                    if mp["ticker"] == pos["ticker"]:
                        actual_pf = float(mp.get("position_fp", 0))
                        break
            except Exception:
                actual_pf = 0.0

            actual_side = "yes" if actual_pf > 0 else "no" if actual_pf < 0 else None
            actual_count = abs(int(actual_pf))

            # Skip if position no longer exists
            if actual_side is None or actual_count == 0:
                print(f"[Kalshi/HEARTBEAT] SKIPPED {pos['ticker']} — position already closed on API")
                continue

            # Cap close count to actual position size (prevent overshoot)
            close_count = min(pos["count"], actual_count)
            close_side = actual_side  # close the side we actually hold

            # Kalshi's order API is YES-denominated for both buys and sells.
            # action["close_price"] is NO-denominated for a NO position (used
            # for PnL bookkeeping), so convert it the same way the buy path
            # does. Sending the raw NO price posts the close at the complement
            # level → never fills → stop-loss/take-profit silently fail.
            if close_side == "yes":
                price_cents = int(round(action["close_price"] * 100))
            else:
                price_cents = int(round((1.0 - action["close_price"]) * 100))
            resp = await client.close_position(
                ticker=pos["ticker"],
                count=close_count,
                side=close_side,
                price_cents=price_cents,
            )
            # Don't mark closed locally — let sync_from_kalshi detect fills
            # on the next cycle. This avoids re-triggering on unfilled orders.
            entry_price = pos.get("entry_price", 0) or 0
            close_px = action["close_price"]
            pnl = round((close_px - entry_price) * close_count, 2)
            log_close_event(
                ticker=pos["ticker"],
                count=close_count,
                side=close_side,
                close_price=close_px,
                reason=action["reason"],
                entry_price=entry_price,
                realized_pnl=pnl,
            )
            print(f"[Kalshi/HEARTBEAT] ORDER PLACED {pos['ticker']} — {action['reason']} — "
                  f"close_price=${close_px:.4f} count={close_count} side={close_side} pnl=${pnl}")
        except Exception as exc:
            print(f"[Kalshi/HEARTBEAT] FAILED to close {pos['ticker']}: {exc}")


asyncio.run(main())
