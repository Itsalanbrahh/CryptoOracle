#!/usr/bin/env python3
"""Paper-account runner for KXBTC15M + Jev + residual self-improvement.

Each cycle:
  1. Label + settle expired paper positions / feature rows
  2. Fetch microstructure (Binance/Bybit OFI)
  3. Jev judgment → residual blend → fee-aware gate
  4. Open virtual trades; log full feature vectors for learning

  python -m crypto_oracle.kalshi.paper_15m_runner
  python -m crypto_oracle.kalshi.paper_15m_runner --loop --interval 90
  python -m crypto_oracle.kalshi.paper_15m_runner --learn   # also run learn job
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv(Path.home() / "crypto_oracle" / ".env")


def _gates() -> dict:
    try:
        from crypto_oracle.kalshi.learn_15m import load_gates
        return load_gates()
    except Exception:
        return {
            "min_edge": float(os.getenv("KALSHI_15M_MIN_EDGE", "0.08")),
            "min_confidence": float(os.getenv("KALSHI_15M_MIN_CONFIDENCE", "0.55")),
            "min_minutes_left": float(os.getenv("KALSHI_15M_MIN_MINUTES_LEFT", "0.5")),
            "max_minutes_left": float(os.getenv("KALSHI_15M_MAX_MINUTES_LEFT", "14.5")),
            "max_spread": float(os.getenv("KALSHI_15M_MAX_SPREAD", "0.08")),
        }


async def run_cycle(*, verbose: bool = True, do_learn: bool = False) -> dict:
    from crypto_oracle.kalshi.belief_15m import blend_belief
    from crypto_oracle.kalshi import feature_store as fs
    from crypto_oracle.kalshi.feeds_ws import collect_live_feeds
    from crypto_oracle.kalshi.jev_15m import evaluate_15m_market
    from crypto_oracle.kalshi.jev_client import jev_enabled
    from crypto_oracle.kalshi.market_data import (
        fetch_funding_rate,
        fetch_realized_vol,
        fetch_recent_1m_closes,
    )
    from crypto_oracle.kalshi.markets import fetch_btc_15m_markets
    from crypto_oracle.kalshi.microstructure import fetch_microstructure
    from crypto_oracle.kalshi.paper_15m import (
        load_account,
        open_paper_trade,
        save_account,
        settle_due_positions,
        summary,
    )
    from crypto_oracle.kalshi.strategy import decide_kalshi_trade
    from crypto_oracle.polymarket.agents import fetch_spot_price

    os.environ.setdefault("KALSHI_15M_ENABLED", "1")
    gates = _gates()

    acct = load_account()
    spot, vol, funding, closes, markets, micro = await asyncio.gather(
        fetch_spot_price(),
        fetch_realized_vol(hours=24),
        fetch_funding_rate(),
        fetch_recent_1m_closes(minutes=20),
        fetch_btc_15m_markets(),
        fetch_microstructure(),
    )

    # Official Kalshi result labels first; spot only as late fallback
    newly_labeled = await fs.label_settled_async(settle_spot=spot)
    settled = settle_due_positions(acct, settle_spot=spot)

    # Live WS trade flow (+ optional Kalshi book if API key present)
    active_ticker = markets[0].ticker if markets else None
    live = await collect_live_feeds(ticker=active_ticker)
    micro = dict(micro or {})
    micro.update(live.as_micro_overlay())
    if live.brti_proxy:
        # Prefer multi-venue WS mid as BRTI proxy for distance features
        spot_for_features = live.brti_proxy
    else:
        spot_for_features = spot

    min_edge = float(gates.get("min_edge", 0.08))
    min_confidence = float(gates.get("min_confidence", 0.55))
    min_minutes = float(gates.get("min_minutes_left", 0.5))
    max_minutes = float(gates.get("max_minutes_left", 14.5))
    max_spread = float(gates.get("max_spread", 0.08))
    max_position = float(os.getenv("KALSHI_MAX_POSITION_USD", "5"))
    maker_mode = os.getenv("KALSHI_MAKER_MODE", "1").strip() == "1"

    decisions: list[dict] = []
    opened: list[str] = []

    for market in markets:
        tte_min = market.hours_to_expiry * 60.0
        spread = max(0.0, market.yes_ask - market.yes_bid)
        if not (min_minutes <= tte_min <= max_minutes):
            decisions.append({
                "ticker": market.ticker,
                "skipped": f"tte={tte_min:.1f}m outside [{min_minutes},{max_minutes}]",
            })
            continue
        if spread > max_spread:
            decisions.append({
                "ticker": market.ticker,
                "skipped": f"spread={spread:.3f} > max_spread={max_spread}",
            })
            continue

        judgment = await evaluate_15m_market(
            market,
            spot=spot_for_features,
            annual_vol=vol,
            funding_rate=funding,
            recent_closes=closes,
        )
        belief = blend_belief(
            market,
            spot=spot_for_features,
            annual_vol=vol,
            funding_rate=funding,
            recent_closes=closes,
            micro=micro,
            jev_p_up=judgment.p_settle_up,
        )

        # Confidence: Jev sharpness + residual agreement
        agree = 1.0 - min(1.0, abs(belief.p_up - belief.jev_p_up) * 4)
        confidence = max(0.20, min(0.95, 0.5 * judgment.confidence + 0.5 * agree))

        decision = decide_kalshi_trade(
            market,
            aggregate=0.0,
            confidence=confidence,
            spot=spot_for_features,
            annual_vol=vol,
            max_position_usd=max_position,
            min_edge=min_edge,
            min_confidence=min_confidence,
            min_strike_distance_pct=0.0,
            momentum_trigger=0.0,
            divergence_cut=1.0,
            maker_mode=maker_mode,
            agg_tilt=0.0,
            implied_prob=belief.p_up,
        )

        # Side policy: probability-aligned + Jev veto on hold
        status = "hold"
        note = decision.reasoning
        if judgment.action == "hold":
            status = "jev_hold"
            note = judgment.reasoning
        elif belief.p_up >= 0.5 and decision.action != "BUY_YES":
            status = "no_yes_edge"
            note = f"p_up={belief.p_up:.3f} decision={decision.action} edge={decision.edge:.3f}"
        elif belief.p_up <= 0.5 and decision.action != "BUY_NO":
            status = "no_no_edge"
            note = f"p_up={belief.p_up:.3f} decision={decision.action} edge={decision.edge:.3f}"
        elif decision.action == "HOLD":
            status = "strategy_hold"
        else:
            pos = open_paper_trade(
                acct,
                ticker=market.ticker,
                side=decision.side,
                count=decision.count,
                entry_price=decision.price,
                strike=decision.strike,
                close_time=market.close_time,
                edge=decision.edge,
                confidence=confidence,
                jev_p_up=judgment.p_settle_up,
                jev_action=judgment.action,
                jev_model=judgment.model,
                spot_at_entry=spot_for_features,
            )
            if pos is None:
                status = "skip_duplicate_or_underfunded"
                note = f"cash=${acct.cash_usd:.2f} cost≈{decision.position_usd}"
            else:
                status = "paper_opened"
                note = (
                    f"{decision.side.upper()} x{decision.count} @ {decision.price:.3f} "
                    f"cost=${pos.cost_usd:.2f} edge={decision.edge:.3f} src={belief.source}"
                )
                opened.append(pos.ticker)

        # Feature store — every decision, including holds (for calibration)
        fs.log_decision({
            "ticker": market.ticker,
            "strike": market.strike,
            "close_time": market.close_time,
            "action": decision.action if status == "paper_opened" else "HOLD",
            "side": decision.side if status == "paper_opened" else None,
            "status": status,
            "p_model": belief.p_up,
            "jev_p_up": belief.jev_p_up,
            "gbm_p_up": belief.gbm_p_up,
            "jev_action": judgment.action,
            "jev_model": judgment.model,
            "belief_source": belief.source,
            "residual": belief.residual.residual,
            "edge": decision.edge,
            "confidence": confidence,
            "yes_ask": market.yes_ask,
            "no_ask": market.no_ask,
            "spot": spot_for_features,
            "spot_raw": spot,
            "brti_proxy": live.brti_proxy,
            "features": belief.features,
            "micro_ok": bool(micro.get("ok")),
            "ws_ok": live.ok,
            "gates": {"min_edge": min_edge, "min_confidence": min_confidence},
        })

        decisions.append({
            "ticker": market.ticker,
            "tte_min": round(tte_min, 2),
            "target": market.strike,
            "yes_ask": market.yes_ask,
            "no_ask": market.no_ask,
            "jev_model": judgment.model,
            "jev_action": judgment.action,
            "jev_p_up": round(belief.jev_p_up, 4),
            "p_model": round(belief.p_up, 4),
            "source": belief.source,
            "ofi": micro.get("combined_ofi"),
            "decision": decision.action,
            "edge": round(decision.edge, 4),
            "status": status,
            "note": note,
        })

    save_account(acct)

    learn_report = None
    if do_learn:
        from crypto_oracle.kalshi.learn_15m import run_async as learn_run
        learn_report = await learn_run(spot=spot)

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "paper",
        "jev_live": jev_enabled(),
        "spot": spot,
        "vol": round(vol, 4),
        "micro": {
            "ok": micro.get("ok"),
            "combined_ofi": micro.get("combined_ofi"),
            "binance_imbalance": micro.get("binance_imbalance"),
            "binance_trade_imbalance": micro.get("binance_trade_imbalance"),
            "venue_disp_bps": micro.get("venue_mid_dispersion_bps"),
            "ws_trade_imbalance": micro.get("ws_trade_imbalance"),
            "ws_ok": live.ok,
            "brti_proxy": live.brti_proxy,
            "kalshi_book_ok": live.kalshi_book.ok,
        },
        "label_counts": fs.label_counts(),
        "gates": gates,
        "newly_labeled": newly_labeled,
        "settled": [
            {"ticker": p.ticker, "pnl": p.pnl_usd, "settled_up": p.settled_up}
            for p in settled
        ],
        "opened": opened,
        "decisions": decisions,
        "account": summary(acct),
        "learn": learn_report,
    }
    if verbose:
        print(json.dumps(out, indent=2, default=str))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="KXBTC15M paper + learning runner")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=float, default=90.0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--cycles", type=int, default=0)
    parser.add_argument("--learn", action="store_true", help="Run learn job each cycle")
    parser.add_argument("--learn-every", type=int, default=5, help="With --loop, learn every N cycles")
    args = parser.parse_args()

    if args.reset:
        from crypto_oracle.kalshi.paper_15m import (
            DEFAULT_BANKROLL,
            PaperAccount,
            _STATE_PATH,
            save_account,
        )
        save_account(PaperAccount(bankroll_usd=DEFAULT_BANKROLL, cash_usd=DEFAULT_BANKROLL))
        print(json.dumps({"reset": True, "bankroll": DEFAULT_BANKROLL, "path": str(_STATE_PATH)}))
        return

    if not args.loop:
        asyncio.run(run_cycle(do_learn=args.learn))
        return

    n = 0
    print(
        f"[paper_15m] looping every {args.interval:.0f}s "
        f"(jev={'ON' if os.getenv('TYPESAFE_API_KEY') or os.getenv('JEV_API_KEY') else 'OFF'})",
        flush=True,
    )
    try:
        while True:
            n += 1
            do_learn = args.learn or (args.learn_every > 0 and n % args.learn_every == 0)
            print(f"\n=== paper cycle {n} (learn={do_learn}) ===", flush=True)
            try:
                asyncio.run(run_cycle(do_learn=do_learn))
            except Exception as exc:
                print(json.dumps({"error": str(exc)}), flush=True)
            if args.cycles and n >= args.cycles:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[paper_15m] stopped", flush=True)


if __name__ == "__main__":
    main()
