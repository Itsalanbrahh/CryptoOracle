"""
Kalshi BTC strategy backtest engine.

Replays the strategy on historical BTC hourly data to measure:
  - Win rate by edge / confidence / time-to-expiry
  - P&L curve across simulated trades
  - Optimal parameter ranges

Uses GBM-based pricing as the agent signal proxy (same model as live strategy)
so backtests are fast and don't require external API calls for agent inference.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist as _ND
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

_BACKTEST_DIR = Path.home() / ".hermes" / "state" / "backtest"
_DEFAULT_DAYS = 60
_VOL_FLOOR = 0.30
_VOL_CAP = 2.50
_VOL_DEFAULT = 0.65

# ── Historical data fetching ──────────────────────────────────────────────────

# Module-level candle cache: avoids repeat Kraken API calls within a single scan.
# Key = days requested, value = (unix_timestamp_fetched, candles_list).
_CANDLE_CACHE: dict[int, tuple[float, list[dict]]] = {}
_CANDLE_CACHE_TTL = 1800  # 30 minutes


async def fetch_historical_btc(days: int = _DEFAULT_DAYS) -> list[dict]:
    """Fetch hourly BTC candles from Kraken for the last N days.

    Results are cached for 30 minutes so multiple agents in a single scan
    share one Kraken API call instead of each making independent requests.
    Returns list of {ts, open, high, low, close} sorted chronologically.
    """
    import time
    import aiohttp
    import asyncio

    cached = _CANDLE_CACHE.get(days)
    if cached:
        fetched_at, candles = cached
        if time.time() - fetched_at < _CANDLE_CACHE_TTL:
            return candles

    # Kraken OHLC: interval=60 (1h)
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XBTUSD", "interval": 60}

    # Kraken returns the last 720 candles per call (30 days)
    # We need multiple calls for > 30 days
    all_rows: list[list] = []
    seen = set()

    async with aiohttp.ClientSession() as session:
        for offset in range(0, days, 28):
            # Kraken doesn't support offset param natively; we use `since`
            # Fetch a larger window and dedupe
            try:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=20)
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                rows = data.get("result", {}).get("XXBTZUSD", [])
                for row in rows:
                    ts = int(row[0])
                    if ts not in seen:
                        seen.add(ts)
                        all_rows.append(row)
            except Exception:
                continue
            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)

    # Sort by timestamp and convert to dicts
    all_rows.sort(key=lambda r: int(r[0]))
    candles = []
    for row in all_rows:
        ts = int(row[0])
        candles.append({
            "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "timestamp": ts,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[6]),
        })

    _CANDLE_CACHE[days] = (time.time(), candles)
    return candles


# ── Volatility estimation ─────────────────────────────────────────────────────


def _hourly_vol(closes: list[float]) -> float:
    """Annualized vol from hourly log returns."""
    if len(closes) < 10:
        return _VOL_DEFAULT
    log_rets = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    if len(log_rets) < 4:
        return _VOL_DEFAULT
    mean = sum(log_rets) / len(log_rets)
    var = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)
    return max(_VOL_FLOOR, min(_VOL_CAP, math.sqrt(max(0.0, var)) * math.sqrt(8760)))


# ── GBM pricing (replicating strategy.py logic) ───────────────────────────────


def _gbm_prob(spot: float, strike: float, hours_to_expiry: float, annual_vol: float) -> float:
    t = max(hours_to_expiry, 0.25) / 8760
    sigma_t = annual_vol * math.sqrt(t)
    if sigma_t < 1e-9:
        return 1.0 if spot >= strike else 0.0
    d2 = math.log(spot / strike) / sigma_t
    return _ND().cdf(d2)


def _gbm_range_prob(spot: float, floor: float, cap: float, hours_to_expiry: float, annual_vol: float) -> float:
    return _gbm_prob(spot, floor, hours_to_expiry, annual_vol) - _gbm_prob(spot, cap, hours_to_expiry, annual_vol)


async def run_backtest(
    days: int = _DEFAULT_DAYS,
    min_edge: float = 0.03,
    min_confidence: float = 0.52,
    max_position: float = 5.0,
    tte_min: float = 2.0,
    tte_max: float = 72.0,
    strike_step: float = 250.0,
    output_path: str | None = None,
) -> dict:
    """Run a full backtest.

    Simulates every possible trade across historical hourly windows,
    records whether the GBM-model-based strategy would win or lose.

    Returns a dict with summary stats and per-trade records.
    """
    import asyncio
    # Ensure we fetch async data properly
    candles = await fetch_historical_btc(days=days)
    if len(candles) < 48:
        return {"error": f"Not enough historical data: {len(candles)} candles"}

    trades: list[dict] = []
    hourly_closes = [c["close"] for c in candles]

    print(f"Backtest: {len(candles)} hourly candles, {days} days")

    for i in range(len(candles)):
        candle = candles[i]
        spot = candle["close"]
        ts = candle["ts"]

        # Use trailing vol
        vol = _hourly_vol(hourly_closes[: i + 1])

        # Simulate a set of possible strikes at $250 intervals
        # around spot ± 15%
        for dir_sign in [-1, 1]:
            for step in range(1, 20):
                strike = spot + dir_sign * step * strike_step
                if strike <= 0:
                    continue
                # Simulate TTE across the full range, including short windows
                hours_to = max(tte_min, min(tte_max, step * 1.0))
                # Mix in some 1-3h entries for diversity
                if step % 3 == 0:
                    hours_to = max(tte_min, min(tte_max, 0.5 + step * 0.15))
                if hours_to < tte_min or hours_to > tte_max:
                    continue

                gbm_yes = _gbm_prob(spot, strike, hours_to, vol)

                # Market price = efficient GBM pricing with tight spread (market is mostly efficient)
                market_yes = gbm_yes

                # Agent belief = GBM + signal noise (agents think they see mispricing)
                # Higher vol = more uncertainty = wider agent disagreement
                agent_signal_scale = min(0.10, max(0.01, (vol - 0.30) * 0.12))
                # Use a deterministic pseudo-random signal based on candle index
                _r = ((i * 7 + step * 13 + dir_sign * 31) % 1001) / 1000  # 0..1
                signal = (_r * 2.0 - 1.0) * agent_signal_scale
                belief_yes = max(0.02, min(0.98, gbm_yes + signal))
                belief_no = 1.0 - belief_yes

                market_no = 1.0 - market_yes

                edge_yes = belief_yes - market_yes
                edge_no = belief_no - market_no

                # Simulate confidence (correlated with vol stability)
                confidence = max(0.40, min(0.95, 0.55 + (0.25 * (1.0 - min(1.0, vol / 1.5)))))

                # Decision: buy YES or NO?
                action = "HOLD"
                exec_price = 0.0
                position_usd = 0.0
                profit_if_win = 0.0
                buy_yes = edge_yes > min_edge and confidence >= min_confidence
                buy_no = edge_no > min_edge and confidence >= min_confidence

                if buy_yes and not buy_no:
                    action = "BUY_YES"
                    exec_price = market_yes + 0.01  # simulate ask spread
                    pos_size = max_position * min(1.0, confidence * (1.0 + edge_yes))
                    count = max(1, int(pos_size / exec_price))
                    position_usd = round(count * exec_price, 2)
                    profit_if_win = round(count * (1.0 - exec_price), 2)
                    edge = edge_yes
                elif buy_no and not buy_yes:
                    action = "BUY_NO"
                    exec_price = market_no + 0.01  # simulate ask spread for NO
                    pos_size = max_position * min(1.0, confidence * (1.0 + edge_no))
                    count = max(1, int(pos_size / exec_price))
                    position_usd = round(count * exec_price, 2)
                    profit_if_win = round(count * (1.0 - exec_price), 2)
                    edge = edge_no
                elif buy_yes and buy_no:
                    # Both edges positive — take the larger one
                    if edge_yes >= edge_no:
                        action = "BUY_YES"
                        exec_price = market_yes + 0.01
                        pos_size = max_position * min(1.0, confidence * (1.0 + edge_yes))
                        count = max(1, int(pos_size / exec_price))
                        position_usd = round(count * exec_price, 2)
                        profit_if_win = round(count * (1.0 - exec_price), 2)
                        edge = edge_yes
                    else:
                        action = "BUY_NO"
                        exec_price = market_no + 0.01
                        pos_size = max_position * min(1.0, confidence * (1.0 + edge_no))
                        count = max(1, int(pos_size / exec_price))
                        position_usd = round(count * exec_price, 2)
                        profit_if_win = round(count * (1.0 - exec_price), 2)
                        edge = edge_no
                else:
                    continue  # HOLD — no trade signal

                # Determine outcome: use the candle at actual expiry, not just i+1.
                # hours_to can range 2–72h; settling on i+1 (1h later) regardless
                # inflates win rates for far-OTM NO trades and distorts PnL.
                settle_offset = max(1, min(int(round(hours_to)), len(candles) - 1 - i))
                settle_idx = i + settle_offset
                if settle_idx < len(candles):
                    settle_price = candles[settle_idx]["close"]
                else:
                    settle_price = spot

                if action == "BUY_YES":
                    won = settle_price >= strike
                else:
                    won = settle_price < strike

                # profit_if_win is already the NET gain (count * (1 - exec_price)).
                # Only deduct position_usd on a loss; do NOT subtract it again on a win.
                pnl = round(profit_if_win, 2) if won else round(-position_usd, 2)

                trade = {
                    "ts": ts,
                    "spot_entry": round(spot, 2),
                    "strike": round(strike, 2),
                    "action": action,
                    "edge": round(edge, 4),
                    "confidence": round(confidence, 4),
                    "hours_to_expiry": round(hours_to, 1),
                    "settle_candle_offset": settle_offset,
                    "position_usd": position_usd,
                    "profit_if_win": profit_if_win,
                    "won": won,
                    "pnl": pnl,
                    "aggregate": round(signal, 4),  # raw agent signal driving the trade
                    "gbm_baseline": round(gbm_yes, 4),
                    "belief_yes": round(belief_yes, 4),
                    "market_yes_price": round(market_yes, 4),
                    "vol": round(vol, 4),
                    "strike_distance_pct": round((strike - spot) / spot * 100, 2),
                }
                trades.append(trade)

    if not trades:
        return {"error": "No trades generated — check backtest params"}

    # Aggregate results
    total = len(trades)
    wins = sum(1 for t in trades if t["won"])
    losses = total - wins
    total_pnl = sum(t["pnl"] for t in trades)
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = sum(t["pnl"] for t in trades if t["pnl"] < 0)

    # Per-side breakdown — critical for understanding YES/NO asymmetry
    yes_trades = [t for t in trades if t["action"] == "BUY_YES"]
    no_trades = [t for t in trades if t["action"] == "BUY_NO"]
    yes_wins = sum(1 for t in yes_trades if t["won"])
    no_wins = sum(1 for t in no_trades if t["won"])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_backtested": days,
        "candles_analyzed": len(candles),
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total, 4) if total else 0,
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / total, 2) if total else 0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(abs(gross_profit / gross_loss), 2) if gross_loss else float("inf"),
        "max_drawdown_pct": 0.0,
        "best_trade": max(trades, key=lambda t: t["pnl"])["pnl"] if trades else 0,
        "worst_trade": min(trades, key=lambda t: t["pnl"])["pnl"] if trades else 0,
        "yes_trades": len(yes_trades),
        "yes_wins": yes_wins,
        "yes_win_rate": round(yes_wins / len(yes_trades), 4) if yes_trades else 0,
        "yes_pnl": round(sum(t["pnl"] for t in yes_trades), 2),
        "yes_avg_exec_price": round(
            sum(t["market_yes_price"] for t in yes_trades) / len(yes_trades), 4
        ) if yes_trades else 0,
        "no_trades": len(no_trades),
        "no_wins": no_wins,
        "no_win_rate": round(no_wins / len(no_trades), 4) if no_trades else 0,
        "no_pnl": round(sum(t["pnl"] for t in no_trades), 2),
        "no_avg_exec_price": round(
            sum(1.0 - t["market_yes_price"] for t in no_trades) / len(no_trades), 4
        ) if no_trades else 0,
        "yes_pct": round(len(yes_trades) / total, 4) if total else 0,
        "no_pct": round(len(no_trades) / total, 4) if total else 0,
        "settings": {
            "min_edge": min_edge,
            "min_confidence": min_confidence,
            "max_position": max_position,
            "tte_min": tte_min,
            "tte_max": tte_max,
        },
    }

    # Win rate by edge bucket
    edge_buckets = {}
    for t in trades:
        bucket = f"{math.floor(t['edge'] / 0.02) * 0.02:.2f}"
        if bucket not in edge_buckets:
            edge_buckets[bucket] = {"count": 0, "wins": 0}
        edge_buckets[bucket]["count"] += 1
        if t["won"]:
            edge_buckets[bucket]["wins"] += 1

    # Win rate by confidence bucket
    conf_buckets = {}
    for t in trades:
        bucket = f"{math.floor(t['confidence'] / 0.05) * 0.05:.2f}"
        if bucket not in conf_buckets:
            conf_buckets[bucket] = {"count": 0, "wins": 0}
        conf_buckets[bucket]["count"] += 1
        if t["won"]:
            conf_buckets[bucket]["wins"] += 1

    # Compute max drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cumulative += t["pnl"]
        if cumulative > peak:
            peak = cumulative
        dd = (peak - cumulative) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    summary["max_drawdown_pct"] = round(max_dd * 100, 2)

    result = {
        "summary": summary,
        "trades": trades,
        "edge_buckets": {k: {"count": v["count"], "wins": v["wins"],
                             "win_rate": round(v["wins"] / v["count"], 3) if v["count"] else 0}
                         for k, v in sorted(edge_buckets.items())},
        "conf_buckets": {k: {"count": v["count"], "wins": v["wins"],
                             "win_rate": round(v["wins"] / v["count"], 3) if v["count"] else 0}
                         for k, v in sorted(conf_buckets.items())},
    }

    # Add TTE bucket breakdown — shows how settlement timing affects win rate
    tte_buckets: dict[str, dict] = {}
    for t in trades:
        tte_key = f"{int(t['hours_to_expiry'])}h"
        if tte_key not in tte_buckets:
            tte_buckets[tte_key] = {"count": 0, "wins": 0, "pnl": 0.0,
                                    "yes": 0, "no": 0}
        b = tte_buckets[tte_key]
        b["count"] += 1
        b["pnl"] = round(b["pnl"] + t["pnl"], 2)
        if t["won"]:
            b["wins"] += 1
        if t["action"] == "BUY_YES":
            b["yes"] += 1
        else:
            b["no"] += 1
    result["tte_buckets"] = {
        k: {**v, "win_rate": round(v["wins"] / v["count"], 3) if v["count"] else 0}
        for k, v in sorted(tte_buckets.items(), key=lambda x: int(x[0].rstrip("h")))
    }

    # Save to file
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str))
        print(f"Backtest saved to {path}")

    return result


async def run_agent_backtest(
    agent_name: str = "MeanReversion",
    days: int = _DEFAULT_DAYS,
    min_confidence: float = 0.0,
    min_score: float = 0.0,
    output_path: str | None = None,
) -> dict:
    """Replay a single named agent on historical data using its actual logic.

    Unlike run_backtest() which uses GBM + pseudo-noise for all agents,
    this function instantiates the real agent class, feeds it actual hourly
    candles via a mock market object, and records its per-trade decisions.

    Supported agent_name values: MeanReversion, MomentumContinuation,
    VolatilitySnapback.

    Settlement uses the candle at hours_to_expiry (same fix as run_backtest).
    PnL: profit_if_win on win, -position_usd on loss (no double-deduction).
    """
    from .agents.edge_agents import (
        MeanReversionAgent,
        MomentumContinuationAgent,
        VolatilitySnapbackAgent,
    )

    _AGENT_MAP = {
        "MeanReversion": MeanReversionAgent,
        "MomentumContinuation": MomentumContinuationAgent,
        "VolatilitySnapback": VolatilitySnapbackAgent,
    }

    if agent_name not in _AGENT_MAP:
        return {"error": f"Unknown agent '{agent_name}'. Choose from: {list(_AGENT_MAP)}"}

    agent = _AGENT_MAP[agent_name]()
    candles = await fetch_historical_btc(days=days)
    if len(candles) < 48:
        return {"error": f"Not enough data: {len(candles)} candles"}

    print(f"Agent backtest ({agent_name}): {len(candles)} candles")

    # We monkey-patch fetch_historical_btc inside edge_agents so the agent
    # sees the same cached candles we already have (avoids re-fetching).
    import crypto_oracle.kalshi.backtest as _bt_module
    _orig = _bt_module.fetch_historical_btc

    async def _cached_fetch(days=_DEFAULT_DAYS):
        return candles

    _bt_module.fetch_historical_btc = _cached_fetch

    trades: list[dict] = []
    tte_default = 6.0  # agent scores don't carry TTE; use a fixed 6h horizon

    try:
        for i in range(24, len(candles)):  # need 24h warmup for MA-based agents
            spot = candles[i]["close"]
            ts = candles[i]["ts"]

            # Provide a minimal market proxy — agents only use candle history
            proxy = type("MockMarket", (), {"strike": spot, "hours_to_expiry": tte_default})()

            result_sig = await agent.run(proxy)
            score = result_sig.get("score", 0.0)
            confidence = result_sig.get("confidence", 0.0)

            if abs(score) < min_score or confidence < min_confidence:
                continue  # agent abstains

            # Translate score to YES/NO: positive = bullish = BUY_YES
            if score > 0:
                action = "BUY_YES"
                exec_price = 0.55 + abs(score) * 0.10  # rough proxy: stronger signal → less cheap YES
                exec_price = min(0.90, max(0.10, exec_price))
            else:
                action = "BUY_NO"
                exec_price = 0.15 + abs(score) * 0.10
                exec_price = min(0.90, max(0.10, exec_price))

            count = max(1, int(5.0 / exec_price))
            position_usd = round(count * exec_price, 2)
            profit_if_win = round(count * (1.0 - exec_price), 2)

            # Settle at TTE candle
            settle_offset = max(1, min(int(round(tte_default)), len(candles) - 1 - i))
            settle_price = candles[i + settle_offset]["close"]

            # Approximate strike: current spot (agent bets on direction, not specific strike)
            strike = spot
            if action == "BUY_YES":
                won = settle_price > strike  # price went up
            else:
                won = settle_price <= strike  # price did not go up

            pnl = round(profit_if_win, 2) if won else round(-position_usd, 2)

            # Track directional accuracy independently from trade outcome:
            # was the agent's signal direction (positive=up, negative=down) correct?
            price_went_up = settle_price > spot
            dir_correct = (score > 0 and price_went_up) or (score < 0 and not price_went_up)

            trades.append({
                "ts": ts,
                "spot_entry": round(spot, 2),
                "action": action,
                "score": round(score, 4),
                "confidence": round(confidence, 4),
                "exec_price": round(exec_price, 4),
                "position_usd": position_usd,
                "profit_if_win": profit_if_win,
                "settle_offset": settle_offset,
                "settle_price": round(settle_price, 2),
                "won": won,
                "direction_correct": dir_correct,
                "pnl": pnl,
                "reasoning": result_sig.get("reasoning", ""),
            })
    finally:
        _bt_module.fetch_historical_btc = _orig

    if not trades:
        return {"error": f"No trades from {agent_name} — try lowering min_score/min_confidence"}

    total = len(trades)
    wins = sum(1 for t in trades if t["won"])
    dir_correct_total = sum(1 for t in trades if t["direction_correct"])
    yes_trades = [t for t in trades if t["action"] == "BUY_YES"]
    no_trades = [t for t in trades if t["action"] == "BUY_NO"]
    yes_wins = sum(1 for t in yes_trades if t["won"])
    no_wins = sum(1 for t in no_trades if t["won"])
    total_pnl = sum(t["pnl"] for t in trades)

    # Score differential on NO trades: good agents are more bearish on winning NO trades
    no_win_scores = [t["score"] for t in no_trades if t["won"]]
    no_loss_scores = [t["score"] for t in no_trades if not t["won"]]
    no_score_diff = (
        (sum(no_win_scores) / len(no_win_scores)) - (sum(no_loss_scores) / len(no_loss_scores))
        if no_win_scores and no_loss_scores else None
    )

    summary = {
        "agent": agent_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "days_backtested": days,
        "total_trades": total,
        "wins": wins,
        "win_rate": round(wins / total, 4),
        "direction_accuracy": round(dir_correct_total / total, 4),  # score sign vs actual price move
        "total_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / total, 2),
        "yes_trades": len(yes_trades),
        "yes_wins": yes_wins,
        "yes_win_rate": round(yes_wins / len(yes_trades), 4) if yes_trades else 0,
        "yes_pnl": round(sum(t["pnl"] for t in yes_trades), 2),
        "no_trades": len(no_trades),
        "no_wins": no_wins,
        "no_win_rate": round(no_wins / len(no_trades), 4) if no_trades else 0,
        "no_pnl": round(sum(t["pnl"] for t in no_trades), 2),
        "yes_pct": round(len(yes_trades) / total, 4) if total else 0,
        "no_pct": round(len(no_trades) / total, 4) if total else 0,
        "no_score_diff_win_vs_loss": round(no_score_diff, 4) if no_score_diff is not None else None,
        "abstain_rate": "not tracked (only fired trades recorded)",
    }

    print(f"\n{agent_name} standalone results ({days}d):")
    print(f"  Trades: {total} | WR: {summary['win_rate']:.1%} | PnL: ${total_pnl:.2f}")
    print(f"  YES: {len(yes_trades)} ({summary['yes_pct']:.1%}) WR={summary['yes_win_rate']:.1%} PnL=${summary['yes_pnl']:.2f}")
    print(f"  NO:  {len(no_trades)} ({summary['no_pct']:.1%}) WR={summary['no_win_rate']:.1%} PnL=${summary['no_pnl']:.2f}")

    result = {"summary": summary, "trades": trades}

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str))
        print(f"Agent backtest saved to {path}")

    return result
