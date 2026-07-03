"""Kalshi BTC trading loop — fetches markets, runs agents, executes."""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from crypto_oracle.polymarket.agents import (
    KnowledgeMarketAgent,
    LinearRegressionMarketAgent,
    MacroMarketAgent,
    TechnicalMarketAgent,
    fetch_spot_price,
    fetch_spot_history,
)
from .client import KalshiClient
from .deribit import implied_prob_above
from .market_data import fetch_funding_rate, fetch_realized_vol, funding_tilt
from .markets import KalshiMarket, fetch_btc_markets, fetch_btc_range_markets, select_target_markets
from .strategy import KalshiDecision, decide_kalshi_trade
from .postmortem import build_entry, log_entry, read_recent
from . import agent_tracker as at
from . import position_manager as pm
from .agents import (
    KronosMarketAgent,
    FibonacciRetracementAgent,
    CandlestickPatternAgent,
    SupportResistanceAgent,
    DynamicSRAgent,
    FairValueGapAgent,
    MomentumContinuationAgent,
    MeanReversionAgent,
    VolatilitySnapbackAgent,
)
from .agents.signal import wrap_agent_result

# Master kill switch: touch this file to halt all new Kalshi orders immediately.
_PAUSED_PATH = Path(__file__).resolve().parent.parent.parent / "PAUSED"

# Daily risk state persisted across cron runs so the cap survives restarts.
_DAILY_STATE_PATH = Path.home() / ".hermes" / "state" / "kalshi_daily_state.json"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw and raw.strip():
        try:
            return float(raw.strip())
        except ValueError:
            pass
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw and raw.strip():
        try:
            return int(raw.strip())
        except ValueError:
            pass
    return default


def _load_daily_state() -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _DAILY_STATE_PATH.exists():
        try:
            state = json.loads(_DAILY_STATE_PATH.read_text())
            if state.get("date") == today:
                return state
        except Exception:
            pass
    return {"date": today, "deployed_usd": 0.0, "trades": 0}


def _save_daily_state(state: dict) -> None:
    _DAILY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DAILY_STATE_PATH.write_text(json.dumps(state))


AGENTS = [
    # Market specialists (pre-existing)
    MacroMarketAgent(),
    KnowledgeMarketAgent(),
    TechnicalMarketAgent(),
    LinearRegressionMarketAgent(),
    # Foundation model (deep learning time-series forecast)
    KronosMarketAgent(),
    # Technical analysis patterns
    CandlestickPatternAgent(),
    SupportResistanceAgent(),
    DynamicSRAgent(),
    FairValueGapAgent(),
    # Fibonacci retracement levels
    FibonacciRetracementAgent(),
    # Proven edge agents (from 60 days of BTC data analysis)
    MomentumContinuationAgent(),
    MeanReversionAgent(),
    VolatilitySnapbackAgent(),
]


async def _run_agents(market: KalshiMarket, spot: float, annual_vol: float | None = None, funding_rate: float | None = None) -> tuple[float, float, dict, float]:
    """Run all specialist agents, return (aggregate, confidence, agent_signals).

    ``agent_signals`` is a dict mapping agent_name -> {"score": float, "confidence": float}
    for individual-agent postmortem analysis.

    If ``annual_vol`` and/or ``funding_rate`` are provided, they are used to apply
    market-conditions confidence modifiers to each agent's raw self-reported confidence.
    This breaks the static-confidence problem: agents that always report ~0.55
    get penalized when vol is high or funding is extreme, and boosted when calm.
    """
    # Build a minimal PolymarketMarket-compatible object for agents
    from crypto_oracle.polymarket.models import PolymarketMarket, PolymarketOutcome
    proxy = PolymarketMarket(
       market_id=market.ticker,
       question=f"Will the price of Bitcoin be above ${market.strike:,.0f}?",
       slug=market.ticker.lower(),
       outcomes=[
           PolymarketOutcome(name="Yes", price=market.mid),
           PolymarketOutcome(name="No", price=1.0 - market.mid),
       ],
    )

    # New Kalshi-specific agents need market data that PolymarketMarket doesn't have.
    _kalshi_ctx = type("KalshiCtx", (), {"strike": market.strike, "hours_to_expiry": market.hours_to_expiry, "spot_price": spot})()

    async def _safe_run(agent):
        """Run an agent, passing kalshi context to agents that accept it."""
        try:
            return await agent.run(proxy, kalshi=_kalshi_ctx)
        except TypeError:
            # Old-style agents don't accept kalshi= kwarg
            return await agent.run(proxy)

    signals = await asyncio.gather(*(_safe_run(agent) for agent in AGENTS))
    
    # Normalize: new agents return dicts, old ones return PolymarketSpecialistSignal
    normalized = []
    for i, s in enumerate(signals):
        agent = AGENTS[i]
        if isinstance(s, dict):
            normalized.append(wrap_agent_result(agent.name, s))
        else:
            normalized.append(s)
    signals = normalized
    
    score_map = {s.agent_name: s.score for s in signals}
    conf_map = {s.agent_name: s.confidence for s in signals}
    agent_signals = {s.agent_name: {"score": s.score, "confidence": s.confidence} for s in signals}

    # ── 13-Agent aggregation ─────────────────────────────────────────────
    # Pre-existing specialists
    macro = score_map.get("MacroMarket", 0.0)
    technical = score_map.get("TechnicalMarket", 0.0)
    knowledge = score_map.get("KnowledgeMarket", 0.0)
    linreg = score_map.get("LinearRegressionMarket", 0.0)
    
    # New agents
    kronos = score_map.get("KronosMarket", 0.0)
    candlestick = score_map.get("CandlestickPatterns", 0.0)
    sr = score_map.get("SupportResistance", 0.0)
    dynamic_sr = score_map.get("DynamicSR", 0.0)
    fvg = score_map.get("FairValueGap", 0.0)
    fib = score_map.get("FibonacciRetracement", 0.0)
    
    # Proven edge agents
    momentum_cont = score_map.get("MomentumContinuation", 0.0)
    vol_snap = score_map.get("VolatilitySnapback", 0.0)
    # MeanReversion is structurally 50/50 YES/NO — its weak signals add
    # directional noise that dilutes the ensemble's NO bias. Zero it out
    # unless the signal is strongly convicted (|score| >= 0.25, meaning
    # a sharp move of ~0.75%+ detected).
    _mean_rev_raw = score_map.get("MeanReversion", 0.0)
    mean_rev = _mean_rev_raw if abs(_mean_rev_raw) >= 0.25 else 0.0

    # Weighted aggregate with agent tracker adjustments
    weights = at.get_agent_weights(agent_signals=agent_signals)
    macro_w = weights.get("MacroMarket", 1.0)
    tech_w = weights.get("TechnicalMarket", 1.0)
    know_w = weights.get("KnowledgeMarket", 1.0)
    linreg_w = weights.get("LinearRegressionMarket", 1.0)

    # All agents get dynamic weights from tracker; defaults to 1.0 until 3+ resolved trades
    kronos_w = weights.get("KronosMarket", 1.0)
    candlestick_w = weights.get("CandlestickPatterns", 1.0)
    sr_w = weights.get("SupportResistance", 1.0)
    dynamic_sr_w = weights.get("DynamicSR", 1.0)
    fvg_w = weights.get("FairValueGap", 1.0)
    fib_w = weights.get("FibonacciRetracement", 1.0)
    momentum_cont_w = weights.get("MomentumContinuation", 1.0)
    mean_rev_w = weights.get("MeanReversion", 1.0)
    vol_snap_w = weights.get("VolatilitySnapback", 1.0)

    # 13-agent aggregate — weights recalibrated from 90-day simulation data:
    # Specialists (30%): Macro 12%, Technical 12%, Knowledge 3%, LinReg 3%
    # New agents (46%): Fib 20%, Candlestick 6%, S/R 5%, DynamicSR 5%, FVG 5%,
    #                   Kronos -5% INVERTED (more bullish on NO wins → flip sign)
    # Proven edge (22%): MomentumContinuation 14%, VolatilitySnapback 5%, MeanReversion 3%
    #
    # Changes from sim analysis (90d, 7,433 trades):
    #   FibonacciRetracement  7%→20%: strongest avg bearish signal (-0.25 avg score)
    #   MacroMarket           8%→12%: second strongest bearish signal (-0.24 avg)
    #   TechnicalMarket      10%→12%: best filter signal (8.8pp WR gap by sign)
    #   KronosMarket         10%→-5%: inverted — bullish on NO winners, bearish on losers
    #   KnowledgeMarket       7%→3%: always bullish (+0.16 avg), fights NO strategy
    #   LinearRegressionMkt   5%→3%: mild bullish bias, counterproductive
    #   VolatilitySnapback   19%→5%: fires only 4.4% of the time, ~0 avg signal
    aggregate = (
        # Specialists (30%)
        macro * macro_w * 0.12
        + technical * tech_w * 0.12
        + knowledge * know_w * 0.03
        + linreg * linreg_w * 0.03
        # New agents (46%) — Kronos inverted: bullish Kronos = bearish signal for NO
        - kronos * kronos_w * 0.05
        + candlestick * candlestick_w * 0.06
        + sr * sr_w * 0.05
        + dynamic_sr * dynamic_sr_w * 0.05
        + fvg * fvg_w * 0.05
        + fib * fib_w * 0.20
        # Proven edge agents (22%)
        + momentum_cont * momentum_cont_w * 0.14
        + mean_rev * mean_rev_w * 0.03
        + vol_snap * vol_snap_w * 0.05
    )

    # Clamp
    aggregate = max(-1.0, min(1.0, aggregate))

    # ── Confidence from aggregate conviction ────────────────────────────────
    # Instead of averaging agents' self-reported static confidences (which all
    # converge to the same narrow band regardless of market conditions), derive
    # confidence from how strongly the 13 agents agree on direction.
    #
    #   |aggregate| = 1.0 → all agents agree → confidence ~0.95
    #   |aggregate| = 0.5 → decent agreement → confidence ~0.63
    #   |aggregate| = 0.3 → weak signal → confidence ~0.50  (threshold cross)
    #   |aggregate| = 0.1 → noise → confidence ~0.37
    #   |aggregate| = 0.0 → split down the middle → confidence ~0.30
    #
    # Vol/funding still apply as a gentler cap so noisy markets dampen conviction
    # without crushing everything to the same floor.
    abs_agg = abs(aggregate)
    confidence = max(0.20, min(0.95, 0.30 + abs_agg * 0.65))
    if annual_vol is not None:
        if annual_vol > 1.00:
            confidence *= 0.80
        elif annual_vol > 0.80:
            confidence *= 0.90
        elif annual_vol < 0.40:
            confidence = min(0.95, confidence * 1.10)
    if funding_rate is not None and isinstance(funding_rate, (int, float)):
        if abs(float(funding_rate)) > 0.0003:
            confidence *= 0.85
    confidence = max(0.20, min(0.95, confidence))

    # Divergence flag for postmortem logging
    divergence_cut = weights.get("divergence_cut", 1.0)

    return aggregate, confidence, agent_signals, divergence_cut


def _select_with_expiry_diversification(
    markets: list[KalshiMarket],
    spot_price: float,
    top_n: int = 4,
    primary_share: int = 3,
    target_no_price: float = 0.30,
) -> list[KalshiMarket]:
    """Select markets across expiry dates for diversification.

    Takes ``primary_share`` from the nearest expiry and the rest from
    the next available expiry, scoring within each group by the same
    criteria as ``select_target_markets``.

    ``target_no_price`` sets the ideal NO contract price to scan for. A lower
    value (e.g. 0.30) favors cheap-NO directional bets with asymmetric payoff
    (risk $0.30 to win $0.70 ≈ 2.3:1) — far better for compounding a small
    account than expensive NO-favorites (risk $0.88 to win $0.12).
    """
    if not markets:
        return []

    from collections import defaultdict

    # Group by expiry date (extracted from close_time)
    buckets: dict[str, list[KalshiMarket]] = defaultdict(list)
    for m in markets:
        if m.close_time:
            expiry_date = m.close_time[:10]  # "2026-06-25" portion
        else:
            expiry_date = "unknown"
        buckets[expiry_date].append(m)

    # Sort expiry dates — nearest first
    sorted_dates = sorted(
        d for d in buckets if d != "unknown"
    )
    if "unknown" in buckets:
        sorted_dates.append("unknown")

    if not sorted_dates:
        return []

    def _score(m: KalshiMarket) -> float:
        """Prefer markets where the NO contract is priced near ``target_no_price``.

        NO price = 1 − YES mid. Targeting a cheap NO (~0.30) means scanning for
        below-spot strikes where a bearish NO bet pays ~2.3:1 — asymmetric
        upside that compounds a small account, and consistent with the bearish
        tech/regime gates. Range markets scored by distance from spot as before.
        """
        p = m.mid
        if p < 0.02 or p > 0.99:
            return 999.0
        if m.is_range:
            return abs(m.bin_center - spot_price) / 500.0
        no_price = 1.0 - p
        return abs(no_price - target_no_price)

    result: list[KalshiMarket] = []

    def _pick_from(date_key: str, count: int) -> list[KalshiMarket]:
        pool = sorted(buckets[date_key], key=_score)
        return pool[:count]

    # Pick primary_share from the nearest expiry
    primary_date = sorted_dates[0]
    primary_picked = min(primary_share, top_n)
    result.extend(_pick_from(primary_date, primary_picked))

    # Fill remaining slots from next expiry (if available)
    remaining = top_n - len(result)
    if remaining > 0 and len(sorted_dates) > 1:
        secondary_date = sorted_dates[1]
        result.extend(_pick_from(secondary_date, remaining))

    # If still below top_n, take more from primary
    remaining = top_n - len(result)
    if remaining > 0:
        extra = _pick_from(primary_date, 100)  # take all scored
        for m in extra:
            if m not in result and remaining > 0:
                result.append(m)
                remaining -= 1

    return result[:top_n]


async def run_kalshi_scan(limit: int = 8, live: bool = False) -> dict:
    """
    Scan Kalshi BTC markets, run signals, decide and optionally execute.
    Returns a summary dict suitable for Telegram delivery.
    """
    # ── Learning: resolve agent stats from any recently expired trades ─────
    try:
        resolved_count = at.resolve_from_postmortem()
        if resolved_count:
            print(f"[agent_tracker] Resolved {resolved_count} trades from postmortem")
    except Exception:
        pass  # non-fatal

    # ── Gate 1: PAUSED file (hard kill switch) ────────────────────────────────
    if _PAUSED_PATH.exists():
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "platform": "kalshi",
            "mode": "paused",
            "gate_blocked": "PAUSED file present — no new orders",
            "trades_executed": 0,
            "total_deployed_usd": 0.0,
            "results": [],
        }

    max_position = _env_float("KALSHI_MAX_POSITION_USD", _env_float("POLYMARKET_MAX_POSITION_USD", 20.0))
    min_edge = _env_float("KALSHI_MIN_EDGE", _env_float("POLYMARKET_MIN_EDGE", 0.03))
    min_confidence = _env_float("KALSHI_MIN_CONFIDENCE", _env_float("POLYMARKET_MIN_CONFIDENCE", 0.52))
    max_daily_risk = _env_float("KALSHI_MAX_DAILY_RISK_USD", 20.0)
    max_entries_per_day = _env_int("KALSHI_MAX_ENTRIES_PER_DAY", 5)
    min_balance = _env_float("KALSHI_MIN_BALANCE_USD", 0.50)
    stop_loss_pct = _env_float("KALSHI_STOP_LOSS_PCT", 0.50)
    take_profit_pct = _env_float("KALSHI_TAKE_PROFIT_PCT", 0.70)
    rebalance_enabled = os.getenv("KALSHI_REBALANCE_ENABLED", "1").strip() == "1"
    rebalance_edge_mult = _env_float("KALSHI_REBALANCE_EDGE_MULTIPLIER", 1.5)
    rebalance_min_hold = _env_float("KALSHI_REBALANCE_MIN_HOLD_MINUTES", 30.0)
    min_strike_dist = _env_float("KALSHI_MIN_STRIKE_DISTANCE_PCT", 2.0)
    # Maker mode: rest entries inside the spread instead of lifting the ask.
    # ~75% lower fees, captures the spread; unfilled orders are canceled by the
    # stale-entry cleanup each scan. Set KALSHI_MAKER_MODE=0 for taker fills.
    maker_mode = os.getenv("KALSHI_MAKER_MODE", "1").strip() == "1"
    # Max belief tilt from the agent aggregate (in probability dollars).
    # Down from 0.15: a big tilt from a ~coin-flip ensemble manufactures edges.
    agg_tilt = _env_float("KALSHI_AGG_TILT", 0.08)
    # Anchor beliefs to Deribit options-implied probabilities (the source the
    # professionals price this ladder from); falls back to GBM when the chain
    # is unavailable. Set KALSHI_DERIBIT_ANCHOR=0 to force GBM-only.
    use_deribit = os.getenv("KALSHI_DERIBIT_ANCHOR", "1").strip() == "1"
    # Momentum gate: block bearish NO entries when 6h momentum trigger exceeds
    # this (0.2 ≈ +0.4% over 6h), and bullish YES below the negative of it.
    momentum_block = _env_float("KALSHI_MOMENTUM_GATE", 0.2)
    # Concentration limits: best-signal-only per scan, bounded total exposure.
    # One scan once fired 4 same-direction entries at once — top-1 per scan
    # plus a total open cap bounds correlated wipeout risk on one BTC move.
    max_entries_per_scan = _env_int("KALSHI_MAX_ENTRIES_PER_SCAN", 1)
    max_open_positions = _env_int("KALSHI_MAX_OPEN_POSITIONS", 4)

    # ── Daily entry state (uses position manager for persistence) ─────────────
    entries_today = pm.get_entry_count_today()
    deployed_today = pm.get_today_deployed_usd()

    # ── Gate 3: live balance check (live mode only) ───────────────────────────
    kalshi_balance_cents: int | None = None
    if live:
        try:
            client = KalshiClient(key_id=os.getenv("KALSHI_API_KEY_ID", ""))
            bal = await client.get_balance()
            kalshi_balance_cents = bal.get("balance", 0)
            balance_usd = kalshi_balance_cents / 100
            if balance_usd < min_balance:
                return {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "platform": "kalshi",
                    "mode": "live",
                    "gate_blocked": f"balance ${balance_usd:.2f} below minimum ${min_balance:.2f}",
                    "trades_executed": 0,
                    "total_deployed_usd": 0.0,
                    "results": [],
                }
            # ── Scale position size with balance (5% pct, no hard cap) ─────────
            # Position = balance × position_pct — grows with the account.
            # E.g. $100 → $5, $240 → $12, $1,000 → $50. No cap.
            # Floor is $0.20 (not $1.00) so a tiny account can still place a
            # 1-contract order without the position exceeding the daily-risk
            # cap. A $1 account previously floored at $1.00 while daily risk
            # was $0.30 → every order was gate-blocked and nothing traded.
            position_size_pct = _env_float("KALSHI_POSITION_SIZE_PCT", 0.05)
            max_position = max(0.20, round(balance_usd * position_size_pct, 2))
            print(f"[Kalshi/SCAN] balance=${balance_usd:.2f} pct={position_size_pct:.0%} "
                  f"scaled_pos=${max_position:.2f}")
            # Daily risk = 30% of balance, floored at $1.00 so a sub-$3 account
            # can actually deploy (30% of $1 is $0.30, below one position).
            max_daily_risk = max(1.0, round(balance_usd * 0.30, 2))
            # Entries = balance / $5 per entry, min 5, max 48 (every 30min scan window)
            max_entries_per_day = max(5, min(48, int(balance_usd / 5)))
            print(f"[Kalshi/SCAN] daily_risk_cap=${max_daily_risk:.2f} entries_per_day={max_entries_per_day}")
        except Exception as exc:
            # Refuse to trade blind. If the balance fetch fails, the
            # balance-scaled position/daily-risk sizing above never runs and
            # the bot would fall through to the raw env defaults (historically
            # $20/position) — catastrophic oversizing on a small account.
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "platform": "kalshi",
                "mode": "live",
                "gate_blocked": f"balance check failed ({str(exc)[:120]}) — refusing to trade with unscaled defaults",
                "trades_executed": 0,
                "total_deployed_usd": 0.0,
                "results": [],
            }

    spot, annual_vol, funding_rate, spot_history, spot_history_14d = await asyncio.gather(
        fetch_spot_price(),
        fetch_realized_vol(hours=24),
        fetch_funding_rate(),
        fetch_spot_history(days=1),   # 24h of hourly data — enough for 6h momentum
        fetch_spot_history(days=14),  # 14-day window for regime detection
    )
    # Funding rate adds a small directional tilt on top of agent aggregate
    fund_tilt = funding_tilt(funding_rate)

    # ── Momentum trigger: compare current spot to 6h ago ───────────────────
    momentum_trigger = 0.0
    if spot_history and len(spot_history) > 1:
        spot_6h_ago = spot_history[0]  # oldest in the window
        if spot_6h_ago > 0:
            pct_change = (spot - spot_6h_ago) / spot_6h_ago
            # Map percentage change to [-1, 1] trigger: ±1% → ±0.5, ±3% → ±1.0
            momentum_trigger = max(-1.0, min(1.0, pct_change * 50))

    # ── Regime detection: 14-day BTC return ─────────────────────────────────
    # The volatility risk premium (core edge of cheap-NO strategy) compresses
    # in strong BTC uptrends. Above-spot NO positions become adversarial when
    # BTC is in a sustained rally — strikes get breached more frequently.
    btc_14d_return = 0.0
    if spot_history_14d and len(spot_history_14d) > 1:
        spot_14d_ago = spot_history_14d[0]
        if spot_14d_ago > 0:
            btc_14d_return = (spot - spot_14d_ago) / spot_14d_ago
    _uptrend_regime = btc_14d_return > 0.10  # BTC up >10% in 14d = strong uptrend

    # Time-of-day liquidity filter: 02:00–12:00 UTC = low-liquidity window
    # (roughly 10pm–8am US Eastern). Kalshi is US-regulated; spreads widen
    # and signal quality drops overnight. Dampen confidence by 15% during this window.
    _hour_utc = datetime.now(timezone.utc).hour
    _liquidity_conf_mult = 0.85 if 2 <= _hour_utc < 12 else 1.0

    directional, range_bins = await asyncio.gather(
        fetch_btc_markets(min_volume=100.0),
        fetch_btc_range_markets(min_volume=50.0),
    )

    # ── Expiration diversification ──────────────────────────────────────────
    # Partition directional markets by expiry date, score within each group,
    # then draw 3 from nearest expiry + 1 from next expiry. Same for range.
    # target_no_price biases selection toward cheap-NO asymmetric-payoff bets
    # (default 0.30 ≈ 2.3:1) — better geometric growth for a small account.
    target_no_price = _env_float("KALSHI_TARGET_NO_PRICE", 0.30)
    half = max(1, limit // 2)
    selected = _select_with_expiry_diversification(directional, spot, top_n=half, primary_share=3, target_no_price=target_no_price)
    selected += _select_with_expiry_diversification(range_bins, spot, top_n=half, primary_share=3, target_no_price=target_no_price)

    results = []
    trades_executed = 0
    total_deployed = 0.0
    closed_positions: list[dict] = []

    # ── Position management: stop-loss / take-profit / rebalance ─────────────
    if live:
        best_new_edge = max(
            (decision.edge for decision in (
                decide_kalshi_trade(
                    m, aggregate=0.0, confidence=0.5, spot=spot,
                    annual_vol=annual_vol, max_position_usd=max_position,
                    min_edge=0.01, min_confidence=0.0,
                    min_strike_distance_pct=min_strike_dist,
                    maker_mode=maker_mode,
                ) for m in selected[:4]
            )),
            default=0.0,
        )
        close_actions = await pm.scan_positions(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            rebalance_min_hold=rebalance_min_hold,
            rebalance_edge_mult=rebalance_edge_mult if rebalance_enabled else 999.0,
            best_new_edge=best_new_edge if rebalance_enabled else 0.0,
        )
        for action in close_actions:
            pos = action["pos"]
            try:
                client = KalshiClient(key_id=os.getenv("KALSHI_API_KEY_ID", ""))
                # Kalshi's order API is YES-denominated for BOTH buys and sells
                # (the buy path sends (1-no_ask)*100 for a NO and reconciles
                # correctly). action["close_price"] is the NO-denominated sell
                # price (for PnL bookkeeping), so convert it the same way the
                # buy path does. Sending the raw NO price here posts the close
                # at the complement level, so it never fills and stop-loss /
                # take-profit silently fail — losers ride to settlement.
                if pos["side"] == "yes":
                    price_cents = int(round(action["close_price"] * 100))
                else:
                    price_cents = int(round((1.0 - action["close_price"]) * 100))
                resp = await client.close_position(
                    ticker=pos["ticker"],
                    count=pos["count"],
                    side=pos["side"],
                    price_cents=price_cents,
                )
                closed = pm.close_position(pos["ticker"], action["reason"], action["close_price"])
                closed_positions.append({
                    "ticker": pos["ticker"],
                    "side": pos["side"],
                    "count": pos["count"],
                    "entry_price": pos["entry_price"],
                    "close_price": action["close_price"],
                    "reason": action["reason"],
                    "pnl": closed["realized_pnl"] if closed else None,
                    "order_id": resp.get("order", {}).get("order_id") or resp.get("order_id"),
                })
            except Exception as exc:
                pass  # non-fatal; skip if close fails

        # ── Cancel stale resting ENTRY orders ─────────────────────────────
        # GTC entry limit orders that didn't fill (common overnight when the
        # book is thin) would otherwise linger and can fill later on a stale
        # signal — worst of all right at the daily settlement boundary. Cancel
        # resting BUY orders each scan so entries are always re-decided fresh.
        # Resting SELL orders (stop-loss / take-profit) are left alone — the
        # heartbeat owns those.
        try:
            client = KalshiClient(key_id=os.getenv("KALSHI_API_KEY_ID", ""))
            canceled_entries = 0
            for o in await client.get_resting_orders():
                if o.get("action") == "buy" and o.get("order_id"):
                    try:
                        await client.cancel_order(o["order_id"])
                        canceled_entries += 1
                    except Exception:
                        pass  # already filled/canceled — non-fatal
            if canceled_entries:
                print(f"[Kalshi/SCAN] Canceled {canceled_entries} stale resting entry order(s)")
        except Exception:
            pass  # non-fatal; resting-order cleanup is best-effort

    # Re-load fresh entry counts after potential closes freed capital
    entries_today = pm.get_entry_count_today()
    deployed_today = pm.get_today_deployed_usd()

    for market in selected:

        aggregate, confidence, agent_signals, divergence_cut = await _run_agents(market, spot, annual_vol=annual_vol, funding_rate=funding_rate)
        # Apply funding rate tilt after agent signals — it's a market-level
        # crowding signal, not tied to any individual market's microstructure.
        tilted_aggregate = max(-1.0, min(1.0, aggregate + fund_tilt))
        # Dampen confidence during low-liquidity hours
        confidence = max(0.20, min(0.95, confidence * _liquidity_conf_mult))
        # Options-implied anchor (chain is cached — one HTTP call per scan).
        # Range markets: P(floor < BTC < cap) = P(>floor) − P(>cap).
        implied_prob = None
        if use_deribit:
            try:
                if market.is_range and market.cap_strike:
                    p_floor = await implied_prob_above(market.strike, market.hours_to_expiry, spot)
                    p_cap = await implied_prob_above(market.cap_strike, market.hours_to_expiry, spot)
                    if p_floor is not None and p_cap is not None:
                        implied_prob = max(0.0, p_floor - p_cap)
                else:
                    implied_prob = await implied_prob_above(market.strike, market.hours_to_expiry, spot)
            except Exception:
                implied_prob = None  # fall back to GBM anchor
        decision = decide_kalshi_trade(
            market,
            aggregate=tilted_aggregate,
            confidence=confidence,
            spot=spot,
            annual_vol=annual_vol,
            max_position_usd=max_position,
            min_edge=min_edge,
            min_confidence=min_confidence,
            min_strike_distance_pct=min_strike_dist,
            momentum_trigger=momentum_trigger,
            divergence_cut=divergence_cut,
            maker_mode=maker_mode,
            agg_tilt=agg_tilt,
            implied_prob=implied_prob,
            momentum_block=momentum_block,
        )

        exec_status = "hold"
        exec_error = None
        order_id = None

        # ── Post-decision signal filters (apply in both live and paper modes) ──
        # These convert actionable signals to HOLD based on simulation findings.
        _tech_score = agent_signals.get("TechnicalMarket", {}).get("score", 0.0)
        if decision.action == "BUY_YES":
            exec_status = "filtered"
            exec_error = "yes_eliminated: YES trades show negative expected value across 90-day simulation (981 trades, -$1,406)"
        elif decision.action == "BUY_NO" and _uptrend_regime and market.strike > spot:
            exec_status = "filtered"
            exec_error = (
                f"regime_filter: BTC +{btc_14d_return:.1%} in 14d; "
                f"above-spot NO adversarial in uptrend (strike=${market.strike:,.0f})"
            )
        elif decision.action == "BUY_NO" and _tech_score >= 0:
            exec_status = "filtered"
            exec_error = (
                f"tech_gate: TechnicalMarket={_tech_score:+.3f} (not bearish); "
                f"NO requires bearish technical confirmation (+8.8pp WR when negative)"
            )

        if exec_status == "filtered":
            pass  # fall through to postmortem logging with exec_status="filtered"
        elif decision.action != "HOLD" and live:
            # ── Gate 4: correlated NO cap — max 2 open NO positions per expiry ─
            _event_ticker = market.ticker.split('-')[0] + '-' + market.ticker.split('-')[1]
            if decision.action == "BUY_NO":
                _open_no_for_event = sum(
                    1 for p in pm.get_open_positions()
                    if p.get("side") == "no" and p.get("event_ticker") == _event_ticker
                )
                _max_no_per_expiry = _env_int("KALSHI_MAX_NO_PER_EXPIRY", 2)
                if _open_no_for_event >= _max_no_per_expiry:
                    exec_status = "hold"
                    exec_error = (
                        f"corr_cap: {_open_no_for_event} NO positions already open "
                        f"for expiry {_event_ticker} (max={_max_no_per_expiry})"
                    )
                    results.append({
                        "ticker": decision.ticker, "strike": decision.strike,
                        "action": "HOLD", "side": "hold",
                        "market_mid": market.mid, "exec_price": decision.price,
                        "count": 0, "position_usd": 0.0, "profit_if_win": 0.0,
                        "confidence": decision.confidence, "edge": decision.edge,
                        "agent_signals": agent_signals, "reasoning": exec_error,
                        "exec_status": exec_status, "exec_error": exec_error, "order_id": None,
                    })
                    continue
            # ── Gate 5a: one entry per scan — take only the best signal ───────
            if trades_executed >= max_entries_per_scan:
                exec_status = "hold"
                exec_error = f"scan_cap: {trades_executed} entry/entries already placed this scan (max={max_entries_per_scan})"
            # ── Gate 5b: total open positions cap ─────────────────────────────
            elif pm.get_open_count() >= max_open_positions:
                exec_status = "hold"
                exec_error = f"open_cap: {pm.get_open_count()} positions open (max={max_open_positions})"
            # ── Gate 6: daily risk cap (entries only) ─────────────────────────
            elif deployed_today + decision.position_usd > max_daily_risk:
                exec_status = "hold"
                exec_error = (
                    f"daily_risk_cap: ${deployed_today:.2f} deployed today "
                    f"+ ${decision.position_usd:.2f} would exceed ${max_daily_risk:.2f} limit"
                )
            # ── Gate 7: max entries per day (exits/closes are not counted) ────
            elif entries_today >= max_entries_per_day:
                exec_status = "hold"
                exec_error = f"max_entries_per_day: {entries_today}/{max_entries_per_day} reached"
            else:
                key_id = os.getenv("KALSHI_API_KEY_ID", "")
                if not key_id:
                    exec_status = "error"
                    exec_error = "KALSHI_API_KEY_ID not set"
                else:
                    try:
                        client = KalshiClient(key_id=key_id)
                        resp = await client.place_order(
                            ticker=decision.ticker,
                            side=decision.side,
                            count=decision.count,
                            price_cents=decision.price_cents,
                        )
                        order_id = resp.get("order", {}).get("order_id") or resp.get("order_id")
                        exec_status = "submitted"
                        trades_executed += 1
                        total_deployed += decision.position_usd
                        # Save position for tracking & stop-loss/take-profit
                        _order_status = (resp.get("order") or {}).get("status", "")
                        pos = pm.make_position(
                            ticker=decision.ticker,
                            side=decision.side,
                            count=decision.count,
                            entry_price=decision.price,
                            strike=decision.strike,
                            event_ticker=market.ticker.split('-')[0] + '-' + market.ticker.split('-')[1],
                            order_id=order_id,
                            edge=decision.edge,
                            confidence=confidence,
                            spot_at_entry=spot,
                        )
                        # Track fill state: a resting (maker) order is NOT a
                        # position yet. sync_from_kalshi clears the flag when
                        # the fill appears on the API; if it never fills, the
                        # entry is marked entry_never_filled instead of being
                        # phantom-logged as a settled (losing) position.
                        pos["order_pending"] = _order_status != "executed"
                        pm.save_new_position(pos)
                        entries_today = pm.get_entry_count_today()
                        deployed_today = pm.get_today_deployed_usd()
                    except Exception as exc:
                        exec_status = "error"
                        exec_error = str(exc)[:200]
        elif exec_status != "filtered" and decision.action != "HOLD":
            exec_status = "paper"

        # ── Postmortem: log every decision ────────────────────────────────────
        _pm_entry = build_entry(
            ticker=decision.ticker,
            strike=decision.strike,
            is_range=market.is_range,
            side=decision.side if decision.action != "HOLD" else None,
            action=decision.action,
            count=decision.count,
            entry_price=decision.price if decision.action != "HOLD" else None,
            position_usd=decision.position_usd,
            profit_if_win=decision.profit_if_win,
            order_id=order_id,
            agent_signals=agent_signals,
            aggregate=aggregate,
            confidence=confidence,
            edge=decision.edge,
            gbm_baseline=decision.gbm_baseline,
            belief_yes=decision.gbm_baseline,  # KalshiDecision.gbm_baseline holds the tilted belief_yes
            market_yes_price=market.mid,
            spot_price=spot,
            realized_vol=annual_vol,
            funding_rate=funding_rate,
            funding_tilt=fund_tilt,
            hours_to_expiry=market.hours_to_expiry,
            gate_blocked=exec_error if exec_status in ("hold", "error") else None,
            daily_deployed_usd=deployed_today,
            daily_trades=entries_today,
            daily_cap_usd=max_daily_risk,
            max_trades=max_entries_per_day,
            balance_usd=kalshi_balance_cents / 100 if kalshi_balance_cents is not None else None,
            exec_status=exec_status,
            exec_error=exec_error,
            reasoning=decision.reasoning,
        )
        log_entry(_pm_entry)

        results.append({
            "ticker": decision.ticker,
            "strike": decision.strike,
            "action": decision.action,
            "side": decision.side,
            "market_mid": market.mid,
            "exec_price": decision.price,
            "count": decision.count,
            "position_usd": decision.position_usd,
            "profit_if_win": decision.profit_if_win,
            "confidence": decision.confidence,
            "edge": decision.edge,
            "agent_signals": agent_signals,
            "reasoning": decision.reasoning,
            "exec_status": exec_status,
            "exec_error": exec_error,
            "order_id": order_id,
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": "kalshi",
        "mode": "live" if live else "paper",
        "spot_price": spot,
        "markets_scanned": len(selected),
        "positions_open": pm.get_open_count(),
        "trades_executed": trades_executed,
        "positions_closed": len(closed_positions),
        "closed_details": closed_positions,
        "total_deployed_usd": total_deployed,
        "daily_deployed_usd": deployed_today,
        "daily_entries": entries_today,
        "daily_entry_cap_usd": max_daily_risk,
        "daily_entry_cap": max_entries_per_day,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "kalshi_balance_usd": kalshi_balance_cents / 100 if kalshi_balance_cents is not None else None,
        "realized_vol_annual": round(annual_vol, 4),
        "funding_rate_8h": round(funding_rate, 6),
        "funding_tilt": round(fund_tilt, 4),
        "momentum_trigger": round(momentum_trigger, 4),
        "btc_14d_return": round(btc_14d_return, 4),
        "uptrend_regime": _uptrend_regime,
        "min_strike_distance_pct": min_strike_dist,
        "target_no_price": target_no_price,
        "results": results,
    }
