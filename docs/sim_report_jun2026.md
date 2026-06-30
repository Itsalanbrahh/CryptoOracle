# Kalshi BTC Bot — Historical Simulation Report

**Generated:** June 29, 2026  
**Agent count:** 13-agent ensemble (live bot)  
**Data window:** 90 days (Mar 31 → Jun 29, 2026 @ 1h sampling)  
**Total ticks:** 2,160  
**Total trades:** 7,433  
**Sim runtime:** ~6 hours wall time  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Methodology & Limitations](#methodology--limitations)
3. [Overall Performance](#overall-performance)
4. [NO vs YES Direction Analysis](#no-vs-yes-direction-analysis)
5. [Edge Band Analysis](#edge-band-analysis)
6. [Agent-by-Agent Breakdown](#agent-by-agent-breakdown)
7. [Agent Activity Rates](#agent-activity-rates)
8. [Agent Weights vs Actual Contribution](#agent-weights-vs-actual-contribution)
9. [Entry Price Analysis (NO Trades)](#entry-price-analysis-no-trades)
10. [Proposed Improvements](#proposed-improvements)
11. [Simulation Bugs Found](#simulation-bugs-found)
12. [Live Bot Bugs Found](#live-bot-bugs-found)
13. [Appendix: Raw Data Summary by Chunk](#appendix-raw-data-summary-by-chunk)

---

## Executive Summary

We ran the full 13-agent live trading ensemble against 90 days of hourly BTC data using a GBM-based market simulator. Key findings:

| Metric | Value |
|---|---|
| Total trades | 7,433 |
| Win rate | 11.7% |
| **Total simulated PnL** | **+$31,944.41** |
| Profit factor | **3.02** |
| Max drawdown | 671.9% (equity curve peak-based) |
| Avg PnL per trade | +$4.30 |

**The system is a profitable tail-risk NO strategy** — 87% of all trades are NO (betting BTC stays below strike). Even at 11.7% WR, the asymmetric payoff structure (cheap entry → $0.85-0.95 payout on win) makes the strategy profitable.

**However, the sim has significant limitations** (see §2). All edge numbers measure disagreement with a GBM model, not real Kalshi market prices.

---

## Methodology & Limitations

### How the simulation works

1. **Market price = GBM probability.** Not real Kalshi order books. The "market" prices trades as `GBM(spot, strike, vol)` — a pure Black-Scholes model.
2. **Edge = agent_belief − GBM_probability.** High edge means agents disagree with the mathematical model, not that they found a real market mispricing.
3. **Every 1 hour**, the sim generates ~5 strikes around current spot (within 0.5‑15%), evaluates all 13 agents, and takes trades meeting `min_edge=0.03` and `min_confidence=0.30`.
4. **Expiry = 14 hours forward.** Settlement uses actual future BTC price from historical data.
5. **Position sizing = $5 × min(1, confidence × (1+edge)).** Same as live bot.

### Critical limitations

- **GBM ≠ Kalshi.** Real Kalshi market prices reflect liquidity, sentiment, and order flow — not just math. Our edge bands measure "agent skill vs GBM," not "agent skill vs real markets."
- **Perfect market assumption.** Every strike is available and liquid. Real Kalshi has gaps and spreads.
- **In-sample analysis.** All the "filter discovery" (TechMarket negative, etc.) uses the same data. Needs out-of-sample validation.
- **Fixed expiry (14h).** Live bot trades diverse expiries from daily binaries to hourly.
- **Agent_perf tracking is broken** in sim_harness.py (see §11).

---

## Overall Performance

### Summary by period

| Period | Trades | WR | PnL | Notes |
|---|---|---|---|---|
| Mar 31 → Apr 22 | 1,999 | 7.9% | -$2,406 | Losing period |
| Apr 23 → May 15 | 1,583 | 7.1% | -$2,300 | Losing period |
| May 15 → Jun 7 | 2,353 | **17.0%** | **+$35,719** | Winning period |
| Jun 7 → Jun 29 | 1,498 | 13.5% | +$931 | Slightly positive |
| **Total** | **7,433** | **11.7%** | **+$31,944** | |

**High variance.** One chunk (May–Jun) accounts for almost all profit. The other three are roughly breakeven or negative. This suggests strong regime dependence — the strategy works in certain market conditions and should probably have a regime-based off switch.

### Yearly breakdown (all data is 2026)

Only 2026 was simulated (90-day window). Four chunks within 2026 show massive variance.

---

## NO vs YES Direction Analysis

| Direction | Trades | % of total | WR | PnL |
|---|---|---|---|---|
| **BUY_NO** | **6,452** | **86.8%** | **12.2%** | **+$33,350.34** |
| BUY_YES | 981 | 13.2% | 8.5% | -$1,405.93 |

**The ensemble is structurally bearish.** When it has conviction, it overwhelmingly favors betting BTC won't exceed a strike. This makes intuitive sense: it's easier for agents to identify overpriced YES markets than underpriced ones.

**YES trades lose money on every metric** — lower count, lower WR, negative PnL. They may be worth eliminating entirely, or at least requiring a much higher edge threshold.

---

## Edge Band Analysis

The edge distribution is **bimodal**, which is an artifact of using GBM as the market price:

```
Band        Trades    WR
[0.02-0.04)     1   100%      ← tiny sample
[0.08-0.10)    17   41.2%     ← best WR band
[0.12-0.14)    52   32.7%     
[0.18-0.20)   170   21.2%     ← declining WR
[0.30-0.32)   265   18.1%     
[0.38-0.40)   368   10.9%     ← bimodal peak (bad trades)
[0.48-0.50)   786    3.7%     ← worst WR
[0.50-0.52)   783    3.7%     
[0.58-0.60)    30   33.3%     ← tiny sample, high variance
```

**Why the bimodal shape:** When GBM says >80% probability in one direction, GBM and agents strongly disagree → edge is high (0.38-0.52). But most of the time GBM is right, so these trades lose frequently (3-10% WR). The few that win pay enormous profits because entry prices are tiny ($0.01-0.05). This is a **tail-risk payoff structure**, not a signal that edge is broken.

**The sweet spot:** Edge 0.08-0.20 (where GBM and agents moderately disagree) produces 21-41% WR. But these trades are fewer and generate less total PnL than the high-edge cluster.

---

## Agent-by-Agent Breakdown

### Overall directional accuracy

| Agent | Correct | Total | Accuracy |
|---|---|---|---|
| KnowledgeMarket | 6,527 | 7,433 | **87.8%** |
| LinearRegressionMarket | 5,279 | 7,433 | 71.0% |
| TechnicalMarket | 4,133 | 7,433 | 55.6% |
| KronosMarket | 3,547 | 7,433 | 47.7% |
| DynamicSR | 3,135 | 7,433 | 42.2% |
| SupportResistance | 2,903 | 7,433 | 39.1% |
| MomentumContinuation | 2,823 | 7,433 | 38.0% |
| MeanReversion | 2,260 | 7,433 | 30.4% |
| FibonacciRetracement | 1,843 | 7,433 | 24.8% |
| FairValueGap | 1,763 | 7,433 | 23.7% |
| CandlestickPatterns | 1,719 | 7,433 | 23.1% |
| MacroMarket | 1,687 | 7,433 | 22.7% |
| VolatilitySnapback | 1,633 | 7,433 | 22.0% |

**Note on this metric:** KnowledgeMarket at 87.8% looks incredible, but this is because it has a **consistent bullish bias** (avg score +0.1586). Since 87% of trades are NO and most NO trades *lose* (BTC goes up), KnowledgeMarket is positive when the trade loses — its "accuracy" is mostly coincidental.

The better metric is the **score differential on NO trades** (next table).

### NO trade signal quality (score differential)

For NO trades (betting BTC stays below strike), a good agent should be **more bearish (negative score) on winning trades** than on losing trades.

| Agent | Win avg | Loss avg | Diff | Quality |
|---|---|---|---|---|
| **TechnicalMarket** | **-0.2098** | **+0.1313** | **-0.3411** | 🏆 **Best** |
| LinearRegressionMarket | +0.0632 | +0.1585 | -0.0953 | Good (less bullish on wins) |
| MomentumContinuation | -0.1695 | -0.1047 | -0.0647 | Good (more bearish on wins) |
| KnowledgeMarket | +0.1463 | +0.2155 | -0.0691 | Good (less bullish on wins) |
| DynamicSR | -0.0827 | -0.0483 | -0.0344 | Modest |
| MacroMarket | -0.2417 | -0.2268 | -0.0149 | Modest |
| SupportResistance | -0.0042 | +0.0029 | -0.0071 | Weak |
| FairValueGap | +0.0062 | +0.0094 | -0.0032 | Negligible |
| VolatilitySnapback | -0.0101 | -0.0089 | -0.0012 | Negligible |
| CandlestickPatterns | -0.1897 | -0.1930 | +0.0033 | Negligible |
| MeanReversion | -0.0039 | -0.0212 | **+0.0173** | ❌ Inverted |
| FibonacciRetracement | -0.2675 | -0.3536 | **+0.0861** | ❌ Inverted |
| **KronosMarket** | **+0.0919** | **-0.0188** | **+0.1107** | ❌ **Worst** |

**Key insight: TechnicalMarket is the single best agent for NO trades.** When it's bearish, the trade wins. When it's bullish, the trade loses. The separation (-0.3411) is 3x larger than any other agent.

**KronosMarket and FibonacciRetracement are inverted indicators** — they're more bullish on winners than losers. Their signals should either be inverted or downweighted for NO decisions.

### Sign-based WR filter (NO trades only)

When we filter NO trades by agent score sign:

| Agent | Negative WR | Positive WR | Diff | Impact |
|---|---|---|---|---|
| **TechnicalMarket** | **16.4%** (3,390) | **7.6%** (3,062) | **+8.8pp** | **Strongest filter** |
| KnowledgeMarket | 27.9% (68) | 12.0% (6,384) | +15.9pp | Tiny sample size |
| MacroMarket | 12.2% (6,451) | 0.0% (1) | +12.2pp | Almost never positive |
| LinearRegressionMarket | 14.6% (1,680) | 11.4% (4,772) | +3.3pp | Modest |
| SupportResistance | 13.2% (3,930) | 10.6% (2,522) | +2.6pp | Modest |
| DynamicSR | 13.3% (3,756) | 10.7% (2,696) | +2.6pp | Modest |
| MomentumContinuation | 12.9% (4,108) | 11.0% (2,344) | +1.9pp | Weak |
| MeanReversion | 13.1% (1,327) | 12.0% (5,125) | +1.1pp | Marginal |
| **KronosMarket** | **9.5%** (2,928) | **14.4%** (3,524) | **-4.9pp** | ❌ **Inverted** |
| FibonacciRetracement | 11.2% (5,231) | 16.4% (1,221) | -5.1pp | ❌ Inverted |

---

## Agent Activity Rates

How often does each agent produce a non-zero score?

| Agent | Active | % Active | Avg score |
|---|---|---|---|
| DynamicSR | 7,433 | **100%** | -0.0236 |
| FibonacciRetracement | 7,433 | **100%** | -0.2503 |
| KnowledgeMarket | 7,433 | **100%** | +0.1586 |
| KronosMarket | 7,433 | **100%** | +0.0656 |
| LinearRegressionMarket | 7,432 | 100% | +0.0718 |
| MacroMarket | 7,433 | **100%** | -0.2391 |
| TechnicalMarket | 7,433 | **100%** | -0.0031 |
| MomentumContinuation | 7,412 | 99.7% | -0.0631 |
| SupportResistance | 7,358 | 99.0% | +0.0190 |
| CandlestickPatterns | 1,627 | 21.9% | -0.1703 |
| **MeanReversion** | **2,756** | **37.1%** | **-0.0029** |
| VolatilitySnapback | 326 | 4.4% | -0.0035 |
| FairValueGap | 117 | 1.6% | +0.0085 |

**MeanReversion fires only 37% of the time** — and when it does, its average score is -0.0029 (essentially zero). Despite carrying **22% of the ensemble weight**, it's the laziest signal in the group.

---

## Agent Weights vs Actual Contribution

Current ensemble weights and their effective contribution:

| Agent | Weight | Avg score | Signal contribution | Effectiveness rank |
|---|---|---|---|---|
| MeanReversion | **0.22** | -0.0029 | **-0.0006** — near zero | 🔻 Worst |
| MomentumContinuation | 0.14 | -0.0631 | -0.0088 | Mid |
| FibonacciRetracement | 0.12 | **-0.2503** | **-0.0300** | 🏆 **Best raw signal** |
| VolatilitySnapback | 0.14 | -0.0035 | -0.0005 | Near zero |
| KronosMarket | 0.10 | +0.0656 | +0.0066 | Counterproductive 🔴 |
| TechnicalMarket | 0.07 | -0.0031 | -0.0002 | Not a raw signal agent, but best filter |
| KnowledgeMarket | 0.07 | +0.1586 | +0.0111 | Counterproductive 🔴 |
| MacroMarket | 0.06 | **-0.2391** | **-0.0143** | 🥈 **Second best raw signal** |
| LinearRegressionMarket | 0.05 | +0.0718 | +0.0036 | Counterproductive |
| DynamicSR | 0.04 | -0.0236 | -0.0009 | Neutral |
| SupportResistance | 0.04 | +0.0190 | +0.0008 | Neutral |
| CandlestickPatterns | 0.03 | -0.1703 | -0.0051 | Rare but strong when active |
| FairValueGap | 0.03 | +0.0085 | +0.0003 | Near zero |

**The problem:** MeanReversion (22%) + VolatilitySnapback (14%) = **36% of the ensemble** contributes effectively nothing. Meanwhile, FibonacciRetracement (12%, best raw signal) and MacroMarket (6%, second best) are underweighted.

---

## Entry Price Analysis (NO Trades)

The cheapest NO trades drive the strategy:

| Entry price | Trades | WR | PnL | Implication |
|---|---|---|---|---|
| < $0.03 | 1,842 | 7.5% | **+$31,909** | 🏆 Profit engine |
| ≥ $0.03 | 4,610 | 15.8% | +$1,441 | Secondary |
| ≥ $0.05 | 4,039 | 17.1% | +$379 | Breakeven+ |
| ≥ $0.08 | 3,365 | 18.9% | -$336 | Losing |
| ≥ $0.10 | 3,025 | 20.0% | -$468 | Losing |

**Key insight:** The absolute cheapest NO entries ($0.01-0.03) have the lowest WR (7.5%) but generate 96% of total PnL. This is because each win on a $0.02 NO contract pays $0.98 — a 49:1 risk/reward. Higher-entry NO trades raise WR but have worse risk/reward because you're risking more capital per trade.

**Taking only NO trades ≥ $0.05 kills the strategy** — +$379 vs +$33,350. The low-WR, cheap-entry trades ARE the strategy.

---

## Proposed Improvements

### Improvement 1: Filter NO trades when TechnicalMarket ≥ 0

- Removes ~3,062 bad NO trades from the dataset
- Retains the profitable trades that pass the TechnicalMarket bearish check
- Projected impact: +$4,453 PnL improvement (from removing TechMarket-positive bleed)
- Implementation: 3-line change in the live loop

```
Approach: Before opening a NO position, check TechnicalMarket's score.
If score >= 0 (bullish), skip the trade entirely.
Only trade NO when TechnicalMarket is negative (bearish).
```

### Improvement 2: Eliminate BUY_YES trades

- YES trades: 981 trades, 8.5% WR, -$1,406 PnL
- Simple fix: don't open YES positions
- Saves $1,406 in losses

### Improvement 3: Rebalance Agent Weights

**Current vs proposed:**

| Agent | Current | Proposed | Rationale |
|---|---|---|---|
| MeanReversion | **0.22** | **0.05** | 63% flatlined, near-zero signal |
| FibonacciRetracement | 0.12 | **0.20** | Strongest bearish signal (-0.25 avg) |
| MacroMarket | 0.06 | **0.12** | Second strongest bearish signal |
| MomentumContinuation | 0.14 | 0.14 | Stable, modest signal |
| TechnicalMarket | 0.07 | **0.12** | Best filter signal, 100% active |
| KronosMarket | 0.10 | **0.05** | Inverted for NO, bullish avg |
| KnowledgeMarket | 0.07 | **0.03** | Always bullish, counterproductive |
| VolatilitySnapback | 0.14 | 0.05 | 96% flatlined |
| LinearRegressionMarket | 0.05 | 0.03 | Bullish avg, counterproductive |
| CandlestickPatterns | 0.03 | 0.06 | Rare but accurate when active |
| DynamicSR | 0.04 | 0.05 | Neutral, 100% active |
| SupportResistance | 0.04 | 0.05 | Neutral, 99% active |
| FairValueGap | 0.03 | 0.05 | Negligible signal |

### Improvement 4: Regime-based trading

- Mar–Apr chunks lost money; May–Jun chunk won big
- Add a trailing PnL gate: pause new entries after N consecutive losing trades or X% drawdown
- Needs further analysis to define the regime trigger

---

## Simulation Bugs Found

### 1. Aggregate not stored in trade dict

**sim_harness.py lines 336-354:** The trade dictionary saves `edge`, `confidence`, `agent_signals`, and `agent_weights` but **not the raw `aggregate` score**. This makes post-hoc analysis harder (must reverse-engineer aggregate from edge + market_yes).

**Fix:** Add `"aggregate": round(aggregate, 4)` to the trade dict.

### 2. Agent performance tracking records wrong metric

**sim_harness.py lines 422-450:** The agent performance tracking records a "win" for every agent when the trade wins, regardless of whether that agent's score was on the correct side of the trade. This means all 13 agents show identical WR (11.7%).

**Fix:** Track per-agent directional accuracy, not combined outcome.

### 3. Direction accuracy calculation is incomplete

**sim_harness.py lines 440-449:** The `dir_accuracy` field only counts correct predictions on WINNING trades, ignoring losses entirely. This overstates accuracy.

**Fix:** Count correct/incorrect on ALL trades.

---

## Live Bot Bugs Found

### Germany vs Paraguay local-expiry garbage collection

**File:** `kalshi_position_heartbeat.py`  
**Type:** Real bug, actively losing money

The heartbeat script has a "local expiry" check that marks positions as settled when the ticker date matches today AND `utc_now.hour >= 17`:

```python
m = re.search(f'({month_abbrs})(\d{2})', event)
if m and m.group() == today_mmdoy and utc_now.hour >= 17:
    p["closed"] = True
    p["close_reason"] = "settled (expired locally)"
```

This logic assumes all markets expire at 5 PM UTC. It's correct for daily BTC binaries but **prematurely garbage-collects sports/advance markets** that settle after evening matches (World Cup, etc.).

**Impact:** The Germany vs Paraguay advance market (KXWCADVANCE-26JUN29GERPAR-GER) was entered at 9:30 PM on June 29 and immediately marked as "expired locally" on the next heartbeat run (after 5 PM UTC), because the ticker contained "JUN29" which matched the date check. The position was logged as a full loss (181 contracts × $0.6507 = $117.78) without waiting for the match outcome.

**Fix:** Exclude non-BTC markets from the local-expiry check, or use actual market settlement time.

### 15-minute binary edge not recorded

The postmortem logger does not capture agent signals for 15-minute binary markets. Two trades totaling $19.27 in losses have no edge data (edge=0.0) — we can't audit whether they had sufficient signal.

---

## Appendix: Raw Data Summary by Chunk

### Chunk 0: Mar 31 → Apr 22

| Metric | Value |
|---|---|
| Trades | 1,999 |
| Win rate | 7.9% |
| PnL | -$2,405.96 |
| Profit factor | 0.45 |
| Gross profit | $1,975.99 |
| Gross loss | $4,381.95 |
| Max drawdown | 671.9% |
| Sim time | 90.4 min |

### Chunk 1: Apr 23 → May 15

| Metric | Value |
|---|---|
| Trades | 1,583 |
| Win rate | 7.1% |
| PnL | -$2,299.79 |
| Sim time | ~89 min |

### Chunk 2: May 15 → Jun 7

| Metric | Value |
|---|---|
| Trades | 2,353 |
| Win rate | **17.0%** |
| PnL | **+$35,719.30** |
| Sim time | ~88 min |

### Chunk 3: Jun 7 → Jun 29

| Metric | Value |
|---|---|
| Trades | 1,498 |
| Win rate | 13.5% |
| PnL | +$930.86 |
| Sim time | ~92 min |

### Combined

| Metric | Value |
|---|---|
| Total trades | 7,433 |
| Win rate | 11.7% |
| Total PnL | +$31,944.41 |
| Profit factor | 3.02 |
| Total sim time | ~6 hours |

---

*Report generated by Hermes Agent from sim_results_seq_0..3.json and sim_trades_seq_0..3.json*  
*Raw data: ~/crypto_oracle/crypto_oracle/kalshi/data/*
