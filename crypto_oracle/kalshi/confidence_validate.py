#!/usr/bin/env python3
"""
Confidence calibration validator for the Kalshi BTC trading bot.

Reads resolved postmortem entries and checks whether stated confidence
matches actual win rates in each bucket. If the model says 65% confidence
but only wins 45% of the time, it is over-confident in that range.

Also reports per-side (YES/NO), per-edge-bucket, and agent tracker stats.

Run after reconcile_positions.py so resolved=True entries are populated.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
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


def _edge_bucket(edge: float) -> str:
    if edge < 0.02:
        return "0–2%"
    if edge < 0.05:
        return "2–5%"
    if edge < 0.10:
        return "5–10%"
    if edge < 0.20:
        return "10–20%"
    return ">20%"


_EDGE_ORDER = ["0–2%", "2–5%", "5–10%", "10–20%", ">20%"]


def _no_price_band(price: float) -> str:
    """Bucket a NO contract entry price into a payoff band."""
    if price < 0.15:
        return "0.00–0.15"
    if price < 0.25:
        return "0.15–0.25"
    if price < 0.35:
        return "0.25–0.35"
    if price < 0.50:
        return "0.35–0.50"
    if price < 0.70:
        return "0.50–0.70"
    return "0.70–1.00"


_NO_BAND_ORDER = ["0.00–0.15", "0.15–0.25", "0.25–0.35", "0.35–0.50", "0.50–0.70", "0.70–1.00"]


def main() -> None:
    from crypto_oracle.kalshi.postmortem import _LOG_PATH
    from crypto_oracle.kalshi import agent_tracker as at

    if not _LOG_PATH.exists():
        print("No postmortem log found. Run reconcile_positions.py first.")
        return

    all_entries: list[dict] = []
    for line in _LOG_PATH.read_text().splitlines():
        if line.strip():
            try:
                all_entries.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                pass

    # Only executable (submitted or paper) BUY trades that are resolved
    resolved = [
        e for e in all_entries
        if e.get("resolved")
        and e.get("action") in ("BUY_YES", "BUY_NO")
        and e.get("exec_status") in ("submitted", "paper")
    ]

    total_entries = len(all_entries)
    total_buys = sum(1 for e in all_entries if e.get("action") in ("BUY_YES", "BUY_NO"))

    print(f"\n{'='*62}")
    print("  KALSHI CONFIDENCE CALIBRATION REPORT")
    print(f"{'='*62}")
    print(f"  Postmortem entries total : {total_entries}")
    print(f"  BUY entries              : {total_buys}")
    print(f"  Resolved (have outcome)  : {len(resolved)}")

    if not resolved:
        print("\n  No resolved trades yet. Run reconcile_positions.py first.")
        print(f"{'='*62}\n")
        return

    # ── Confidence calibration ──────────────────────────────────────────────
    # Round confidence to 5% buckets
    conf_buckets: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0.0})
    for e in resolved:
        conf = e.get("confidence", 0.0) or 0.0
        bucket = round(conf * 20) / 20  # nearest 5%
        label = f"{bucket:.0%}"
        conf_buckets[label]["total"] += 1
        conf_buckets[label]["pnl"] += e.get("resolved_pnl_usd") or 0.0
        if e.get("resolved_itm"):
            conf_buckets[label]["wins"] += 1

    print(f"\n  CONFIDENCE CALIBRATION (n={len(resolved)})")
    print(f"  {'Conf':>6}  {'N':>5}  {'WinRate':>8}  {'Expected':>9}  {'Gap':>7}  {'PnL':>8}")
    print(f"  {'-'*56}")
    for label in sorted(conf_buckets):
        d = conf_buckets[label]
        n = d["total"]
        wr = d["wins"] / n
        expected = float(label.rstrip("%")) / 100
        gap = wr - expected
        pnl = d["pnl"]
        flag = " ⚠" if abs(gap) > 0.10 else ""
        print(
            f"  {label:>6}  {n:>5}  {wr:>8.1%}  {expected:>9.1%}  {gap:>+7.1%}  ${pnl:>7.2f}{flag}"
        )

    # ── Per-side breakdown ──────────────────────────────────────────────────
    print(f"\n  BY SIDE")
    print(f"  {'Side':>5}  {'N':>5}  {'WinRate':>8}  {'AvgConf':>8}  {'AvgEdge':>8}  {'PnL':>8}")
    print(f"  {'-'*56}")
    for side in ("yes", "no"):
        side_trades = [e for e in resolved if e.get("side") == side]
        if not side_trades:
            continue
        n = len(side_trades)
        wins = sum(1 for e in side_trades if e.get("resolved_itm"))
        wr = wins / n
        avg_conf = sum(e.get("confidence") or 0 for e in side_trades) / n
        avg_edge = sum(e.get("edge") or 0 for e in side_trades) / n
        total_pnl = sum(e.get("resolved_pnl_usd") or 0 for e in side_trades)
        print(f"  {side.upper():>5}  {n:>5}  {wr:>8.1%}  {avg_conf:>8.1%}  {avg_edge:>8.1%}  ${total_pnl:>7.2f}")

    # ── Edge calibration ────────────────────────────────────────────────────
    edge_buckets: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0.0})
    for e in resolved:
        label = _edge_bucket(e.get("edge") or 0.0)
        edge_buckets[label]["total"] += 1
        edge_buckets[label]["pnl"] += e.get("resolved_pnl_usd") or 0.0
        if e.get("resolved_itm"):
            edge_buckets[label]["wins"] += 1

    print(f"\n  EDGE CALIBRATION")
    print(f"  {'Edge':>6}  {'N':>5}  {'WinRate':>8}  {'PnL':>8}")
    print(f"  {'-'*36}")
    for label in _EDGE_ORDER:
        if label not in edge_buckets:
            continue
        d = edge_buckets[label]
        n = d["total"]
        wr = d["wins"] / n
        pnl = d["pnl"]
        print(f"  {label:>6}  {n:>5}  {wr:>8.1%}  ${pnl:>7.2f}")

    # ── NO price band: which payoff band actually made money ─────────────────
    # This is the key report for a small account: a NO@0.30 (2.3:1) needs only
    # ~31% win rate to break even, while a NO@0.85 (0.18:1) needs ~85%. Ranking
    # bands by realized PnL shows where the real edge (and survivable variance) is.
    no_resolved = [
        e for e in resolved
        if e.get("side") == "no" and e.get("entry_price") not in (None, 0)
    ]
    if no_resolved:
        band_buckets: dict[str, dict] = defaultdict(
            lambda: {"wins": 0, "total": 0, "pnl": 0.0, "payoff_sum": 0.0}
        )
        for e in no_resolved:
            price = e["entry_price"]
            label = _no_price_band(price)
            b = band_buckets[label]
            b["total"] += 1
            b["pnl"] += e.get("resolved_pnl_usd") or 0.0
            b["payoff_sum"] += e.get("payoff_ratio") or ((1.0 - price) / price if price else 0.0)
            if e.get("resolved_itm"):
                b["wins"] += 1

        print(f"\n  NO PRICE BAND (which payoff band made money)")
        print(f"  {'Band':>11}  {'N':>4}  {'WinRate':>8}  {'Payoff':>7}  {'Breakeven':>9}  {'PnL':>8}")
        print(f"  {'-'*60}")
        for label in _NO_BAND_ORDER:
            if label not in band_buckets:
                continue
            d = band_buckets[label]
            n = d["total"]
            wr = d["wins"] / n
            avg_payoff = d["payoff_sum"] / n
            # Breakeven win rate for a payoff of b:1 is 1/(1+b)
            breakeven = 1.0 / (1.0 + avg_payoff) if avg_payoff > 0 else 0.0
            edge_flag = " ✓" if wr > breakeven else " ✗"
            print(
                f"  {label:>11}  {n:>4}  {wr:>8.1%}  {avg_payoff:>6.2f}:1  "
                f"{breakeven:>9.1%}  ${d['pnl']:>7.2f}{edge_flag}"
            )

    # ── Filtered counterfactuals: do the signal filters actually help? ──────
    # The paper resolver settles trades that filters blocked. If a filter's
    # blocked trades would have LOST money, the filter is earning its keep;
    # if they would have won, the filter is discarding real edge.
    filtered_resolved = [
        e for e in all_entries
        if e.get("resolved")
        and e.get("exec_status") == "filtered"
        and e.get("action") in ("BUY_YES", "BUY_NO")
    ]
    if filtered_resolved:
        filt_buckets: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0.0})
        for e in filtered_resolved:
            # Filter name is the token before ":" in exec_error
            name = (e.get("exec_error") or "unknown").split(":")[0]
            b = filt_buckets[name]
            b["total"] += 1
            b["pnl"] += e.get("resolved_pnl_usd") or 0.0
            if e.get("resolved_itm"):
                b["wins"] += 1

        print(f"\n  FILTER COUNTERFACTUALS (what blocked trades would have done)")
        print(f"  {'Filter':>16}  {'N':>4}  {'WinRate':>8}  {'PnL-if-traded':>13}  verdict")
        print(f"  {'-'*62}")
        for name, d in sorted(filt_buckets.items()):
            n = d["total"]
            wr = d["wins"] / n
            verdict = "filter helps ✓" if d["pnl"] < 0 else "filter costs edge ✗"
            print(f"  {name:>16}  {n:>4}  {wr:>8.1%}  ${d['pnl']:>12.2f}  {verdict}")

    # ── GBM baseline calibration ────────────────────────────────────────────
    # Uses resolved_yes_outcome (did BTC finish above the strike?), recorded by
    # the paper resolver on EVERY directional entry including HOLDs — so this
    # section has a large sample independent of trade decisions. Falls back to
    # deriving the YES outcome from side+resolved_itm for API-resolved trades.
    gbm_entries = []
    for e in all_entries:
        if e.get("belief_yes") is None:
            continue
        yes_outcome = e.get("resolved_yes_outcome")
        if yes_outcome is None and e.get("resolved") and e.get("side") in ("yes", "no"):
            itm = e.get("resolved_itm")
            if itm is not None:
                yes_outcome = itm if e["side"] == "yes" else (not itm)
        if yes_outcome is not None:
            gbm_entries.append((e["belief_yes"], bool(yes_outcome)))
    if gbm_entries:
        gbm_buckets: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
        for bv, yes_won in gbm_entries:
            bucket = round(bv * 10) / 10  # nearest 10%
            label = f"{bucket:.0%}"
            gbm_buckets[label]["total"] += 1
            if yes_won:
                gbm_buckets[label]["wins"] += 1

        print(f"\n  GBM BASELINE vs ACTUAL (P(BTC>strike): belief vs outcome, n={len(gbm_entries)})")
        print(f"  {'GBM':>6}  {'N':>5}  {'ActualWR':>9}")
        print(f"  {'-'*26}")
        for label in sorted(gbm_buckets):
            d = gbm_buckets[label]
            n = d["total"]
            wr = d["wins"] / n
            flag = " ⚠" if abs(wr - float(label.rstrip("%")) / 100) > 0.15 else ""
            print(f"  {label:>6}  {n:>5}  {wr:>9.1%}{flag}")

    # ── Overall summary ─────────────────────────────────────────────────────
    total_wins = sum(1 for e in resolved if e.get("resolved_itm"))
    total_pnl = sum(e.get("resolved_pnl_usd") or 0 for e in resolved)
    overall_wr = total_wins / len(resolved)
    avg_conf_overall = sum(e.get("confidence") or 0 for e in resolved) / len(resolved)

    print(f"\n  OVERALL")
    for status in ("submitted", "paper"):
        st = [e for e in resolved if e.get("exec_status") == status]
        if st:
            st_wins = sum(1 for e in st if e.get("resolved_itm"))
            st_pnl = sum(e.get("resolved_pnl_usd") or 0 for e in st)
            label = "live " if status == "submitted" else "paper"
            print(f"  {label}      : {st_wins}/{len(st)} won  (${st_pnl:+.2f})")
    print(f"  Win rate   : {overall_wr:.1%}  ({total_wins}/{len(resolved)})")
    print(f"  Avg conf   : {avg_conf_overall:.1%}")
    print(f"  Calibration gap: {overall_wr - avg_conf_overall:+.1%}  "
          f"({'over-confident' if overall_wr < avg_conf_overall else 'under-confident'})")
    print(f"  Total PnL  : ${total_pnl:.2f}")

    # ── Agent tracker stats ─────────────────────────────────────────────────
    print(f"\n{at.summary_text()}")
    print(f"{'='*62}\n")


main()
