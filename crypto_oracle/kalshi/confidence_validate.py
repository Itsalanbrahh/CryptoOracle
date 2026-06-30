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

    # ── GBM baseline calibration ────────────────────────────────────────────
    gbm_entries = [e for e in resolved if e.get("belief_yes") is not None]
    if gbm_entries:
        gbm_buckets: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
        for e in gbm_entries:
            bv = e["belief_yes"]
            bucket = round(bv * 10) / 10  # nearest 10%
            label = f"{bucket:.0%}"
            gbm_buckets[label]["total"] += 1
            if e.get("resolved_itm"):
                gbm_buckets[label]["wins"] += 1

        print(f"\n  GBM BASELINE vs ACTUAL (YES outcomes)")
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
    print(f"  Win rate   : {overall_wr:.1%}  ({total_wins}/{len(resolved)})")
    print(f"  Avg conf   : {avg_conf_overall:.1%}")
    print(f"  Calibration gap: {overall_wr - avg_conf_overall:+.1%}  "
          f"({'over-confident' if overall_wr < avg_conf_overall else 'under-confident'})")
    print(f"  Total PnL  : ${total_pnl:.2f}")

    # ── Agent tracker stats ─────────────────────────────────────────────────
    print(f"\n{at.summary_text()}")
    print(f"{'='*62}\n")


main()
