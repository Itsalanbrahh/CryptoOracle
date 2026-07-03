#!/usr/bin/env python3
"""
Does running more concurrent positions actually produce better returns?

Reads the postmortem log and buckets every resolved BUY entry two ways:

  1. CONCURRENCY AT ENTRY — how many other positions were open (entered,
     not yet expired) at the moment this trade was placed. Tests the claim
     "we earn the most running N open positions."

  2. SAME-SCAN CLUSTER SIZE — how many entries were placed in the same scan
     (within 5 minutes of each other). Tests "4 at once" vs "staggered":
     same max exposure, different entry timing.

For each bucket: trade count, win rate, total PnL, average PnL per trade,
and worst single-scan cluster loss (the tail-risk number a max-return
comparison hides).

Interpretation caution: high concurrency happens in target-rich regimes,
which are also high-return regimes — correlation isn't causation. Compare
avg PnL PER TRADE across buckets, not total PnL (more trades mechanically
produce more total in good times).

Run on the deployment box:  python3 crypto_oracle/kalshi/analyze_concurrency.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
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

_SCAN_CLUSTER_WINDOW_SEC = 300  # entries within 5 min = same scan


def _parse_ts(raw: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def main() -> None:
    from crypto_oracle.kalshi.postmortem import _LOG_PATH

    if not _LOG_PATH.exists():
        print("No postmortem log found.")
        return

    trades: list[dict] = []
    for line in _LOG_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if (
            e.get("resolved")
            and e.get("action") in ("BUY_YES", "BUY_NO")
            and e.get("resolved_pnl_usd") is not None
            and e.get("exec_status") in ("submitted", "paper")
        ):
            ts = _parse_ts(e.get("ts", ""))
            tte = e.get("hours_to_expiry")
            if ts is None or tte is None:
                continue
            trades.append({
                "ts": ts,
                "expiry": ts + timedelta(hours=float(tte)),
                "pnl": float(e["resolved_pnl_usd"]),
                "won": bool(e.get("resolved_itm")),
                "status": e.get("exec_status"),
            })

    if not trades:
        print("No resolved BUY trades in the postmortem yet.")
        return

    trades.sort(key=lambda t: t["ts"])
    print(f"Resolved trades analyzed: {len(trades)} "
          f"(live={sum(1 for t in trades if t['status'] == 'submitted')}, "
          f"paper={sum(1 for t in trades if t['status'] == 'paper')})\n")

    # ── 1. Concurrency at entry ─────────────────────────────────────────────
    def bucket_label(n: int) -> str:
        return f"{n}" if n < 5 else "5+"

    conc: dict[str, dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for i, t in enumerate(trades):
        open_count = 1  # this trade itself
        for j, o in enumerate(trades):
            if j != i and o["ts"] <= t["ts"] < o["expiry"]:
                open_count += 1
        b = conc[bucket_label(open_count)]
        b["n"] += 1
        b["pnl"] += t["pnl"]
        if t["won"]:
            b["wins"] += 1

    print("CONCURRENCY AT ENTRY (positions open when the trade was placed)")
    print(f"{'Open':>5}  {'N':>5}  {'WinRate':>8}  {'TotalPnL':>9}  {'AvgPnL/trade':>13}")
    print("-" * 50)
    for label in ["1", "2", "3", "4", "5+"]:
        if label not in conc:
            continue
        d = conc[label]
        print(f"{label:>5}  {d['n']:>5}  {d['wins']/d['n']:>8.1%}  "
              f"${d['pnl']:>8.2f}  ${d['pnl']/d['n']:>12.3f}")

    # ── 2. Same-scan cluster size ───────────────────────────────────────────
    clusters: list[list[dict]] = []
    for t in trades:
        if clusters and (t["ts"] - clusters[-1][-1]["ts"]).total_seconds() <= _SCAN_CLUSTER_WINDOW_SEC:
            clusters[-1].append(t)
        else:
            clusters.append([t])

    clus: dict[str, dict] = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0, "clusters": 0, "worst_cluster": 0.0})
    for c in clusters:
        label = bucket_label(len(c))
        d = clus[label]
        d["clusters"] += 1
        cluster_pnl = sum(t["pnl"] for t in c)
        d["worst_cluster"] = min(d["worst_cluster"], cluster_pnl)
        for t in c:
            d["n"] += 1
            d["pnl"] += t["pnl"]
            if t["won"]:
                d["wins"] += 1

    print("\nSAME-SCAN CLUSTER SIZE (entries placed within 5 min)")
    print(f"{'Size':>5}  {'Scans':>6}  {'N':>5}  {'WinRate':>8}  {'AvgPnL/trade':>13}  {'WorstCluster':>13}")
    print("-" * 62)
    for label in ["1", "2", "3", "4", "5+"]:
        if label not in clus:
            continue
        d = clus[label]
        print(f"{label:>5}  {d['clusters']:>6}  {d['n']:>5}  {d['wins']/d['n']:>8.1%}  "
              f"${d['pnl']/d['n']:>12.3f}  ${d['worst_cluster']:>12.2f}")

    print(
        "\nHow to read this:\n"
        "  - If avg PnL/trade HOLDS UP (or improves) at concurrency 3-4, higher\n"
        "    caps are justified — raise KALSHI_MAX_OPEN_POSITIONS.\n"
        "  - If avg PnL/trade at cluster size 3-4 beats size 1, same-scan\n"
        "    clustering genuinely helps — raise KALSHI_MAX_ENTRIES_PER_SCAN.\n"
        "  - If total PnL peaks at 4 open but avg/trade FALLS, the regime was\n"
        "    doing the work, not the concurrency — keep staggered entries.\n"
        "  - WorstCluster is the tail: what one bad BTC move cost when several\n"
        "    same-direction entries were placed together."
    )


main()
