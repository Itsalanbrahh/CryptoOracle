# Kalshi BTC trading — cron setup

The Kalshi pipeline runs as three OS-cron jobs (separate from the in-process
APScheduler that drives the oracle/Telegram side). Install or refresh them with:

```bash
crypto_oracle/kalshi/install_kalshi_cron.sh            # install / refresh
crypto_oracle/kalshi/install_kalshi_cron.sh list       # show installed jobs
crypto_oracle/kalshi/install_kalshi_cron.sh uninstall  # remove them
```

The installer is idempotent — every job line is tagged `# kalshi-oracle`, so
re-running refreshes those lines without touching the rest of your crontab.

## Jobs

| Job | Default schedule | Script | Purpose |
| --- | --- | --- | --- |
| Entry scan | every 30 min (`*/30 * * * *`) | `kalshi_live_trade.sh` | Runs the 13-agent ensemble and places NO trades that clear the gates. |
| Position heartbeat | every 15 min (`*/15 * * * *`) | `kalshi_position_heartbeat.sh` | Stop-loss / take-profit / expiry settlement on open positions. |
| Confidence calibration | daily 18:00 (`0 18 * * *`) | `kalshi_confidence_calibration.sh` | Reconciles live trades with the API, resolves paper/filtered decisions against actual BTC settlement (`resolve_paper_trades.py`), then prints the calibration report: confidence buckets, NO-price bands, filter counterfactuals, GBM calibration. |

Override any schedule via env when installing, e.g.:

```bash
KALSHI_SCAN_CRON='*/20 * * * *' KALSHI_CAL_HOUR=19 \
  crypto_oracle/kalshi/install_kalshi_cron.sh
```

## 15-minute UP/DOWN (KXBTC15M) + Jev

Installed by `install_kalshi_cron.sh`:

| Job | Default | Script |
| --- | --- | --- |
| 15m paper cycle | every 2 min | `kalshi_15m_paper.sh` |
| 15m learn | :17 past each hour | `kalshi_15m_learn.sh` |

Labels prefer official Kalshi `result` / `expiration_value` (BRTI settlement). Each paper cycle samples Coinbase/Kraken trade websockets (~6s) and REST OFI.

## 15-minute UP/DOWN (KXBTC15M) + Jev

Hourly `*/30` scans miss most of a 15-minute window. For `KXBTC15M`, run a
dedicated paper/live scanner every 1–2 minutes:

```bash
*/2 * * * * cd /path/to/CryptoOracle && .venv/bin/python -m crypto_oracle.kalshi.kalshi_15m_scan >> ~/.hermes/logs/kalshi_15m.log 2>&1
```

Requires `TYPESAFE_API_KEY` (or `JEV_API_KEY`). Without a key the scanner still
runs using a GBM-distance heuristic and tags `model=heuristic-fallback` in the
postmortem so you can tell live Jev apart from the fallback.

Jev answers two questions per open 15m contract:

1. **Noul** `settle_up` — P(ending BRTI ≥ opening target)
2. **Choice** `action` — `buy_yes` / `buy_no` / `hold`

The fee-aware edge gate (`KALSHI_15M_MIN_EDGE`, default 0.08) still must clear
before an order is sized. Hourly-only filters (YES elimination, TechnicalMarket
NO gate, 2% strike-distance) do **not** apply to 15m.

## Timing note

Calibration's `18:00` is intended as **UTC** — just after the 17:00 UTC daily
BTC settlement. If the box is not on UTC, set `KALSHI_CAL_HOUR` to the local
hour matching 18:00 UTC (or set the box timezone to UTC). The job is
idempotent, so the exact hour only changes when the daily report lands, not
what it processes.

## What the installer does

1. Creates `~/.hermes/logs/` and `~/.hermes/scripts/`.
2. Copies the canonical `kalshi_live_trade.py` and `kalshi_position_heartbeat.py`
   into `~/.hermes/scripts/` (where the `.sh` wrappers exec them). The
   calibration wrapper runs from the repo checkout directly.
3. Marks the `.sh` wrappers executable.
4. Installs the three tagged cron jobs.

## Logs

Each job appends to its own log:

- `~/.hermes/logs/kalshi_live_trade.log`
- `~/.hermes/logs/kalshi_heartbeat.log`
- `~/.hermes/logs/kalshi_calibration.log` — the calibration report, including
  the **NO PRICE BAND** table that shows which payoff band actually made money
  (use it to tune `KALSHI_TARGET_NO_PRICE`).
