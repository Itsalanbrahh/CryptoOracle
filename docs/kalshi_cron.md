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
