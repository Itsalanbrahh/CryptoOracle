# Project Emerald — TODO

Items are roughly ordered by impact. Check off as completed.

---

## Active / In Progress

- [ ] **ruelashub.com/oracle** — publish web app publicly under ruelashub.com (domain purchased)

---

## High Impact

- [ ] **Prompt caching on Claude API calls** — both oracles send full system prompts on every run with no `cache_control`; adding Anthropic prompt caching would cut input token costs ~70% at 30-min intervals
- [ ] **Fix StockOracle strategy propagation** — `StockOracle.run()` saves strategy state but never calls `update_auto_trade_settings()`, so the stock oracle's self-tuned threshold/amount is never reflected in auto-trade decisions (unlike `CryptoOracle` which does this correctly at orchestrator.py:136–146)
- [ ] **BaseOracle shared class** — `_apply_weights`, `_parse_rec`, `_parse_strategy`, and `_synthesise` skeleton are copy-pasted between `orchestrator.py` and `stock_oracle.py`; extract ~150 lines into a shared base class

---

## Medium Impact

- [ ] **Rename `robinhood_orders` table** — still named `robinhood_orders` in `db.py` schema (leftover from earlier version); rename to `orders` or `alpaca_orders`
- [ ] **Price trigger for stocks** — `price_trigger_job` only monitors crypto watchlist; add a parallel loop over `get_stock_watchlist()` so stocks get the same rapid-response oracle trigger
- [ ] **Singleton Anthropic client** — each oracle run creates a new `AsyncAnthropic` client (new HTTP connection pool); use a module-level singleton instead
- [ ] **Actual portfolio-% position sizing** — the synthesis prompt tells Claude to recommend "15-20% of portfolio" but auto-trader ignores it and uses a fixed dollar amount; fetch account equity from Alpaca and compute real dollar value of the recommended %

---

## Lower Impact / Nice to Have

- [ ] **Stop-loss / take-profit** — positions only exit on oracle SELL signal; add background check to auto-exit if position loses >N% or gains >N% between runs
- [ ] **PDF reports on stock oracle runs** — `_send_scheduled_report()` is only called from `oracle_run_job()`, not `stock_oracle_run_job()`; stock signals never appear in scheduled PDFs
- [ ] **Fix `.env.example` missing keys** — `STOCK_INTERVAL_MINUTES`, `ORACLE_PRICE_TRIGGER_PCT`, `AUTO_TRADE_AMOUNT_USD`, and `FINNHUB_API_KEY` are used in code but not documented in `.env.example`

---

## Completed

- [x] Raise max trade amount from $500 to $20,000 (orchestrator.py, stock_oracle.py, models/db.py, alpaca/client.py, api/router.py, .env, .env.example)
- [x] Update README to reflect dual-market (crypto + stocks), all Telegram commands, full API reference, correct cost table, config reference
- [x] Fix Kronos OHLCV data source — Binance geo-blocked (HTTP 451 in US); switched to Kraken primary (721 rows, full OHLCV, no auth) + CoinGecko OHLC fallback
- [x] Fix Kronos short-circuit — was calling Claude with empty `{}` forecast on every failure (wasting API tokens); now returns NEUTRAL immediately with no Claude call
- [x] Add `extra: dict` field to AgentSignal — carries Kronos forecast path + historical OHLCV for dashboard
- [x] Add Kronos forecast chart to dashboard — 30-day historical price + 7-day forecast line + high/low band, powered by Chart.js; shows method tag (Kronos model vs GBM fallback)
