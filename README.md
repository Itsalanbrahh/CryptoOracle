# CryptoOracle (Project Emerald)

Autonomous multi-agent trading intelligence system for crypto and US equities.
Self-improving: agent weights, confidence thresholds, and trade sizes evolve automatically based on live P&L performance.

---

## What It Does

| Layer | Crypto | Stocks |
|-------|--------|--------|
| Data agents | 7 (Kronos, Macro, Micro, Volume, OnChain, Sentiment, Technical) | 3 (Technical, Macro, Sentiment) |
| Oracle | Claude synthesis → BUY / SELL / HOLD | Claude synthesis → LONG / SHORT / HOLD |
| Auto-trade | Market buy / close position | Long entry / short entry / position flip |
| Scheduling | Every N min (24/7) + price-movement trigger | Every N min (market hours only) |
| Alerts | Telegram push + morning brief + PDF report | Telegram push on signal |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CryptoOracle                                   │
│                                                                         │
│  ┌──── Crypto: 7 Specialist Agents ────┐  ┌── Stocks: 3 Agents ──────┐ │
│  │  Kronos    Monte Carlo / quant      │  │  Technical  OHLCV/yfinance│ │
│  │  Macro     Fed / DXY / fear&greed   │  │  Macro      Fed / macro   │ │
│  │  Micro     Order book / bid-ask     │  │  Sentiment  Finnhub/news  │ │
│  │  Volume    OBV / volume profile     │  └──────────────┬────────────┘ │
│  │  OnChain   Hash rate / mempool      │                 │              │
│  │  Sentiment Headlines / social       │        ┌────────▼───────┐      │
│  │  Technical EMA / RSI / MACD / BB   │        │  StockOracle   │      │
│  └───────────────┬─────────────────────┘        └────────┬───────┘      │
│                  │ AgentSignal[]                          │              │
│        ┌─────────▼──────────┐                   ┌────────▼───────┐      │
│        │   CryptoOracle     │                   │ StockAutoTrader│      │
│        │   (Orchestrator)   │                   │ long/short/flip│      │
│        └─────────┬──────────┘                   └────────────────┘      │
│                  │ MasterRecommendation                                  │
│      ┌───────────┼───────────────────────┐                              │
│      ▼           ▼                       ▼                              │
│  ┌────────┐  ┌────────┐          ┌──────────────┐                       │
│  │ SQLite │  │FastAPI │          │  APScheduler │                       │
│  │  + WS  │  │  /docs │          │  5 jobs      │                       │
│  └────────┘  └────┬───┘          └──────┬───────┘                       │
│                   │                     │                               │
│             ┌─────▼──────┐     ┌────────▼────────┐                      │
│             │  Dashboard │     │  Telegram Bot   │                      │
│             │ (React SPA)│     │  + PDF reports  │                      │
│             └────────────┘     └────────┬────────┘                      │
│                                         │                               │
│                                ┌────────▼────────┐                      │
│                                │  Alpaca API     │                      │
│                                │  paper / live   │                      │
│                                └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘

Data sources:
  Crypto  — CoinGecko, Binance, Blockchain.com, Mempool.space,
             alternative.me (Fear & Greed), NewsAPI / RSS
  Stocks  — yfinance (OHLCV), Finnhub (company news), NewsAPI / RSS

Scheduler jobs (APScheduler, SQLite-backed, survives restarts):
  • oracle_run       — full 7-agent crypto run for all watchlist symbols
  • stock_oracle_run — 3-agent stock run (market hours only)
  • heartbeat        — Telegram status ping
  • price_trigger    — checks every 5 min, fires oracle if crypto moved ≥ ORACLE_PRICE_TRIGGER_PCT
  • market_open      — morning brief pushed at 9:30 AM ET daily
```

---

## Self-Improving Strategy

After every oracle run Claude outputs updated metadata that is persisted per-symbol:

| Field | What changes |
|-------|-------------|
| `agent_weights` | Boost agents that called last 2+ moves right (up to 1.8×); cut agents wrong 2+ in a row (down to 0.4×) |
| `confidence_threshold` | Lowers toward 0.55 on a hot streak (>65% win rate); raises toward 0.75 when struggling |
| `auto_trade_amount` | +$50 per 3-trade winning streak, −$50 per losing streak; clamped $25–$20,000 |
| `strategy_notes` | Rolling 2–3 sentence log of what's working |

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd crypto_oracle
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Required: ANTHROPIC_API_KEY + ALPACA_API_KEY + ALPACA_SECRET_KEY
# Optional: TELEGRAM_BOT_TOKEN, FINNHUB_API_KEY

# 3. Init database
make db-init

# 4. Run
make run

# Dashboard → http://localhost:8000
# API docs  → http://localhost:8000/docs
```

---

## Alpaca Setup (Paper Trading)

1. Create account at https://alpaca.markets
2. Go to **Paper Account** → **API Keys** → Generate
3. Add to `.env`:
   ```
   ALPACA_API_KEY=PK...
   ALPACA_SECRET_KEY=...
   ALPACA_PAPER=true
   ```
4. Paper account starts with $100,000 virtual cash
5. Live trading: set `ALPACA_PAPER=false` (use extreme caution)

**Supported crypto symbols:** BTC, ETH, SOL, DOGE, AVAX, LTC, BCH, LINK, UNI, AAVE

**Supported stock symbols:** Any fractional-eligible US equity (NVDA, TSLA, AAPL, MSFT, etc.)

---

## Telegram Setup

1. Message `@BotFather` → `/newbot` → copy token to `.env` as `TELEGRAM_BOT_TOKEN`
2. Start your bot and send `/start` — it registers your chat ID automatically
3. Optional: restrict access via `TELEGRAM_ALLOWED_CHAT_IDS=123456,789012`

### Commands

**Crypto Oracle**
```
/status              latest recommendation for all watchlist symbols
/status BTC          specific symbol
/run                 trigger fresh analysis (5-min cooldown)
/history             last 5 recommendations
```

**Crypto Trades**
```
/buy BTC 200         market buy $200 of BTC
/sell BTC            close entire BTC position
/pnl                 crypto trade P&L summary
```

**Stock Trading (long/short)**
```
/stocks              stock watchlist + latest signals
/long NVDA 500       go LONG $500 NVDA
/short TSLA 300      go SHORT $300 TSLA
/cover NVDA          close/cover position (long or short)
/stockpnl            stock trade P&L summary
/addstock AAPL       add to stock watchlist
/removestock AAPL    remove from stock watchlist
```

**Portfolio & Reports**
```
/portfolio           Alpaca account summary (equity, buying power, P&L)
/report              generate and send a PDF dashboard report now
```

**Auto-Trade**
```
/autotrade on        enable auto-trading
/autotrade off       disable auto-trading
/autotrade 500       set trade size to $500 (max $20,000)
```

**Intelligence & Settings**
```
/strategy BTC        agent weights, per-agent accuracy, and strategy notes
/watchlist           show crypto watchlist
/watch ETH           add crypto symbol
/unwatch ETH         remove crypto symbol
/alerts on|off       toggle proactive alerts
/interval 60         change oracle run interval (minutes)
```

Any free-text message is handled by the Claude conversational agent with full chat history context.

---

## Scheduled Reports

Every crypto oracle run automatically generates and sends a PDF dashboard report to all registered Telegram chats. The PDF includes:
- Current action and confidence for all symbols
- Per-agent signal breakdown (signal, confidence, summary)
- Recent recommendation history and outcomes
- Agent accuracy leaderboard

Use `/report` at any time to trigger a report on demand.

---

## Cost Breakdown

Costs scale with oracle interval. Current `.env` default is **30 minutes**.

| Component              | Every 30m (~$) | Every 4h (~$) |
|------------------------|----------------|---------------|
| Claude API (Crypto Oracle × 2 symbols) | ~$90/month | ~$25/month |
| Claude API (Stock Oracle × 2 symbols, market hours) | ~$40/month | ~$12/month |
| Claude API (Telegram conversations) | ~$8/month | ~$8/month |
| Finnhub API | $0 (free tier) | $0 |
| NewsAPI | $0 (RSS fallback) | $0 |
| Alpaca Paper Trading | $0 | $0 |
| CoinGecko / alternative.me | $0 (free tier) | $0 |
| yfinance / Binance | $0 (public) | $0 |
| Hosting (Mac Mini M4) | $0 | $0 |

**Tip:** Run on a Mac Mini M4 with Tailscale for remote access. At 30-min intervals budget ~$140/month in API costs; at 4h intervals ~$45/month.

---

## Configuration Reference

```bash
# ── Core AI ───────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...

# ── Alpaca ────────────────────────────────────────────────────────
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
ALPACA_PAPER=true
ALPACA_MAX_ORDER_USD=20000        # hard cap per order (default $20,000)

# ── Telegram ──────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_CHAT_IDS=        # optional whitelist, comma-separated

# ── Optional data sources ─────────────────────────────────────────
FINNHUB_API_KEY=                  # Finnhub for stock news (free tier available)
NEWSAPI_KEY=                      # NewsAPI for headlines (RSS fallback if blank)

# ── Scheduler ────────────────────────────────────────────────────
ORACLE_INTERVAL_MINUTES=30        # crypto oracle frequency
STOCK_INTERVAL_MINUTES=30         # stock oracle frequency (market hours only)
HEARTBEAT_INTERVAL_MINUTES=120    # Telegram status ping frequency
ALERT_THRESHOLD=0.60              # min confidence to push a proactive alert
ORACLE_PRICE_TRIGGER_PCT=1.5      # fire oracle immediately on crypto price move ≥ this %
AUTO_TRADE_AMOUNT_USD=200         # default starting auto-trade size

# ── Watchlists ───────────────────────────────────────────────────
WATCHLIST=BTC,ETH                 # crypto symbols to track

# ── Feature Flags ────────────────────────────────────────────────
SKIP_KRONOS=false                 # skip Monte Carlo agent (saves ~5s per run)
SKIP_ALPACA=false                 # disable all Alpaca / trading features
```

---

## API Reference

### Health & Recommendations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Uptime, WS connection count |
| GET | `/api/recommendations` | Latest recommendation per symbol |
| GET | `/api/recommendations/{symbol}` | Latest for one symbol |
| GET | `/api/recommendations/{symbol}/history` | History (limit param, max 200) |
| POST | `/api/run/{symbol}` | Trigger crypto oracle run (5-min cooldown) |

### Portfolio & Orders
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/portfolio` | Alpaca account summary |
| GET | `/api/portfolio/crypto` | Open crypto positions |
| GET | `/api/portfolio/orders` | Open orders |
| POST | `/api/order` | Stage a crypto order (returns pending ID) |
| POST | `/api/order/confirm/{id}` | Execute staged order |
| DELETE | `/api/order/{id}` | Cancel open order |

### Crypto Trades
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/trades` | Trade history (optional `?symbol=BTC`) |
| GET | `/api/trades/stats` | Aggregate P&L stats |

### Auto-Trade Settings
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/settings/auto-trade` | Current auto-trade settings |
| POST | `/api/settings/auto-trade` | Update enabled / amount / threshold |

### Strategy
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/strategy/{symbol}` | Agent weights, accuracy, win rate |

### Stocks
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks/watchlist` | Stock watchlist |
| POST | `/api/stocks/watchlist/{symbol}` | Add stock symbol |
| DELETE | `/api/stocks/watchlist/{symbol}` | Remove stock symbol |
| GET | `/api/stocks/positions` | Open stock positions |
| GET | `/api/stocks/market-open` | Is NYSE market currently open? |
| POST | `/api/stocks/run/{symbol}` | Trigger stock oracle run |
| POST | `/api/stocks/order` | Execute stock order (long or short) |
| POST | `/api/stocks/close/{symbol}` | Close/cover stock position |
| GET | `/api/stocks/trades` | Stock trade history |
| GET | `/api/stocks/trades/stats` | Stock aggregate P&L stats |

### Real-time
| Method | Endpoint | Description |
|--------|----------|-------------|
| WS | `/ws/feed` | Real-time stream: `recommendation`, `stock_recommendation`, `trade`, `stock_trade` events |

---

## Disclaimer

This software is for educational and research purposes only.
It does not constitute financial advice. Crypto and equity markets are highly volatile.
Past signals do not guarantee future performance.
Paper trading results do not reflect real-market execution quality or slippage.
Always do your own research before making any investment decisions.
