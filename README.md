# CryptoOracle

Autonomous multi-agent crypto trading intelligence system.
7 specialist AI agents → orchestrator synthesis → FastAPI backend → React dashboard → Telegram bot → Alpaca paper trading.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CryptoOracle                             │
│                                                                 │
│  ┌──────────── 7 Specialist Agents ─────────────┐              │
│  │  📊 Kronos    Monte Carlo / quant stats       │              │
│  │  🌍 Macro     Fed / DXY / news / fear&greed   │              │
│  │  🔬 Micro     Order book / bid-ask / depth    │              │
│  │  📦 Volume    OBV / volume profile             │              │
│  │  ⛓  OnChain  Hash rate / mempool / tx volume  │              │
│  │  💬 Sentiment Fear&greed / headlines / social │              │
│  │  📈 Technical EMA / RSI / MACD / Bollinger    │              │
│  └──────────────────────┬────────────────────────┘              │
│                         │ AgentSignal[]                         │
│               ┌─────────▼──────────┐                           │
│               │    Orchestrator    │  Claude synthesis          │
│               └─────────┬──────────┘                           │
│                         │ MasterRecommendation                  │
│          ┌──────────────┼──────────────────────┐               │
│          ▼              ▼                       ▼               │
│    ┌───────────┐  ┌───────────┐        ┌──────────────┐        │
│    │  SQLite   │  │  FastAPI  │        │  APScheduler │        │
│    │   (db)    │  │  + WS     │        │  heartbeat   │        │
│    └───────────┘  └─────┬─────┘        └──────┬───────┘        │
│                         │                     │                 │
│                  ┌──────▼──────┐      ┌───────▼────────┐       │
│                  │   React     │      │  Telegram Bot  │       │
│                  │  Dashboard  │      │  + Claude conv │       │
│                  └─────────────┘      └───────┬────────┘       │
│                                               │                 │
│                                       ┌───────▼────────┐       │
│                                       │  Alpaca Paper  │       │
│                                       │  Trading API   │       │
│                                       └────────────────┘       │
└─────────────────────────────────────────────────────────────────┘

Data flow:
  Agents fetch data (CoinGecko, Binance, Blockchain.com, Mempool.space,
  alternative.me, NewsAPI) → Claude analysis → AgentSignal
  Orchestrator collects signals → Claude synthesis → MasterRecommendation
  Saved to SQLite → broadcast via WebSocket → push to Telegram
```

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd crypto_oracle
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — minimum required: ANTHROPIC_API_KEY + ALPACA_API_KEY + ALPACA_SECRET_KEY

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
2. Navigate to **Paper Account** → **API Keys** → Generate
3. Add to `.env`:
   ```
   ALPACA_API_KEY=PK...
   ALPACA_SECRET_KEY=...
   ALPACA_PAPER=true
   ```
4. Paper account starts with $100,000 virtual cash
5. To go live: set `ALPACA_PAPER=false` (use extreme caution)

Supported crypto symbols: BTC, ETH, SOL, DOGE, AVAX, LTC, BCH, LINK, UNI, AAVE

---

## Telegram Setup

1. Message `@BotFather` on Telegram → `/newbot`
2. Copy the token to `.env` as `TELEGRAM_BOT_TOKEN`
3. Start your bot and send `/start`
4. The bot registers your chat ID automatically

To restrict access to specific users, set `TELEGRAM_ALLOWED_CHAT_IDS=123456,789012`

**Commands:**
```
/status          latest oracle recommendation
/status BTC      specific symbol
/run             trigger fresh analysis (30-min cooldown)
/history         last 5 recommendations
/portfolio       Alpaca account summary
/watchlist       show watchlist
/watch ETH       add symbol
/unwatch ETH     remove symbol
/alerts on|off   toggle proactive alerts
/interval 120    set oracle run interval in minutes
```

Any free-text message is handled by the Claude conversational agent.

---

## Cost Breakdown

| Component              | Every 4h (default) | Every 1h       |
|------------------------|-------------------|----------------|
| Claude API (Oracle)    | ~$25/month        | ~$95/month     |
| Claude API (Telegram)  | ~$8/month         | ~$8/month      |
| NewsAPI                | $0 (RSS fallback) | $0             |
| Alpaca Paper Trading   | $0                | $0             |
| CoinGecko / alt.me     | $0 (free tier)    | $0             |
| Binance data           | $0 (public API)   | $0             |
| Hosting (Mac Mini M4)  | $0                | $0             |
| Hosting (Cloud VPS)    | $24/month (alt)   | $24/month (alt)|
| **TOTAL (Mac Mini)**   | **~$33/month**    | **~$103/month**|
| **TOTAL (Cloud VPS)**  | **~$57/month**    | **~$127/month**|

Recommendation: Run on your Mac Mini M4 with Tailscale for ~$1/day at 4h intervals.

---

## Scheduling Config

```bash
ORACLE_INTERVAL_MINUTES=240    # 4h default — full 7-agent oracle run
HEARTBEAT_INTERVAL_MINUTES=360 # 6h default — Telegram status ping
ALERT_THRESHOLD=0.70           # only alert on signals ≥ 70% confidence
```

Change interval live via Telegram: `/interval 120`

Scheduler jobs survive restarts (APScheduler SQLAlchemy job store → same SQLite DB).

---

## Feature Flags

```bash
SKIP_KRONOS=false   # set true to skip Monte Carlo agent (saves ~5s per run)
SKIP_ALPACA=false   # set true to disable all trading features
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Uptime + WS connection count |
| GET | `/api/recommendations` | Latest rec per symbol |
| GET | `/api/recommendations/{symbol}` | Latest for one symbol |
| GET | `/api/recommendations/{symbol}/history` | History (limit param) |
| POST | `/api/run/{symbol}` | Trigger oracle run |
| GET | `/api/portfolio` | Alpaca account summary |
| GET | `/api/portfolio/crypto` | Open crypto positions |
| GET | `/api/portfolio/orders` | Open orders |
| POST | `/api/order` | Create order (pending confirmation) |
| POST | `/api/order/confirm/{id}` | Confirm and execute order |
| DELETE | `/api/order/{id}` | Cancel open order |
| GET | `/api/watchlist` | Current watchlist |
| WS | `/ws/feed` | Real-time recommendation stream |

---

## Disclaimer

This software is for educational and research purposes only.
It does not constitute financial advice. Crypto markets are highly volatile.
Past signals do not guarantee future performance.
Paper trading results do not reflect real-market execution.
Always do your own research before making any investment decisions.
