"""CryptoOracle — unified entry point.

Startup sequence:
  1. Load .env and validate required vars
  2. init_db()
  3. Start APScheduler
  4. Start Telegram bot (async, in same event loop)
  5. Start FastAPI / uvicorn
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_env() -> None:
    required = ["ANTHROPIC_API_KEY"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"[FATAL] Missing required env vars: {', '.join(missing)}")
        print("        Copy .env.example to .env and fill in your keys.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    from crypto_oracle.models.db import init_db
    from crypto_oracle.scheduler.heartbeat import setup_scheduler
    from crypto_oracle.telegram.bot import start_bot, stop_bot

    await init_db()

    scheduler = setup_scheduler()
    scheduler.start()

    try:
        await start_bot()
    except Exception as exc:
        print(f"[WARN] Telegram bot failed to start: {exc} — server continuing without it")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    interval = int(os.getenv("ORACLE_INTERVAL_MINUTES", "240"))
    paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
    tg_enabled = bool(os.getenv("TELEGRAM_BOT_TOKEN"))

    print(
        f"\n{'='*60}\n"
        f"  CryptoOracle is running\n"
        f"  Dashboard  : http://{host}:{port}\n"
        f"  API docs   : http://{host}:{port}/docs\n"
        f"  Alpaca     : {'paper' if paper else 'LIVE'} trading\n"
        f"  Telegram   : {'active' if tg_enabled else 'disabled (no token)'}\n"
        f"  Oracle runs every {interval} minutes\n"
        f"{'='*60}\n"
    )

    yield

    # Shutdown
    scheduler.shutdown(wait=False)
    await stop_bot()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    _validate_env()

    app = FastAPI(
        title="CryptoOracle",
        description="Multi-agent crypto trading intelligence system",
        version="2.0.0",
        lifespan=lifespan,
    )

    allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
    wildcard = allowed_origins_raw == "*"
    origins = ["*"] if wildcard else [o.strip() for o in allowed_origins_raw.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from crypto_oracle.api.router import router
    app.include_router(router)

    from crypto_oracle.api.websocket import manager
    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/ws/feed")
    async def websocket_feed(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                await asyncio.wait_for(websocket.receive_text(), timeout=60)
        except (WebSocketDisconnect, asyncio.TimeoutError, Exception):
            await manager.disconnect(websocket)

    # Serve React dashboard
    dashboard_path = Path(__file__).parent / "dashboard" / "index.html"

    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard():
        if dashboard_path.exists():
            return HTMLResponse(content=dashboard_path.read_text())
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "crypto_oracle.main:app",
        host=host,
        port=port,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=10,
    )
