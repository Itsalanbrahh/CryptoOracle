"""WebSocket connection manager and broadcast helpers."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket

from crypto_oracle.utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts messages."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info("WS client connected. Total: %d", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            try:
                self._connections.remove(ws)
            except ValueError:
                pass
        logger.info("WS client disconnected. Total: %d", len(self._connections))

    async def broadcast(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload)
        dead: list[WebSocket] = []
        async with self._lock:
            connections = list(self._connections)

        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    try:
                        self._connections.remove(ws)
                    except ValueError:
                        pass
            logger.debug("Removed %d dead WS connections", len(dead))

    async def ping_all(self) -> None:
        """Send ping to all clients; remove unresponsive ones."""
        dead: list[WebSocket] = []
        async with self._lock:
            connections = list(self._connections)

        for ws in connections:
            try:
                await ws.send_text(json.dumps({"type": "ping"}))
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    try:
                        self._connections.remove(ws)
                    except ValueError:
                        pass

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Global singleton
manager = ConnectionManager()
