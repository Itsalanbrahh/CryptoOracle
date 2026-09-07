"""
Test that mutating FastAPI trading routes require a valid API key header.
- When APP_API_KEY is set, routes must reject requests lacking the correct header.
- When APP_LOCAL_DEV=true AND APP_API_KEY is unset, routes are open (local dev only).
- Read-only routes (/health, GET /api/recommendations) must remain accessible without key.
"""
import os
import sys
from unittest.mock import patch

import pytest


def _make_client():
    """Create a TestClient with freshly reloaded router.

    NOTE: security.py reads env at *call* time (os.getenv inside require_api_key),
    so the caller must patch os.environ *around each request*, not around the client
    construction.  This helper only reloads modules so the router is fresh.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Reload the API modules so the router is freshly imported
    for mod_name in list(sys.modules.keys()):
        if "crypto_oracle.api" in mod_name:
            del sys.modules[mod_name]

    from crypto_oracle.api.router import router as fresh_router
    app = FastAPI()
    app.include_router(fresh_router)
    return TestClient(app, raise_server_exceptions=False)


class TestApiKeyProtection:
    """API key security on mutating routes."""

    def test_post_order_rejected_without_key_when_api_key_set(self):
        """POST /api/order must return 403 when API key is configured but header is missing."""
        client = _make_client()
        env = {"APP_API_KEY": "secret123", "SKIP_ALPACA": "true"}
        with patch.dict(os.environ, env, clear=False):
            resp = client.post(
                "/api/order",
                json={"symbol": "BTC", "side": "buy", "amount_usd": 100},
            )
        assert resp.status_code in (401, 403), (
            f"Expected 401/403, got {resp.status_code}: {resp.text}"
        )

    def test_post_order_accepted_with_correct_key(self):
        """POST /api/order must not return 401/403 when correct API key is supplied."""
        client = _make_client()
        env = {"APP_API_KEY": "secret123", "SKIP_ALPACA": "true"}
        with patch.dict(os.environ, env, clear=False):
            resp = client.post(
                "/api/order",
                json={"symbol": "BTC", "side": "buy", "amount_usd": 100},
                headers={"X-API-Key": "secret123"},
            )
        # Should NOT be 401/403 — may be 503 (Alpaca disabled) or 400, that's fine
        assert resp.status_code not in (401, 403), (
            f"Correct key rejected: {resp.status_code}: {resp.text}"
        )

    def test_post_order_rejected_with_wrong_key(self):
        """POST /api/order must return 401/403 if wrong API key supplied."""
        client = _make_client()
        env = {"APP_API_KEY": "secret123", "SKIP_ALPACA": "true"}
        with patch.dict(os.environ, env, clear=False):
            resp = client.post(
                "/api/order",
                json={"symbol": "BTC", "side": "buy", "amount_usd": 100},
                headers={"X-API-Key": "wrongkey"},
            )
        assert resp.status_code in (401, 403), (
            f"Wrong key should be rejected, got {resp.status_code}"
        )

    def test_get_health_is_public(self):
        """GET /api/health must remain accessible without API key."""
        client = _make_client()
        env = {"APP_API_KEY": "secret123", "SKIP_ALPACA": "true"}
        with patch.dict(os.environ, env, clear=False):
            resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_get_recommendations_is_public(self):
        """GET /api/recommendations must remain accessible without API key."""
        client = _make_client()
        env = {"APP_API_KEY": "secret123", "SKIP_ALPACA": "true"}
        with patch.dict(os.environ, env, clear=False):
            resp = client.get("/api/recommendations")
        # May succeed (empty list) or 502 if no DB — just not 401/403
        assert resp.status_code not in (401, 403)

    def test_fails_closed_when_api_key_unset_and_not_local_dev(self):
        """When APP_API_KEY is unset and APP_LOCAL_DEV is not 'true', mutating routes must fail closed."""
        client = _make_client()
        # Use clear=True so APP_API_KEY and APP_LOCAL_DEV are definitely absent
        clean_env = {"SKIP_ALPACA": "true"}
        with patch.dict(os.environ, clean_env, clear=True):
            resp = client.post(
                "/api/order",
                json={"symbol": "BTC", "side": "buy", "amount_usd": 100},
            )
        assert resp.status_code in (401, 403), (
            f"Without APP_API_KEY and not in local dev, route must fail closed. Got {resp.status_code}"
        )

    def test_local_dev_mode_allows_unauthenticated(self):
        """When APP_LOCAL_DEV=true and APP_API_KEY is unset, mutating routes are open (dev only)."""
        client = _make_client()
        dev_env = {"APP_LOCAL_DEV": "true", "SKIP_ALPACA": "true"}
        with patch.dict(os.environ, dev_env, clear=True):
            resp = client.post(
                "/api/order",
                json={"symbol": "BTC", "side": "buy", "amount_usd": 100},
            )
        # Must not be blocked by auth (503 Alpaca disabled is OK)
        assert resp.status_code not in (401, 403), (
            f"Local dev should bypass auth, got {resp.status_code}"
        )
