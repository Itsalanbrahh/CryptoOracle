"""FastAPI security dependencies for CryptoOracle."""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _is_local_dev() -> bool:
    return os.getenv("APP_LOCAL_DEV", "").lower() == "true"


def _configured_api_key() -> Optional[str]:
    return os.getenv("APP_API_KEY", "").strip() or None


async def require_api_key(
    api_key_header: Optional[str] = Security(_API_KEY_HEADER),
) -> None:
    """FastAPI dependency that enforces the APP_API_KEY on mutating routes.

    Behaviour:
    - If APP_LOCAL_DEV=true and APP_API_KEY is unset: allow all (local dev convenience).
    - If APP_API_KEY is set: require the X-API-Key header to match exactly.
    - If APP_API_KEY is unset and APP_LOCAL_DEV is not 'true': fail closed (403).
    """
    configured = _configured_api_key()

    if configured is None:
        # No key configured
        if _is_local_dev():
            return  # Open in explicit local dev mode
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key not configured. Set APP_API_KEY or enable APP_LOCAL_DEV=true for local development.",
        )

    if api_key_header != configured:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
