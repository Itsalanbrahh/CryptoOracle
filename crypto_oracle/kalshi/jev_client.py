"""Thin async client for TypeSafe Jev (System One).

Uses raw HTTP via aiohttp so we don't hard-require ``typesafe-sdk`` at import
time. Set ``TYPESAFE_API_KEY`` (or ``JEV_API_KEY``) to enable live calls.
"""
from __future__ import annotations

import os
from typing import Any

import aiohttp

TYPESAFE_BASE_URL = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
DEFAULT_MODEL = os.getenv("JEV_MODEL", os.getenv("TYPESAFE_DEFAULT_MODEL", "jev-latest"))


class JevError(RuntimeError):
    """Raised when a Jev / System One call fails."""


def jev_api_key() -> str:
    return (
        os.getenv("TYPESAFE_API_KEY", "").strip()
        or os.getenv("JEV_API_KEY", "").strip()
    )


def jev_enabled() -> bool:
    """True when a key is present and KALSHI_JEV_15M is not explicitly off."""
    if os.getenv("KALSHI_JEV_15M", "1").strip() == "0":
        return False
    return bool(jev_api_key())


async def system_one(
    state: Any,
    questions: dict[str, dict],
    *,
    model: str | None = None,
    timeout_s: float = 20.0,
) -> dict:
    """
    POST /v1/systemone and return the raw JSON body.

    ``questions`` uses the HTTP shape, e.g.::

        {
          "settle_up": {"type": "noul", "instructions": "..."},
          "action": {
              "type": "choice",
              "instructions": "...",
              "criteria": {"buy_yes": "...", "buy_no": "...", "hold": "..."},
          },
        }
    """
    key = jev_api_key()
    if not key:
        raise JevError("TYPESAFE_API_KEY / JEV_API_KEY not set")

    payload = {
        "state": state,
        "model": model or DEFAULT_MODEL,
        "questions": questions,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    url = f"{TYPESAFE_BASE_URL}/v1/systemone"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                data = await resp.json(content_type=None)
                if resp.status >= 400:
                    raise JevError(f"Jev HTTP {resp.status}: {data}")
                return data if isinstance(data, dict) else {"raw": data}
    except JevError:
        raise
    except Exception as exc:
        raise JevError(f"Jev request failed: {exc}") from exc
