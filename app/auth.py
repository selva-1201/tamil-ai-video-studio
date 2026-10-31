"""
Mandatory Password / Bearer Token Authentication (`app/auth.py`)

Because the studio UI is exposed via a public `cloudflared` (or `ngrok`) tunnel from Kaggle,
every `/api/*` endpoint enforces password/token authentication fetched from Kaggle Secrets
(`STUDIO_UI_PASSWORD` or `STUDIO_API_TOKEN`).
"""

from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Header, HTTPException, Query, status

from pipeline.drive import fetch_kaggle_secret


def get_expected_token() -> str:
    """
    Read `STUDIO_UI_PASSWORD` or `STUDIO_API_TOKEN` from Kaggle Secrets / environment.
    If neither is set in local dev mode, defaults to `tamil-studio-local`.
    """
    secret = fetch_kaggle_secret("STUDIO_UI_PASSWORD") or fetch_kaggle_secret("STUDIO_API_TOKEN")
    return secret if secret else "tamil-studio-local"


def verify_studio_auth(
    authorization: Optional[str] = Header(default=None),
    x_studio_token: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
) -> str:
    """
    FastAPI dependency verifying the studio password/token via:
    1. `Authorization: Bearer <token>`
    2. `X-Studio-Token: <token>`
    3. `?token=<token>` query param (needed for browser `EventSource` SSE and `<video>`/`<audio>` streaming)
    """
    expected = get_expected_token()
    candidate = ""
    if authorization and authorization.lower().startswith("bearer "):
        candidate = authorization[7:].strip()
    elif x_studio_token:
        candidate = x_studio_token.strip()
    elif token:
        candidate = token.strip()

    if not candidate or not secrets.compare_digest(candidate, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Studio password/token. Set STUDIO_UI_PASSWORD in Kaggle Secrets.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return candidate
