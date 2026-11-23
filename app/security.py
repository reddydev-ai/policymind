"""Small fail-closed shared-key guard for the demo API.

Replace this with an identity provider, per-user authorization, document ACLs,
audit logging and HTTPS at the edge before any multi-user deployment.
"""
from __future__ import annotations
import secrets
from fastapi import Header, HTTPException
from .config import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if not expected:
        raise HTTPException(status_code=503, detail="API authentication not configured.")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Unauthorized.")
