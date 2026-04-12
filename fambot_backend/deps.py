from __future__ import annotations

import os

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth

from fambot_backend.firebase_init import init_firebase

_bearer = HTTPBearer(auto_error=False)


async def firebase_uid(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    if os.environ.get("FAMBOT_SKIP_AUTH") == "1":
        # Local-only escape hatch; never enable in production.
        return os.environ.get("FAMBOT_DEV_UID", "dev-user")
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    init_firebase()
    try:
        decoded = auth.verify_id_token(creds.credentials)
    except Exception as exc:  # firebase raises various subclasses
        raise HTTPException(status_code=401, detail="Invalid or expired ID token") from exc
    uid = decoded.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Token missing uid")
    return uid
