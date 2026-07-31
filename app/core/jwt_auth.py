# app/core/jwt_auth.py — issue/verify signed session tokens.
#
# Claim structure (see report for the frontend contract):
#   sub    — str(user.id)
#   email  — user.email
#   role   — "admin" | "member"
#   status — "approved" (only approved users are ever issued a token)
#   iat    — issued-at, unix seconds
#   exp    — expiry, unix seconds (settings.JWT_EXPIRE_MINUTES from iat)
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt

from app.core.config import settings


def create_access_token(user) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role,
        "status": user.status,
        "iat": now,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Raises jwt.ExpiredSignatureError / jwt.InvalidTokenError on failure."""
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
