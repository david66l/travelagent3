"""Lightweight JWT authentication — FastAPI dependency, zero external service.

Architecture choice: FastAPI Depends (not Starlette middleware).
- Dependencies are testable, composable, and align with FastAPI's design.
- Middleware is reserved for cross-cutting concerns (request-id, logging).

Token flow:
  POST /api/v1/auth/guest  →  guest token (24h, no credentials)
  POST /api/v1/auth/login  →  access + refresh token
  All protected endpoints →  Authorization: Bearer <token>

Design decisions:
- HS256 for single-service deployment (simpler than RS256)
- Short-lived access tokens (30 min) + longer refresh tokens (7 days)
- optional_user for public/guest-friendly endpoints (returns None if no token)
- require_user for authenticated-only endpoints (raises 401)
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Optional

from core.clock import utc_now_naive

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from jwt.exceptions import InvalidTokenError

from core.settings import settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Security scheme
# --------------------------------------------------------------------------- #

security = HTTPBearer(auto_error=False)


# --------------------------------------------------------------------------- #
# Token payload
# --------------------------------------------------------------------------- #


class TokenPayload:
    """Decoded JWT claims."""

    __slots__ = ("sub", "role", "exp", "iat", "jti")

    def __init__(self, sub: str, role: str, exp: int, iat: int, jti: str = ""):
        self.sub = sub
        self.role = role
        self.exp = exp
        self.iat = iat
        self.jti = jti

    @property
    def is_expired(self) -> bool:
        return self.exp < int(time.time())


# --------------------------------------------------------------------------- #
# Token creation
# --------------------------------------------------------------------------- #


def create_access_token(
    sub: str,
    role: str = "user",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a short-lived access token."""
    now = utc_now_naive()
    expire = now + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    return jwt.encode(
        {
            "sub": sub,
            "role": role,
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "jti": _rand_id(),
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(sub: str, role: str = "user") -> str:
    """Create a long-lived refresh token."""
    now = utc_now_naive()
    expire = now + timedelta(days=settings.refresh_token_expire_days)
    return jwt.encode(
        {
            "sub": sub,
            "role": role,
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "type": "refresh",
            "jti": _rand_id(),
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


def create_guest_token(session_id: str) -> str:
    """Create a guest token (24h lifetime, no login required)."""
    now = utc_now_naive()
    expire = now + timedelta(hours=settings.guest_token_expire_hours)
    return jwt.encode(
        {
            "sub": f"guest:{session_id}",
            "role": "guest",
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "jti": _rand_id(),
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


# --------------------------------------------------------------------------- #
# Dependency: require authenticated user
# --------------------------------------------------------------------------- #


async def require_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> TokenPayload:
    """Require a valid access token. Raises 401 if missing or invalid."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _decode_token(credentials.credentials)


# --------------------------------------------------------------------------- #
# Dependency: optional user (guest-friendly)
# --------------------------------------------------------------------------- #


async def optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[TokenPayload]:
    """Return the user if a valid token is present, otherwise None."""
    if credentials is None:
        return None
    try:
        return _decode_token(credentials.credentials)
    except HTTPException:
        return None


# --------------------------------------------------------------------------- #
# Role checkers
# --------------------------------------------------------------------------- #


class RequireRole:
    """Dependency factory: require specific role(s)."""

    def __init__(self, *roles: str):
        self.roles = set(roles)

    async def __call__(
        self,
        user: TokenPayload = Depends(require_user),
    ) -> TokenPayload:
        if user.role not in self.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {self.roles}",
            )
        return user


# --------------------------------------------------------------------------- #
# Internal
# --------------------------------------------------------------------------- #


def _decode_token(token: str) -> TokenPayload:
    """Decode and validate a JWT. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    sub = payload.get("sub", "")
    role = payload.get("role", "user")
    exp = payload.get("exp", 0)
    iat = payload.get("iat", 0)
    jti = payload.get("jti", "")

    if not sub:
        raise HTTPException(status_code=401, detail="Token missing subject")

    return TokenPayload(sub=sub, role=role, exp=exp, iat=iat, jti=jti)


def _rand_id(length: int = 16) -> str:
    import secrets

    return secrets.token_hex(length)
