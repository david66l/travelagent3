"""Security utilities: password hashing and JWT tokens."""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import bcrypt
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

from core.exceptions import UnauthorizedException
from core.redis_client import redis_client
from core.settings import settings


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _user_ban_key(user_id: str) -> str:
    return f"jwt_banned_user:{user_id}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: UUID, role: str = "user") -> str:
    """Create a short-lived access token."""
    expire = _now() + timedelta(minutes=settings.access_token_expire_minutes)
    claims = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "jti": str(uuid4()),
        "exp": expire,
        "iat": _now(),
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(user_id: UUID, role: str = "user") -> str:
    """Create a longer-lived refresh token."""
    expire = _now() + timedelta(days=settings.refresh_token_expire_days)
    claims = {
        "sub": str(user_id),
        "role": role,
        "type": "refresh",
        "jti": str(uuid4()),
        "exp": expire,
        "iat": _now(),
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_guest_token(user_id: UUID, device_fingerprint: str) -> str:
    """Create a 24h guest token bound to a device fingerprint."""
    expire = _now() + timedelta(hours=settings.guest_token_expire_hours)
    claims = {
        "sub": str(user_id),
        "role": "guest",
        "type": "guest",
        "device_fingerprint": device_fingerprint,
        "jti": str(uuid4()),
        "exp": expire,
        "iat": _now(),
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT.

    Raises:
        UnauthorizedException: With PRD codes TOKEN_EXPIRED / TOKEN_INVALID.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except ExpiredSignatureError as exc:
        raise UnauthorizedException("Token expired", code="TOKEN_EXPIRED") from exc
    except InvalidTokenError as exc:
        raise UnauthorizedException("Invalid token", code="TOKEN_INVALID") from exc

    if "sub" not in payload or "type" not in payload:
        raise UnauthorizedException("Malformed token", code="TOKEN_INVALID")

    return payload


def _remaining_ttl(exp: int) -> int:
    """Return seconds until exp, or 0 if already expired."""
    exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
    delta = (exp_dt - _now()).total_seconds()
    return max(int(delta), 0)


async def blacklist_token(token: str) -> None:
    """Add a token hash to the Redis blacklist with its remaining TTL."""
    try:
        payload = decode_token(token)
    except UnauthorizedException as exc:
        if exc.code != "TOKEN_EXPIRED":
            raise
        # Allow blacklisting expired tokens (logout edge case).
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": False},
        )
    ttl = _remaining_ttl(payload["exp"])
    if ttl <= 0:
        return
    key = f"jwt_blacklist:{_token_hash(token)}"
    await redis_client.set(key, "1", ttl=ttl)


async def is_token_blacklisted(token: str) -> bool:
    """Check whether a token hash is blacklisted."""
    key = f"jwt_blacklist:{_token_hash(token)}"
    return await redis_client.get(key) is not None


async def is_user_banned(user_id: str) -> bool:
    """True when admin has revoked all tokens for this user."""
    return await redis_client.get(_user_ban_key(user_id)) is not None


async def ban_user_tokens(user_id: str, ttl_seconds: int | None = None) -> None:
    """Ban all outstanding tokens for a user until max token lifetime elapses."""
    if ttl_seconds is None:
        ttl_seconds = max(
            settings.refresh_token_expire_days * 86400,
            settings.guest_token_expire_hours * 3600,
        )
    await redis_client.set(_user_ban_key(user_id), "1", ttl=ttl_seconds)


__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "create_guest_token",
    "decode_token",
    "blacklist_token",
    "is_token_blacklisted",
    "is_user_banned",
    "ban_user_tokens",
]
