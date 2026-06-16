"""Security utilities: password hashing and JWT tokens."""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import bcrypt
from jose import JWTError, jwt

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
        UnauthorizedException: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise UnauthorizedException("Invalid or expired token") from exc

    if "sub" not in payload or "type" not in payload:
        raise UnauthorizedException("Malformed token")

    return payload


def _remaining_ttl(exp: int) -> int:
    """Return seconds until exp, or 0 if already expired."""
    exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
    delta = (exp_dt - _now()).total_seconds()
    return max(int(delta), 0)


async def blacklist_token(token: str) -> None:
    """Add a token hash to the Redis blacklist with its remaining TTL."""
    payload = decode_token(token)
    ttl = _remaining_ttl(payload["exp"])
    if ttl <= 0:
        return
    key = f"jwt_blacklist:{_token_hash(token)}"
    await redis_client.set(key, "1", ttl=ttl)


async def is_token_blacklisted(token: str) -> bool:
    """Check whether a token hash is blacklisted."""
    key = f"jwt_blacklist:{_token_hash(token)}"
    return await redis_client.get(key) is not None


__all__ = [
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "create_guest_token",
    "decode_token",
    "blacklist_token",
    "is_token_blacklisted",
]
