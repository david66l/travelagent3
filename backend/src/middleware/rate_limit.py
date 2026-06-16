"""Rate limiting middleware."""

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from core.exceptions import RateLimitException
from core.rate_limit import check_rate_limit, get_client_ip, rate_limit_key
from core.redis_client import redis_client
from core.responses import error_response
from core.security import decode_token
from core.settings import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce IP / user / guest sliding-window rate limits."""

    # Paths that should not consume rate-limit budget.
    EXEMPT_PATHS = {"/health", "/healthz", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        kind, identifier, limit = self._classify(request)
        key = rate_limit_key(kind=kind, identifier=identifier)
        try:
            await check_rate_limit(
                redis_client,
                key=key,
                limit=limit,
                window_seconds=settings.cache_ttl_rate_limit,
            )
        except RateLimitException as exc:
            response = error_response(
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
                details=exc.details,
            )
            retry_after = exc.details.get("retry_after") if exc.details else None
            if retry_after:
                response.headers["Retry-After"] = str(retry_after)
            return response
        return await call_next(request)

    def _classify(self, request: Request):
        """Classify the request and return (kind, identifier, limit)."""
        auth = request.headers.get("Authorization", "")
        token = None
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()

        if token:
            try:
                payload = decode_token(token)
                token_type = payload.get("type")
                if token_type == "guest":
                    fingerprint = payload.get("device_fingerprint") or "unknown"
                    return "guest", fingerprint, settings.rate_limit_guest_per_minute

                user_id = payload.get("sub") or "unknown"
                role = payload.get("role")
                if role == "admin":
                    # Admins share the user bucket; can be upgraded to exempt later.
                    return "user", user_id, settings.rate_limit_user_per_minute
                return "user", user_id, settings.rate_limit_user_per_minute
            except Exception:
                # Malformed/expired token: fall back to IP limiting.
                pass

        return "ip", get_client_ip(request), settings.rate_limit_ip_per_minute


def setup_rate_limit(app: FastAPI) -> None:
    """Register rate limiting middleware on the app."""
    app.add_middleware(RateLimitMiddleware)
