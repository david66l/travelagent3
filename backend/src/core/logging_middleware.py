"""Structured request logging middleware with correlation IDs.

Adds ``request_id`` and ``latency_ms`` to every log entry within a request,
without requiring per-endpoint code changes.  Uses contextvars internally
so it works correctly under asyncio concurrency.

To enable, add to FastAPI app:
    app.add_middleware(LoggingMiddleware)
"""

from __future__ import annotations

import logging
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("travel_agent.http")

# contextvars so parallel requests don't cross-contaminate
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_request_path: ContextVar[str] = ContextVar("request_path", default="")
_request_method: ContextVar[str] = ContextVar("request_method", default="")


# Public accessors for use in any module
def get_correlation_id() -> str:
    return _correlation_id.get()


def get_request_path() -> str:
    return _request_path.get()


def get_request_method() -> str:
    return _request_method.get()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Inject correlation ID and emit structured request logs."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Use incoming X-Request-ID or generate a new one
        cid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        _correlation_id.set(cid)
        _request_path.set(request.url.path)
        _request_method.set(request.method)

        start = time.monotonic()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = cid
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            level = logging.WARNING if status_code >= 500 else (
                logging.INFO if status_code >= 400 else logging.DEBUG
            )
            logger.log(
                level,
                "%s %s → %d (%dms)",
                request.method, request.url.path, status_code, latency_ms,
                extra={
                    "request_id": cid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "latency_ms": latency_ms,
                    "client_ip": request.client.host if request.client else "",
                    "user_agent": request.headers.get("user-agent", "")[:200],
                },
            )
