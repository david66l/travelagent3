"""HTTP request metrics middleware."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from core.metrics import record_http_request


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return str(route.path)
    return request.url.path


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for Prometheus."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path in ("/metrics", "/api/v1/metrics"):
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        record_http_request(
            request.method,
            _route_template(request),
            response.status_code,
            duration,
        )
        return response
