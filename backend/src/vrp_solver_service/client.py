"""Async HTTP client for the standalone VRP solver service."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from core.settings import settings
from vrp_solver_service.models import SolverRequest, SolverResponse

logger = logging.getLogger(__name__)

DEFAULT_VRP_BASE_URL = "http://localhost:8001"
_REQUEST_TIMEOUT = 30.0


class VRPSolverClient:
    """Call the VRP solver service asynchronously from LangGraph / main backend."""

    def __init__(self, base_url: str | None = None, client: httpx.AsyncClient | None = None):
        self._base_url = (base_url or settings.vrp_solver_url).rstrip("/")
        self._client = client

    async def solve(self, request: SolverRequest) -> SolverResponse:
        """Send a planning request to the VRP solver service."""
        client = self._client or httpx.AsyncClient(timeout=_REQUEST_TIMEOUT)
        try:
            response = await client.post(
                f"{self._base_url}/solve",
                json=request.model_dump(),
            )
            response.raise_for_status()
            return SolverResponse(**response.json())
        except Exception as exc:
            logger.warning("VRP service call failed: %s", exc)
            raise
        finally:
            if self._client is None and isinstance(client, httpx.AsyncClient):
                await client.aclose()

    async def health(self) -> dict[str, Any]:
        """Check VRP service health."""
        client = self._client or httpx.AsyncClient(timeout=5.0)
        try:
            response = await client.get(f"{self._base_url}/health")
            response.raise_for_status()
            return response.json()
        finally:
            if self._client is None and isinstance(client, httpx.AsyncClient):
                await client.aclose()
