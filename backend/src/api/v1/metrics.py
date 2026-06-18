"""Prometheus metrics endpoint (PRD §7 / §12.7)."""

from fastapi import APIRouter, Response

from core.metrics import prometheus_content_type, render_prometheus

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(
        content=render_prometheus(),
        media_type=prometheus_content_type(),
    )
