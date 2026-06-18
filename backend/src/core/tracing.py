"""OpenTelemetry bootstrap (PRD §12.7)."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from core.settings import settings

logger = logging.getLogger(__name__)


def setup_tracing(app: FastAPI) -> None:
    """Configure OTLP export and FastAPI instrumentation when enabled."""
    if not settings.otel_enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OpenTelemetry packages missing; tracing disabled")
        return

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": "2.0.0",
        }
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry tracing enabled -> %s", settings.otel_exporter_endpoint)
