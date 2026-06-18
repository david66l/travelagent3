# Backend Dockerfile for TravelAgent2 (M5)
# Multi-stage build for smaller production FastAPI image.
# Image: ghcr.io/travelagent/travel-agent-backend:latest

# ── Builder stage ──
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency definitions
COPY pyproject.toml ./

# Install production dependencies into a virtual environment
RUN uv venv /opt/venv && \
    uv pip install --no-cache -e "." --python /opt/venv/bin/python

# ── Production stage ──
FROM python:3.13-slim AS production

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Copy application code, data, and migrations
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup seed_data/ ./seed_data/
COPY --chown=appuser:appgroup alembic.ini ./
COPY --chown=appuser:appgroup migrations/ ./migrations/

# Ensure tmp dirs are writable for prometheus multiprocess and uvicorn
RUN mkdir -p /tmp/prometheus && chown -R appuser:appgroup /app /tmp/prometheus

USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/api/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
