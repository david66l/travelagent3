# Backend Dockerfile for TravelAgent2
# Multi-stage build for smaller production image

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

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy application code
COPY src/ ./src/
COPY seed_data/ ./seed_data/

# Set ownership
RUN chown -R appuser:appgroup /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/api/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
