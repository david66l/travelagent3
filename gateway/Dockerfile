# Gateway Dockerfile for TravelAgent2 (M5)
# Multi-stage build for Go API gateway.
# Image: ghcr.io/travelagent/travel-agent-gateway:latest

# ── Build stage ──
FROM golang:1.23-alpine AS build

WORKDIR /src

# Install git for private modules (if any)
RUN apk add --no-cache git ca-certificates

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /gateway ./cmd/gateway

# ── Production stage ──
FROM alpine:3.20

# Install CA certificates for TLS upstream calls
RUN apk add --no-cache ca-certificates wget && \
    addgroup -g 10001 -S appgroup && \
    adduser -u 10001 -S appuser -G appgroup

WORKDIR /app

COPY --from=build /gateway /app/gateway
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080

ENV GATEWAY_LISTEN=:8080
ENV GATEWAY_PORT=8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 -O /dev/null http://localhost:8080/api/v1/health || exit 1

CMD ["/app/gateway"]
