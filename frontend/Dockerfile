# Frontend Dockerfile for TravelAgent2 (M5)
# Multi-stage build for optimized Next.js production image.
# Requires Next.js standalone output (next.config.js -> output: 'standalone').
# Image: ghcr.io/travelagent/travel-agent-frontend:latest

# ── Builder stage ──
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (better layer caching)
COPY package*.json ./
RUN npm ci

# Copy source and build
COPY . .
# Ensure public/ exists (Next.js standalone output expects it)
RUN mkdir -p public
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production
RUN npm run build

# ── Production stage ──
FROM node:20-alpine AS production

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

# Create non-root user
RUN addgroup -g 1000 -S nodejs && adduser -S nextjs -u 1000

# Copy only necessary files from builder
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs

EXPOSE 3000

# Health check (use node built-in http to avoid busybox wget issues in alpine)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD node -e "require('http').get('http://127.0.0.1:3000', (r) => process.exit(r.statusCode === 200 ? 0 : 1)).on('error', () => process.exit(1))"

CMD ["node", "server.js"]
