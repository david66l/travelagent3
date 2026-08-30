# Frontend Dockerfile for TravelAgent2 (M5)
# Multi-stage build for optimized Next.js production image.
# Requires Next.js standalone output (next.config.js -> output: 'standalone').
# Image: ghcr.io/travelagent/travel-agent-frontend:latest

# ── Builder stage ──
FROM node:24-alpine AS builder

WORKDIR /app

ARG INTERNAL_API_URL=http://backend:8000
ENV INTERNAL_API_URL=$INTERNAL_API_URL

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
FROM node:24-alpine AS production

WORKDIR /app

# Apply distribution security fixes and remove package-manager tooling that is
# unnecessary for the immutable runtime image (and otherwise adds vendored
# packages to the production attack surface).
RUN apk upgrade --no-cache && \
    rm -rf /usr/local/lib/node_modules/npm \
           /usr/local/bin/npm \
           /usr/local/bin/npx \
           /usr/local/lib/node_modules/corepack \
           /usr/local/bin/corepack \
           /usr/local/bin/pnpm \
           /usr/local/bin/pnpx \
           /usr/local/bin/yarn \
           /usr/local/bin/yarnpkg

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

# Copy only necessary files from builder
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static
COPY --from=builder --chown=node:node /app/public ./public

USER node

EXPOSE 3000

# Health check (use node built-in http to avoid busybox wget issues in alpine)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD node -e "require('http').get('http://127.0.0.1:3000/login', (r) => process.exit(r.statusCode === 200 ? 0 : 1)).on('error', () => process.exit(1))"

CMD ["node", "server.js"]
