# ============================================================
# Frontend Dockerfile — HalluScan React UI
# Author: Chalani Dinitha (20211032)
# ============================================================

# Stage 1: Build React app
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY frontend/ ./

# Build production bundle
RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:alpine

# Copy built app to nginx
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
