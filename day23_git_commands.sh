#!/bin/bash
# ============================================================
# DAY 23 — Git Commands Script
# Docker Containerization
# Branch: deploy/production-config
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 23 — Docker Containerization"
echo "=============================================="
echo ""

git checkout main
git pull origin main
git checkout -b deploy/production-config
echo "Switched to: deploy/production-config"
echo ""

# COMMIT 1: Backend Dockerfile
git add backend/Dockerfile
git commit -m "deploy(docker): add backend Dockerfile

- Base: python:3.11-slim
- Installs system deps: gcc, g++, curl
- Installs Python deps from requirements.txt
- Downloads spaCy en_core_web_sm model
- Exposes port 8000
- Health check: GET /api/health every 30s
- CMD: uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo "Commit 1: backend/Dockerfile"

# COMMIT 2: Frontend Dockerfile + nginx
git add frontend/Dockerfile frontend/nginx.conf
git commit -m "deploy(docker): add frontend Dockerfile with nginx

Multi-stage build:
  Stage 1 (builder): node:20-alpine
    - npm ci installs dependencies
    - npm run build creates production bundle
  Stage 2 (server): nginx:alpine
    - Serves /dist from Stage 1
    - nginx.conf: SPA routing, API proxy, static caching
    - Proxies /api/ to http://backend:8000
    - Exposes port 80"

echo "Commit 2: frontend/Dockerfile + nginx.conf"

# COMMIT 3: docker-compose
git add docker-compose.yml
git commit -m "deploy(docker): add docker-compose with 3 services

Services:
  backend:  FastAPI + OPT-1.3b on port 8000
    - 8GB memory limit for model
    - model_cache volume (avoid re-download)
    - data/ and experiments/ volumes mounted

  frontend: React + nginx on port 80
    - depends_on backend
    - /api/ proxied to backend:8000

  mlflow: Experiment tracking on port 5000
    - mlruns/ volume mounted

Network: halluscan-network (bridge)
Volumes: model_cache (HuggingFace cache)

Usage:
  docker-compose up --build     # build and start
  docker-compose up -d          # background
  docker-compose down           # stop"

echo "Commit 3: docker-compose.yml"

# COMMIT 4: dockerignore
git add .dockerignore
git commit -m "deploy(docker): add .dockerignore

Excludes from Docker build context:
  - venv/, node_modules/ (large, not needed)
  - .git/, .env (security)
  - __pycache__, *.pyc (compiled files)
  - mlruns/, *.log (runtime files)
  - frontend/dist/, frontend/.vite/ (build artifacts)"

echo "Commit 4: .dockerignore"

# COMMIT 5: git script
git add day23_git_commands.sh
git commit -m "chore(day23): add Day 23 git commands script"
echo "Commit 5: day23_git_commands.sh"

git push origin deploy/production-config

echo ""
echo "=============================================="
echo "  Day 23 COMPLETE!"
echo "=============================================="
echo ""
echo "To test Docker (requires Docker Desktop):"
echo ""
echo "  1. Install Docker Desktop from docker.com"
echo "  2. Run: docker-compose up --build"
echo "  3. Wait ~5 minutes for model download"
echo "  4. Visit: http://localhost (frontend)"
echo "  5. Visit: http://localhost:8000/docs (API)"
echo "  6. Visit: http://localhost:5000 (MLflow)"
echo ""
echo "GitHub → PR deploy/production-config → main → Merge"
echo ""
git log --oneline -6
