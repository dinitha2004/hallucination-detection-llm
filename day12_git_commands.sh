#!/bin/bash
# ============================================================
# DAY 12 — Git Commands Script
# FastAPI Server & Endpoints
# Branch: feat/fastapi-backend (continues Day 11)
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 12 — FastAPI Server & Endpoints"
echo "=============================================="
echo ""

# STEP 1: Stay on feat/fastapi-backend branch
git checkout feat/fastapi-backend 2>/dev/null || {
    git checkout main
    git pull origin main
    git checkout -b feat/fastapi-backend
}
echo "On branch: feat/fastapi-backend"
echo ""

# COMMIT 1: Pydantic schemas
git add backend/schemas.py
git commit -m "feat(api): add Pydantic request and response schemas

- PromptRequest: prompt (1-2000 chars) + max_new_tokens (10-500)
- ConfigUpdateRequest: hallucination_threshold, suspicious_threshold
- AnnotatedTokenResponse: token, position, score, risk_level, is_eat,
  is_flagged, entity_type
- DetectionResponse: full annotated output for React frontend
- HealthResponse: server + pipeline status
- ConfigResponse: current thresholds and weights
- ConfigUpdateResponse: confirmation of config change

All models use Pydantic v2 Field with descriptions for /docs
FR12: confidence score displayed for flagged content
FR14: configurable thresholds via request model"

echo "Commit 1 done: schemas.py"
echo ""

# COMMIT 2: FastAPI main application
git add backend/main.py
git commit -m "feat(api): implement FastAPI server with all endpoints

Endpoints:
  GET  /              → API root and links
  POST /api/detect    → run hallucination detection pipeline
  GET  /api/health    → server + pipeline status check
  GET  /api/config    → current threshold and weight config
  POST /api/config    → update thresholds at runtime (FR14)

Features:
  - CORS middleware for React frontend (localhost:3000)
  - Lifespan context: pipeline.initialize() on server startup
  - Auto-generated /docs (Swagger UI)
  - 503 if pipeline not ready, 400 if empty prompt
  - DetectionResponse maps all AnnotatedToken fields
  - Async endpoints for non-blocking handling

Addresses:
  FR1: accept user prompt as input
  FR7-FR12: detection results returned to frontend
  FR14: POST /api/config for runtime threshold update
  NFR1: async server for real-time performance
  NFR5: prompt not logged (LOG_PROMPTS=false default)"

echo "Commit 2 done: main.py"
echo ""

# COMMIT 3: git script
git add day12_git_commands.sh
git commit -m "chore(day12): add Day 12 git commands script"
echo "Commit 3 done: day12_git_commands.sh"
echo ""

# Push to GitHub
git push origin feat/fastapi-backend

echo ""
echo "=============================================="
echo "  Day 12 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "To start the server:"
echo "  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Then visit:"
echo "  http://localhost:8000/docs   ← Swagger UI (auto-generated)"
echo "  http://localhost:8000/api/health"
echo ""
echo "Test with curl:"
echo "  curl http://localhost:8000/api/health"
echo '  curl -X POST http://localhost:8000/api/detect \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"prompt": "Einstein was born in 1879"}'"'"''
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/fastapi-backend to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 13"
echo ""
git log --oneline -8
