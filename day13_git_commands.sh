#!/bin/bash
# ============================================================
# DAY 13 — Git Commands Script
# React Frontend Setup & Layout
# Branch: feat/react-frontend
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 13 — React Frontend Setup & Layout"
echo "=============================================="
echo ""

# STEP 1: Create new branch from main
git checkout main
git pull origin main
git checkout -b feat/react-frontend
echo "Switched to branch: feat/react-frontend"
echo ""

# COMMIT 1: Project setup files
git add frontend/package.json frontend/vite.config.js \
        frontend/tailwind.config.js frontend/postcss.config.js \
        frontend/index.html
git commit -m "feat(frontend): initialize Vite + React + Tailwind project

- package.json: react 18, axios, lucide-react, tailwindcss, vite
- vite.config.js: dev server on port 3000, proxy /api → port 8000
- tailwind.config.js: custom brand/danger/warning/safe colors,
  JetBrains Mono + DM Sans fonts
- postcss.config.js: tailwind + autoprefixer
- index.html: Google Fonts import (DM Sans, JetBrains Mono)"

echo "Commit 1 done: project config files"
echo ""

# COMMIT 2: Core app files
git add frontend/src/main.jsx \
        frontend/src/styles/globals.css \
        frontend/src/api/client.js \
        frontend/src/App.jsx
git commit -m "feat(frontend): add app entry, global styles, API client, layout

- main.jsx: React 18 root render
- globals.css: Tailwind directives + token risk CSS utilities
    .token-hallucinated: red highlight + ring
    .token-suspicious: yellow highlight
    .token-safe: transparent (no highlight)
- client.js: axios wrapper for all 4 API endpoints
    detectHallucination(), getHealth(), getConfig(), updateConfig()
    120s timeout for slow CPU inference
- App.jsx: two-panel layout
    LEFT:  QueryInput (sidebar, 384px)
    RIGHT: Results area (TokenDisplay + ScorePanel + WarningBanner)
    Header: HalluScan brand + API ready/offline status pill
    States: checking, ready, loading, error, results"

echo "Commit 2 done: app entry + layout"
echo ""

# COMMIT 3: Components
git add frontend/src/components/QueryInput.jsx \
        frontend/src/components/TokenDisplay.jsx \
        frontend/src/components/ScorePanel.jsx \
        frontend/src/components/WarningBanner.jsx
git commit -m "feat(frontend): add all UI components

QueryInput.jsx:
  - Textarea with ⌘+Enter shortcut
  - Max tokens slider (10-100)
  - Detect Hallucination submit button with loading state
  - 5 example prompts clickable

TokenDisplay.jsx (Gap 2 visualisation):
  - Renders each token with risk-level colour coding
  - Hover tooltip: score, entity_type, risk_level
  - EAT token chips summary below output
  - Only EAT tokens shown in red/yellow (key novelty)

ScorePanel.jsx:
  - Overall risk progress bar with colour coding
  - Scoring formula display: 0.4×entropy + 0.4×wass + 0.2×tsv
  - Stats grid: Flagged / EAT Tokens / Total Tokens
  - EAT token score table with per-token breakdown

WarningBanner.jsx:
  - Red alert shown only when hallucination_detected=true
  - Lists flagged token names
  - Shows overall risk score"

echo "Commit 3 done: all components"
echo ""

# COMMIT 4: git script
git add day13_git_commands.sh
git commit -m "chore(day13): add Day 13 git commands script"
echo "Commit 4 done: day13_git_commands.sh"
echo ""

# Push to GitHub
git push origin feat/react-frontend

echo ""
echo "=============================================="
echo "  Day 13 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "To run the React frontend:"
echo "  cd frontend"
echo "  npm install"
echo "  npm run dev"
echo ""
echo "Then visit: http://localhost:3000"
echo ""
echo "Make sure FastAPI is ALSO running:"
echo "  uvicorn backend.main:app --reload --port 8000"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/react-frontend to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 14"
echo ""
git log --oneline -8
