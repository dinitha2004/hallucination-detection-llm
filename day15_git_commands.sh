#!/bin/bash
# ============================================================
# DAY 15 — Git Commands Script
# React Frontend — Config Panel & Polish
# Branch: feat/react-frontend
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 15 — Config Panel & Experiment Log"
echo "=============================================="
echo ""

git checkout feat/react-frontend 2>/dev/null || git checkout -b feat/react-frontend
echo "On branch: feat/react-frontend"
echo ""

# COMMIT 1: New components
git add frontend/src/components/ConfigPanel.jsx \
        frontend/src/components/ExperimentLog.jsx
git commit -m "feat(frontend): add ConfigPanel and ExperimentLog components

ConfigPanel.jsx (FR14):
  - Slider for HALLUCINATION_THRESHOLD (0.10 to 0.90)
  - Slider for SUSPICIOUS_THRESHOLD_LOW (0.10 to 0.80)
  - Calls POST /api/config on Apply Changes
  - Shows current model info (layers, weights, model name)
  - Success feedback: button turns green for 2 seconds
  - Loads current config from GET /api/config on mount

ExperimentLog.jsx:
  - Table of all past queries in this session
  - Columns: prompt, risk score, EAT count, flagged count, ms
  - Risk score color coded (red/yellow/green)
  - Expand/collapse when > 5 entries
  - Shows total query count"

echo "Commit 1 done: ConfigPanel + ExperimentLog"
echo ""

# COMMIT 2: Updated App.jsx
git add frontend/src/App_updated.jsx
git commit -m "feat(frontend): update App with tabs, config panel, experiment log

- Three-tab navigation: Results | Config | Log(N)
- Results tab: full detection flow with error handling
- Config tab: ConfigPanel with live threshold sliders
- Log tab: ExperimentLog showing all session queries
- Improved error display with Pipeline step info
- History state: every query added to experiment log
- Better loading message showing pipeline steps
- Backend offline detection with helpful message"

echo "Commit 2 done: Updated App.jsx"
echo ""

# COMMIT 3: git script
git add day15_git_commands.sh
git commit -m "chore(day15): add Day 15 git commands script"
echo "Commit 3 done: day15_git_commands.sh"
echo ""

# Push
git push origin feat/react-frontend

echo ""
echo "=============================================="
echo "  Day 15 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "IMPORTANT: Replace App.jsx with App_updated.jsx:"
echo "  cp frontend/src/App_updated.jsx frontend/src/App.jsx"
echo ""
echo "Then restart the frontend:"
echo "  cd frontend && npm run dev"
echo ""
echo "New features at http://localhost:3000:"
echo "  - Click 'config' tab → threshold sliders"
echo "  - Click 'log(N)' tab → experiment history"
echo "  - Better error messages"
echo "  - Full loading state with pipeline steps"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/react-frontend to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 16"
echo ""
git log --oneline -8
