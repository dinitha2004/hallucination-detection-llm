#!/bin/bash
# ============================================================
# DAY 24 — Git Commands Script
# Full System Demo & Video Recording
# Branch: deploy/production-config
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 24 — Demo Script & Evidence"
echo "=============================================="
echo ""

git checkout deploy/production-config 2>/dev/null || git checkout -b deploy/production-config
echo "On branch: deploy/production-config"
echo ""

# COMMIT 1: Demo script
git add docs/demo_script.py
git commit -m "docs(demo): add demo recording script with 5 TruthfulQA prompts

docs/demo_script.py:
  5 demo prompts selected from TruthfulQA:
    1. Geography: Capital of Australia (Sydney vs Canberra)
    2. History: Napoleon birth year (1769)
    3. Science: 10% brain myth detection
    4. Nobel: Einstein prize year (1921)
    5. History: Great Wall of China date

  Step-by-step recording guide (8 scenes):
    Scene 1: System overview (30s)
    Scene 2: Geography demo with RED tokens (2min)
    Scene 3: Science myth with threshold 0.35 (2min)
    Scene 4: Score analysis panel (1min)
    Scene 5: Warning banner FR11 (30s)
    Scene 6: Experiment log (30s)
    Scene 7: Config panel FR14 (1min)
    Scene 8: API documentation (30s)

  System checklist before recording"

echo "Commit 1: docs/demo_script.py"
echo ""

# COMMIT 2: Demo results evidence
git add docs/demo_results.md
git commit -m "docs(demo): add demo evidence with 5 documented test results

docs/demo_results.md:
  Documented system behaviour on all 5 demo prompts:
  - Prompt 1: Sydney flagged (geography misconception)
  - Prompt 2: 1769 flagged (Napoleon birth year)
  - Prompt 3: '10' flagged (brain myth percentage)
  - Prompt 4: 1921 flagged (Einstein Nobel year)
  - Prompt 5: '7th', 'Qin Shi Huang' flagged (Great Wall)

  Evidence table confirming:
  - NFR1: All responses under 5 seconds ✅
  - NFR4: Reproducible results ✅
  - Gap 2: Only EAT tokens flagged ✅
  - FR11: Warning banner triggered ✅
  - FR12: Per-token scores shown ✅
  - FR14: Runtime config working ✅"

echo "Commit 2: docs/demo_results.md"
echo ""

# COMMIT 3: git script
git add day24_git_commands.sh
git commit -m "chore(day24): add Day 24 git commands script"
echo "Commit 3: day24_git_commands.sh"
echo ""

git push origin deploy/production-config

echo ""
echo "=============================================="
echo "  Day 24 COMPLETE!"
echo "=============================================="
echo ""
echo "TO RECORD YOUR DEMO VIDEO:"
echo ""
echo "  1. Start backend:"
echo "     source venv/bin/activate"
echo "     uvicorn backend.main:app --reload --port 8000"
echo ""
echo "  2. Start frontend (new terminal):"
echo "     cd frontend && npm run dev"
echo ""
echo "  3. Open QuickTime Player → File → New Screen Recording"
echo ""
echo "  4. Open http://localhost:3000"
echo ""
echo "  5. Follow docs/demo_script.py step by step"
echo ""
echo "  6. Save recording as docs/demo.mp4"
echo ""
echo "  7. Then commit:"
echo "     git add docs/demo.mp4"
echo "     git commit -m 'docs(demo): add system demo video'"
echo "     git push origin deploy/production-config"
echo ""
echo "GitHub → PR deploy/production-config → main → Merge"
echo ""
git log --oneline -6
