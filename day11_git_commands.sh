#!/bin/bash
# ============================================================
# DAY 11 — Git Commands Script
# Detection Pipeline Orchestrator
# Branch: feat/fastapi-backend
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 11 — Detection Pipeline Orchestrator"
echo "=============================================="
echo ""

# STEP 1: Create new branch from main
git checkout main
git pull origin main
git checkout -b feat/fastapi-backend
echo "Switched to branch: feat/fastapi-backend"
echo ""

# COMMIT 1: Detection pipeline
git add backend/pipeline/detection_pipeline.py
git commit -m "feat(pipeline): implement full detection pipeline orchestrator

Complete 5-step hallucination detection pipeline:
  Step 1: Module A — EAT detection on prompt + generated text
  Step 2: Inference Engine — generate tokens + attach hooks
  Step 3: Module B — extract TBG states + INSIDE feature clipping
  Step 4: Module C — Wasserstein shifts + TSV deviation per token
  Step 5: Module D — entropy + aggregate scores + span-level mapping

DetectionPipeline class:
  - initialize(): load model + all modules, train TSV if needed
  - run(prompt): execute full pipeline → DetectionOutput
  - get_config(): return current pipeline configuration
  - update_threshold(): runtime threshold update (FR14)
  - Singleton pattern: one instance per server session

Key design decisions:
  - Hooks attached before generation, removed immediately after
  - do_sample=False for deterministic output (NFR4)
  - TSV auto-trained with synthetic data if no saved vector
  - EAT detection on BOTH prompt and generated text

Addresses ALL functional requirements:
  FR1-FR12, Gap 1 (during generation), Gap 2 (EAT-only flagging)
  NFR1 (real-time), NFR4 (reliability), NFR8 (resource efficiency)"

echo "Commit 1 done: detection_pipeline.py"
echo ""

# COMMIT 2: Pipeline tests
git add backend/tests/test_pipeline.py
git commit -m "test(pipeline): add pipeline unit tests

- TestDetectionPipelineUnit: instantiation, singleton, run before init,
  threshold update before init, get_config before init
- TestDetectionOutputStructure: to_dict fields, AnnotatedToken API format,
  flagged_tokens property, eat_tokens property
- TestPipelineStepsLogic: EAT detection step, scoring step,
  span mapping EAT-only, NFR1 processing time, NFR4 reliability,
  FR14 configurable threshold

All tests run without loading model (fast unit tests)
NFR10: comprehensive test coverage for pipeline logic"

echo "Commit 2 done: test_pipeline.py"
echo ""

# COMMIT 3: git script
git add day11_git_commands.sh
git commit -m "chore(day11): add Day 11 git commands script"
echo "Commit 3 done: day11_git_commands.sh"
echo ""

# Run all tests
echo "Running all tests..."
pytest backend/tests/ -q --ignore=backend/tests/test_module_b_integration.py 2>&1 | tail -3
echo ""

# Push to GitHub
git push origin feat/fastapi-backend

echo ""
echo "=============================================="
echo "  Day 11 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Tests to run on your Mac:"
echo "  1. Test pipeline end-to-end (loads real model ~2 min):"
echo "     python backend/pipeline/detection_pipeline.py"
echo ""
echo "  2. Run unit tests (fast, no model):"
echo "     pytest backend/tests/test_pipeline.py -v"
echo ""
echo "  3. Run ALL tests:"
echo "     pytest backend/tests/ -v"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/fastapi-backend to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 12"
echo ""
git log --oneline -8
