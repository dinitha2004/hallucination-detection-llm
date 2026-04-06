#!/bin/bash
# ============================================================
# DAY 10 — Git Commands Script
# Module D: Span-Level Mapper & Output Renderer
# Branch: feat/module-d-scoring (continues Day 9)
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 10 — Span-Level Mapper & Output Renderer"
echo "=============================================="
echo ""

# STEP 1: Switch to correct branch
git checkout main
git pull origin main
git checkout -b feat/module-d-span-mapper
echo "Switched to branch: feat/module-d-span-mapper"
echo ""

# COMMIT 1: Span-level mapper implementation
git add backend/modules/span_level_mapper.py
git commit -m "feat(module-d): implement span-level mapper and output renderer

KEY NOVELTY of Gap 2:
  span_level_mapper() flags ONLY tokens that are BOTH:
    1. An EAT (Exact Answer Token from Module A)
    2. Above hallucination threshold (from Module D scoring)
  Standard approach: flags any token above threshold
  This approach: flags only factual entities above threshold

Functions:
- span_level_mapper(): takes EAT positions + TokenScores, applies
  EAT filter — non-EAT tokens are NEVER flagged regardless of score
- build_annotated_output(): assembles DetectionOutput for API
- format_highlighted_text(): [TOKEN*] marking for logs/display
- AnnotatedToken dataclass: token, score, risk, is_eat, is_flagged
- DetectionOutput dataclass: full API response structure
  with to_dict() for JSON serialization

Addresses:
  Gap 2: fine-grained span-level hallucination localization
  FR9: identify exact hallucinated token or span
  FR10: highlight detected hallucinated part in output
  FR11: warning when hallucinated content detected
  FR12: confidence score for flagged content"

echo "Commit 1 done: span_level_mapper.py"
echo ""

# COMMIT 2: Tests
git add backend/tests/test_span_mapper.py
git commit -m "test(module-d): add span-level mapper tests + end-to-end pipeline test

- TestSpanLevelMapper: key novelty (EAT+threshold only), non-EAT never
  flagged, EAT with low score not flagged, empty EATs, all flagged
- TestBuildAnnotatedOutput: DetectionOutput type, num_flagged, detected
  true/false, to_dict, processing_time, flagged_tokens property
- TestFormatHighlightedText: flagged marked, returns string
- TestAnnotatedTokenDataclass: should_highlight, to_dict keys, singleton
- TestEndToEnd: full pipeline from hidden states to annotated output
  verifies non-EAT tokens never flagged, structure correct

This is the complete Module D test suite (Day 9 + Day 10 tests)"

echo "Commit 2 done: test_span_mapper.py"
echo ""

# COMMIT 3: git script
git add day10_git_commands.sh
git commit -m "chore(day10): add Day 10 git commands script"
echo "Commit 3 done: day10_git_commands.sh"
echo ""

# COMMIT 4: Run all tests to confirm everything still passes
echo "Running all tests before push..."
pytest backend/tests/ -q 2>&1 | tail -5
echo ""

# COMMIT 5: Push to GitHub
git push origin feat/module-d-span-mapper

echo ""
echo "=============================================="
echo "  Day 10 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Tests to run on your Mac:"
echo "  1. Test Span Mapper directly:"
echo "     python backend/modules/span_level_mapper.py"
echo ""
echo "  2. Run ALL tests (all modules):"
echo "     pytest backend/tests/ -v"
echo ""
echo "  Expected: ALL tests passing across all modules!"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/module-d-span-mapper to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 11"
echo ""
git log --oneline -8
