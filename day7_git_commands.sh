#!/bin/bash
# ============================================================
# DAY 7 — Git Commands Script
# Module C: HalluShift Analyzer
# Branch: feat/module-c-hallushift
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 7 — Module C: HalluShift Analyzer"
echo "=============================================="
echo ""

# STEP 1: Create new branch from main
git checkout main
git pull origin main
git checkout -b feat/module-c-hallushift
echo "Switched to branch: feat/module-c-hallushift"
echo ""

# COMMIT 1: Add Module C HalluShift implementation
git add backend/modules/module_c_hallushift.py
git commit -m "feat(module-c): implement HalluShift distribution shift analyzer

- calculate_wasserstein_distance(): uses scipy.stats.wasserstein_distance
  between consecutive layer hidden state distributions
- calculate_cosine_similarity(): directional agreement between layers
  using dot product / (norm_a * norm_b)
- compute_distribution_shift(): sliding window (size=2) across layers
  for one token position → {(layer_i, layer_j): ShiftScore}
- compute_all_token_shifts(): processes all generated tokens
- get_max_shift_for_token(): max magnitude across all layer pairs
- get_average_wasserstein(): avg Wasserstein for Module D formula
- ShiftScore dataclass: layer_pair, wasserstein, cosine, magnitude
- Singleton pattern: one analyzer instance per session

Layer pairs analyzed for [18, 20, 22]:
  Window 1: (18, 20) → ShiftScore
  Window 2: (20, 22) → ShiftScore

Addresses:
  Gap 1: distribution shifts detect hallucination fingerprints
  FR5: analyse distribution shifts across decoding steps
  Scoring formula: 0.4 × wasserstein component"

echo "Commit 1 done: module_c_hallushift.py"
echo ""

# COMMIT 2: Add unit tests
git add backend/tests/test_module_c.py
git commit -m "test(module-c): add unit tests for HalluShift analyzer

- TestWassersteinDistance: positive, zero for same vector, float type,
  similar < different, numpy arrays accepted
- TestCosineSimilarity: same=1, opposite=-1, perpendicular=0,
  always in [-1,1], zero vector handled
- TestDistributionShift: correct pairs, ShiftScore objects,
  values in valid ranges, empty/single layer edge cases
- TestAllTokenShifts: length matches tokens, max/avg utilities
- TestShiftScoreDataclass: creation, is_high_shift, singleton

All tests use dummy tensors — no model loading needed
NFR10: reproducibility through comprehensive test coverage"

echo "Commit 2 done: test_module_c.py"
echo ""

# COMMIT 3: Add day7 git script
git add day7_git_commands.sh
git commit -m "chore(day7): add Day 7 git commands script"
echo "Commit 3 done: day7_git_commands.sh"
echo ""

# COMMIT 4: Push to GitHub
git push origin feat/module-c-hallushift

echo ""
echo "=============================================="
echo "  Day 7 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Tests to run on your Mac:"
echo "  1. Test Module C directly:"
echo "     python backend/modules/module_c_hallushift.py"
echo ""
echo "  2. Run unit tests:"
echo "     pytest backend/tests/test_module_c.py -v"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/module-c-hallushift to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 8"
echo ""
git log --oneline -8
