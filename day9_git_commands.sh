#!/bin/bash
# ============================================================
# DAY 9 — Git Commands Script
# Module D: Semantic Entropy & Score Aggregation
# Branch: feat/module-d-scoring
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 9 — Module D: Scoring Engine"
echo "=============================================="
echo ""

# STEP 1: Create new branch from main
git checkout main
git pull origin main
git checkout -b feat/module-d-scoring
echo "Switched to branch: feat/module-d-scoring"
echo ""

# COMMIT 1: Module D scoring engine
git add backend/modules/module_d_scoring.py
git commit -m "feat(module-d): implement semantic entropy and score aggregation

- calculate_semantic_entropy(): H = -sum(p*log(p)) over softmax
  of hidden state absolute values, normalized to [0,1]
- aggregate_hallucination_score(): weighted formula
  score = 0.4×entropy + 0.4×wasserstein_norm + 0.2×tsv_deviation
- apply_threshold(): 3-zone classification
    SAFE:         score < 0.45
    SUSPICIOUS:   0.45 ≤ score < 0.65
    HALLUCINATED: score ≥ 0.65
- score_token(): complete token analysis → TokenScore dataclass
- score_all_tokens(): batch scoring for full generated sequence
- get_overall_risk(): max score across all tokens
- update_threshold(): runtime config update (FR14)
- TokenScore dataclass: token, position, entropy, wasserstein,
  tsv_deviation, hallucination_score, is_flagged, is_eat, risk_level
- Singleton pattern

Addresses:
  FR4: analyse semantic entropy
  FR7: combine signals into unified hallucination score
  FR8: detect hallucinations during generation
  FR12: display confidence score for flagged content
  FR14: configurable threshold"

echo "Commit 1 done: module_d_scoring.py"
echo ""

# COMMIT 2: Unit tests
git add backend/tests/test_module_d.py
git commit -m "test(module-d): add unit tests for scoring engine

- TestSemanticEntropy: float, [0,1], uniform>peaked, empty=0.5, multilayer
- TestAggregateHallucinationScore: [0,1] always, high>0.5, low<0.5,
  float return, weight formula verified
- TestApplyThreshold: hallucinated/safe/suspicious zones, tuple return,
  at-threshold behaviour
- TestScoreToken: TokenScore type, correct fields, [0,1] score,
  is_eat flag, to_dict serializable
- TestTokenScoreDataclass: properties, default risk level
- TestScoringEngineConfig: threshold update, invalid ignored, get_config,
  overall risk empty, singleton

NFR10: reproducibility through comprehensive test coverage"

echo "Commit 2 done: test_module_d.py"
echo ""

# COMMIT 3: git script
git add day9_git_commands.sh
git commit -m "chore(day9): add Day 9 git commands script"
echo "Commit 3 done: day9_git_commands.sh"
echo ""

# COMMIT 4: Push
git push origin feat/module-d-scoring

echo ""
echo "=============================================="
echo "  Day 9 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Tests to run on your Mac:"
echo "  1. Test Module D directly:"
echo "     python backend/modules/module_d_scoring.py"
echo ""
echo "  2. Run unit tests:"
echo "     pytest backend/tests/test_module_d.py -v"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/module-d-scoring to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 10"
echo ""
git log --oneline -8
