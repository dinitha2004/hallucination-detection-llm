#!/bin/bash
# ============================================================
# DAY 8 — Git Commands Script
# Module C: TSV (Truthfulness Separator Vector)
# Branch: feat/module-c-hallushift (continues Day 7 branch)
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 8 — Module C: TSV Training & Steering"
echo "=============================================="
echo ""

# STEP 1: Switch to correct branch
git checkout main
git pull origin main
git checkout -b feat/module-c-tsv
echo "Switched to branch: feat/module-c-tsv"
echo ""

# COMMIT 1: Add TSV trainer implementation
git add backend/modules/tsv_trainer.py
git commit -m "feat(module-c): implement TSV training and latent space steering

- TSVTrainer class with:
    - compute_tsv(): trains LogisticRegression probe on contrastive pairs
      (50 truthful + 50 hallucinated hidden states, AI Assumption A5)
    - apply_tsv_steering(): projects hidden state onto TSV direction
      returns truthfulness score in [0.0, 1.0] via sigmoid
    - get_tsv_deviation(): deviation = 1 - truthfulness (for Module D)
    - generate_synthetic_training_data(): for development/testing
    - save_tsv() / _load_tsv(): persist to data/memory_bank/tsv_vector.npy
- TSV extraction: probe.coef_[0] normalized to unit vector
- Sigmoid scaling: projection × 5.0 for good score separation
- Singleton pattern + auto-save after training
- NFR10: random_state=42 for reproducible training

Research connection:
  Finalized technique Step 3: Apply TSV to steer latent representations
  Score formula: 0.4×entropy + 0.4×wasserstein + 0.2×tsv_deviation
  FR6: apply truthfulness separator vectors"

echo "Commit 1 done: tsv_trainer.py"
echo ""

# COMMIT 2: Add TSV unit tests
git add backend/tests/test_tsv.py
git commit -m "test(module-c): add unit tests for TSV trainer

- TestTSVTrainerSetup: instantiation, singleton, initial state, summary
- TestSyntheticDataGeneration: counts, torch tensors, hidden size,
  reproducibility with seed, truthful != hallucinated
- TestComputeTSV: numpy return, correct shape (64,), unit norm,
  is_trained after compute, empty input graceful handling
- TestApplyTSVSteering: float return, [0,1] range, truthful > hallucinated,
  torch/numpy input, untrained returns 0.5
- TestGetTSVDeviation: complement of score, [0,1] range, hallucinated > truthful
- All tests use hidden_size=64 (fast, same API as real 2048)

NFR10: reproducibility via fixed seed and comprehensive test coverage
Gap 1: TSV projection is scalar float — feeds into Module D formula"

echo "Commit 2 done: test_tsv.py"
echo ""

# COMMIT 3: Add day8 git script
git add day8_git_commands.sh
git commit -m "chore(day8): add Day 8 git commands script"
echo "Commit 3 done: day8_git_commands.sh"
echo ""

# COMMIT 4: Push to GitHub
git push origin feat/module-c-tsv

echo ""
echo "=============================================="
echo "  Day 8 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Tests to run on your Mac:"
echo "  1. Test TSV directly:"
echo "     python backend/modules/tsv_trainer.py"
echo ""
echo "  2. Run unit tests:"
echo "     pytest backend/tests/test_tsv.py -v"
echo ""
echo "  Expected output:"
echo "    Truthful state score:     > 0.5"
echo "    Hallucinated state score: < 0.5"
echo "    TSV saved to data/memory_bank/tsv_vector.npy"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/module-c-tsv to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 9"
echo ""
git log --oneline -8
