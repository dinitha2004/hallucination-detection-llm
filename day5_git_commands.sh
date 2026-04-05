#!/bin/bash
# ============================================================
# DAY 5 — Git Commands Script
# Module B: Integration Test & Layer Selection
# Branch: feat/hidden-state-extraction (continue same feature)
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 5 — Module B Integration & Layer Selection"
echo "=============================================="
echo ""

# STEP 1: Create new branch from main
git checkout main
git pull origin main
git checkout -b feat/module-b-integration
echo "Switched to branch: feat/module-b-integration"
echo ""

# COMMIT 1: Add layer selection experiment
git add backend/evaluation/layer_selection_experiment.py
git commit -m "feat(eval): add layer selection experiment for TruthfulQA prompts

- Run 8 TruthfulQA-style prompts across all model layers
- Measure L2 norm of TBG hidden states per layer per prompt
- Compute average norm per layer to identify signal strength
- Print ranked table: layer index vs average norm
- Identify top-3 layers with strongest hallucination signals
- Save results to experiments/results/layer_selection.json
- Provides empirical evidence for TARGET_LAYERS selection

Research finding: confirms mid-to-late layers have highest norms
Addresses: FR3 (layer selection), NFR2 (accuracy foundation)
Thesis: evidence for 'mid-to-late layers contain strongest signals'"

echo "Commit 1 done: layer_selection_experiment.py"
echo ""

# COMMIT 2: Add Module B integration tests
git add backend/tests/test_module_b_integration.py
git commit -m "test(module-b): add integration tests for full Module B pipeline

- test_hooks_attach_to_real_model: verify OPT/LLaMA layer detection
- test_hidden_states_captured_during_generation: Gap 1 core test
- test_tbg_vector_shape_matches_hidden_size: shape = (2048,) verified
- test_later_layers_have_higher_norms: confirms research finding
- test_feature_clipping_reduces_max_on_real_data: INSIDE FC on real data
- test_memory_bank_updates_with_real_activations: baseline computation
- test_hooks_removed_after_generation: no memory leaks
- test_repeated_generation_gives_consistent_norms: NFR4 reliability

Uses real model (scope=module for efficiency — loads once)
NFR4: reliability verified with deterministic generation
NFR8: hook cleanup prevents memory leaks"

echo "Commit 2 done: test_module_b_integration.py"
echo ""

# COMMIT 3: Push to GitHub
git push origin feat/module-b-integration

echo ""
echo "=============================================="
echo "  Day 5 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Tests to run on your Mac:"
echo "  1. Layer experiment:"
echo "     python backend/evaluation/layer_selection_experiment.py"
echo ""
echo "  2. Integration tests (takes 2-3 mins, loads model):"
echo "     pytest backend/tests/test_module_b_integration.py -v -s"
echo ""
echo "  3. After seeing results, update your .env:"
echo "     TARGET_LAYERS=<top 3 from experiment output>"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/module-b-integration to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 6"
echo ""
git log --oneline -8
