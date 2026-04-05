#!/bin/bash
# ============================================================
# DAY 4 — Git Commands Script
# Module B: Hidden State Extraction
# Branch: feat/hidden-state-extraction
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 4 — Module B: Hidden State Extraction"
echo "=============================================="
echo ""

# STEP 1: Create new branch from main
git checkout main
git pull origin main
git checkout -b feat/hidden-state-extraction
echo "Switched to branch: feat/hidden-state-extraction"
echo ""

# COMMIT 1: Add Module B core implementation
git add backend/modules/module_b_hidden.py
git commit -m "feat(module-b): implement hidden state extraction with TBG probing

- Attach PyTorch register_forward_hook() to target layers (12, 16, 20)
- Implement HiddenStateExtractor class with:
    - attach_hooks(): secret listeners on target layers
    - extract_tbg_hidden_state(): TBG position per token per layer
    - extract_layer_activations(): all captured states by layer
    - get_all_tbg_vectors_for_token(): cross-layer profile per token
- Implement INSIDE Feature Clipping (apply_inside_feature_clipping):
    - Clips extreme activations at 95th percentile threshold
    - Suppresses overconfident hallucinations (from INSIDE paper)
    - Uses memory bank baseline when available
- Implement Memory Bank:
    - Stores reference activations from truthful answers
    - Computes per-layer clipping thresholds
    - Persists to disk for reuse across sessions
- Singleton pattern: one extractor instance per session

Addresses:
  Gap 1: detection during generation via TBG probing
  FR3: extract hidden states from selected LLM layers
  NFR8: efficient resource usage via selective layer extraction"

echo "Commit 1 done: module_b_hidden.py"
echo ""

# COMMIT 2: Add unit tests for Module B
git add backend/tests/test_module_b.py
git commit -m "test(module-b): add comprehensive unit tests for hidden state extraction

- TestHiddenStateExtractorUnit: instantiation, TBG extraction, shapes
- TestInsideFeatureClipping: clipping reduces extremes, shape preserved,
  None input handled, 95th percentile verified, memory bank thresholds
- TestMemoryBank: truthful update, threshold computation after update
- TestGetSummary: structure, initial values, singleton pattern
- All tests run WITHOUT loading model (fast, CI-friendly)

NFR10: Reproducibility - test suite ensures consistent module behaviour"

echo "Commit 2 done: test_module_b.py"
echo ""

# COMMIT 3: Push to GitHub
git push origin feat/hidden-state-extraction

echo ""
echo "=============================================="
echo "  Day 4 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/hidden-state-extraction to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then run the full test:"
echo "  python backend/modules/module_b_hidden.py"
echo "  pytest backend/tests/test_module_b.py -v"
echo ""
git log --oneline -8
