#!/bin/bash
# ============================================================
# DAY 2 — Git Commands Script
# Run each section one at a time as you complete each task
# Author: Chalani Dinitha (20211032)
# ============================================================
#
# HOW TO USE:
#   Open Terminal in your project folder.
#   Copy and paste each SECTION below one at a time.
#   Each section = one commit to GitHub.
# ============================================================

echo ""
echo "=============================================="
echo "  Day 2 — LLM Model Loading Git Commits"
echo "=============================================="
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIRST: Make sure Day 1 is merged into main
# Run this only if you haven't merged Day 1 yet
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# git checkout main
# git merge setup/project-structure
# git push origin main


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: Create and switch to Day 2 branch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
git checkout main
git checkout -b feat/llm-inference-engine
echo "✓ Switched to branch: feat/llm-inference-engine"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMIT 1: Add model_loader.py
# Do this after you have model_loader.py in place
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
git add backend/llm/model_loader.py
git commit -m "feat(llm): add ModelLoader class with hidden state output enabled

- Load LLM using AutoModelForCausalLM with output_hidden_states=True
- Auto-detect device (CPU/CUDA)
- Validate TARGET_LAYERS against model's actual layer count
- Singleton pattern: model loaded once, reused everywhere
- Supports OPT-1.3b (default) and LLaMA-3.2-3B-Instruct
- Handles missing pad_token gracefully (OPT quirk)

Addresses: Gap 1 - real-time hallucination detection requires
access to internal hidden states during generation
FR3: system shall extract hidden states from selected LLM layers"

echo "✓ Commit 1 done: model_loader.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMIT 2: Add inference_engine.py
# Do this after you have inference_engine.py in place
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
git add backend/llm/inference_engine.py
git commit -m "feat(llm): add InferenceEngine for token generation and hidden state capture

- Run model.generate() with return_dict_in_generate=True
- Capture hidden states for every generated token across TARGET_LAYERS
- Implement TBG (Token Before Generating) position extraction
- Return structured GenerationResult dataclass with:
    - generated_text, tokens, token_ids
    - hidden_states_by_layer: {layer_idx: [tensor per token]}
    - generation_time_ms
- Uses torch.no_grad() for memory-efficient inference
- Individual token decoding for span-level mapping (Gap 2)

Addresses: FR2, FR3, FR8 - token generation with hidden state capture"

echo "✓ Commit 2 done: inference_engine.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMIT 3: Add unit tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
git add backend/tests/test_model_loader.py
git commit -m "test(llm): add unit tests for ModelLoader and GenerationResult

- Test ModelLoader instantiation and singleton pattern
- Test error raised before model is loaded (prevents pipeline bugs)
- Test config values have correct types
- Test weight sum equals 1.0 (calibration guard for scoring formula)
- Test GenerationResult dataclass: success, num_tokens, hidden states
- All tests run without downloading any model weights (fast CI-friendly)

NFR10: Reproducibility - tests ensure consistent behaviour"

echo "✓ Commit 3 done: test_model_loader.py"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMIT 4: Push branch to GitHub
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
git push origin feat/llm-inference-engine

echo ""
echo "=============================================="
echo "  ✅ Day 2 COMPLETE — All commits pushed!"
echo "=============================================="
echo ""
echo "What you just committed:"
echo "  1. backend/llm/model_loader.py  — LLM loading with hidden states"
echo "  2. backend/llm/inference_engine.py — Token generation + state capture"
echo "  3. backend/tests/test_model_loader.py — Unit tests"
echo ""
echo "Next: Go to GitHub and create a Pull Request:"
echo "  feat/llm-inference-engine → main"
echo ""
echo "Then come back and say 'Start Day 3'"
echo ""
git log --oneline -5
