#!/bin/bash
# ============================================================
# DAY 3 — Git Commands
# Branch: feat/llm-inference-engine (same as Day 2)
# Day 3 plan says: same branch, add inference engine test
# Author: Chalani Dinitha (20211032)
# ============================================================
#
# HOW TO USE:
#   Run each section one at a time after completing each task.
# ============================================================

echo ""
echo "=============================================="
echo "  Day 3 — Token Generation & Inference Engine"
echo "=============================================="
echo ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: Switch to the Day 3 branch
# Day 3 plan says: feat/llm-inference-engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
git checkout main
git pull origin main
git checkout -b feat/day3-inference-test
echo "✓ Switched to branch: feat/day3-inference-test"
echo ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMIT 1: Add inference engine test script
# Run AFTER: python backend/llm/test_inference_engine.py passes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
git add backend/llm/test_inference_engine.py
git commit -m "feat(llm): add Day 3 inference engine test with token logging

- Tokenize prompt and display every token ID + text
- Run model.generate() with return_dict_in_generate=True
- Capture sequences AND hidden_states from model output  
- Implement decode_tokens: convert token IDs back to words
- Log every generated token with ID and cumulative text
- Verify hidden_states present: steps x layers confirmed
- Test TBG (Token Before Generating) position extraction
- Test InferenceEngine class end-to-end with real prompt

Test prompt: 'What is the capital of France?'
Day 3 deliverable: inference engine generates tokens + returns hidden states

Addresses: FR2 (receive tokens during decoding), FR3 (extract hidden states)
Gap 1: hidden states captured DURING generation, not after"

echo "✓ Commit 1 done: test_inference_engine.py"
echo ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMMIT 2: Add Day 3 git commands script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
git add day3_git_commands.sh
git commit -m "chore(day3): add Day 3 git commands script"
echo "✓ Commit 2 done: day3_git_commands.sh"
echo ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: Push to GitHub
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
git push origin feat/day3-inference-test

echo ""
echo "=============================================="
echo "  ✅ Day 3 COMPLETE — All commits pushed!"
echo "=============================================="
echo ""
echo "What you committed:"
echo "  1. backend/llm/test_inference_engine.py"
echo "     - Full token generation test"
echo "     - Every token logged with ID"
echo "     - Hidden states verified"
echo ""
echo "Next steps:"
echo "  1. Go to GitHub → Create Pull Request"
echo "     feat/day3-inference-test → main"
echo "  2. Merge the PR"
echo "  3. Come back and say 'Start Day 4'"
echo ""
git log --oneline -6
