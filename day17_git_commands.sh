#!/bin/bash
# ============================================================
# DAY 17 — Git Commands Script
# Evaluation Metrics (FR13)
# Branch: feat/evaluation-pipeline
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 17 — Evaluation Metrics (FR13)"
echo "=============================================="
echo ""

git checkout main
git pull origin main
git checkout feat/evaluation-pipeline 2>/dev/null || git checkout -b feat/evaluation-pipeline
echo "On branch: feat/evaluation-pipeline"
echo ""

# COMMIT 1: Metrics implementation
git add backend/evaluation/metrics.py
git commit -m "feat(eval): implement token-level and span-level evaluation metrics

HallucinationMetrics class:

confusion_matrix_stats(flagged, actually_wrong, total):
  → TP: flagged AND actually wrong
  → FP: flagged BUT NOT wrong (false alarm)
  → TN: not flagged AND not wrong (correct silence)
  → FN: not flagged BUT actually wrong (missed)

token_level_f1(flagged, actually_wrong, total):
  → Precision = TP / (TP + FP)
  → Recall    = TP / (TP + FN)
  → F1        = 2×P×R / (P + R)
  → Accuracy  = (TP + TN) / total
  → Returns TokenLevelResult dataclass

span_precision_recall(detected_spans, correct_wrong_spans):
  → span_hit: any detected span matches a wrong span
  → exact_match: detected spans == correct wrong spans
  → partial_match: Jaccard similarity [0, 1]
  → Returns SpanLevelResult dataclass

aggregate(token_results, span_results):
  → Macro averages (per-sample then average)
  → Micro precision/recall/F1 (aggregate TP/FP/FN first)
  → Span hit rate and exact match rate
  → Hallucination detection rate
  → Returns AggregateMetrics dataclass

save_results(): saves to experiments/results/evaluation_results.json

Addresses:
  FR13: evaluate system detection accuracy on benchmarks
  Used in Days 18+ for TruthfulQA + TriviaQA evaluation"

echo "Commit 1 done: metrics.py"
echo ""

# COMMIT 2: git script
git add day17_git_commands.sh
git commit -m "chore(day17): add Day 17 git commands script"
echo "Commit 2 done: day17_git_commands.sh"
echo ""

# Push
git push origin feat/evaluation-pipeline

echo ""
echo "=============================================="
echo "  Day 17 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Steps to run on your Mac:"
echo ""
echo "  1. Test the metrics:"
echo "     python backend/evaluation/metrics.py"
echo ""
echo "  Expected output:"
echo "    TEST 1: Perfect Detection — F1=1.0 PASS"
echo "    TEST 2: Missed Hallucination — F1=0.0 PASS"
echo "    TEST 3: False Positive — Precision=0 PASS"
echo "    TEST 4: Span-Level — exact match PASS"
echo "    TEST 5: Aggregate — 3 samples PASS"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/evaluation-pipeline to main"
echo "  Merge it, then:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 18"
echo ""
git log --oneline -6
