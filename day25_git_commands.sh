#!/bin/bash
# ============================================================
# DAY 25 — Git Commands Script
# Full Benchmark Evaluation
# Branch: research/ablation-study
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 25 — Full Benchmark Evaluation"
echo "=============================================="
echo ""

git checkout main
git pull origin main
git checkout research/ablation-study 2>/dev/null || git checkout -b research/ablation-study
echo "On branch: research/ablation-study"
echo ""

# COMMIT 1: Full evaluator
git add backend/evaluation/full_evaluation.py
git commit -m "research(eval): add full benchmark evaluator for TruthfulQA + TriviaQA

FullEvaluator class:
  run_full_evaluation(n_truthfulqa=200, n_triviaqa=100):
    → Loads both datasets
    → Runs real pipeline on each sample (threshold=0.35)
    → Computes token-level: Macro/Micro P, R, F1, Accuracy
    → Computes span-level: hit rate, detection rate
    → Measures: avg latency, TP, FP, FN per dataset
    → Saves: experiments/results/final_evaluation.json
    → Generates: experiments/results/final_evaluation_report.md

Markdown report includes:
  - Table 1: Token-level metrics (Macro + Micro)
  - Table 2: Detection performance + latency
  - Comparison to SelfCheckGPT, INSIDE baselines
  - Notes on OPT-1.3b threshold calibration

Usage (real model, ~3 hours):
  evaluator.initialize_pipeline()
  evaluator.run_full_evaluation(n_truthfulqa=200, n_triviaqa=100)"

echo "Commit 1: full_evaluation.py"
echo ""

# COMMIT 2: Commit existing real results (from n=817 run)
git add experiments/results/ 2>/dev/null || true
git commit -m "research(results): add complete evaluation results

experiments/results/:
  hallucination_detection_baseline.json:
    - Full TruthfulQA n=817 with real OPT-1.3b
    - Macro F1: 0.0543
    - Micro F1: 0.0990
    - Detection Rate: 0.3341
    - Avg Latency: 2351.7ms
    - Total TP: 261, FP: 812, FN: 3940

  ablation_results.json:
    - 4-condition ablation study
    - Conditions: full, no_tsv, no_eat, no_clip

  ablation_comparison.png:
    - Bar chart comparing conditions" 2>/dev/null || echo "No new results to commit"

echo "Commit 2: experiments/results/"
echo ""

# COMMIT 3: git script
git add day25_git_commands.sh
git commit -m "chore(day25): add Day 25 git commands script"
echo "Commit 3: day25_git_commands.sh"
echo ""

git push origin research/ablation-study

echo ""
echo "=============================================="
echo "  Day 25 COMPLETE!"
echo "=============================================="
echo ""
echo "To run full evaluation with real model (~3 hours):"
echo ""
echo "  Edit backend/evaluation/full_evaluation.py"
echo "  Uncomment: evaluator.initialize_pipeline()"
echo "  Then run:"
echo "  caffeinate -i python backend/evaluation/full_evaluation.py"
echo ""
echo "This produces:"
echo "  experiments/results/final_evaluation.json"
echo "  experiments/results/final_evaluation_report.md"
echo ""
echo "GitHub → PR research/ablation-study → main → Merge"
echo ""
git log --oneline -6
