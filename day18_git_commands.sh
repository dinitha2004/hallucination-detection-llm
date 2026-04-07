#!/bin/bash
# ============================================================
# DAY 18 — Git Commands Script
# MLflow Experiment Runner (NFR10)
# Branch: feat/evaluation-pipeline
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 18 — MLflow Experiment Runner"
echo "=============================================="
echo ""

git checkout main
git pull origin main
git checkout feat/evaluation-pipeline 2>/dev/null || git checkout -b feat/evaluation-pipeline
echo "On branch: feat/evaluation-pipeline"
echo ""

# COMMIT 1: Experiment runner
git add backend/evaluation/experiment_runner.py
git commit -m "feat(eval): add MLflow experiment runner with parameter and metric logging

ExperimentRunner class:
  initialize_pipeline(): loads detection pipeline
  run_experiment(config): full evaluation loop
    → loops over TruthfulQA/TriviaQA samples
    → runs pipeline.run(question) per sample
    → checks answer correctness vs incorrect_answers
    → computes TP/FP/TN/FN per sample
    → logs per-sample: f1, precision, recall, latency_ms
    → logs aggregate: avg_f1, micro_f1, span_hit_rate, detection_rate
    → saves to experiments/results/{experiment_name}.json

ExperimentConfig dataclass:
  experiment_name, dataset, n_samples, threshold,
  target_layers, weight_entropy, weight_wasserstein, weight_tsv,
  max_new_tokens, description

MLflow logging:
  mlflow.set_experiment() → creates named experiment
  mlflow.start_run() → begins tracking
  mlflow.log_param() → records hyperparameters
  mlflow.log_metric(step=i) → per-sample metrics
  mlflow.end_run() → saves run

Results table printed with thesis-ready format:
  Macro/Micro Precision, Recall, F1, Accuracy
  Span hit rate, Detection rate, Avg latency

Addresses:
  NFR10: experiment reproducibility via MLflow tracking
  FR13: evaluate detection accuracy on benchmarks
  FR14: configurable threshold tested across runs"

echo "Commit 1 done: experiment_runner.py"
echo ""

# COMMIT 2: git script
git add day18_git_commands.sh
git commit -m "chore(day18): add Day 18 git commands script"
echo "Commit 2 done"
echo ""

# Push
git push origin feat/evaluation-pipeline

echo ""
echo "=============================================="
echo "  Day 18 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Steps to run on your Mac:"
echo ""
echo "  1. Install MLflow:"
echo "     pip install mlflow"
echo ""
echo "  2. Test experiment runner (simulation mode):"
echo "     python backend/evaluation/experiment_runner.py"
echo ""
echo "  3. Start MLflow UI:"
echo "     mlflow ui"
echo "     → visit http://localhost:5000"
echo ""
echo "  4. To run REAL experiment (loads model ~3 min):"
echo "     Edit experiment_runner.py __main__ section:"
echo "       runner.initialize_pipeline()  # uncomment"
echo "       n_samples=50                  # change from 10"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/evaluation-pipeline to main"
echo "  Merge it, then:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 19"
echo ""
git log --oneline -6
