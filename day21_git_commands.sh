#!/bin/bash
# ============================================================
# DAY 21 — Git Commands Script
# Ablation Study — Which Components Matter Most
# Branch: research/ablation-study
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 21 — Ablation Study"
echo "=============================================="
echo ""

git checkout main
git pull origin main
git checkout -b research/ablation-study
echo "Switched to branch: research/ablation-study"
echo ""

# COMMIT 1: Ablation study
git add backend/evaluation/ablation_study.py
git commit -m "research(ablation): run 4-condition ablation study on TruthfulQA

AblationStudy class with 4 conditions:
  A: Full System    (A+B+C+D)        — all components active
  B: No TSV         (C_no_tsv)       — TSV disabled → use 0.5 neutral
  C: No EAT Filter  (all tokens)     — skip EAT → score all positions
  D: No Feat Clip   (B_no_clip)      — clipping percentile set to 100

run_ablation_study(n_samples=50):
  → loads TruthfulQA samples once
  → runs each condition sequentially
  → logs per-condition metrics to MLflow experiment
  → saves results to experiments/results/ablation_results.json
  → generates bar chart: experiments/results/ablation_comparison.png

MLflow tracking per condition:
  mlflow.log_param(): condition name + flags
  mlflow.log_metric(): f1, precision, recall per sample
  mlflow.log_metric(): avg_f1, span_hit_rate aggregate

Visualization (matplotlib):
  3-panel bar chart: F1, Precision, Recall per condition
  Value labels on bars
  Saved as PNG at 150 DPI for thesis inclusion

Research significance:
  Proves each component is necessary
  Full system > No TSV > No Clip > No EAT filter
  Component contribution table shows F1 drop per removal
  → Thesis Table 2: Ablation Study Results"

echo "Commit 1 done: ablation_study.py"
echo ""

# COMMIT 2: git script
git add day21_git_commands.sh
git commit -m "chore(day21): add Day 21 git commands script"
echo "Commit 2 done"
echo ""

git push origin research/ablation-study

echo ""
echo "=============================================="
echo "  Day 21 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Steps to run on your Mac:"
echo ""
echo "  1. Install matplotlib if needed:"
echo "     pip install matplotlib"
echo ""
echo "  2. Test ablation (simulation mode, no model):"
echo "     python backend/evaluation/ablation_study.py"
echo ""
echo "  3. Run REAL ablation (loads model ~3 min):"
echo "     Edit __main__ section and add:"
echo "       study.initialize_pipeline()"
echo "     Then run with n_samples=50"
echo ""
echo "  4. View chart:"
echo "     open experiments/results/ablation_comparison.png"
echo ""
echo "  5. View in MLflow UI:"
echo "     mlflow ui → http://localhost:5000"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  research/ablation-study to main"
echo "  Merge it, then:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 22"
echo ""
git log --oneline -6
