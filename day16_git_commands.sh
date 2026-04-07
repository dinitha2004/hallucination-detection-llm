#!/bin/bash
# ============================================================
# DAY 16 — Git Commands Script
# Dataset Loading — TruthfulQA & TriviaQA
# Branch: feat/evaluation-pipeline
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 16 — TruthfulQA & TriviaQA Loaders"
echo "=============================================="
echo ""

# STEP 1: Create new branch from main
git checkout main
git pull origin main
git checkout -b feat/evaluation-pipeline
echo "Switched to branch: feat/evaluation-pipeline"
echo ""

# COMMIT 1: Dataset loader
git add backend/evaluation/dataset_loader.py
git commit -m "feat(eval): add TruthfulQA and TriviaQA dataset loaders

DatasetLoader class:
  load_truthfulqa(n=100): loads from HuggingFace truthful_qa/generation
    → returns List[TruthfulQASample]
    → fields: question, correct_answer, incorrect_answers, category
    → fallback: 10 hardcoded samples if no internet

  load_triviaqa(n=100): loads from HuggingFace trivia_qa/rc.nocontext
    → returns List[TriviaQASample]
    → fields: question, answer, aliases
    → fallback: 10 hardcoded samples if no internet

  get_dataset_stats(): summary statistics for both datasets
  Singleton pattern: one loader instance per session
  Caching: datasets cached after first load (no re-download)

TruthfulQASample dataclass:
  question, correct_answer, incorrect_answers, category, source
  to_prompt(): formats question for LLM completion

TriviaQASample dataclass:
  question, answer, aliases, source
  to_prompt(): formats question for LLM completion

Research connection:
  TruthfulQA → tests hallucination resistance (817 trick questions)
  TriviaQA   → tests factual recall (95,000 Q&A pairs)
  Both used for thesis evaluation in Days 17-18"

echo "Commit 1 done: dataset_loader.py"
echo ""

# COMMIT 2: git script
git add day16_git_commands.sh
git commit -m "chore(day16): add Day 16 git commands script"
echo "Commit 2 done: day16_git_commands.sh"
echo ""

# Push
git push origin feat/evaluation-pipeline

echo ""
echo "=============================================="
echo "  Day 16 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Steps to run on your Mac:"
echo ""
echo "  1. Install datasets library:"
echo "     pip install datasets"
echo ""
echo "  2. Test the loaders:"
echo "     python backend/evaluation/dataset_loader.py"
echo ""
echo "  Expected output:"
echo "    TruthfulQA — 100 samples loaded"
echo "    TriviaQA   — 100 samples loaded"
echo "    5 samples printed from each"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/evaluation-pipeline to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 17"
echo ""
git log --oneline -6
