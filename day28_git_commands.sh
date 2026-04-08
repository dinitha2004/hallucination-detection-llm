#!/bin/bash
# ============================================================
# DAY 28 — Git Commands Script
# Chapter 6 Implementation Evidence
# Branch: thesis/writing-support
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 28 — Chapter 6 Implementation Evidence"
echo "=============================================="
echo ""

git checkout main
git pull origin main
git checkout -b thesis/writing-support
echo "On branch: thesis/writing-support"
echo ""

# COMMIT 1: Evidence script
git add experiments/notebooks/chapter6_implementation_evidence.py
git commit -m "thesis(ch6): add Chapter 6 implementation evidence generator

experiments/notebooks/chapter6_implementation_evidence.py:
  Generates 3 output files for thesis Chapter 6:

  docs/git_history.txt:
    Full git commit history (all commits since Day 1)
    Shows 22+ days of implementation journey
    Evidence for examiner of systematic development

  docs/sample_detection_output.json:
    Real pipeline output with annotated tokens
    Shows: score, risk_level, is_eat, is_flagged per token
    Evidence for FR9, FR12, Gap 2 implementation

  docs/chapter6_evidence.md:
    Comprehensive Chapter 6 evidence document including:
    - ASCII architecture diagram (A→B→C→D)
    - Key code listings for all 4 modules
    - Sample token-level detection output with legend
    - API endpoint demonstration
    - Test evidence summary (194 tests)
    - Implementation summary table"

echo "Commit 1: chapter6_implementation_evidence.py"
echo ""

# COMMIT 2: Generated docs
git add docs/git_history.txt 2>/dev/null || true
git add docs/sample_detection_output.json 2>/dev/null || true
git add docs/chapter6_evidence.md 2>/dev/null || true
git commit -m "thesis(ch6): add generated Chapter 6 evidence documents

  docs/git_history.txt — full commit history
  docs/sample_detection_output.json — real detection example
  docs/chapter6_evidence.md — complete evidence markdown" \
  2>/dev/null || echo "Run script first to generate docs"

echo "Commit 2: generated docs"
echo ""

# COMMIT 3: git script
git add day28_git_commands.sh
git commit -m "chore(day28): add Day 28 git commands script"
echo "Commit 3: day28_git_commands.sh"
echo ""

git push origin thesis/writing-support

echo ""
echo "=============================================="
echo "  Day 28 COMPLETE!"
echo "=============================================="
echo ""
echo "Steps:"
echo "  1. Run the evidence generator:"
echo "     python experiments/notebooks/chapter6_implementation_evidence.py"
echo ""
echo "  2. Check generated files:"
echo "     cat docs/chapter6_evidence.md"
echo "     cat docs/git_history.txt"
echo ""
echo "  3. Commit the generated files:"
echo "     git add docs/"
echo "     git commit -m 'thesis(ch6): add generated evidence docs'"
echo "     git push origin thesis/writing-support"
echo ""
echo "GitHub → PR thesis/writing-support → main → Merge"
echo ""
git log --oneline -6
