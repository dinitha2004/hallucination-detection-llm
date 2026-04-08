#!/bin/bash
# ============================================================
# DAY 22 — Git Commands Script
# Documentation & README
# Branch: docs/readme-and-docs
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 22 — Documentation & README"
echo "=============================================="
echo ""

git checkout main
git pull origin main
git checkout -b docs/readme-and-docs
echo "Switched to branch: docs/readme-and-docs"
echo ""

# COMMIT 1: README
git add README.md
git commit -m "docs: comprehensive README with setup, architecture, examples

- Project overview with research gap table
- ASCII architecture diagram (A→B→C→D pipeline)
- Quick start: backend + frontend setup instructions
- Example detection output table
- Test running instructions (unit + integration)
- Evaluation commands (datasets, metrics, MLflow)
- Configuration reference (.env settings)
- Full project structure tree
- API reference summary table
- Research results table (NFR1, NFR4)
- Dependencies table"

echo "Commit 1 done: README.md"
echo ""

# COMMIT 2: Architecture doc
git add docs/architecture.md
git commit -m "docs: add architecture guide for all 4 modules

docs/architecture.md:
  - Module A: EAT Detection with spaCy NER
  - Module B: Hidden State Extraction + INSIDE clipping
  - Module C: HalluShift (Wasserstein + cosine) + TSV probe
  - Module D: Scoring formula + Span-Level Mapper key novelty
  - Full pipeline execution flow (9 steps)
  - Layer selection experiment results table"

echo "Commit 2 done: docs/architecture.md"
echo ""

# COMMIT 3: API reference
git add docs/api_reference.md
git commit -m "docs: add complete API reference with request/response examples

docs/api_reference.md:
  - GET /: API root
  - GET /api/health: server status
  - POST /api/detect: full detection with annotated_tokens example
  - GET /api/config: current thresholds and weights
  - POST /api/config: runtime threshold update (FR14)
  - Error codes table"

echo "Commit 3 done: docs/api_reference.md"
echo ""

# COMMIT 4: Contributing guide
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md with branching and commit conventions

- Branch naming strategy (feat/, test/, research/, docs/, fix/)
- Commit message format: type(scope): description
- All commit types with examples
- Pull request process
- Test requirements (100% pass rate maintained)"

echo "Commit 4 done: CONTRIBUTING.md"
echo ""

# COMMIT 5: git script
git add day22_git_commands.sh
git commit -m "chore(day22): add Day 22 git commands script"
echo "Commit 5 done"
echo ""

git push origin docs/readme-and-docs

echo ""
echo "=============================================="
echo "  Day 22 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  docs/readme-and-docs to main"
echo "  Merge it, then:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Your GitHub repo now has:"
echo "  README.md         ← Professional project overview"
echo "  docs/architecture.md  ← All 4 modules explained"
echo "  docs/api_reference.md ← API documentation"
echo "  CONTRIBUTING.md   ← Branching + commit rules"
echo ""
echo "Then say: Start Day 23"
echo ""
git log --oneline -8
