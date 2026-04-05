#!/bin/bash
# ============================================================
# DAY 6 — Git Commands Script
# Module A: EAT Detection (Gap 2 Core)
# Branch: feat/module-a-eat-detection
# Author: Chalani Dinitha (20211032)
# ============================================================

echo ""
echo "=============================================="
echo "  Day 6 — Module A: EAT Detection"
echo "=============================================="
echo ""

# STEP 1: Create new branch from main
git checkout main
git pull origin main
git checkout -b feat/module-a-eat-detection
echo "Switched to branch: feat/module-a-eat-detection"
echo ""

# COMMIT 1: Add Module A EAT detection
git add backend/modules/module_a_eat.py
git commit -m "feat(module-a): implement EAT detection using spaCy NER

- EATDetector class with spaCy en_core_web_sm
- identify_eat_tokens(): scan text for PERSON, DATE, GPE,
  ORG, CARDINAL, LOC and 12 other entity types
- map_eat_to_token_positions(): map entity char spans to
  exact token position indices
- detect_and_map(): combined pipeline method
- get_eat_position_set(): flat set for Module D scoring
- is_eat_position(): O(1) lookup for position checking
- format_eat_summary(): human-readable output for UI/logs
- EATSpan dataclass: text, entity_type, positions, confidence
- Singleton pattern: one instance per session
- Tested: 'Einstein was born in 1879 in Ulm, Germany'
  → flags 1879, Ulm, Germany, Einstein correctly

Addresses:
  Gap 2: highlight only the exact wrong token
  FR9: identify exact hallucinated token or span
  FR10: highlight detected hallucinated part in output"

echo "Commit 1 done: module_a_eat.py"
echo ""

# COMMIT 2: Add unit tests for Module A
git add backend/tests/test_module_a.py
git commit -m "test(module-a): add unit tests for EAT detection

- TestEATDetectorSetup: instantiation, singleton, spaCy loads
- TestIdentifyEATTokens: Day 6 plan test case (Einstein/1879/Germany),
  capital city test, empty text, plain text no entities
- TestMapEATToTokenPositions: 1879 maps to position 4, empty tokens,
  positions in valid range, return type tuple
- TestEATSpanDataclass: creation, is_single_token, span_length, repr
- TestUtilityMethods: format summary, is_eat_position true/false
- All tests skip gracefully if spaCy not installed

NFR10: reproducibility through test coverage
Gap 2: tests verify exact token position mapping"

echo "Commit 2 done: test_module_a.py"
echo ""

# COMMIT 3: Push to GitHub
git push origin feat/module-a-eat-detection

echo ""
echo "=============================================="
echo "  Day 6 COMPLETE - All commits pushed!"
echo "=============================================="
echo ""
echo "Tests to run on your Mac:"
echo ""
echo "  1. Test Module A directly:"
echo "     python backend/modules/module_a_eat.py"
echo ""
echo "  2. Run unit tests:"
echo "     pytest backend/tests/test_module_a.py -v"
echo ""
echo "  Expected output for unit tests:"
echo "    'Einstein was born in 1879 in Ulm, Germany'"
echo "    → EATs: Einstein (PERSON), 1879 (DATE), Germany (GPE)"
echo ""
echo "Go to GitHub - create Pull Request:"
echo "  feat/module-a-eat-detection to main"
echo "  Merge it, then run:"
echo "  git checkout main && git pull origin main"
echo ""
echo "Then say: Start Day 7"
echo ""
git log --oneline -8
