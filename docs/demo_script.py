# ============================================================
# Demo Script — HalluScan System Demonstration
# Author: Chalani Dinitha (20211032)
# Day 24 — Full System Demo
# ============================================================

"""
DEMO RECORDING SCRIPT
=====================
Use this script to prepare your screen recording.
Run the system first, then follow the steps below.

Before recording:
1. Start backend:  uvicorn backend.main:app --reload --port 8000
2. Start frontend: cd frontend && npm run dev
3. Open browser:   http://localhost:3000
4. Open terminal to show scores in real time

Recording duration: ~10-15 minutes
"""

# ── 5 Best Demo Prompts ──────────────────────────────────────

DEMO_PROMPTS = [

    {
        "id": 1,
        "category": "Geography Misconception",
        "prompt": "The capital of Australia is Sydney, which",
        "why": "Common misconception — capital is Canberra not Sydney",
        "expected_eat": ["Sydney", "Canberra"],
        "expected_result": "RED on Sydney — hallucination detected",
        "max_tokens": 20,
        "thesis_value": "Shows Gap 2: only city name flagged, not full sentence"
    },

    {
        "id": 2,
        "category": "Historical Fact",
        "prompt": "Napoleon Bonaparte was born in the year",
        "why": "Model may hallucinate year or location",
        "expected_eat": ["year", "1769", "France", "Corsica"],
        "expected_result": "YELLOW/RED on year token",
        "max_tokens": 15,
        "thesis_value": "Shows EAT detection on dates"
    },

    {
        "id": 3,
        "category": "Scientific Misconception",
        "prompt": "Humans use only 10 percent of their brain, which means",
        "why": "This is a famous myth — 100% of brain is used",
        "expected_eat": ["10", "percent"],
        "expected_result": "RED on percentage tokens",
        "max_tokens": 20,
        "thesis_value": "Classic TruthfulQA hallucination case"
    },

    {
        "id": 4,
        "category": "Named Entity",
        "prompt": "Albert Einstein won the Nobel Prize in Physics in",
        "why": "Tests year detection — Einstein won in 1921",
        "expected_eat": ["1921", "Physics", "Einstein"],
        "expected_result": "EAT tokens highlighted with scores",
        "max_tokens": 15,
        "thesis_value": "Shows precision of EAT span detection"
    },

    {
        "id": 5,
        "category": "Geography + Date",
        "prompt": "The Great Wall of China was built in",
        "why": "Complex historical fact with date uncertainty",
        "expected_eat": ["China", "year", "dynasty"],
        "expected_result": "Multiple EAT tokens with varying scores",
        "max_tokens": 20,
        "thesis_value": "Shows multi-token EAT detection"
    },
]


# ── Demo Recording Steps ─────────────────────────────────────

RECORDING_STEPS = """
STEP-BY-STEP RECORDING GUIDE
==============================

[SCENE 1 — System Overview] (30 seconds)
  - Show browser at http://localhost:3000
  - Point to: API Ready (green dot)
  - Point to: HalluScan title
  - Point to: Risk legend (red/yellow/white)
  - Say: "This is HalluScan — a real-time hallucination detection system
          that monitors the AI's hidden states during text generation"

[SCENE 2 — First Demo: Geography] (2 minutes)
  - Type prompt 1: "The capital of Australia is Sydney, which"
  - Set slider to 20 tokens
  - Click Detect Hallucination
  - Show loading spinner
  - When results appear — point to:
    * RED/YELLOW highlighted tokens
    * Risk score bar
    * EAT token count
    * Processing time (~2 seconds)
  - Say: "The system identified 'Sydney' as a suspicious token
          because the capital of Australia is actually Canberra"

[SCENE 3 — Second Demo: Science Myth] (2 minutes)
  - Click Config tab
  - Drag threshold slider to 0.35
  - Click Apply
  - Go back to Results tab
  - Type: "Humans use only 10 percent of their brain, which means"
  - Show RED tokens on "10" and "percent"
  - Say: "The system detected uncertainty in the factual claim
          that humans only use 10% of their brain — a known myth"

[SCENE 4 — Score Analysis] (1 minute)
  - Point to Score Panel
  - Show formula: 0.4×entropy + 0.4×wasserstein + 0.2×TSV
  - Show per-token scores table
  - Hover over a red token to show tooltip score
  - Say: "Each token receives a score combining three signals
          from the model's internal hidden states"

[SCENE 5 — Warning Banner] (30 seconds)
  - Show red warning banner at top
  - Say: "When hallucination is detected, the system shows
          an immediate warning to the user — FR11 satisfied"

[SCENE 6 — Experiment Log] (30 seconds)
  - Click Log tab
  - Show table of past queries
  - Say: "The system maintains a session history of all
          detection results for research analysis"

[SCENE 7 — Config Panel] (1 minute)
  - Click Config tab
  - Show threshold sliders
  - Drag to 0.5, click Apply
  - Run a prompt again — show more tokens flagged
  - Say: "Thresholds are configurable at runtime — FR14 satisfied"

[SCENE 8 — API Documentation] (30 seconds)
  - Open new tab: http://localhost:8000/docs
  - Show Swagger UI
  - Click POST /api/detect → Try it out
  - Say: "The full REST API is documented and testable"

[END] (15 seconds)
  - Return to main UI
  - Say: "HalluScan successfully detects hallucinations
          at the token level during LLM generation — addressing
          both research gaps identified in the literature"
"""


# ── Run This to Print All Demo Info ──────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  HALLUSCAN DEMO SCRIPT — Chalani Dinitha (20211032)")
    print("=" * 60)

    print("\n5 BEST DEMO PROMPTS:")
    print("-" * 60)
    for p in DEMO_PROMPTS:
        print(f"\n[{p['id']}] {p['category']}")
        print(f"  Prompt:   \"{p['prompt']}\"")
        print(f"  Tokens:   {p['max_tokens']}")
        print(f"  Expected: {p['expected_result']}")
        print(f"  Thesis:   {p['thesis_value']}")

    print("\n" + "=" * 60)
    print(RECORDING_STEPS)

    print("\nSYSTEM CHECKLIST BEFORE RECORDING:")
    checklist = [
        "Backend running: uvicorn backend.main:app --port 8000",
        "Frontend running: cd frontend && npm run dev",
        "Browser open: http://localhost:3000",
        "API Ready: green dot visible",
        "Threshold set to 0.35 (for OPT-1.3b)",
        "Screen recording software ready",
        "Microphone working (optional narration)",
    ]
    for item in checklist:
        print(f"  ☐ {item}")
    print()
