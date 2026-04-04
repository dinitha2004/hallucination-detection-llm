"""
config.py — Central Configuration for Hallucination Detection System
====================================================================
All tunable parameters live here. Change values in .env file,
not directly in this file. This keeps your experiments reproducible.

Author: Chalani Dinitha (20211032)
Project: Fine-Grained Hallucination Detection Using Hidden States
"""

import os
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


# ── Model Configuration ───────────────────────────────────────────────────────

# Primary model: LLaMA-3.2-3B-Instruct
# Requires HuggingFace account + access request at:
# https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
#
# Fallback model: OPT-1.3B (use while waiting for LLaMA approval)
# Identical API, free, no approval needed.
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/opt-1.3b")

# Your HuggingFace token (required for LLaMA models)
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Device: "cuda" for GPU, "cpu" for CPU only
# The system will auto-detect GPU if available
DEVICE = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"

# Maximum new tokens to generate per prompt
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "100"))


# ── Hidden State Extraction Configuration (Module B) ─────────────────────────

# Which layers to extract hidden states from (mid-to-late layers)
# For OPT-1.3B (24 layers total): use layers 12, 16, 20
# For LLaMA-3.2-3B (32 layers total): use layers 16, 20, 24, 28
# AI-Added Assumption A4: adjusted based on model size
_raw_layers = os.getenv("TARGET_LAYERS", "12,16,20")
TARGET_LAYERS = [int(x) for x in _raw_layers.split(",")]

# Feature clipping percentile for INSIDE mechanism
# Activations above this percentile are clipped to suppress overconfidence
# AI-Added Assumption: 95th percentile as starting point for tuning
FEATURE_CLIP_PERCENTILE = float(os.getenv("FEATURE_CLIP_PERCENTILE", "95"))


# ── Hallucination Detection Configuration (Module D) ─────────────────────────

# Main threshold: tokens with score above this are flagged as hallucinated
# Range: 0.0 to 1.0 | Default: 0.65
# AI-Added Assumption A3: 0.65 is the starting point; tuned via ablation study
HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.65"))

# Suspicious zone: tokens in this range are shown as "uncertain" (yellow)
SUSPICIOUS_THRESHOLD_LOW = float(os.getenv("SUSPICIOUS_THRESHOLD_LOW", "0.45"))

# Score aggregation weights (must sum to 1.0)
# AI-Added Assumption A2: equal weight to entropy and shift, less to TSV
WEIGHT_ENTROPY = float(os.getenv("WEIGHT_ENTROPY", "0.4"))
WEIGHT_WASSERSTEIN = float(os.getenv("WEIGHT_WASSERSTEIN", "0.4"))
WEIGHT_TSV = float(os.getenv("WEIGHT_TSV", "0.2"))


# ── Distribution Shift Configuration (Module C) ───────────────────────────────

# Sliding window size for layer-to-layer shift analysis
# Your framework specifies optimal window size = 2
SHIFT_WINDOW_SIZE = int(os.getenv("SHIFT_WINDOW_SIZE", "2"))

# TSV training data size (contrastive pairs)
# AI-Added Assumption A5: 50 truthful + 50 hallucinated pairs
TSV_TRAINING_PAIRS = int(os.getenv("TSV_TRAINING_PAIRS", "100"))

# Path to saved TSV vector
TSV_VECTOR_PATH = os.getenv("TSV_VECTOR_PATH", "./data/memory_bank/tsv_vector.npy")

# Path to memory bank for feature clipping baseline
MEMORY_BANK_PATH = os.getenv("MEMORY_BANK_PATH", "./data/memory_bank/activation_baseline.npy")


# ── Server Configuration ──────────────────────────────────────────────────────
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))


# ── MLflow Experiment Tracking ────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "hallucination_detection_v1")


# ── Logging & Privacy ─────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# NFR5 Security: Set to False by default — prompts are NOT saved to disk
LOG_PROMPTS = os.getenv("LOG_PROMPTS", "false").lower() == "true"


# ── Validation ────────────────────────────────────────────────────────────────
def validate_config():
    """
    Check that configuration values are valid.
    Call this at application startup.
    """
    errors = []

    if not (0.0 <= HALLUCINATION_THRESHOLD <= 1.0):
        errors.append(f"HALLUCINATION_THRESHOLD must be between 0.0 and 1.0, got {HALLUCINATION_THRESHOLD}")

    weight_sum = WEIGHT_ENTROPY + WEIGHT_WASSERSTEIN + WEIGHT_TSV
    if abs(weight_sum - 1.0) > 0.001:
        errors.append(f"Weights must sum to 1.0, got {weight_sum:.3f}")

    if SHIFT_WINDOW_SIZE < 1:
        errors.append(f"SHIFT_WINDOW_SIZE must be >= 1, got {SHIFT_WINDOW_SIZE}")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True


# ── Print summary (helpful for debugging) ────────────────────────────────────
def print_config():
    """Print current configuration. Useful when starting the server."""
    print("=" * 60)
    print("  Hallucination Detection System — Configuration")
    print("=" * 60)
    print(f"  Model:              {MODEL_NAME}")
    print(f"  Device:             {DEVICE}")
    print(f"  Target Layers:      {TARGET_LAYERS}")
    print(f"  Threshold:          {HALLUCINATION_THRESHOLD}")
    print(f"  Score Weights:      entropy={WEIGHT_ENTROPY}, wasserstein={WEIGHT_WASSERSTEIN}, tsv={WEIGHT_TSV}")
    print(f"  Max New Tokens:     {MAX_NEW_TOKENS}")
    print(f"  Log Prompts:        {LOG_PROMPTS}  (NFR5 privacy)")
    print("=" * 60)


if __name__ == "__main__":
    # Quick test: run this file directly to check your config
    # Command: python backend/config.py
    validate_config()
    print_config()
    print("\nConfig validation passed!")
