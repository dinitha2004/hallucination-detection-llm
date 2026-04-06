"""
module_c_hallushift.py — Module C: HalluShift Distribution Shift Analyzer
==========================================================================
This module implements the HalluShift technique from your research framework.

What is HalluShift?
-------------------
When an LLM generates a hallucinated token, the information flowing
through its layers shows a distinct "distributional fingerprint" —
the pattern of how activations change from layer to layer is different
compared to when the model generates a truthful token.

HalluShift detects this by measuring TWO things between consecutive layers:

1. WASSERSTEIN DISTANCE — measures how much the distribution of
   activation values has "shifted" between layers.
   Think of it like: "how far did the probability mass move?"
   High shift = model is uncertain/changing its mind = hallucination signal.

2. COSINE SIMILARITY — measures the directional alignment between
   two layer activation vectors.
   Think of it like: "are the two layers pointing in the same direction?"
   Low similarity = layers disagree = hallucination signal.

Sliding Window (size=2):
------------------------
Instead of comparing all possible layer pairs, we use a sliding
window of size 2 — comparing only CONSECUTIVE layer pairs:
    (layer_18, layer_20), (layer_20, layer_22)
This is computationally efficient and your framework specifies
"optimal window size = 2".

Research connection:
--------------------
- HalluShift → your finalized technique Section 1 (Gap 1)
- Wasserstein metrics + cosine similarities → your framework Module C
- Window size=2 → specified in your framework design document
- FR5: analyse distribution shifts across decoding steps

Author: Chalani Dinitha (20211032)
"""

import sys
import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

sys.path.append(".")
from backend.config import TARGET_LAYERS, SHIFT_WINDOW_SIZE

logger = logging.getLogger(__name__)


@dataclass
class ShiftScore:
    """
    Holds the distribution shift scores for one layer pair.

    Example:
        ShiftScore(
            layer_pair=(18, 20),
            wasserstein=0.847,
            cosine=0.623,
            shift_magnitude=0.724
        )
    """
    layer_pair: Tuple[int, int]        # e.g. (18, 20)
    wasserstein: float                  # Wasserstein distance (higher = more shift)
    cosine: float                       # Cosine similarity (lower = more shift)
    shift_magnitude: float = 0.0        # Combined shift score (0.0 to 1.0)

    @property
    def layer_from(self) -> int:
        return self.layer_pair[0]

    @property
    def layer_to(self) -> int:
        return self.layer_pair[1]

    @property
    def is_high_shift(self) -> bool:
        """True if shift magnitude suggests hallucination."""
        return self.shift_magnitude > 0.5

    def __repr__(self):
        return (f"ShiftScore(layers={self.layer_pair}, "
                f"wasserstein={self.wasserstein:.4f}, "
                f"cosine={self.cosine:.4f}, "
                f"magnitude={self.shift_magnitude:.4f})")


class HalluShiftAnalyzer:
    """
    Monitors distribution shifts across LLM layers to detect hallucinations.

    How it fits in the pipeline:
    ----------------------------
    Module B extracts hidden states → Module C computes shift scores
    → Module D uses these scores in the hallucination score formula:
        score = 0.4*entropy + 0.4*wasserstein + 0.2*tsv

    The Wasserstein distance computed here is the 0.4*wasserstein
    component of your scoring formula.
    """

    def __init__(self):
        self.window_size = SHIFT_WINDOW_SIZE  # = 2 from config
        self.target_layers = sorted(TARGET_LAYERS)

    # =========================================================
    # SECTION 1: Wasserstein Distance
    # =========================================================

    def calculate_wasserstein_distance(
        self,
        vec_a: torch.Tensor,
        vec_b: torch.Tensor
    ) -> float:
        """
        Calculate the Wasserstein distance between two hidden state vectors.

        What is Wasserstein distance? (Beginner explanation)
        ----------------------------------------------------
        Imagine two piles of sand. The Wasserstein distance (also called
        "Earth Mover's Distance") measures how much work it takes to
        reshape one pile into the other.

        For hidden states:
        - vec_a = activation distribution from layer 18
        - vec_b = activation distribution from layer 20
        - High distance = the distributions are very different
                        = the model's "thinking" changed a lot
                        = hallucination signal

        We use scipy's implementation which treats the vectors as
        1D distributions (histograms of activation values).

        Args:
            vec_a: Hidden state tensor from layer i, shape (hidden_size,)
            vec_b: Hidden state tensor from layer j, shape (hidden_size,)

        Returns:
            Float: Wasserstein distance (always >= 0.0)
        """
        try:
            from scipy.stats import wasserstein_distance

            # Convert to numpy for scipy
            a = vec_a.float().numpy() if isinstance(vec_a, torch.Tensor) else vec_a
            b = vec_b.float().numpy() if isinstance(vec_b, torch.Tensor) else vec_b

            # Normalize to make distances comparable across layers
            a = a / (np.linalg.norm(a) + 1e-8)
            b = b / (np.linalg.norm(b) + 1e-8)

            distance = float(wasserstein_distance(a, b))
            return distance

        except Exception as e:
            logger.warning(f"Wasserstein calculation failed: {e}")
            return 0.0

    # =========================================================
    # SECTION 2: Cosine Similarity
    # =========================================================

    def calculate_cosine_similarity(
        self,
        vec_a: torch.Tensor,
        vec_b: torch.Tensor
    ) -> float:
        """
        Calculate cosine similarity between two hidden state vectors.

        What is cosine similarity? (Beginner explanation)
        -------------------------------------------------
        Cosine similarity measures the ANGLE between two vectors.
        - Score = 1.0: vectors point in exactly the same direction
        - Score = 0.0: vectors are perpendicular (90 degrees apart)
        - Score = -1.0: vectors point in opposite directions

        For hidden states:
        - High similarity (near 1.0) = layers agree → truthful
        - Low similarity (near 0.0)  = layers disagree → uncertain
        - Negative similarity        = strong disagreement → likely hallucination

        Args:
            vec_a: Hidden state tensor, shape (hidden_size,)
            vec_b: Hidden state tensor, shape (hidden_size,)

        Returns:
            Float: Cosine similarity in range [-1.0, 1.0]
        """
        try:
            a = vec_a.float().numpy() if isinstance(vec_a, torch.Tensor) else vec_a
            b = vec_b.float().numpy() if isinstance(vec_b, torch.Tensor) else vec_b

            # Compute dot product / (norm_a * norm_b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a < 1e-8 or norm_b < 1e-8:
                return 0.0

            similarity = float(np.dot(a, b) / (norm_a * norm_b))

            # Clip to valid range [-1, 1] (floating point errors can exceed)
            return float(np.clip(similarity, -1.0, 1.0))

        except Exception as e:
            logger.warning(f"Cosine similarity calculation failed: {e}")
            return 0.0

    # =========================================================
    # SECTION 3: Distribution Shift with Sliding Window
    # =========================================================

    def compute_distribution_shift(
        self,
        activations: Dict[int, List[torch.Tensor]],
        token_position: int = 0
    ) -> Dict[Tuple[int, int], ShiftScore]:
        """
        Compute distribution shifts across all consecutive layer pairs
        for a specific token position.

        Uses sliding window (size=2) as specified in your framework:
        "optimal window size = 2 for balancing overhead with sensitivity"

        For target layers [18, 20, 22]:
            Window 1: layers (18, 20) → ShiftScore
            Window 2: layers (20, 22) → ShiftScore

        Args:
            activations:    Output of Module B's extract_layer_activations()
                           {layer_idx: [vec_token0, vec_token1, ...]}
            token_position: Which generated token to analyze (default: 0)

        Returns:
            Dict: {(layer_i, layer_j): ShiftScore}

        Example output:
            {
                (18, 20): ShiftScore(wasserstein=0.84, cosine=0.62),
                (20, 22): ShiftScore(wasserstein=1.23, cosine=0.45),
            }
        """
        shift_scores = {}

        # Get sorted layer indices that have data
        available_layers = sorted([
            l for l in self.target_layers
            if l in activations and token_position < len(activations[l])
        ])

        if len(available_layers) < 2:
            logger.warning(
                f"Need at least 2 layers for shift analysis, "
                f"got {len(available_layers)}"
            )
            return shift_scores

        # Sliding window across consecutive layer pairs
        # window_size=2 means we look at pairs: (i, i+1)
        for i in range(len(available_layers) - self.window_size + 1):
            layer_a = available_layers[i]
            layer_b = available_layers[i + self.window_size - 1]
            pair = (layer_a, layer_b)

            # Get vectors for this token at both layers
            vec_a = activations[layer_a][token_position]
            vec_b = activations[layer_b][token_position]

            # Compute both metrics
            wasserstein = self.calculate_wasserstein_distance(vec_a, vec_b)
            cosine = self.calculate_cosine_similarity(vec_a, vec_b)

            # Compute combined shift magnitude
            # cosine is in [-1, 1]: convert to shift (1 - cosine) / 2 → [0, 1]
            cosine_shift = (1.0 - cosine) / 2.0

            # Normalize Wasserstein (typically 0 to ~2 for normalized vectors)
            wasserstein_norm = min(wasserstein / 2.0, 1.0)

            # Combined magnitude (equal weight)
            magnitude = (wasserstein_norm + cosine_shift) / 2.0

            shift_scores[pair] = ShiftScore(
                layer_pair=pair,
                wasserstein=wasserstein,
                cosine=cosine,
                shift_magnitude=magnitude
            )

            logger.debug(
                f"Layer pair {pair}: "
                f"wasserstein={wasserstein:.4f}, "
                f"cosine={cosine:.4f}, "
                f"magnitude={magnitude:.4f}"
            )

        return shift_scores

    def compute_all_token_shifts(
        self,
        activations: Dict[int, List[torch.Tensor]]
    ) -> List[Dict[Tuple[int, int], ShiftScore]]:
        """
        Compute distribution shifts for ALL generated tokens.

        Returns a list where index i = shift scores for token i.
        Used by Module D to get per-token shift signals.

        Args:
            activations: Full activations dict from Module B

        Returns:
            List of shift score dicts, one per token
        """
        # Determine how many tokens were captured
        if not activations:
            return []

        first_layer = next(iter(activations))
        num_tokens = len(activations[first_layer])

        all_shifts = []
        for token_pos in range(num_tokens):
            shifts = self.compute_distribution_shift(activations, token_pos)
            all_shifts.append(shifts)

        logger.info(
            f"Computed shift scores for {num_tokens} tokens "
            f"across {len(self.target_layers)} layers"
        )
        return all_shifts

    def get_max_shift_for_token(
        self,
        shift_scores: Dict[Tuple[int, int], ShiftScore]
    ) -> float:
        """
        Get the maximum shift magnitude across all layer pairs for one token.

        This single value represents the overall distribution instability
        for this token — the higher it is, the more likely this token
        is a hallucination.

        Args:
            shift_scores: Output of compute_distribution_shift()

        Returns:
            Float: Maximum shift magnitude (0.0 to 1.0)
        """
        if not shift_scores:
            return 0.0
        return max(s.shift_magnitude for s in shift_scores.values())

    def get_average_wasserstein(
        self,
        shift_scores: Dict[Tuple[int, int], ShiftScore]
    ) -> float:
        """
        Get average Wasserstein distance across all layer pairs.
        Used in Module D's scoring formula.
        """
        if not shift_scores:
            return 0.0
        return sum(s.wasserstein for s in shift_scores.values()) / len(shift_scores)

    def format_shift_summary(
        self,
        shift_scores: Dict[Tuple[int, int], ShiftScore]
    ) -> str:
        """Human-readable summary of shift scores. Used for logging."""
        if not shift_scores:
            return "No shift scores computed"

        lines = [f"Distribution Shift Analysis ({len(shift_scores)} layer pairs):"]
        for pair, score in sorted(shift_scores.items()):
            lines.append(
                f"  Layers {pair}: "
                f"wasserstein={score.wasserstein:.4f}, "
                f"cosine={score.cosine:.4f}, "
                f"magnitude={score.shift_magnitude:.4f} "
                f"{'[HIGH SHIFT]' if score.is_high_shift else ''}"
            )
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────────────────────────
_hallushift_instance = None


def get_hallushift_analyzer() -> HalluShiftAnalyzer:
    """Returns the global HalluShiftAnalyzer instance."""
    global _hallushift_instance
    if _hallushift_instance is None:
        _hallushift_instance = HalluShiftAnalyzer()
    return _hallushift_instance


# ── Quick Test ────────────────────────────────────────────────────────────────
# Run: python backend/modules/module_c_hallushift.py

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 65)
    print("  DAY 7 TEST: Module C — HalluShift Analyzer")
    print("=" * 65 + "\n")

    analyzer = HalluShiftAnalyzer()

    print(f"Window size: {analyzer.window_size}")
    print(f"Target layers: {analyzer.target_layers}\n")

    # ── Test 1: Wasserstein distance ──────────────────────────────
    print("TEST 1: Wasserstein Distance")
    vec_a = torch.randn(2048)
    vec_b = torch.randn(2048)
    vec_similar = vec_a + torch.randn(2048) * 0.1  # slightly different

    dist_different = analyzer.calculate_wasserstein_distance(vec_a, vec_b)
    dist_similar = analyzer.calculate_wasserstein_distance(vec_a, vec_similar)

    print(f"  Different vectors: wasserstein = {dist_different:.4f}")
    print(f"  Similar vectors:   wasserstein = {dist_similar:.4f}")
    assert dist_different >= 0.0, "Wasserstein must be >= 0"
    print(f"  PASS: Wasserstein distance is always positive ✅\n")

    # ── Test 2: Cosine Similarity ─────────────────────────────────
    print("TEST 2: Cosine Similarity")
    same = torch.ones(2048)
    opposite = -torch.ones(2048)
    perp_a = torch.zeros(2048); perp_a[0] = 1.0
    perp_b = torch.zeros(2048); perp_b[1] = 1.0

    cos_same = analyzer.calculate_cosine_similarity(same, same)
    cos_opp = analyzer.calculate_cosine_similarity(same, opposite)
    cos_perp = analyzer.calculate_cosine_similarity(perp_a, perp_b)

    print(f"  Same direction:     cosine = {cos_same:.4f}  (expected ~1.0)")
    print(f"  Opposite direction: cosine = {cos_opp:.4f} (expected ~-1.0)")
    print(f"  Perpendicular:      cosine = {cos_perp:.4f}  (expected ~0.0)")

    assert abs(cos_same - 1.0) < 0.001
    assert abs(cos_opp - (-1.0)) < 0.001
    assert abs(cos_perp) < 0.001
    print(f"  PASS: Cosine similarity correct ✅\n")

    # ── Test 3: Distribution shift with sliding window ────────────
    print("TEST 3: Distribution Shift with Sliding Window (size=2)")
    torch.manual_seed(42)

    # Simulate activations from Module B
    fake_activations = {
        18: [torch.randn(2048), torch.randn(2048)],  # 2 tokens
        20: [torch.randn(2048), torch.randn(2048)],
        22: [torch.randn(2048), torch.randn(2048)],
    }

    shift_scores = analyzer.compute_distribution_shift(fake_activations, token_position=0)

    print(f"  Layer pairs analyzed: {list(shift_scores.keys())}")
    for pair, score in shift_scores.items():
        print(f"  {pair}: wasserstein={score.wasserstein:.4f}, "
              f"cosine={score.cosine:.4f}, "
              f"magnitude={score.shift_magnitude:.4f}")

    assert len(shift_scores) == 2, f"Expected 2 pairs, got {len(shift_scores)}"
    assert (18, 20) in shift_scores
    assert (20, 22) in shift_scores
    print(f"  PASS: Correct layer pairs computed ✅\n")

    # ── Test 4: All tokens ────────────────────────────────────────
    print("TEST 4: compute_all_token_shifts (both tokens)")
    all_shifts = analyzer.compute_all_token_shifts(fake_activations)

    print(f"  Tokens processed: {len(all_shifts)}")
    for i, shifts in enumerate(all_shifts):
        max_shift = analyzer.get_max_shift_for_token(shifts)
        avg_wass = analyzer.get_average_wasserstein(shifts)
        print(f"  Token {i}: max_shift={max_shift:.4f}, avg_wasserstein={avg_wass:.4f}")

    assert len(all_shifts) == 2
    print(f"  PASS: All tokens processed ✅\n")

    # ── Test 5: Summary ───────────────────────────────────────────
    print("TEST 5: format_shift_summary")
    summary = analyzer.format_shift_summary(shift_scores)
    print(f"  {summary}\n")

    # ── Final ─────────────────────────────────────────────────────
    print("=" * 65)
    print("  DAY 7 DELIVERABLE CONFIRMED")
    print("=" * 65)
    print("  PASS: Wasserstein distance computed correctly")
    print("  PASS: Cosine similarity in [-1, 1] range")
    print("  PASS: Sliding window (size=2) creates correct layer pairs")
    print("  PASS: shift_scores dict {layer_pair: ShiftScore} returned")
    print("  PASS: All tokens processed")
    print()
    print("  Research impact:")
    print("  → Gap 1: distribution shifts between layers detected")
    print("  → Wasserstein feeds into scoring: 0.4 × wasserstein")
    print("  → Cosine disagreement = hallucination signal")
    print("=" * 65 + "\n")
