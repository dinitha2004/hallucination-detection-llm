"""
module_d_scoring.py — Module D: Semantic Entropy & Score Aggregation
====================================================================
This is the FINAL scoring engine of your hallucination detection system.

What this module does:
----------------------
1. SEMANTIC ENTROPY: Measures uncertainty in the hidden state distribution.
   A high entropy = model is uncertain = more likely hallucinating.
   Formula: H = -sum(p * log(p)) over softmax of hidden state norms.

2. SCORE AGGREGATION: Combines three signals into ONE hallucination score:
   score = 0.4 × entropy + 0.4 × wasserstein + 0.2 × tsv_deviation
   (From your research framework — AI-Added Assumption A2)

3. THRESHOLD APPLICATION: Flags a token as hallucinated if its score
   exceeds HALLUCINATION_THRESHOLD (default: 0.65, configurable).

Research connection:
--------------------
- Semantic Entropy → your finalized technique Step 4
- Score aggregation → your framework Module D
- Threshold → HALLUCINATION_THRESHOLD from config.py (FR14)
- FR4: analyse semantic entropy
- FR7: combine multiple signals into unified score
- FR8: detect hallucinations before full response completed

Author: Chalani Dinitha (20211032)
"""

import sys
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.append(".")
from backend.config import (
    HALLUCINATION_THRESHOLD,
    SUSPICIOUS_THRESHOLD_LOW,
    WEIGHT_ENTROPY,
    WEIGHT_WASSERSTEIN,
    WEIGHT_TSV,
    TARGET_LAYERS,
)

logger = logging.getLogger(__name__)


# ── Token Risk Level ──────────────────────────────────────────────────────────
class RiskLevel:
    """Risk levels for token classification."""
    SAFE        = "safe"        # score < SUSPICIOUS_THRESHOLD_LOW
    SUSPICIOUS  = "suspicious"  # SUSPICIOUS < score < HALLUCINATION_THRESHOLD
    HALLUCINATED = "hallucinated"  # score > HALLUCINATION_THRESHOLD


@dataclass
class TokenScore:
    """
    Complete hallucination analysis for one generated token.

    This is the output object that Module D produces for each token.
    The detection pipeline uses this to build the annotated response.
    """
    token: str                          # The token text e.g. " Paris"
    position: int                       # Index in generated sequence
    entropy: float = 0.0               # Semantic entropy score
    wasserstein: float = 0.0           # Average Wasserstein from Module C
    tsv_deviation: float = 0.5         # TSV deviation from Module C
    hallucination_score: float = 0.0   # Combined score (0.0 to 1.0)
    is_flagged: bool = False            # True if above threshold
    is_eat: bool = False                # True if this is an EAT position
    risk_level: str = RiskLevel.SAFE   # "safe", "suspicious", "hallucinated"

    @property
    def is_hallucinated(self) -> bool:
        return self.risk_level == RiskLevel.HALLUCINATED

    @property
    def is_suspicious(self) -> bool:
        return self.risk_level == RiskLevel.SUSPICIOUS

    def to_dict(self) -> dict:
        """Serializable format for API response."""
        return {
            "token": self.token,
            "position": self.position,
            "entropy": round(self.entropy, 4),
            "wasserstein": round(self.wasserstein, 4),
            "tsv_deviation": round(self.tsv_deviation, 4),
            "hallucination_score": round(self.hallucination_score, 4),
            "is_flagged": self.is_flagged,
            "is_eat": self.is_eat,
            "risk_level": self.risk_level,
        }


class ScoringEngine:
    """
    Computes hallucination scores per token by combining:
    - Semantic Entropy (from hidden states)
    - Wasserstein distance (from Module C HalluShift)
    - TSV deviation (from Module C TSV Trainer)

    Into a single score using your research formula:
        score = 0.4×entropy + 0.4×wasserstein + 0.2×tsv_deviation
    """

    def __init__(self):
        self.threshold = HALLUCINATION_THRESHOLD
        self.suspicious_low = SUSPICIOUS_THRESHOLD_LOW
        self.w_entropy = WEIGHT_ENTROPY
        self.w_wasserstein = WEIGHT_WASSERSTEIN
        self.w_tsv = WEIGHT_TSV
        self.target_layers = TARGET_LAYERS

    # =========================================================
    # SECTION 1: Semantic Entropy
    # =========================================================

    def calculate_semantic_entropy(
        self,
        hidden_states: Dict[int, torch.Tensor]
    ) -> float:
        """
        Calculate semantic entropy from hidden state distributions.

        What is semantic entropy? (Beginner explanation)
        -------------------------------------------------
        The hidden state is a vector of 2048 numbers. We can treat
        the ABSOLUTE VALUES of these numbers as an "energy distribution"
        — some dimensions have high energy, others have low energy.

        If all dimensions have similar energy → HIGH entropy → uncertain
        If a few dimensions dominate → LOW entropy → confident

        Formula:
            1. Take absolute values of hidden state: |h|
            2. Apply softmax to get probability distribution: p = softmax(|h|)
            3. Compute entropy: H = -sum(p × log(p + ε))

        Higher entropy = the model is uncertain = hallucination signal.

        Args:
            hidden_states: Dict {layer_idx: tensor of shape (hidden_size,)}
                          One vector per target layer for this token.

        Returns:
            Float: entropy score, normalized to [0.0, 1.0]
        """
        if not hidden_states:
            return 0.5  # neutral when no data

        entropies = []

        for layer_idx, h in hidden_states.items():
            if h is None:
                continue

            # Convert to numpy
            if isinstance(h, torch.Tensor):
                h_np = h.float().numpy()
            else:
                h_np = np.array(h, dtype=np.float32)

            # Step 1: Take absolute values (all values must be non-negative)
            h_abs = np.abs(h_np)

            # Step 2: Apply softmax to get probability distribution
            # Subtract max for numerical stability
            h_shifted = h_abs - h_abs.max()
            exp_h = np.exp(h_shifted)
            p = exp_h / (exp_h.sum() + 1e-10)

            # Step 3: Compute entropy H = -sum(p × log(p))
            entropy = float(-np.sum(p * np.log(p + 1e-10)))

            # Normalize: max entropy for size n is log(n)
            max_entropy = np.log(len(p))
            normalized = entropy / (max_entropy + 1e-10)

            entropies.append(float(np.clip(normalized, 0.0, 1.0)))

        if not entropies:
            return 0.5

        # Average across all target layers
        return float(np.mean(entropies))

    # =========================================================
    # SECTION 2: Score Aggregation
    # =========================================================

    def aggregate_hallucination_score(
        self,
        entropy: float,
        wasserstein: float,
        tsv_deviation: float
    ) -> float:
        """
        Combine the three signals into one unified hallucination score.

        Your research scoring formula (AI-Added Assumption A2):
            score = 0.4 × entropy
                  + 0.4 × wasserstein_normalized
                  + 0.2 × tsv_deviation

        All inputs and outputs are in range [0.0, 1.0].
        Higher score = more likely hallucinated.

        Args:
            entropy:          Semantic entropy score [0, 1]
            wasserstein:      Average Wasserstein distance (raw, will normalize)
            tsv_deviation:    TSV deviation score [0, 1]

        Returns:
            Float: aggregated hallucination score [0.0, 1.0]
        """
        # Normalize Wasserstein to [0, 1]
        # (Wasserstein for normalized vectors is typically 0 to ~0.5)
        wasserstein_norm = float(np.clip(wasserstein / 0.5, 0.0, 1.0))

        # Weighted combination
        score = (
            self.w_entropy * entropy
            + self.w_wasserstein * wasserstein_norm
            + self.w_tsv * tsv_deviation
        )

        # Clip to valid range
        return float(np.clip(score, 0.0, 1.0))

    # =========================================================
    # SECTION 3: Threshold Application
    # =========================================================

    def apply_threshold(self, score: float) -> Tuple[bool, str]:
        """
        Apply the hallucination threshold to classify a token.

        Three zones:
        - SAFE:         score < SUSPICIOUS_THRESHOLD_LOW (default 0.45)
        - SUSPICIOUS:   SUSPICIOUS < score < HALLUCINATION_THRESHOLD (default 0.65)
        - HALLUCINATED: score > HALLUCINATION_THRESHOLD

        This threshold is configurable via config.py (FR14).
        The React UI will show:
            SAFE        → normal text (no highlight)
            SUSPICIOUS  → yellow highlight
            HALLUCINATED → red highlight

        Args:
            score: Aggregated hallucination score [0, 1]

        Returns:
            Tuple: (is_flagged: bool, risk_level: str)
        """
        if score >= self.threshold:
            return True, RiskLevel.HALLUCINATED
        elif score >= self.suspicious_low:
            return False, RiskLevel.SUSPICIOUS
        else:
            return False, RiskLevel.SAFE

    # =========================================================
    # SECTION 4: Full Token Scoring
    # =========================================================

    def score_token(
        self,
        token: str,
        position: int,
        hidden_states_per_layer: Dict[int, torch.Tensor],
        wasserstein_avg: float = 0.0,
        tsv_deviation: float = 0.5,
        is_eat: bool = False
    ) -> TokenScore:
        """
        Compute the complete hallucination score for one token.

        This is the main function called by the detection pipeline
        for each generated token.

        Args:
            token:                  The token text (e.g. " Paris")
            position:               Token position index (0, 1, 2, ...)
            hidden_states_per_layer: {layer_idx: vector} from Module B
            wasserstein_avg:        Average Wasserstein from Module C
            tsv_deviation:          TSV deviation from Module C
            is_eat:                 True if this position is an EAT

        Returns:
            TokenScore with all scores and classification
        """
        # Step 1: Calculate semantic entropy
        entropy = self.calculate_semantic_entropy(hidden_states_per_layer)

        # Step 2: Aggregate into unified score
        score = self.aggregate_hallucination_score(
            entropy, wasserstein_avg, tsv_deviation
        )

        # Step 3: Apply threshold
        is_flagged, risk_level = self.apply_threshold(score)

        return TokenScore(
            token=token,
            position=position,
            entropy=entropy,
            wasserstein=wasserstein_avg,
            tsv_deviation=tsv_deviation,
            hallucination_score=score,
            is_flagged=is_flagged,
            is_eat=is_eat,
            risk_level=risk_level,
        )

    def score_all_tokens(
        self,
        tokens: List[str],
        activations: Dict[int, List[torch.Tensor]],
        all_shift_scores: List[Dict],
        tsv_deviations: List[float],
        eat_positions: set
    ) -> List[TokenScore]:
        """
        Score ALL generated tokens in one pass.

        This is called by the detection pipeline after generation.

        Args:
            tokens:           List of generated token strings
            activations:      {layer_idx: [vec_tok0, vec_tok1, ...]}
            all_shift_scores: List of shift score dicts per token
            tsv_deviations:   List of TSV deviation scores per token
            eat_positions:    Set of EAT token position indices

        Returns:
            List[TokenScore]: one score per token
        """
        token_scores = []

        for pos, token in enumerate(tokens):
            # Get hidden states for this token across all layers
            hidden_states_for_token = {}
            for layer_idx, vectors in activations.items():
                if pos < len(vectors):
                    hidden_states_for_token[layer_idx] = vectors[pos]

            # Get Wasserstein average for this token
            wasserstein_avg = 0.0
            if pos < len(all_shift_scores) and all_shift_scores[pos]:
                shift_dict = all_shift_scores[pos]
                if shift_dict:
                    wasserstein_avg = sum(
                        s.wasserstein for s in shift_dict.values()
                    ) / len(shift_dict)

            # Get TSV deviation for this token
            tsv_dev = tsv_deviations[pos] if pos < len(tsv_deviations) else 0.5

            # Is this token an EAT?
            is_eat = pos in eat_positions

            # Score this token
            token_score = self.score_token(
                token=token,
                position=pos,
                hidden_states_per_layer=hidden_states_for_token,
                wasserstein_avg=wasserstein_avg,
                tsv_deviation=tsv_dev,
                is_eat=is_eat,
            )
            token_scores.append(token_score)

        flagged = sum(1 for s in token_scores if s.is_flagged)
        logger.info(
            f"Scored {len(tokens)} tokens: "
            f"{flagged} flagged as hallucinated"
        )
        return token_scores

    # =========================================================
    # SECTION 5: Utilities
    # =========================================================

    def get_overall_risk(self, token_scores: List[TokenScore]) -> float:
        """
        Compute overall hallucination risk for the full response.
        Returns the maximum score across all tokens.
        """
        if not token_scores:
            return 0.0
        return max(s.hallucination_score for s in token_scores)

    def update_threshold(self, new_threshold: float):
        """
        Update the hallucination threshold at runtime.
        Used by the API's POST /api/config endpoint (FR14).
        """
        if 0.0 <= new_threshold <= 1.0:
            self.threshold = new_threshold
            logger.info(f"Threshold updated to {new_threshold}")
        else:
            logger.warning(f"Invalid threshold {new_threshold} — must be in [0, 1]")

    def get_config(self) -> dict:
        """Return current scoring configuration."""
        return {
            "hallucination_threshold": self.threshold,
            "suspicious_threshold_low": self.suspicious_low,
            "weight_entropy": self.w_entropy,
            "weight_wasserstein": self.w_wasserstein,
            "weight_tsv": self.w_tsv,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_scoring_engine_instance = None


def get_scoring_engine() -> ScoringEngine:
    """Returns the global ScoringEngine instance."""
    global _scoring_engine_instance
    if _scoring_engine_instance is None:
        _scoring_engine_instance = ScoringEngine()
    return _scoring_engine_instance


# ── Quick Test ────────────────────────────────────────────────────────────────
# Run: python backend/modules/module_d_scoring.py

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 65)
    print("  DAY 9 TEST: Module D — Semantic Entropy & Score Aggregation")
    print("=" * 65 + "\n")

    engine = ScoringEngine()

    print(f"Config: threshold={engine.threshold}, "
          f"weights=({engine.w_entropy}, {engine.w_wasserstein}, {engine.w_tsv})\n")

    # TEST 1: Semantic Entropy
    print("TEST 1: calculate_semantic_entropy()")

    # High entropy: uniform distribution (model uncertain)
    uniform_states = {18: torch.ones(2048)}
    entropy_high = engine.calculate_semantic_entropy(uniform_states)

    # Low entropy: peaked distribution (model confident)
    peaked = torch.zeros(2048); peaked[0] = 100.0
    peaked_states = {18: peaked}
    entropy_low = engine.calculate_semantic_entropy(peaked_states)

    print(f"  Uniform (uncertain):  entropy = {entropy_high:.4f}  (expected HIGH)")
    print(f"  Peaked  (confident):  entropy = {entropy_low:.4f}  (expected LOW)")
    assert entropy_high > entropy_low
    assert 0.0 <= entropy_high <= 1.0
    assert 0.0 <= entropy_low <= 1.0
    print(f"  PASS: Uniform > Peaked, both in [0,1] ✅\n")

    # TEST 2: Score Aggregation
    print("TEST 2: aggregate_hallucination_score()")

    score_high = engine.aggregate_hallucination_score(
        entropy=0.9, wasserstein=0.4, tsv_deviation=0.8
    )
    score_low = engine.aggregate_hallucination_score(
        entropy=0.1, wasserstein=0.05, tsv_deviation=0.1
    )
    print(f"  High signals:  score = {score_high:.4f}  (expected HIGH)")
    print(f"  Low signals:   score = {score_low:.4f}  (expected LOW)")
    assert score_high > score_low
    assert 0.0 <= score_high <= 1.0
    assert 0.0 <= score_low <= 1.0
    print(f"  PASS: High > Low, both in [0,1] ✅\n")

    # TEST 3: Threshold
    print("TEST 3: apply_threshold()")
    flagged, level = engine.apply_threshold(0.8)
    safe, safe_level = engine.apply_threshold(0.2)
    susp, susp_level = engine.apply_threshold(0.55)

    print(f"  Score 0.80 → flagged={flagged}, level='{level}'")
    print(f"  Score 0.20 → flagged={safe},  level='{safe_level}'")
    print(f"  Score 0.55 → flagged={susp}, level='{susp_level}'")

    assert flagged is True and level == "hallucinated"
    assert safe is False and safe_level == "safe"
    assert susp is False and susp_level == "suspicious"
    print(f"  PASS: Three-zone classification correct ✅\n")

    # TEST 4: Full token scoring
    print("TEST 4: score_token()")
    torch.manual_seed(0)
    hidden = {18: torch.randn(2048), 20: torch.randn(2048), 22: torch.randn(2048)}

    ts = engine.score_token(
        token=" Paris",
        position=0,
        hidden_states_per_layer=hidden,
        wasserstein_avg=0.3,
        tsv_deviation=0.4,
        is_eat=True
    )
    print(f"  Token: '{ts.token}'  position={ts.position}")
    print(f"  entropy={ts.entropy:.4f}, wasserstein={ts.wasserstein:.4f}")
    print(f"  hallucination_score={ts.hallucination_score:.4f}")
    print(f"  risk_level='{ts.risk_level}', is_eat={ts.is_eat}")
    assert isinstance(ts.hallucination_score, float)
    assert 0.0 <= ts.hallucination_score <= 1.0
    assert ts.is_eat is True
    print(f"  PASS: TokenScore computed correctly ✅\n")

    # TEST 5: Configurable threshold (FR14)
    print("TEST 5: update_threshold() — FR14 researcher configuration")
    engine.update_threshold(0.5)
    assert engine.threshold == 0.5
    flagged_new, _ = engine.apply_threshold(0.55)
    assert flagged_new is True
    engine.update_threshold(0.65)  # restore
    print(f"  PASS: Threshold updated dynamically ✅\n")

    # TEST 6: Singleton
    print("TEST 6: Singleton pattern")
    e1 = get_scoring_engine()
    e2 = get_scoring_engine()
    assert e1 is e2
    print(f"  PASS: Same instance returned ✅\n")

    print("=" * 65)
    print("  DAY 9 DELIVERABLE CONFIRMED")
    print("=" * 65)
    print("  PASS: Semantic entropy computed from hidden states")
    print("  PASS: Score formula: 0.4×entropy + 0.4×wass + 0.2×tsv")
    print("  PASS: Threshold applied → 3 risk zones")
    print("  PASS: TokenScore dataclass with all fields")
    print("  PASS: Configurable threshold (FR14)")
    print()
    print("  Research impact:")
    print("  → Gap 1: hallucination score computed per token during generation")
    print("  → All three signals combined into one interpretable score")
    print("  → Threshold separates hallucinated from safe tokens")
    print("=" * 65 + "\n")
