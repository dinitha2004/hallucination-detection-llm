"""
tsv_trainer.py — Module C: Truthfulness Separator Vector (TSV)
==============================================================
This implements the TSV from your finalized research technique:

"Apply a Truthfulness Separator Vector (TSV) to steering the latent
representations, enhancing the separation between truthful and
hallucinated embeddings."

What is the TSV?
----------------
The TSV is a direction in the hidden state space that points from
hallucinated representations toward truthful ones. Think of it like
a compass needle that always points toward "truth" in the model's
internal space.

How it works:
-------------
1. TRAIN: Collect hidden states from truthful answers (+1)
   and hallucinated answers (-1). Train a LogisticRegression
   linear probe to separate them.

2. EXTRACT: The probe's weight vector becomes the TSV —
   the direction in embedding space that best separates
   truthful from hallucinated representations.

3. APPLY (Steering): Project any new hidden state onto the TSV
   direction. The projection score tells us how "truthful"
   that hidden state is:
   - High positive score → likely truthful
   - Near zero / negative → likely hallucinated

Research connection:
--------------------
- TSV → your finalized technique Step 3
- Latent space steering → your framework Module C
- 50+50 contrastive pairs → AI-Added Assumption A5
- TSV saved to data/memory_bank/tsv_vector.npy
- FR6: apply truthfulness separator vectors

Author: Chalani Dinitha (20211032)
"""

import sys
import logging
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict

sys.path.append(".")
from backend.config import TSV_VECTOR_PATH, TSV_TRAINING_PAIRS, TARGET_LAYERS

logger = logging.getLogger(__name__)


class TSVTrainer:
    """
    Trains and applies the Truthfulness Separator Vector.

    The TSV is a lightweight linear probe (LogisticRegression)
    trained on contrastive pairs of:
        - truthful hidden states  (label = 1)
        - hallucinated hidden states (label = 0)

    Once trained, the weight vector of the probe becomes the TSV.
    Any hidden state can then be projected onto this vector to get
    a truthfulness score between 0.0 and 1.0.
    """

    def __init__(self):
        self._tsv_vector = None          # numpy array: (hidden_size,)
        self._probe = None               # sklearn LogisticRegression
        self._is_trained = False
        self._hidden_size = None
        self._training_layer = None      # which layer the TSV was trained on

        # Try loading saved TSV from disk
        self._load_tsv()

    # =========================================================
    # SECTION 1: TSV Training
    # =========================================================

    def compute_tsv(
        self,
        truthful_states: List[torch.Tensor],
        hallucinated_states: List[torch.Tensor],
        layer_idx: int = None
    ) -> np.ndarray:
        """
        Train the TSV using contrastive hidden state pairs.

        From your research (AI-Added Assumption A5):
        "Minimum viable training: 50 truthful + 50 hallucinated pairs"

        The training data comes from TruthfulQA:
        - truthful_states: hidden states when model answers correctly
        - hallucinated_states: hidden states when model answers incorrectly

        Args:
            truthful_states:     List of hidden state tensors (label=1)
                                 Each tensor shape: (hidden_size,)
            hallucinated_states: List of hidden state tensors (label=0)
                                 Each tensor shape: (hidden_size,)
            layer_idx:           Which layer these states came from

        Returns:
            numpy array: TSV direction vector, shape (hidden_size,)
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn not installed! Run: pip install scikit-learn")
            return None

        if not truthful_states or not hallucinated_states:
            logger.error("Need both truthful and hallucinated states to train TSV")
            return None

        logger.info(
            f"Training TSV with {len(truthful_states)} truthful + "
            f"{len(hallucinated_states)} hallucinated states"
        )

        # ── Step 1: Prepare training data ────────────────────────
        # Convert to numpy and create labels
        X_truthful = np.stack([
            s.float().numpy() if isinstance(s, torch.Tensor) else s
            for s in truthful_states
        ])
        X_hallucinated = np.stack([
            s.float().numpy() if isinstance(s, torch.Tensor) else s
            for s in hallucinated_states
        ])

        X = np.vstack([X_truthful, X_hallucinated])
        y = np.array([1] * len(truthful_states) + [0] * len(hallucinated_states))

        # ── Step 2: Normalize features ────────────────────────────
        # StandardScaler ensures all dimensions contribute equally
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Step 3: Train LogisticRegression probe ────────────────
        # max_iter=1000 ensures convergence even with small datasets
        probe = LogisticRegression(
            max_iter=1000,
            C=1.0,           # regularization strength
            solver="lbfgs",  # good for small datasets
            random_state=42  # reproducibility (NFR10)
        )
        probe.fit(X_scaled, y)

        train_accuracy = probe.score(X_scaled, y)
        logger.info(f"TSV probe training accuracy: {train_accuracy:.3f}")

        # ── Step 4: Extract TSV direction ─────────────────────────
        # The weight vector of the probe IS the TSV.
        # It points in the direction that best separates truthful from hallucinated.
        tsv_vector = probe.coef_[0]  # shape: (hidden_size,)

        # Normalize to unit length (makes projection scores comparable)
        tsv_norm = np.linalg.norm(tsv_vector)
        if tsv_norm > 1e-8:
            tsv_vector = tsv_vector / tsv_norm

        # Store internally
        self._tsv_vector = tsv_vector
        self._probe = probe
        self._is_trained = True
        self._hidden_size = tsv_vector.shape[0]
        self._training_layer = layer_idx

        logger.info(
            f"TSV trained successfully! "
            f"Shape: {tsv_vector.shape}, "
            f"Norm: {np.linalg.norm(tsv_vector):.4f}, "
            f"Train accuracy: {train_accuracy:.3f}"
        )

        # Auto-save after training
        self.save_tsv()

        return tsv_vector

    # =========================================================
    # SECTION 2: TSV Steering (Projection)
    # =========================================================

    def apply_tsv_steering(
        self,
        hidden_state: torch.Tensor
    ) -> float:
        """
        Project a hidden state onto the TSV direction to get
        a truthfulness score.

        What does "projection" mean? (Beginner explanation)
        ---------------------------------------------------
        Imagine the TSV is an arrow pointing toward "truth".
        Projection measures how much your hidden state "agrees"
        with that direction:
        - Strong positive projection → hidden state points toward truth
        - Zero projection → hidden state is perpendicular to truth direction
        - Negative projection → hidden state points away from truth

        We normalize the score to [0, 1] using a sigmoid so it can
        be combined with other signals in Module D's formula.

        Args:
            hidden_state: Tensor of shape (hidden_size,)

        Returns:
            Float: truthfulness score in [0.0, 1.0]
                   Higher = more truthful
                   Lower = more likely hallucinated
        """
        if not self._is_trained or self._tsv_vector is None:
            logger.warning("TSV not trained yet — returning neutral score 0.5")
            return 0.5

        # Convert to numpy
        if isinstance(hidden_state, torch.Tensor):
            h = hidden_state.float().numpy()
        else:
            h = np.array(hidden_state, dtype=np.float32)

        # Normalize the hidden state
        h_norm = np.linalg.norm(h)
        if h_norm > 1e-8:
            h = h / h_norm

        # Project onto TSV direction (dot product)
        projection = float(np.dot(h, self._tsv_vector))

        # Convert to probability using sigmoid
        # sigmoid maps any real number to (0, 1)
        # projection > 0 → > 0.5 → more truthful
        # projection < 0 → < 0.5 → more hallucinated
        score = float(1.0 / (1.0 + np.exp(-projection * 5.0)))

        logger.debug(
            f"TSV projection: {projection:.4f} → "
            f"truthfulness score: {score:.4f}"
        )

        return score

    def get_tsv_deviation(self, hidden_state: torch.Tensor) -> float:
        """
        Get how much this hidden state DEVIATES from truthfulness.
        Used in Module D's scoring formula as the TSV component.

        Deviation = 1.0 - truthfulness_score
        Higher deviation = more likely hallucinated

        Returns:
            Float: deviation score in [0.0, 1.0]
                   Higher = more hallucinated
        """
        truthfulness = self.apply_tsv_steering(hidden_state)
        return 1.0 - truthfulness

    # =========================================================
    # SECTION 3: Synthetic Training Data Generator
    # =========================================================

    def generate_synthetic_training_data(
        self,
        n_pairs: int = None,
        hidden_size: int = 2048
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate synthetic contrastive pairs for TSV training.

        This is used when real TruthfulQA data is not yet available
        (e.g., during development and testing).

        In your real evaluation (Day 16+), this will be replaced by
        actual hidden states extracted from TruthfulQA examples.

        Strategy:
        ---------
        - Truthful states: centered around a "truthful" prototype
          with small random noise
        - Hallucinated states: centered around a different direction
          with small random noise
        - This creates a linearly separable dataset the TSV can learn

        Args:
            n_pairs:     Number of pairs (default: TSV_TRAINING_PAIRS//2)
            hidden_size: Dimension of hidden states

        Returns:
            Tuple of (truthful_states, hallucinated_states)
        """
        if n_pairs is None:
            n_pairs = TSV_TRAINING_PAIRS // 2  # 50 truthful + 50 hallucinated

        torch.manual_seed(42)  # NFR10: reproducibility

        # Create two distinct "prototype" directions
        truthful_direction = torch.randn(hidden_size)
        truthful_direction = truthful_direction / truthful_direction.norm()

        hallucinated_direction = -truthful_direction + torch.randn(hidden_size) * 0.3
        hallucinated_direction = (
            hallucinated_direction / hallucinated_direction.norm()
        )

        # Generate samples around each prototype
        truthful_states = [
            (truthful_direction * 10 + torch.randn(hidden_size) * 2)
            for _ in range(n_pairs)
        ]

        hallucinated_states = [
            (hallucinated_direction * 10 + torch.randn(hidden_size) * 2)
            for _ in range(n_pairs)
        ]

        logger.info(
            f"Generated {n_pairs} synthetic pairs "
            f"(hidden_size={hidden_size})"
        )
        return truthful_states, hallucinated_states

    # =========================================================
    # SECTION 4: Save & Load
    # =========================================================

    def save_tsv(self, path: str = None):
        """Save TSV vector to disk for reuse across sessions."""
        save_path = path or TSV_VECTOR_PATH
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        if self._tsv_vector is not None:
            np.save(save_path, self._tsv_vector)
            logger.info(f"TSV vector saved to {save_path}")

    def _load_tsv(self):
        """Load TSV vector from disk if it exists."""
        path = Path(TSV_VECTOR_PATH)
        if path.exists():
            try:
                self._tsv_vector = np.load(str(path))
                self._is_trained = True
                self._hidden_size = self._tsv_vector.shape[0]
                logger.info(
                    f"TSV loaded from {path}, "
                    f"shape: {self._tsv_vector.shape}"
                )
            except Exception as e:
                logger.warning(f"Could not load TSV: {e}")

    # =========================================================
    # SECTION 5: Properties
    # =========================================================

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def tsv_vector(self) -> Optional[np.ndarray]:
        return self._tsv_vector

    @property
    def hidden_size(self) -> Optional[int]:
        return self._hidden_size

    def get_summary(self) -> dict:
        return {
            "is_trained": self._is_trained,
            "hidden_size": self._hidden_size,
            "training_layer": self._training_layer,
            "tsv_path": TSV_VECTOR_PATH,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_tsv_trainer_instance = None


def get_tsv_trainer() -> TSVTrainer:
    """Returns the global TSVTrainer instance."""
    global _tsv_trainer_instance
    if _tsv_trainer_instance is None:
        _tsv_trainer_instance = TSVTrainer()
    return _tsv_trainer_instance


# ── Quick Test ────────────────────────────────────────────────────────────────
# Run: python backend/modules/tsv_trainer.py

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 65)
    print("  DAY 8 TEST: Module C — TSV Trainer")
    print("=" * 65 + "\n")

    trainer = TSVTrainer()

    # Step 1: Generate synthetic training data
    print("STEP 1: Generating synthetic contrastive pairs...")
    truthful, hallucinated = trainer.generate_synthetic_training_data(
        n_pairs=50, hidden_size=2048
    )
    print(f"  Truthful states:     {len(truthful)} × {truthful[0].shape}")
    print(f"  Hallucinated states: {len(hallucinated)} × {hallucinated[0].shape}\n")

    # Step 2: Train TSV
    print("STEP 2: Training TSV (LogisticRegression probe)...")
    tsv = trainer.compute_tsv(truthful, hallucinated, layer_idx=20)
    print(f"  TSV shape:  {tsv.shape}")
    print(f"  TSV norm:   {np.linalg.norm(tsv):.4f}  (should be ~1.0)")
    print(f"  TSV saved:  {TSV_VECTOR_PATH}\n")

    # Step 3: Test apply_tsv_steering
    print("STEP 3: Testing apply_tsv_steering()...")

    # Truthful state (similar to training truthful direction)
    truthful_test = truthful[0]
    score_truthful = trainer.apply_tsv_steering(truthful_test)

    # Hallucinated state (similar to training hallucinated direction)
    hallucinated_test = hallucinated[0]
    score_hallucinated = trainer.apply_tsv_steering(hallucinated_test)

    print(f"  Truthful state score:     {score_truthful:.4f}  (expected > 0.5)")
    print(f"  Hallucinated state score: {score_hallucinated:.4f}  (expected < 0.5)")

    assert score_truthful > score_hallucinated, \
        "Truthful score should be higher than hallucinated score!"
    print(f"  PASS: Truthful > Hallucinated ✅\n")

    # Step 4: Test get_tsv_deviation
    print("STEP 4: Testing get_tsv_deviation()...")
    dev_truthful = trainer.get_tsv_deviation(truthful_test)
    dev_hallucinated = trainer.get_tsv_deviation(hallucinated_test)

    print(f"  Truthful deviation:     {dev_truthful:.4f}  (expected low)")
    print(f"  Hallucinated deviation: {dev_hallucinated:.4f}  (expected high)")
    assert dev_hallucinated > dev_truthful
    print(f"  PASS: Hallucinated deviation > Truthful deviation ✅\n")

    # Step 5: Test score is scalar float in [0, 1]
    print("STEP 5: Testing score is scalar float in [0, 1]...")
    test_vec = torch.randn(2048)
    score = trainer.apply_tsv_steering(test_vec)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    print(f"  Score: {score:.4f} — is float in [0,1] ✅\n")

    # Step 6: Test singleton
    print("STEP 6: Testing singleton pattern...")
    t1 = get_tsv_trainer()
    t2 = get_tsv_trainer()
    assert t1 is t2
    print(f"  PASS: Same instance returned ✅\n")

    # Summary
    print("=" * 65)
    print("  DAY 8 DELIVERABLE CONFIRMED")
    print("=" * 65)
    print(f"  PASS: 50+50 contrastive pairs generated")
    print(f"  PASS: TSV trained (LogisticRegression probe)")
    print(f"  PASS: TSV shape: {tsv.shape}")
    print(f"  PASS: apply_tsv_steering() returns scalar in [0,1]")
    print(f"  PASS: get_tsv_deviation() — deviation score works")
    print(f"  PASS: TSV saved to {TSV_VECTOR_PATH}")
    print()
    print("  Research impact:")
    print("  → Latent space steering: hidden states separated by truth")
    print("  → TSV deviation feeds into: score = 0.4×entropy + 0.4×wass + 0.2×tsv")
    print("  → Ready for Module D (scoring) on Day 9")
    print("=" * 65 + "\n")
