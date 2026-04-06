"""
test_tsv.py — Unit Tests for TSV Trainer (Day 8)
=================================================
Tests all functions in tsv_trainer.py.
Uses synthetic data — no model loading needed.

HOW TO RUN:
    pytest backend/tests/test_tsv.py -v

Author: Chalani Dinitha (20211032)
Day 8 Deliverable: TSV vector trained and saved, steering works
"""

import sys
import pytest
import numpy as np
import torch

sys.path.append(".")


class TestTSVTrainerSetup:
    """Tests for TSVTrainer initialization."""

    def test_instantiation(self):
        """TSVTrainer creates without errors."""
        from backend.modules.tsv_trainer import TSVTrainer
        trainer = TSVTrainer()
        assert trainer is not None

    def test_singleton_returns_same_instance(self):
        """get_tsv_trainer() always returns same instance."""
        from backend.modules.tsv_trainer import get_tsv_trainer
        t1 = get_tsv_trainer()
        t2 = get_tsv_trainer()
        assert t1 is t2

    def test_initial_state_before_training(self):
        """Before training, is_trained depends on whether TSV file exists."""
        from backend.modules.tsv_trainer import TSVTrainer
        trainer = TSVTrainer.__new__(TSVTrainer)
        trainer._tsv_vector = None
        trainer._probe = None
        trainer._is_trained = False
        trainer._hidden_size = None
        trainer._training_layer = None
        assert trainer.is_trained is False
        assert trainer.tsv_vector is None

    def test_get_summary_structure(self):
        """get_summary() returns dict with required keys."""
        from backend.modules.tsv_trainer import TSVTrainer
        trainer = TSVTrainer()
        summary = trainer.get_summary()
        assert "is_trained" in summary
        assert "hidden_size" in summary
        assert "training_layer" in summary
        assert "tsv_path" in summary


class TestSyntheticDataGeneration:
    """Tests for generate_synthetic_training_data()."""

    def setup_method(self):
        from backend.modules.tsv_trainer import TSVTrainer
        self.trainer = TSVTrainer()

    def test_returns_correct_counts(self):
        """Should return n_pairs truthful and n_pairs hallucinated."""
        truthful, hallucinated = self.trainer.generate_synthetic_training_data(
            n_pairs=10, hidden_size=64
        )
        assert len(truthful) == 10
        assert len(hallucinated) == 10

    def test_returns_torch_tensors(self):
        """Generated states should be torch.Tensor objects."""
        truthful, hallucinated = self.trainer.generate_synthetic_training_data(
            n_pairs=5, hidden_size=64
        )
        for t in truthful:
            assert isinstance(t, torch.Tensor)
        for h in hallucinated:
            assert isinstance(h, torch.Tensor)

    def test_correct_hidden_size(self):
        """Generated tensors should have correct hidden_size dimension."""
        hidden_size = 128
        truthful, hallucinated = self.trainer.generate_synthetic_training_data(
            n_pairs=5, hidden_size=hidden_size
        )
        for t in truthful:
            assert t.shape[0] == hidden_size
        for h in hallucinated:
            assert h.shape[0] == hidden_size

    def test_reproducible_with_seed(self):
        """Same seed should produce same synthetic data."""
        t1, h1 = self.trainer.generate_synthetic_training_data(
            n_pairs=5, hidden_size=64
        )
        t2, h2 = self.trainer.generate_synthetic_training_data(
            n_pairs=5, hidden_size=64
        )
        # Tensors should be identical (same seed=42 used internally)
        assert torch.allclose(t1[0], t2[0])

    def test_truthful_and_hallucinated_are_different(self):
        """Truthful and hallucinated prototypes should be different."""
        truthful, hallucinated = self.trainer.generate_synthetic_training_data(
            n_pairs=5, hidden_size=64
        )
        # Means should be different
        mean_t = torch.stack(truthful).mean(0)
        mean_h = torch.stack(hallucinated).mean(0)
        cosine = float(torch.nn.functional.cosine_similarity(
            mean_t.unsqueeze(0), mean_h.unsqueeze(0)
        ))
        assert cosine < 0.9, \
            "Truthful and hallucinated means should be clearly different"


class TestComputeTSV:
    """Tests for compute_tsv() — the TSV training function."""

    def setup_method(self):
        from backend.modules.tsv_trainer import TSVTrainer
        self.trainer = TSVTrainer()
        # Generate small dataset for fast tests
        self.truthful, self.hallucinated = \
            self.trainer.generate_synthetic_training_data(
                n_pairs=30, hidden_size=64
            )

    def test_tsv_returns_numpy_array(self):
        """compute_tsv() should return a numpy array."""
        tsv = self.trainer.compute_tsv(
            self.truthful, self.hallucinated
        )
        assert isinstance(tsv, np.ndarray)

    def test_tsv_correct_shape(self):
        """TSV shape should match hidden_size."""
        tsv = self.trainer.compute_tsv(self.truthful, self.hallucinated)
        assert tsv.shape == (64,)

    def test_tsv_is_unit_vector(self):
        """TSV should be normalized to unit length."""
        tsv = self.trainer.compute_tsv(self.truthful, self.hallucinated)
        norm = float(np.linalg.norm(tsv))
        assert abs(norm - 1.0) < 0.001, \
            f"TSV norm should be 1.0, got {norm:.4f}"

    def test_is_trained_after_compute(self):
        """After training, is_trained should be True."""
        self.trainer.compute_tsv(self.truthful, self.hallucinated)
        assert self.trainer.is_trained is True

    def test_empty_truthful_returns_none(self):
        """Empty truthful list should return None gracefully."""
        result = self.trainer.compute_tsv([], self.hallucinated)
        assert result is None

    def test_empty_hallucinated_returns_none(self):
        """Empty hallucinated list should return None gracefully."""
        result = self.trainer.compute_tsv(self.truthful, [])
        assert result is None


class TestApplyTSVSteering:
    """Tests for apply_tsv_steering() — the projection function."""

    def setup_method(self):
        from backend.modules.tsv_trainer import TSVTrainer
        self.trainer = TSVTrainer()
        # Train with synthetic data
        truthful, hallucinated = self.trainer.generate_synthetic_training_data(
            n_pairs=30, hidden_size=64
        )
        self.trainer.compute_tsv(truthful, hallucinated)
        self.truthful = truthful
        self.hallucinated = hallucinated

    def test_returns_float(self):
        """apply_tsv_steering() must return a float."""
        test_vec = torch.randn(64)
        score = self.trainer.apply_tsv_steering(test_vec)
        assert isinstance(score, float)

    def test_score_in_zero_to_one(self):
        """Score must be in [0.0, 1.0]."""
        for _ in range(20):
            test_vec = torch.randn(64)
            score = self.trainer.apply_tsv_steering(test_vec)
            assert 0.0 <= score <= 1.0, \
                f"Score {score} not in [0, 1]"

    def test_truthful_scores_higher_than_hallucinated(self):
        """
        Truthful states should score higher than hallucinated states
        on average. This is the core property of the TSV.
        """
        truthful_scores = [
            self.trainer.apply_tsv_steering(t) for t in self.truthful[:10]
        ]
        hallucinated_scores = [
            self.trainer.apply_tsv_steering(h) for h in self.hallucinated[:10]
        ]

        avg_truthful = sum(truthful_scores) / len(truthful_scores)
        avg_hallucinated = sum(hallucinated_scores) / len(hallucinated_scores)

        print(f"\n  Avg truthful score: {avg_truthful:.4f}")
        print(f"  Avg hallucinated score: {avg_hallucinated:.4f}")

        assert avg_truthful > avg_hallucinated, \
            f"Truthful avg ({avg_truthful:.4f}) should > hallucinated ({avg_hallucinated:.4f})"

    def test_accepts_torch_tensor(self):
        """Should accept torch.Tensor input."""
        vec = torch.randn(64)
        score = self.trainer.apply_tsv_steering(vec)
        assert isinstance(score, float)

    def test_accepts_numpy_array(self):
        """Should accept numpy array input."""
        vec = np.random.randn(64).astype(np.float32)
        score = self.trainer.apply_tsv_steering(vec)
        assert isinstance(score, float)

    def test_untrained_returns_neutral_score(self):
        """Untrained TSV should return 0.5 (neutral)."""
        from backend.modules.tsv_trainer import TSVTrainer
        fresh = TSVTrainer.__new__(TSVTrainer)
        fresh._tsv_vector = None
        fresh._is_trained = False
        fresh._hidden_size = None
        fresh._probe = None
        fresh._training_layer = None

        score = fresh.apply_tsv_steering(torch.randn(64))
        assert score == 0.5


class TestGetTSVDeviation:
    """Tests for get_tsv_deviation()."""

    def setup_method(self):
        from backend.modules.tsv_trainer import TSVTrainer
        self.trainer = TSVTrainer()
        truthful, hallucinated = self.trainer.generate_synthetic_training_data(
            n_pairs=30, hidden_size=64
        )
        self.trainer.compute_tsv(truthful, hallucinated)
        self.truthful = truthful
        self.hallucinated = hallucinated

    def test_deviation_is_complement_of_score(self):
        """deviation = 1 - steering_score."""
        vec = torch.randn(64)
        score = self.trainer.apply_tsv_steering(vec)
        deviation = self.trainer.get_tsv_deviation(vec)
        assert abs((score + deviation) - 1.0) < 0.001

    def test_deviation_in_zero_to_one(self):
        """Deviation must be in [0.0, 1.0]."""
        for _ in range(10):
            vec = torch.randn(64)
            dev = self.trainer.get_tsv_deviation(vec)
            assert 0.0 <= dev <= 1.0

    def test_hallucinated_has_higher_deviation(self):
        """Hallucinated states should have higher deviation than truthful."""
        dev_truthful = [
            self.trainer.get_tsv_deviation(t) for t in self.truthful[:10]
        ]
        dev_hallucinated = [
            self.trainer.get_tsv_deviation(h) for h in self.hallucinated[:10]
        ]

        avg_dev_t = sum(dev_truthful) / len(dev_truthful)
        avg_dev_h = sum(dev_hallucinated) / len(dev_hallucinated)

        assert avg_dev_h > avg_dev_t, \
            f"Hallucinated deviation ({avg_dev_h:.4f}) should > truthful ({avg_dev_t:.4f})"


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
