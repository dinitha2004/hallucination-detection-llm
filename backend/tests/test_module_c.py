"""
test_module_c.py — Unit Tests for Module C: HalluShift Analyzer
================================================================
Tests all functions in module_c_hallushift.py.
All tests use dummy tensors — no model loading needed.

HOW TO RUN:
    pytest backend/tests/test_module_c.py -v

Author: Chalani Dinitha (20211032)
Day 7 Deliverable: Wasserstein & cosine shift scores computed per layer pair
"""

import sys
import pytest
import torch
import numpy as np

sys.path.append(".")


class TestWassersteinDistance:
    """Tests for calculate_wasserstein_distance()."""

    def setup_method(self):
        from backend.modules.module_c_hallushift import HalluShiftAnalyzer
        self.analyzer = HalluShiftAnalyzer()

    def test_wasserstein_is_always_positive(self):
        """Wasserstein distance must always be >= 0."""
        a = torch.randn(2048)
        b = torch.randn(2048)
        dist = self.analyzer.calculate_wasserstein_distance(a, b)
        assert dist >= 0.0, f"Expected >= 0, got {dist}"

    def test_wasserstein_same_vector_is_zero(self):
        """Distance between identical vectors should be 0."""
        a = torch.randn(2048)
        dist = self.analyzer.calculate_wasserstein_distance(a, a)
        assert dist < 0.001, f"Expected ~0, got {dist}"

    def test_wasserstein_returns_float(self):
        """Return type must be float."""
        a = torch.randn(2048)
        b = torch.randn(2048)
        result = self.analyzer.calculate_wasserstein_distance(a, b)
        assert isinstance(result, float)

    def test_wasserstein_different_vectors_positive(self):
        """Different random vectors should have positive distance."""
        torch.manual_seed(0)
        a = torch.randn(2048)
        b = torch.randn(2048)
        dist = self.analyzer.calculate_wasserstein_distance(a, b)
        assert dist > 0.0

    def test_wasserstein_similar_vectors_smaller_than_different(self):
        """
        More similar vectors should have smaller Wasserstein distance
        than very different vectors.
        """
        torch.manual_seed(42)
        base = torch.randn(2048)
        similar = base + torch.randn(2048) * 0.01    # tiny noise
        different = torch.randn(2048)                 # completely different

        dist_similar = self.analyzer.calculate_wasserstein_distance(base, similar)
        dist_different = self.analyzer.calculate_wasserstein_distance(base, different)

        assert dist_similar < dist_different, \
            f"Similar ({dist_similar:.4f}) should be < different ({dist_different:.4f})"

    def test_wasserstein_accepts_numpy_arrays(self):
        """Should also work with numpy arrays, not just torch tensors."""
        a = np.random.randn(2048).astype(np.float32)
        b = np.random.randn(2048).astype(np.float32)
        dist = self.analyzer.calculate_wasserstein_distance(a, b)
        assert isinstance(dist, float)
        assert dist >= 0.0


class TestCosineSimilarity:
    """Tests for calculate_cosine_similarity()."""

    def setup_method(self):
        from backend.modules.module_c_hallushift import HalluShiftAnalyzer
        self.analyzer = HalluShiftAnalyzer()

    def test_cosine_same_direction_is_one(self):
        """Identical vectors have cosine similarity = 1.0."""
        v = torch.ones(2048)
        result = self.analyzer.calculate_cosine_similarity(v, v)
        assert abs(result - 1.0) < 0.001

    def test_cosine_opposite_direction_is_minus_one(self):
        """Opposite vectors have cosine similarity = -1.0."""
        a = torch.ones(2048)
        b = -torch.ones(2048)
        result = self.analyzer.calculate_cosine_similarity(a, b)
        assert abs(result - (-1.0)) < 0.001

    def test_cosine_perpendicular_is_zero(self):
        """Perpendicular vectors have cosine similarity = 0.0."""
        a = torch.zeros(2048); a[0] = 1.0
        b = torch.zeros(2048); b[1] = 1.0
        result = self.analyzer.calculate_cosine_similarity(a, b)
        assert abs(result) < 0.001

    def test_cosine_always_in_minus_one_to_one(self):
        """Result must always be in [-1.0, 1.0]."""
        for _ in range(10):
            a = torch.randn(2048)
            b = torch.randn(2048)
            result = self.analyzer.calculate_cosine_similarity(a, b)
            assert -1.0 <= result <= 1.0, \
                f"Cosine {result} not in [-1, 1]"

    def test_cosine_returns_float(self):
        """Return type must be float."""
        a = torch.randn(2048)
        b = torch.randn(2048)
        result = self.analyzer.calculate_cosine_similarity(a, b)
        assert isinstance(result, float)

    def test_cosine_zero_vector_returns_zero(self):
        """Zero vector should return 0.0 gracefully."""
        a = torch.zeros(2048)
        b = torch.randn(2048)
        result = self.analyzer.calculate_cosine_similarity(a, b)
        assert result == 0.0


class TestDistributionShift:
    """Tests for compute_distribution_shift() with sliding window."""

    def setup_method(self):
        from backend.modules.module_c_hallushift import HalluShiftAnalyzer
        self.analyzer = HalluShiftAnalyzer()

        # Create fake activations for 3 layers, 2 tokens each
        torch.manual_seed(0)
        self.fake_activations = {
            18: [torch.randn(2048), torch.randn(2048)],
            20: [torch.randn(2048), torch.randn(2048)],
            22: [torch.randn(2048), torch.randn(2048)],
        }

    def test_returns_correct_layer_pairs(self):
        """
        With target layers [18, 20, 22] and window=2,
        should return pairs (18,20) and (20,22).
        """
        shifts = self.analyzer.compute_distribution_shift(
            self.fake_activations, token_position=0
        )
        assert (18, 20) in shifts
        assert (20, 22) in shifts
        assert len(shifts) == 2

    def test_shift_scores_are_shift_score_objects(self):
        """Values in the dict must be ShiftScore objects."""
        from backend.modules.module_c_hallushift import ShiftScore
        shifts = self.analyzer.compute_distribution_shift(
            self.fake_activations, token_position=0
        )
        for score in shifts.values():
            assert isinstance(score, ShiftScore)

    def test_wasserstein_in_shift_score_is_positive(self):
        """Wasserstein in ShiftScore must be >= 0."""
        shifts = self.analyzer.compute_distribution_shift(
            self.fake_activations, token_position=0
        )
        for score in shifts.values():
            assert score.wasserstein >= 0.0

    def test_cosine_in_shift_score_in_valid_range(self):
        """Cosine in ShiftScore must be in [-1, 1]."""
        shifts = self.analyzer.compute_distribution_shift(
            self.fake_activations, token_position=0
        )
        for score in shifts.values():
            assert -1.0 <= score.cosine <= 1.0

    def test_magnitude_in_zero_to_one(self):
        """Shift magnitude must be in [0, 1]."""
        shifts = self.analyzer.compute_distribution_shift(
            self.fake_activations, token_position=0
        )
        for score in shifts.values():
            assert 0.0 <= score.shift_magnitude <= 1.0, \
                f"Magnitude {score.shift_magnitude} not in [0, 1]"

    def test_empty_activations_returns_empty_dict(self):
        """Empty activations returns empty dict."""
        shifts = self.analyzer.compute_distribution_shift({})
        assert shifts == {}

    def test_single_layer_returns_empty_dict(self):
        """Only one layer — cannot compute pair shift."""
        shifts = self.analyzer.compute_distribution_shift(
            {18: [torch.randn(2048)]}
        )
        assert shifts == {}

    def test_token_position_second_token(self):
        """Should work for token_position=1 as well."""
        shifts = self.analyzer.compute_distribution_shift(
            self.fake_activations, token_position=1
        )
        assert len(shifts) == 2


class TestAllTokenShifts:
    """Tests for compute_all_token_shifts()."""

    def setup_method(self):
        from backend.modules.module_c_hallushift import HalluShiftAnalyzer
        self.analyzer = HalluShiftAnalyzer()

    def test_returns_list_of_same_length_as_tokens(self):
        """Should return one shift dict per token."""
        torch.manual_seed(0)
        activations = {
            18: [torch.randn(2048) for _ in range(5)],
            20: [torch.randn(2048) for _ in range(5)],
            22: [torch.randn(2048) for _ in range(5)],
        }
        all_shifts = self.analyzer.compute_all_token_shifts(activations)
        assert len(all_shifts) == 5

    def test_returns_empty_for_empty_activations(self):
        """Empty activations returns empty list."""
        result = self.analyzer.compute_all_token_shifts({})
        assert result == []

    def test_get_max_shift_for_token(self):
        """get_max_shift_for_token returns float in [0, 1]."""
        torch.manual_seed(1)
        activations = {
            18: [torch.randn(2048)],
            20: [torch.randn(2048)],
            22: [torch.randn(2048)],
        }
        shifts = self.analyzer.compute_distribution_shift(activations, 0)
        max_shift = self.analyzer.get_max_shift_for_token(shifts)
        assert isinstance(max_shift, float)
        assert 0.0 <= max_shift <= 1.0

    def test_get_average_wasserstein(self):
        """get_average_wasserstein returns positive float."""
        torch.manual_seed(2)
        activations = {
            18: [torch.randn(2048)],
            20: [torch.randn(2048)],
            22: [torch.randn(2048)],
        }
        shifts = self.analyzer.compute_distribution_shift(activations, 0)
        avg_wass = self.analyzer.get_average_wasserstein(shifts)
        assert isinstance(avg_wass, float)
        assert avg_wass >= 0.0


class TestShiftScoreDataclass:
    """Tests for the ShiftScore dataclass."""

    def test_shift_score_creation(self):
        from backend.modules.module_c_hallushift import ShiftScore
        s = ShiftScore(layer_pair=(18, 20), wasserstein=0.5, cosine=0.7)
        assert s.layer_from == 18
        assert s.layer_to == 20

    def test_is_high_shift_true(self):
        from backend.modules.module_c_hallushift import ShiftScore
        s = ShiftScore((18, 20), 1.0, 0.0, shift_magnitude=0.8)
        assert s.is_high_shift is True

    def test_is_high_shift_false(self):
        from backend.modules.module_c_hallushift import ShiftScore
        s = ShiftScore((18, 20), 0.1, 0.9, shift_magnitude=0.2)
        assert s.is_high_shift is False

    def test_singleton_returns_same_instance(self):
        from backend.modules.module_c_hallushift import get_hallushift_analyzer
        a1 = get_hallushift_analyzer()
        a2 = get_hallushift_analyzer()
        assert a1 is a2


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
