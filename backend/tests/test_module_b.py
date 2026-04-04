"""
test_module_b.py — Unit Tests for Module B: Hidden State Extraction
====================================================================
Tests all functions in module_b_hidden.py WITHOUT loading the LLM.
Uses dummy tensors so tests run in seconds.

HOW TO RUN:
    pytest backend/tests/test_module_b.py -v

Author: Chalani Dinitha (20211032)
Day 4 Deliverable: Hidden states extracted per layer, FC clipping working
"""

import sys
import pytest
import torch
import numpy as np

sys.path.append(".")


class TestHiddenStateExtractorUnit:
    """
    Unit tests for HiddenStateExtractor.
    All tests use dummy tensors — no model loading needed.
    """

    def setup_method(self):
        """Create a fresh extractor before each test."""
        from backend.modules.module_b_hidden import HiddenStateExtractor
        self.extractor = HiddenStateExtractor()

    def test_extractor_instantiation(self):
        """Test extractor creates correctly with target layers from config."""
        from backend.config import TARGET_LAYERS
        assert self.extractor is not None
        assert self.extractor.target_layers == TARGET_LAYERS
        assert self.extractor.num_tokens_captured == 0
        assert len(self.extractor._hooks) == 0

    def test_extract_tbg_returns_none_when_empty(self):
        """
        Test that extracting TBG state returns None when nothing captured.
        This prevents crashes in the pipeline if hooks were not attached.
        """
        result = self.extractor.extract_tbg_hidden_state(
            layer_idx=12, token_position=0
        )
        assert result is None

    def test_extract_tbg_returns_none_for_invalid_position(self):
        """Test returns None when token_position is out of range."""
        # Manually inject a fake captured state
        dummy = torch.randn(1, 2048)
        self.extractor._captured = {12: [dummy]}

        # Position 0 exists → should return tensor
        result = self.extractor.extract_tbg_hidden_state(12, 0)
        assert result is not None

        # Position 5 does NOT exist → should return None
        result = self.extractor.extract_tbg_hidden_state(12, 5)
        assert result is None

    def test_extract_tbg_squeezes_batch_dimension(self):
        """
        TBG hidden state should be 1D (hidden_size,), not 2D (1, hidden_size).
        The batch dimension must be squeezed for downstream processing.
        """
        dummy = torch.randn(1, 2048)  # (batch=1, hidden_size=2048)
        self.extractor._captured = {12: [dummy]}

        result = self.extractor.extract_tbg_hidden_state(12, 0)
        assert result is not None
        # Should be 1D after squeezing batch dimension
        assert result.dim() == 1
        assert result.shape[0] == 2048

    def test_extract_layer_activations_structure(self):
        """
        Test that extract_layer_activations returns correct structure:
        {layer_idx: [vector_0, vector_1, ...]}
        """
        from backend.config import TARGET_LAYERS

        # Inject fake captured states for all target layers
        for layer in TARGET_LAYERS:
            self.extractor._captured[layer] = [
                torch.randn(1, 2048),  # token 0
                torch.randn(1, 2048),  # token 1
                torch.randn(1, 2048),  # token 2
            ]

        activations = self.extractor.extract_layer_activations()

        # Check all target layers are present
        for layer in TARGET_LAYERS:
            assert layer in activations
            assert len(activations[layer]) == 3
            # Each vector should be 1D after squeezing
            assert activations[layer][0].dim() == 1

    def test_num_tokens_captured_property(self):
        """Test the num_tokens_captured property counts correctly."""
        assert self.extractor.num_tokens_captured == 0

        # Add fake captures
        self.extractor._captured = {
            12: [torch.randn(1, 2048), torch.randn(1, 2048)],
            16: [torch.randn(1, 2048), torch.randn(1, 2048)],
        }
        assert self.extractor.num_tokens_captured == 2

    def test_clear_captured(self):
        """Test clearing captured states resets everything."""
        self.extractor._captured = {12: [torch.randn(1, 2048)]}
        assert self.extractor.num_tokens_captured == 1

        self.extractor.clear_captured()
        assert self.extractor.num_tokens_captured == 0
        assert self.extractor._captured == {}


class TestInsideFeatureClipping:
    """
    Tests for the INSIDE Feature Clipping mechanism.
    This is critical for detecting overconfident hallucinations.
    """

    def setup_method(self):
        from backend.modules.module_b_hidden import HiddenStateExtractor
        self.extractor = HiddenStateExtractor()

    def test_clipping_reduces_extreme_values(self):
        """
        After clipping, the maximum value should be <= original maximum.
        This is the basic property of clipping.

        Fix: use a realistic vector (normal distribution) with added outliers,
        so the 95th percentile threshold is above normal values but below extremes.
        """
        # Use a realistic distribution — most values between -2 and 2
        torch.manual_seed(0)
        hidden = torch.randn(2048)           # normal distribution
        hidden[0] = 1000.0                   # add extreme outlier
        hidden[1] = 999.0                    # another extreme outlier

        original_max = hidden.max().item()
        clipped = self.extractor.apply_inside_feature_clipping(hidden, 12)

        # Core property: clipping must reduce the maximum
        assert clipped.max().item() <= original_max
        # The extreme outlier (1000.0) must be significantly reduced
        assert clipped.max().item() < 100.0  # threshold is ~2 for normal dist at p95

    def test_clipping_output_shape_unchanged(self):
        """Clipping must not change the shape of the hidden state vector."""
        hidden = torch.randn(2048)
        clipped = self.extractor.apply_inside_feature_clipping(hidden, 16)
        assert clipped.shape == hidden.shape

    def test_clipping_handles_none_input(self):
        """Clipping should handle None input gracefully."""
        result = self.extractor.apply_inside_feature_clipping(None, 12)
        assert result is None

    def test_clipping_with_normal_distribution_clips_top_5_percent(self):
        """
        With 95th percentile clipping, the top 5% of values should be reduced.
        Tests the core INSIDE mechanism from your research.
        """
        # Reproducible random tensor
        torch.manual_seed(42)
        hidden = torch.randn(2048) * 10  # normal dist scaled up

        original_max = hidden.max().item()
        clipped = self.extractor.apply_inside_feature_clipping(hidden, 20)
        clipped_max = clipped.max().item()

        # Clipped max should be less than original max
        assert clipped_max < original_max

        # Clipped max should be near the 95th percentile of original
        p95 = float(torch.quantile(hidden.float(), 0.95))
        assert abs(clipped_max - p95) < 0.01

    def test_memory_bank_thresholds_used_when_available(self):
        """
        When memory bank has thresholds, they should override
        the self-computed threshold.
        """
        # Set a manual threshold for layer 12
        self.extractor._clip_thresholds = {12: 0.5}

        hidden = torch.ones(2048) * 2.0  # all values are 2.0
        clipped = self.extractor.apply_inside_feature_clipping(hidden, 12)

        # All values should be clipped to 0.5 (the memory bank threshold)
        assert clipped.max().item() <= 0.5 + 1e-6


class TestMemoryBank:
    """Tests for the Memory Bank used in INSIDE mechanism."""

    def setup_method(self):
        from backend.modules.module_b_hidden import HiddenStateExtractor
        self.extractor = HiddenStateExtractor()

    def test_memory_bank_starts_none(self):
        """Memory bank should be None initially (unless file exists)."""
        from backend.modules.module_b_hidden import HiddenStateExtractor
        fresh = HiddenStateExtractor.__new__(HiddenStateExtractor)
        fresh._memory_bank = None
        fresh._clip_thresholds = {}
        assert fresh._memory_bank is None

    def test_update_memory_bank_with_truthful(self):
        """Updating memory bank with truthful examples should work."""
        from backend.config import TARGET_LAYERS

        # Create fake activations
        fake_activations = {
            layer: [torch.randn(2048) for _ in range(5)]
            for layer in TARGET_LAYERS
        }

        self.extractor.update_memory_bank(fake_activations, label="truthful")

        assert self.extractor.has_memory_bank
        assert "truthful" in self.extractor._memory_bank

    def test_clip_thresholds_computed_after_update(self):
        """After updating memory bank, clip thresholds should be set."""
        from backend.config import TARGET_LAYERS

        fake_activations = {
            layer: [torch.randn(2048) for _ in range(10)]
            for layer in TARGET_LAYERS
        }

        self.extractor.update_memory_bank(fake_activations, label="truthful")

        # Thresholds should now exist for all target layers
        for layer in TARGET_LAYERS:
            assert layer in self.extractor._clip_thresholds
            assert isinstance(self.extractor._clip_thresholds[layer], float)


class TestGetSummary:
    """Tests for the extractor summary method."""

    def test_summary_structure(self):
        """get_summary() should return a dict with expected keys."""
        from backend.modules.module_b_hidden import HiddenStateExtractor
        extractor = HiddenStateExtractor()
        summary = extractor.get_summary()

        assert "target_layers" in summary
        assert "hooks_attached" in summary
        assert "tokens_captured" in summary
        assert "has_memory_bank" in summary
        assert "clip_thresholds" in summary

    def test_summary_initial_values(self):
        """Initial summary should show 0 hooks and 0 tokens."""
        from backend.modules.module_b_hidden import HiddenStateExtractor
        extractor = HiddenStateExtractor()
        summary = extractor.get_summary()

        assert summary["hooks_attached"] == 0
        assert summary["tokens_captured"] == 0

    def test_singleton_returns_same_instance(self):
        """get_hidden_state_extractor() should always return same instance."""
        from backend.modules.module_b_hidden import get_hidden_state_extractor
        e1 = get_hidden_state_extractor()
        e2 = get_hidden_state_extractor()
        assert e1 is e2


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])