"""
test_model_loader.py — Unit Tests for Model Loader (Day 2)
============================================================
These tests verify that the model loader works correctly.

HOW TO RUN:
    pytest backend/tests/test_model_loader.py -v

IMPORTANT: These tests use a tiny dummy model so they run fast
without needing to download gigabytes of weights. The real model
is tested by running model_loader.py directly.

Author: Chalani Dinitha (20211032)
"""

import sys
import pytest
import torch

sys.path.append(".")


class TestModelLoaderUnit:
    """
    Unit tests for ModelLoader.
    These run WITHOUT loading the actual LLM (too slow for unit tests).
    They test the logic and structure of the class.
    """

    def test_model_loader_instantiation(self):
        """Test that ModelLoader can be created without errors."""
        from backend.llm.model_loader import ModelLoader
        loader = ModelLoader()
        assert loader is not None
        assert loader.is_loaded() is False

    def test_get_model_raises_before_load(self):
        """
        Test that calling get_model() before load() raises a clear error.
        This prevents confusing errors later in the pipeline.
        """
        from backend.llm.model_loader import ModelLoader
        loader = ModelLoader()
        with pytest.raises(RuntimeError, match="Model not loaded yet"):
            loader.get_model()

    def test_get_tokenizer_raises_before_load(self):
        """Test that calling get_tokenizer() before load() raises a clear error."""
        from backend.llm.model_loader import ModelLoader
        loader = ModelLoader()
        with pytest.raises(RuntimeError, match="Tokenizer not loaded yet"):
            loader.get_tokenizer()

    def test_get_model_info_not_loaded(self):
        """Test model info returns correct status when not loaded."""
        from backend.llm.model_loader import ModelLoader
        loader = ModelLoader()
        info = loader.get_model_info()
        assert info["loaded"] is False

    def test_singleton_returns_same_instance(self):
        """
        Test that get_model_loader() always returns the same object.
        This confirms the singleton pattern works correctly.
        """
        from backend.llm.model_loader import get_model_loader
        loader1 = get_model_loader()
        loader2 = get_model_loader()
        assert loader1 is loader2, "Singleton should return the same instance"

    def test_config_imported_correctly(self):
        """Test that config values are accessible and have correct types."""
        from backend.config import (
            MODEL_NAME, TARGET_LAYERS, HALLUCINATION_THRESHOLD,
            WEIGHT_ENTROPY, WEIGHT_WASSERSTEIN, WEIGHT_TSV
        )
        assert isinstance(MODEL_NAME, str)
        assert len(MODEL_NAME) > 0
        assert isinstance(TARGET_LAYERS, list)
        assert len(TARGET_LAYERS) > 0
        assert all(isinstance(l, int) for l in TARGET_LAYERS)
        assert 0.0 <= HALLUCINATION_THRESHOLD <= 1.0

    def test_config_weights_sum_to_one(self):
        """
        Test that score aggregation weights sum to 1.0.
        If they don't, scores will be miscalibrated.
        This directly tests your research's scoring formula.
        """
        from backend.config import WEIGHT_ENTROPY, WEIGHT_WASSERSTEIN, WEIGHT_TSV
        total = WEIGHT_ENTROPY + WEIGHT_WASSERSTEIN + WEIGHT_TSV
        assert abs(total - 1.0) < 0.001, (
            f"Weights must sum to 1.0, got {total:.4f}. "
            f"Check WEIGHT_ENTROPY, WEIGHT_WASSERSTEIN, WEIGHT_TSV in config.py"
        )

    def test_config_validation_passes(self):
        """Test that the config validation function passes with default values."""
        from backend.config import validate_config
        assert validate_config() is True


class TestGenerationResultDataclass:
    """Tests for the GenerationResult dataclass."""

    def test_default_values(self):
        """Test that GenerationResult initializes with correct defaults."""
        from backend.llm.inference_engine import GenerationResult
        result = GenerationResult()
        assert result.generated_text == ""
        assert result.tokens == []
        assert result.token_ids == []
        assert result.hidden_states_by_layer == {}
        assert result.error is None

    def test_success_property_false_when_no_tokens(self):
        """Test that success=False when no tokens were generated."""
        from backend.llm.inference_engine import GenerationResult
        result = GenerationResult()
        assert result.success is False

    def test_success_property_true_with_tokens(self):
        """Test that success=True when tokens are present and no error."""
        from backend.llm.inference_engine import GenerationResult
        result = GenerationResult(
            tokens=["hello", " world"],
            generated_text="hello world"
        )
        assert result.success is True

    def test_success_false_with_error(self):
        """Test that success=False if there's an error, even with tokens."""
        from backend.llm.inference_engine import GenerationResult
        result = GenerationResult(
            tokens=["hello"],
            error="Something went wrong"
        )
        assert result.success is False

    def test_num_tokens_property(self):
        """Test the num_tokens property counts correctly."""
        from backend.llm.inference_engine import GenerationResult
        result = GenerationResult(tokens=["The", " sky", " is", " blue"])
        assert result.num_tokens == 4

    def test_hidden_states_structure(self):
        """Test that hidden states can be stored and retrieved by layer."""
        from backend.llm.inference_engine import GenerationResult
        dummy_tensor = torch.randn(2048)  # fake hidden state vector
        result = GenerationResult()
        result.hidden_states_by_layer = {
            16: [dummy_tensor, dummy_tensor],
            20: [dummy_tensor],
        }
        assert 16 in result.hidden_states_by_layer
        assert len(result.hidden_states_by_layer[16]) == 2
        assert result.hidden_states_by_layer[20][0].shape == torch.Size([2048])


# ── Run tests directly ────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
