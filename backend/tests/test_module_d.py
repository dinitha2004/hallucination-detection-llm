"""
test_module_d.py — Unit Tests for Module D: Scoring Engine
===========================================================
Tests all functions in module_d_scoring.py.
All tests use dummy tensors — no model loading needed.

HOW TO RUN:
    pytest backend/tests/test_module_d.py -v

Author: Chalani Dinitha (20211032)
Day 9 Deliverable: Hallucination score computed per token, threshold applied
"""

import sys
import pytest
import torch
import numpy as np

sys.path.append(".")


class TestSemanticEntropy:
    """Tests for calculate_semantic_entropy()."""

    def setup_method(self):
        from backend.modules.module_d_scoring import ScoringEngine
        self.engine = ScoringEngine()

    def test_entropy_returns_float(self):
        """Entropy must be a float."""
        h = {18: torch.randn(2048)}
        result = self.engine.calculate_semantic_entropy(h)
        assert isinstance(result, float)

    def test_entropy_in_zero_to_one(self):
        """Entropy must be in [0, 1]."""
        for _ in range(10):
            h = {18: torch.randn(2048)}
            result = self.engine.calculate_semantic_entropy(h)
            assert 0.0 <= result <= 1.0, f"Entropy {result} not in [0,1]"

    def test_uniform_higher_than_peaked(self):
        """
        Uniform distribution = high entropy (model uncertain).
        Peaked distribution = low entropy (model confident).
        """
        uniform = {18: torch.ones(2048)}
        peaked = torch.zeros(2048)
        peaked[0] = 1000.0
        peaked_d = {18: peaked}

        entropy_uniform = self.engine.calculate_semantic_entropy(uniform)
        entropy_peaked = self.engine.calculate_semantic_entropy(peaked_d)

        assert entropy_uniform > entropy_peaked, \
            f"Uniform ({entropy_uniform:.4f}) should > peaked ({entropy_peaked:.4f})"

    def test_empty_dict_returns_neutral(self):
        """Empty hidden states returns neutral 0.5."""
        result = self.engine.calculate_semantic_entropy({})
        assert result == 0.5

    def test_multiple_layers_averaged(self):
        """Multiple layers should give a valid averaged entropy."""
        h = {
            18: torch.randn(2048),
            20: torch.randn(2048),
            22: torch.randn(2048),
        }
        result = self.engine.calculate_semantic_entropy(h)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestAggregateHallucinationScore:
    """Tests for aggregate_hallucination_score()."""

    def setup_method(self):
        from backend.modules.module_d_scoring import ScoringEngine
        self.engine = ScoringEngine()

    def test_score_in_zero_to_one(self):
        """Score must always be in [0, 1]."""
        combos = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.5, 0.3, 0.4),
            (0.9, 0.8, 0.7),
        ]
        for e, w, t in combos:
            score = self.engine.aggregate_hallucination_score(e, w, t)
            assert 0.0 <= score <= 1.0, \
                f"Score {score} not in [0,1] for ({e},{w},{t})"

    def test_high_signals_give_high_score(self):
        """High entropy + high wasserstein + high tsv → high score."""
        score = self.engine.aggregate_hallucination_score(
            entropy=0.9, wasserstein=0.4, tsv_deviation=0.9
        )
        assert score > 0.5, f"Expected > 0.5, got {score:.4f}"

    def test_low_signals_give_low_score(self):
        """Low entropy + low wasserstein + low tsv → low score."""
        score = self.engine.aggregate_hallucination_score(
            entropy=0.05, wasserstein=0.01, tsv_deviation=0.05
        )
        assert score < 0.5, f"Expected < 0.5, got {score:.4f}"

    def test_returns_float(self):
        """Return type is float."""
        score = self.engine.aggregate_hallucination_score(0.5, 0.2, 0.3)
        assert isinstance(score, float)

    def test_weight_formula_correct(self):
        """
        Verify the formula: 0.4×entropy + 0.4×wass_norm + 0.2×tsv
        With wasserstein=0.0 and tsv=0.0, score should ≈ 0.4×entropy.
        """
        entropy = 0.5
        score = self.engine.aggregate_hallucination_score(
            entropy=entropy, wasserstein=0.0, tsv_deviation=0.0
        )
        expected = 0.4 * entropy
        assert abs(score - expected) < 0.01, \
            f"Expected {expected:.4f}, got {score:.4f}"


class TestApplyThreshold:
    """Tests for apply_threshold()."""

    def setup_method(self):
        from backend.modules.module_d_scoring import ScoringEngine
        self.engine = ScoringEngine()

    def test_above_threshold_is_hallucinated(self):
        """Score above threshold → flagged, level=hallucinated."""
        flagged, level = self.engine.apply_threshold(0.9)
        assert flagged is True
        assert level == "hallucinated"

    def test_below_suspicious_is_safe(self):
        """Score below suspicious threshold → not flagged, level=safe."""
        flagged, level = self.engine.apply_threshold(0.1)
        assert flagged is False
        assert level == "safe"

    def test_in_suspicious_zone(self):
        """Score between suspicious and hallucination → suspicious."""
        flagged, level = self.engine.apply_threshold(0.55)
        assert flagged is False
        assert level == "suspicious"

    def test_returns_tuple(self):
        """Return type is tuple (bool, str)."""
        result = self.engine.apply_threshold(0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_exactly_at_threshold_is_hallucinated(self):
        """Score exactly at threshold → flagged."""
        flagged, level = self.engine.apply_threshold(
            self.engine.threshold
        )
        assert flagged is True


class TestScoreToken:
    """Tests for score_token() — full token scoring."""

    def setup_method(self):
        from backend.modules.module_d_scoring import ScoringEngine
        self.engine = ScoringEngine()

    def test_returns_token_score_object(self):
        """score_token() returns a TokenScore instance."""
        from backend.modules.module_d_scoring import TokenScore
        torch.manual_seed(0)
        hidden = {18: torch.randn(2048)}
        result = self.engine.score_token(
            token=" Paris", position=0,
            hidden_states_per_layer=hidden
        )
        assert isinstance(result, TokenScore)

    def test_token_score_has_correct_token(self):
        """TokenScore.token matches input."""
        hidden = {18: torch.randn(2048)}
        ts = self.engine.score_token(
            token=" Berlin", position=3,
            hidden_states_per_layer=hidden
        )
        assert ts.token == " Berlin"
        assert ts.position == 3

    def test_hallucination_score_in_range(self):
        """hallucination_score must be in [0, 1]."""
        hidden = {18: torch.randn(2048), 20: torch.randn(2048)}
        ts = self.engine.score_token(
            token=" test", position=0,
            hidden_states_per_layer=hidden,
            wasserstein_avg=0.2,
            tsv_deviation=0.3
        )
        assert 0.0 <= ts.hallucination_score <= 1.0

    def test_is_eat_flag_set_correctly(self):
        """is_eat flag should match the input argument."""
        hidden = {18: torch.randn(2048)}
        ts_eat = self.engine.score_token(
            " 1879", 4, hidden, is_eat=True
        )
        ts_not_eat = self.engine.score_token(
            " was", 1, hidden, is_eat=False
        )
        assert ts_eat.is_eat is True
        assert ts_not_eat.is_eat is False

    def test_to_dict_serializable(self):
        """to_dict() returns a JSON-serializable dictionary."""
        hidden = {18: torch.randn(2048)}
        ts = self.engine.score_token(" Paris", 0, hidden)
        d = ts.to_dict()
        assert isinstance(d, dict)
        assert "token" in d
        assert "hallucination_score" in d
        assert "risk_level" in d
        assert "is_eat" in d


class TestTokenScoreDataclass:
    """Tests for TokenScore dataclass."""

    def test_is_hallucinated_property(self):
        from backend.modules.module_d_scoring import TokenScore, RiskLevel
        ts = TokenScore("Paris", 0, risk_level=RiskLevel.HALLUCINATED)
        assert ts.is_hallucinated is True
        assert ts.is_suspicious is False

    def test_is_suspicious_property(self):
        from backend.modules.module_d_scoring import TokenScore, RiskLevel
        ts = TokenScore("Paris", 0, risk_level=RiskLevel.SUSPICIOUS)
        assert ts.is_suspicious is True
        assert ts.is_hallucinated is False

    def test_default_risk_is_safe(self):
        from backend.modules.module_d_scoring import TokenScore, RiskLevel
        ts = TokenScore("Paris", 0)
        assert ts.risk_level == RiskLevel.SAFE


class TestScoringEngineConfig:
    """Tests for configuration and utility methods."""

    def setup_method(self):
        from backend.modules.module_d_scoring import ScoringEngine
        self.engine = ScoringEngine()

    def test_update_threshold_valid(self):
        """Valid threshold update should work."""
        self.engine.update_threshold(0.5)
        assert self.engine.threshold == 0.5
        self.engine.update_threshold(0.65)  # restore

    def test_update_threshold_invalid_ignored(self):
        """Invalid threshold should be ignored."""
        original = self.engine.threshold
        self.engine.update_threshold(1.5)  # invalid
        assert self.engine.threshold == original

    def test_get_config_returns_dict(self):
        """get_config() returns dict with all keys."""
        config = self.engine.get_config()
        assert "hallucination_threshold" in config
        assert "weight_entropy" in config
        assert "weight_wasserstein" in config
        assert "weight_tsv" in config

    def test_get_overall_risk_empty(self):
        """Empty list returns 0.0 risk."""
        assert self.engine.get_overall_risk([]) == 0.0

    def test_singleton_returns_same_instance(self):
        """get_scoring_engine() always returns same instance."""
        from backend.modules.module_d_scoring import get_scoring_engine
        e1 = get_scoring_engine()
        e2 = get_scoring_engine()
        assert e1 is e2


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
