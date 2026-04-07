"""
test_pipeline.py — Unit Tests for Detection Pipeline (Day 11)
=============================================================
Tests the DetectionPipeline without loading the real model.
Uses mocked components for fast, reliable testing.

HOW TO RUN:
    pytest backend/tests/test_pipeline.py -v

Author: Chalani Dinitha (20211032)
"""

import sys
import pytest
import torch
from unittest.mock import MagicMock, patch

sys.path.append(".")


class TestDetectionPipelineUnit:
    """Unit tests for DetectionPipeline — no model loading."""

    def test_instantiation(self):
        """Pipeline creates without errors."""
        from backend.pipeline.detection_pipeline import DetectionPipeline
        pipeline = DetectionPipeline()
        assert pipeline is not None
        assert pipeline.is_initialized is False

    def test_singleton_returns_same_instance(self):
        """get_detection_pipeline() returns same instance."""
        from backend.pipeline.detection_pipeline import get_detection_pipeline
        p1 = get_detection_pipeline()
        p2 = get_detection_pipeline()
        assert p1 is p2

    def test_run_without_init_returns_empty_output(self):
        """Running pipeline before init returns safe empty output."""
        from backend.pipeline.detection_pipeline import DetectionPipeline
        pipeline = DetectionPipeline()
        output = pipeline.run("test prompt")
        assert output is not None
        assert output.generated_text == "" or "Error" in output.generated_text

    def test_update_threshold_before_init_does_not_crash(self):
        """Updating threshold before init should not crash."""
        from backend.pipeline.detection_pipeline import DetectionPipeline
        pipeline = DetectionPipeline()
        # Should not raise — just silently skip
        pipeline.update_threshold(0.5)

    def test_get_config_before_init(self):
        """get_config() works even before initialization."""
        from backend.pipeline.detection_pipeline import DetectionPipeline
        pipeline = DetectionPipeline()
        config = pipeline.get_config()
        assert isinstance(config, dict)


class TestDetectionOutputStructure:
    """Tests for DetectionOutput and AnnotatedToken structures."""

    def test_detection_output_to_dict(self):
        """DetectionOutput.to_dict() returns correct structure."""
        from backend.modules.span_level_mapper import DetectionOutput
        output = DetectionOutput(
            generated_text="Paris is in France",
            overall_risk=0.8,
            num_flagged=1,
            num_eat_tokens=2,
            hallucination_detected=True,
            processing_time_ms=1234.5,
        )
        d = output.to_dict()
        assert d["generated_text"] == "Paris is in France"
        assert d["overall_risk"] == 0.8
        assert d["num_flagged"] == 1
        assert d["hallucination_detected"] is True
        assert d["processing_time_ms"] == 1234.5

    def test_annotated_token_to_dict(self):
        """AnnotatedToken.to_dict() returns all required API fields."""
        from backend.modules.span_level_mapper import AnnotatedToken
        from backend.modules.module_d_scoring import RiskLevel
        t = AnnotatedToken(
            token=" Paris",
            position=5,
            hallucination_score=0.85,
            risk_level=RiskLevel.HALLUCINATED,
            is_eat=True,
            is_flagged=True,
            entity_type="GPE",
        )
        d = t.to_dict()
        assert d["token"] == " Paris"
        assert d["position"] == 5
        assert d["hallucination_score"] == 0.85
        assert d["risk_level"] == "hallucinated"
        assert d["is_eat"] is True
        assert d["is_flagged"] is True
        assert d["entity_type"] == "GPE"

    def test_detection_output_flagged_tokens_property(self):
        """flagged_tokens property returns only flagged tokens."""
        from backend.modules.span_level_mapper import DetectionOutput, AnnotatedToken
        from backend.modules.module_d_scoring import RiskLevel

        tokens = [
            AnnotatedToken(" The", 0, 0.1, RiskLevel.SAFE, False, False),
            AnnotatedToken(" 1879", 1, 0.9, RiskLevel.HALLUCINATED, True, True),
            AnnotatedToken(" in", 2, 0.1, RiskLevel.SAFE, False, False),
        ]
        output = DetectionOutput(
            generated_text="The 1879 in",
            annotated_tokens=tokens,
            num_flagged=1,
            hallucination_detected=True,
        )
        assert len(output.flagged_tokens) == 1
        assert output.flagged_tokens[0].token == " 1879"

    def test_detection_output_eat_tokens_property(self):
        """eat_tokens property returns only EAT tokens."""
        from backend.modules.span_level_mapper import DetectionOutput, AnnotatedToken
        from backend.modules.module_d_scoring import RiskLevel

        tokens = [
            AnnotatedToken(" The", 0, 0.1, RiskLevel.SAFE, False, False),
            AnnotatedToken(" Paris", 1, 0.5, RiskLevel.SUSPICIOUS, True, False),
            AnnotatedToken(" 1879", 2, 0.9, RiskLevel.HALLUCINATED, True, True),
        ]
        output = DetectionOutput(
            generated_text="The Paris 1879",
            annotated_tokens=tokens,
        )
        assert len(output.eat_tokens) == 2


class TestPipelineStepsLogic:
    """
    Tests for the logic of each pipeline step
    without running the actual model.
    """

    def test_step1_eat_detection_called(self):
        """Module A EAT detector should be called with the prompt."""
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()

        if not detector.is_loaded:
            pytest.skip("spaCy not installed")

        # Test that EAT detection works on a factual prompt
        spans = detector.identify_eat_tokens(
            "Einstein was born in 1879 in Germany"
        )
        assert len(spans) > 0
        entity_types = {s.entity_type for s in spans}
        # Should find at least PERSON or DATE
        assert entity_types & {"PERSON", "DATE", "GPE", "CARDINAL"}

    def test_step5_scoring_produces_token_scores(self):
        """Module D scoring should produce one score per token."""
        from backend.modules.module_d_scoring import ScoringEngine
        engine = ScoringEngine()

        tokens = [" Paris", " is", " in", " France"]
        eat_positions = {0, 3}

        token_scores = []
        for i, token in enumerate(tokens):
            hidden = {18: torch.randn(2048), 20: torch.randn(2048)}
            ts = engine.score_token(
                token=token,
                position=i,
                hidden_states_per_layer=hidden,
                wasserstein_avg=0.1,
                tsv_deviation=0.3,
                is_eat=(i in eat_positions),
            )
            token_scores.append(ts)

        assert len(token_scores) == 4
        for ts in token_scores:
            assert 0.0 <= ts.hallucination_score <= 1.0

    def test_span_mapping_only_eats_flagged(self):
        """Span mapper should only flag EAT positions."""
        from backend.modules.module_d_scoring import TokenScore, RiskLevel
        from backend.modules.span_level_mapper import SpanLevelMapper

        mapper = SpanLevelMapper()

        # Create scores where non-EAT has high score
        token_scores = [
            TokenScore(" was", 0, hallucination_score=0.9,
                      is_flagged=True, risk_level=RiskLevel.HALLUCINATED),
            TokenScore(" Paris", 1, hallucination_score=0.9,
                      is_flagged=True, risk_level=RiskLevel.HALLUCINATED),
        ]
        eat_positions = {1}  # only Paris is EAT

        annotated = mapper.span_level_mapper(token_scores, eat_positions)

        # "was" should NOT be flagged even with high score
        assert not annotated[0].is_flagged
        # "Paris" should be flagged (EAT + high score)
        assert annotated[1].is_flagged

    def test_nfr1_processing_time_recorded(self):
        """NFR1: Processing time must be recorded in DetectionOutput."""
        from backend.modules.span_level_mapper import DetectionOutput
        output = DetectionOutput(
            generated_text="test",
            processing_time_ms=1500.0,
        )
        assert output.processing_time_ms == 1500.0
        d = output.to_dict()
        assert "processing_time_ms" in d

    def test_nfr4_same_prompt_same_structure(self):
        """
        NFR4 Reliability: same input should produce same output structure.
        The scoring engine should be deterministic.
        """
        from backend.modules.module_d_scoring import ScoringEngine
        engine = ScoringEngine()

        torch.manual_seed(42)
        hidden = {18: torch.randn(2048)}

        score1 = engine.score_token(" Paris", 0, hidden)
        score2 = engine.score_token(" Paris", 0, hidden)

        assert score1.hallucination_score == score2.hallucination_score
        assert score1.risk_level == score2.risk_level

    def test_fr14_threshold_configurable(self):
        """FR14: Threshold should be configurable at runtime."""
        from backend.pipeline.detection_pipeline import DetectionPipeline
        from backend.modules.module_d_scoring import get_scoring_engine

        pipeline = DetectionPipeline()
        pipeline._scoring_engine = get_scoring_engine()
        pipeline._span_mapper = MagicMock()
        pipeline._span_mapper.threshold = 0.65

        pipeline.update_threshold(0.5)
        assert pipeline._scoring_engine.threshold == 0.5

        # Restore
        pipeline.update_threshold(0.65)


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
