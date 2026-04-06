"""
test_span_mapper.py — Unit Tests for Span-Level Mapper (Day 10)
================================================================
Tests the span_level_mapper and build_annotated_output functions.
Includes end-to-end test from hidden states → annotated tokens.

HOW TO RUN:
    pytest backend/tests/test_span_mapper.py -v

Author: Chalani Dinitha (20211032)
Day 10 Deliverable: All 4 modules complete and individually tested
"""

import sys
import pytest
import torch

sys.path.append(".")


def make_token_score(token, pos, score, flagged=None, is_eat=False):
    """Helper to create a TokenScore for testing."""
    from backend.modules.module_d_scoring import TokenScore, RiskLevel
    if flagged is None:
        flagged = score >= 0.65
    risk = RiskLevel.HALLUCINATED if flagged else (
        RiskLevel.SUSPICIOUS if score >= 0.45 else RiskLevel.SAFE
    )
    return TokenScore(
        token=token,
        position=pos,
        hallucination_score=score,
        is_flagged=flagged,
        is_eat=is_eat,
        risk_level=risk,
    )


class TestSpanLevelMapper:
    """Tests for span_level_mapper() — the key novelty of Gap 2."""

    def setup_method(self):
        from backend.modules.span_level_mapper import SpanLevelMapper
        self.mapper = SpanLevelMapper()

    def test_key_novelty_eat_plus_threshold_only(self):
        """
        KEY TEST: Only tokens that are BOTH EAT AND above threshold
        should be flagged. This is your core research novelty.

        Scenario: "Einstein was born in 1879 in Germany"
        - Einstein (pos 0): EAT + LOW score → NOT flagged
        - was (pos 1): NOT EAT → NOT flagged
        - 1879 (pos 4): EAT + HIGH score → FLAGGED ✅
        - Germany (pos 6): EAT + HIGH score → FLAGGED ✅
        """
        token_scores = [
            make_token_score(" Einstein", 0, 0.20),  # EAT + low → not flagged
            make_token_score(" was",      1, 0.10),  # not EAT
            make_token_score(" born",     2, 0.12),  # not EAT
            make_token_score(" in",       3, 0.08),  # not EAT
            make_token_score(" 1879",     4, 0.85),  # EAT + high → FLAGGED
            make_token_score(" in",       5, 0.09),  # not EAT
            make_token_score(" Germany",  6, 0.78),  # EAT + high → FLAGGED
        ]
        eat_positions = {0, 4, 6}  # Einstein, 1879, Germany

        annotated = self.mapper.span_level_mapper(token_scores, eat_positions)

        # Einstein: EAT but low score → NOT flagged (key novelty!)
        assert not annotated[0].is_flagged, \
            "Einstein has low score → should NOT be flagged even though it's EAT"
        assert annotated[0].is_eat

        # "was": not EAT → NOT flagged
        assert not annotated[1].is_flagged
        assert not annotated[1].is_eat

        # 1879: EAT + high score → FLAGGED
        assert annotated[4].is_flagged
        assert annotated[4].is_eat

        # Germany: EAT + high score → FLAGGED
        assert annotated[6].is_flagged
        assert annotated[6].is_eat

    def test_non_eat_never_flagged_even_high_score(self):
        """
        A non-EAT token with a HIGH score should NOT be flagged.
        This prevents false positives on filler words.
        """
        token_scores = [
            make_token_score(" the",  0, 0.95),  # HIGH score but NOT EAT
            make_token_score(" Paris", 1, 0.90),  # HIGH score + EAT
        ]
        eat_positions = {1}  # only Paris is EAT

        annotated = self.mapper.span_level_mapper(token_scores, eat_positions)

        assert not annotated[0].is_flagged, \
            "Non-EAT token should NOT be flagged regardless of score"
        assert annotated[1].is_flagged, \
            "EAT + high score → should be flagged"

    def test_eat_with_low_score_not_flagged(self):
        """EAT token with low score should NOT be flagged."""
        token_scores = [
            make_token_score(" Paris", 0, 0.10),  # EAT but LOW score
        ]
        eat_positions = {0}

        annotated = self.mapper.span_level_mapper(token_scores, eat_positions)
        assert not annotated[0].is_flagged
        assert annotated[0].is_eat

    def test_returns_correct_count(self):
        """Returns same number of tokens as input."""
        token_scores = [make_token_score(f" tok{i}", i, 0.5) for i in range(7)]
        eat_positions = {0, 2, 4}

        annotated = self.mapper.span_level_mapper(token_scores, eat_positions)
        assert len(annotated) == 7

    def test_returns_annotated_token_objects(self):
        """Returns list of AnnotatedToken objects."""
        from backend.modules.span_level_mapper import AnnotatedToken
        token_scores = [make_token_score(" Paris", 0, 0.8)]
        eat_positions = {0}

        annotated = self.mapper.span_level_mapper(token_scores, eat_positions)
        assert isinstance(annotated[0], AnnotatedToken)

    def test_empty_eat_positions_nothing_flagged(self):
        """If no EAT positions, nothing should be flagged."""
        token_scores = [
            make_token_score(" Paris", 0, 0.9),
            make_token_score(" 1879",  1, 0.85),
        ]
        annotated = self.mapper.span_level_mapper(token_scores, set())

        for t in annotated:
            assert not t.is_flagged

    def test_all_eat_high_score_all_flagged(self):
        """When all tokens are EAT with high scores, all are flagged."""
        token_scores = [
            make_token_score(" Paris",  0, 0.9),
            make_token_score(" France", 1, 0.8),
        ]
        eat_positions = {0, 1}

        annotated = self.mapper.span_level_mapper(token_scores, eat_positions)
        assert annotated[0].is_flagged
        assert annotated[1].is_flagged


class TestBuildAnnotatedOutput:
    """Tests for build_annotated_output()."""

    def setup_method(self):
        from backend.modules.span_level_mapper import SpanLevelMapper
        self.mapper = SpanLevelMapper()

    def make_annotated(self, flagged_positions=None):
        """Helper to create annotated token list."""
        from backend.modules.span_level_mapper import AnnotatedToken
        from backend.modules.module_d_scoring import RiskLevel
        tokens = [" Einstein", " was", " born", " in", " 1879"]
        # NEW (correct)
        if flagged_positions is None:
            flagged_positions = {4}
        result = []
        for i, t in enumerate(tokens):
            flagged = i in flagged_positions
            result.append(AnnotatedToken(
                token=t, position=i,
                hallucination_score=0.9 if flagged else 0.1,
                risk_level=RiskLevel.HALLUCINATED if flagged else RiskLevel.SAFE,
                is_eat=flagged, is_flagged=flagged
            ))
        return result

    def test_returns_detection_output(self):
        """Returns a DetectionOutput object."""
        from backend.modules.span_level_mapper import DetectionOutput
        annotated = self.make_annotated()
        output = self.mapper.build_annotated_output("test text", annotated)
        assert isinstance(output, DetectionOutput)

    def test_num_flagged_correct(self):
        """num_flagged matches actual flagged count."""
        annotated = self.make_annotated(flagged_positions={4})
        output = self.mapper.build_annotated_output("test", annotated)
        assert output.num_flagged == 1

    def test_hallucination_detected_true(self):
        """hallucination_detected is True when any token is flagged."""
        annotated = self.make_annotated(flagged_positions={4})
        output = self.mapper.build_annotated_output("test", annotated)
        assert output.hallucination_detected is True

    def test_hallucination_detected_false_when_none(self):
        """hallucination_detected is False when nothing flagged."""
        annotated = self.make_annotated(flagged_positions=set())
        output = self.mapper.build_annotated_output("test", annotated)
        assert output.hallucination_detected is False

    def test_to_dict_serializable(self):
        """to_dict() returns JSON-serializable dict."""
        annotated = self.make_annotated()
        output = self.mapper.build_annotated_output("test text", annotated)
        d = output.to_dict()
        assert isinstance(d, dict)
        assert "generated_text" in d
        assert "annotated_tokens" in d
        assert "overall_risk" in d
        assert "hallucination_detected" in d
        assert isinstance(d["annotated_tokens"], list)

    def test_processing_time_stored(self):
        """Processing time is stored correctly."""
        annotated = self.make_annotated()
        output = self.mapper.build_annotated_output("test", annotated, 1234.5)
        assert output.processing_time_ms == 1234.5

    def test_flagged_tokens_property(self):
        """flagged_tokens property returns only flagged tokens."""
        annotated = self.make_annotated(flagged_positions={4})
        output = self.mapper.build_annotated_output("test", annotated)
        assert len(output.flagged_tokens) == 1
        assert output.flagged_tokens[0].position == 4


class TestFormatHighlightedText:
    """Tests for format_highlighted_text()."""

    def setup_method(self):
        from backend.modules.span_level_mapper import SpanLevelMapper
        self.mapper = SpanLevelMapper()

    def test_flagged_token_marked(self):
        """Flagged tokens should appear with special marking."""
        from backend.modules.span_level_mapper import AnnotatedToken
        from backend.modules.module_d_scoring import RiskLevel
        tokens = [
            AnnotatedToken(" The", 0, 0.1, RiskLevel.SAFE, False, False),
            AnnotatedToken(" 1879", 1, 0.9, RiskLevel.HALLUCINATED, True, True),
        ]
        result = self.mapper.format_highlighted_text(tokens)
        assert "1879" in result
        assert "The" in result

    def test_returns_string(self):
        """Returns a string."""
        from backend.modules.span_level_mapper import AnnotatedToken
        from backend.modules.module_d_scoring import RiskLevel
        tokens = [AnnotatedToken(" Paris", 0, 0.5, RiskLevel.SAFE, False, False)]
        result = self.mapper.format_highlighted_text(tokens)
        assert isinstance(result, str)


class TestAnnotatedTokenDataclass:
    """Tests for AnnotatedToken dataclass."""

    def test_should_highlight_true(self):
        """should_highlight is True only when is_flagged AND is_eat."""
        from backend.modules.span_level_mapper import AnnotatedToken
        from backend.modules.module_d_scoring import RiskLevel
        t = AnnotatedToken("Paris", 0, 0.9, RiskLevel.HALLUCINATED, True, True)
        assert t.should_highlight is True

    def test_should_highlight_false_not_eat(self):
        """should_highlight is False if not EAT (even if flagged)."""
        from backend.modules.span_level_mapper import AnnotatedToken
        from backend.modules.module_d_scoring import RiskLevel
        t = AnnotatedToken("Paris", 0, 0.9, RiskLevel.HALLUCINATED, False, True)
        assert t.should_highlight is False

    def test_to_dict_has_all_keys(self):
        """to_dict() includes all required keys for frontend."""
        from backend.modules.span_level_mapper import AnnotatedToken
        from backend.modules.module_d_scoring import RiskLevel
        t = AnnotatedToken("Paris", 0, 0.9, RiskLevel.HALLUCINATED, True, True)
        d = t.to_dict()
        for key in ["token", "position", "hallucination_score",
                    "risk_level", "is_eat", "is_flagged"]:
            assert key in d

    def test_singleton_returns_same_instance(self):
        """get_span_mapper() always returns same instance."""
        from backend.modules.span_level_mapper import get_span_mapper
        m1 = get_span_mapper()
        m2 = get_span_mapper()
        assert m1 is m2


class TestEndToEnd:
    """
    End-to-end test: from hidden states → annotated tokens.
    This is the full pipeline test for Day 10.
    """

    def test_end_to_end_pipeline(self):
        """
        Full test: hidden states → entropy → score → span mapping
        → annotated output.

        This tests that Modules A, B, C, D all work together.
        """
        from backend.modules.module_d_scoring import ScoringEngine
        from backend.modules.span_level_mapper import SpanLevelMapper

        engine = ScoringEngine()
        mapper = SpanLevelMapper()

        # Simulate tokens and hidden states
        tokens = [" Einstein", " was", " born", " in", " 1879"]
        eat_positions = {0, 4}  # Einstein and 1879 are EATs

        # Simulate scores — 1879 is "hallucinated"
        token_scores = []
        for i, token in enumerate(tokens):
            hidden = {18: torch.randn(2048), 20: torch.randn(2048)}
            # Force high score for 1879
            wasserstein = 0.4 if i == 4 else 0.05
            tsv_dev = 0.8 if i == 4 else 0.2
            ts = engine.score_token(
                token=token,
                position=i,
                hidden_states_per_layer=hidden,
                wasserstein_avg=wasserstein,
                tsv_deviation=tsv_dev,
                is_eat=(i in eat_positions),
            )
            token_scores.append(ts)

        # Apply span-level mapping
        annotated = mapper.span_level_mapper(token_scores, eat_positions)

        # Build output
        output = mapper.build_annotated_output(
            "Einstein was born in 1879", annotated
        )

        # Verify structure
        assert len(output.annotated_tokens) == 5
        assert output.num_eat_tokens == 2
        assert isinstance(output.overall_risk, float)
        assert 0.0 <= output.overall_risk <= 1.0

        # Verify "was" (non-EAT) is never flagged
        was_token = output.annotated_tokens[1]
        assert was_token.token == " was"
        assert not was_token.is_flagged
        assert not was_token.is_eat

        print(f"\n  End-to-end test passed!")
        print(f"  Tokens: {[t.token for t in output.annotated_tokens]}")
        print(f"  Flagged: {[t.token for t in output.flagged_tokens]}")
        print(f"  Overall risk: {output.overall_risk:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
