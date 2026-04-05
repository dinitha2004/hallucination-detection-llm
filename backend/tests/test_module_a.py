"""
test_module_a.py — Unit Tests for Module A: EAT Detection
==========================================================
Tests all functions in module_a_eat.py WITHOUT loading
the LLM. Uses spaCy only (lightweight, fast).

HOW TO RUN:
    pytest backend/tests/test_module_a.py -v

Author: Chalani Dinitha (20211032)
Day 6 Deliverable: EAT tokens correctly identified & mapped to positions
"""

import sys
import pytest

sys.path.append(".")


# ── Check spaCy available ──────────────────────────────────────────────────────
def spacy_available():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


skip_if_no_spacy = pytest.mark.skipif(
    not spacy_available(),
    reason="spaCy en_core_web_sm not installed. Run: python -m spacy download en_core_web_sm"
)


class TestEATDetectorSetup:
    """Tests for EATDetector initialization."""

    def test_detector_instantiation(self):
        """EATDetector creates without errors."""
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()
        assert detector is not None

    def test_singleton_returns_same_instance(self):
        """get_eat_detector() always returns same instance."""
        from backend.modules.module_a_eat import get_eat_detector
        d1 = get_eat_detector()
        d2 = get_eat_detector()
        assert d1 is d2

    @skip_if_no_spacy
    def test_spacy_loads_correctly(self):
        """spaCy model loads and is_loaded returns True."""
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()
        assert detector.is_loaded is True

    def test_eat_entity_types_defined(self):
        """EAT_ENTITY_TYPES contains expected entity types."""
        from backend.modules.module_a_eat import EAT_ENTITY_TYPES
        assert "PERSON" in EAT_ENTITY_TYPES
        assert "DATE" in EAT_ENTITY_TYPES
        assert "GPE" in EAT_ENTITY_TYPES
        assert "ORG" in EAT_ENTITY_TYPES
        assert "CARDINAL" in EAT_ENTITY_TYPES


class TestIdentifyEATTokens:
    """
    Tests for identify_eat_tokens() — the main NER detection function.
    Day 6 plan test: 'Einstein was born in 1879 in Ulm, Germany'
    should flag 1879, Ulm, Germany
    """

    @skip_if_no_spacy
    def test_day6_plan_test_case(self):
        """
        EXACT test from Day 6 plan:
        'Einstein was born in 1879 in Ulm, Germany'
        should flag 1879, Ulm, Germany (and Einstein)
        """
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()

        text = "Einstein was born in 1879 in Ulm, Germany"
        spans = detector.identify_eat_tokens(text)

        detected_texts = [s.text for s in spans]
        print(f"\nDetected: {detected_texts}")

        # Einstein should be detected as PERSON
        assert any("Einstein" in t for t in detected_texts), \
            f"Einstein not detected. Got: {detected_texts}"

        # 1879 should be detected as DATE or CARDINAL
        assert any("1879" in t for t in detected_texts), \
            f"1879 not detected. Got: {detected_texts}"

        # Germany should be detected as GPE
        assert any("Germany" in t for t in detected_texts), \
            f"Germany not detected. Got: {detected_texts}"

    @skip_if_no_spacy
    def test_capital_city_detection(self):
        """Test detection of capital city entities."""
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()

        spans = detector.identify_eat_tokens(
            "The capital of France is Paris"
        )
        detected = [s.text for s in spans]

        assert any("France" in t or "Paris" in t for t in detected), \
            f"Geographic entities not found. Got: {detected}"

    @skip_if_no_spacy
    def test_empty_text_returns_empty_list(self):
        """Empty text returns empty list without crashing."""
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()

        result = detector.identify_eat_tokens("")
        assert result == []

        result = detector.identify_eat_tokens("   ")
        assert result == []

    @skip_if_no_spacy
    def test_plain_text_no_entities(self):
        """Text with no named entities returns empty or minimal list."""
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()

        spans = detector.identify_eat_tokens(
            "the sky is blue and the grass is green"
        )
        # No named entities should be found
        entity_types = [s.entity_type for s in spans]
        assert "PERSON" not in entity_types
        assert "GPE" not in entity_types

    @skip_if_no_spacy
    def test_returns_list_of_eat_spans(self):
        """Return type is List[EATSpan] with correct structure."""
        from backend.modules.module_a_eat import EATDetector, EATSpan
        detector = EATDetector()

        spans = detector.identify_eat_tokens("Paris is in France")
        assert isinstance(spans, list)
        for span in spans:
            assert isinstance(span, EATSpan)
            assert isinstance(span.text, str)
            assert isinstance(span.entity_type, str)
            assert span.entity_type in __import__(
                'backend.modules.module_a_eat',
                fromlist=['EAT_ENTITY_TYPES']
            ).EAT_ENTITY_TYPES


class TestMapEATToTokenPositions:
    """
    Tests for map_eat_to_token_positions() — maps entity spans to tokens.
    This is critical for Gap 2: we need exact token indices to highlight.
    """

    @skip_if_no_spacy
    def test_einstein_1879_maps_correctly(self):
        """
        Test that 1879 maps to correct token position.
        tokens = ['Einstein', ' was', ' born', ' in', ' 1879', ...]
                   pos 0       pos 1   pos 2    pos 3   pos 4
        '1879' should map to position 4.
        """
        from backend.modules.module_a_eat import EATDetector

        detector = EATDetector()
        text = "Einstein was born in 1879"
        tokens = ["Einstein", " was", " born", " in", " 1879"]

        spans, positions = detector.detect_and_map(text, tokens)

        detected_texts = [s.text for s in spans]
        print(f"\nDetected spans: {spans}")
        print(f"EAT positions: {positions}")

        # 1879 should map to position 4
        year_spans = [s for s in spans if "1879" in s.text]
        if year_spans:
            assert 4 in year_spans[0].token_positions, \
                f"1879 should be at position 4, got {year_spans[0].token_positions}"

    @skip_if_no_spacy
    def test_empty_tokens_returns_empty_positions(self):
        """Empty token list returns empty position set."""
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()

        spans, positions = detector.detect_and_map("Paris is in France", [])
        assert positions == set()

    @skip_if_no_spacy
    def test_position_set_is_subset_of_token_range(self):
        """
        All EAT positions must be valid token indices (0 to len(tokens)-1).
        This prevents index-out-of-bounds errors in Module D.
        """
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()

        text = "Einstein was born in 1879 in Ulm, Germany"
        tokens = ["Einstein", " was", " born", " in",
                  " 1879", " in", " Ulm", ",", " Germany"]

        spans, positions = detector.detect_and_map(text, tokens)

        for pos in positions:
            assert 0 <= pos < len(tokens), \
                f"Position {pos} out of range [0, {len(tokens)-1}]"

    @skip_if_no_spacy
    def test_detect_and_map_returns_tuple(self):
        """detect_and_map returns (List[EATSpan], set)."""
        from backend.modules.module_a_eat import EATDetector, EATSpan
        detector = EATDetector()

        result = detector.detect_and_map(
            "Paris is in France",
            ["Paris", " is", " in", " France"]
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        spans, positions = result
        assert isinstance(spans, list)
        assert isinstance(positions, set)


class TestEATSpanDataclass:
    """Tests for the EATSpan dataclass."""

    def test_eat_span_creation(self):
        """EATSpan creates with required fields."""
        from backend.modules.module_a_eat import EATSpan
        span = EATSpan(
            text="Einstein",
            entity_type="PERSON",
            start_char=0,
            end_char=8,
            token_positions=[0]
        )
        assert span.text == "Einstein"
        assert span.entity_type == "PERSON"
        assert span.token_positions == [0]

    def test_is_single_token_property(self):
        """is_single_token returns True only for single-position spans."""
        from backend.modules.module_a_eat import EATSpan

        single = EATSpan("Paris", "GPE", 0, 5, token_positions=[2])
        assert single.is_single_token is True

        multi = EATSpan("New York", "GPE", 0, 8, token_positions=[2, 3])
        assert multi.is_single_token is False

    def test_span_length_property(self):
        """span_length returns number of tokens in span."""
        from backend.modules.module_a_eat import EATSpan

        s = EATSpan("1879", "DATE", 0, 4, token_positions=[4])
        assert s.span_length == 1

        s2 = EATSpan("New York City", "GPE", 0, 13, token_positions=[1, 2, 3])
        assert s2.span_length == 3

    def test_repr_contains_text_and_type(self):
        """__repr__ includes text and entity_type."""
        from backend.modules.module_a_eat import EATSpan
        span = EATSpan("1879", "DATE", 0, 4, token_positions=[4])
        r = repr(span)
        assert "1879" in r
        assert "DATE" in r


class TestUtilityMethods:
    """Tests for utility methods."""

    @skip_if_no_spacy
    def test_format_eat_summary_no_spans(self):
        """format_eat_summary with empty list returns 'No EATs detected'."""
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()
        result = detector.format_eat_summary([])
        assert "No EATs" in result

    @skip_if_no_spacy
    def test_format_eat_summary_with_spans(self):
        """format_eat_summary with spans returns readable string."""
        from backend.modules.module_a_eat import EATDetector, EATSpan
        detector = EATDetector()

        spans = [
            EATSpan("Einstein", "PERSON", 0, 8, token_positions=[0]),
            EATSpan("1879", "DATE", 20, 24, token_positions=[4]),
        ]
        result = detector.format_eat_summary(spans)
        assert "Einstein" in result
        assert "1879" in result
        assert "2 EAT" in result

    @skip_if_no_spacy
    def test_is_eat_position_true(self):
        """is_eat_position returns True for positions in the set."""
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()
        eat_positions = {0, 4, 6}
        assert detector.is_eat_position(0, eat_positions) is True
        assert detector.is_eat_position(4, eat_positions) is True

    @skip_if_no_spacy
    def test_is_eat_position_false(self):
        """is_eat_position returns False for positions not in the set."""
        from backend.modules.module_a_eat import EATDetector
        detector = EATDetector()
        eat_positions = {0, 4, 6}
        assert detector.is_eat_position(1, eat_positions) is False
        assert detector.is_eat_position(99, eat_positions) is False


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
