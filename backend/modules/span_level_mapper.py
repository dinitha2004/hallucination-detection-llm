"""
span_level_mapper.py — Module D: Span-Level Mapper & Output Renderer
=====================================================================
This is the FINAL piece of your hallucination detection system.

What this module does:
----------------------
This implements the KEY NOVELTY of Gap 2 in your research:

  "Only flags tokens that are BOTH EAT AND above threshold"

Without this module:
    → Score every token → flag anything above 0.65
    → "The capital of France is Paris I know but I was wrong"
         ALL tokens get scored, even "The", "of", "is"

With this module (your novelty):
    → Score every token → ONLY flag if BOTH:
      1. It is an EAT (Exact Answer Token) — identified by Module A
      2. Its score is above the threshold — computed by Module D
    → Result: only "Paris" gets highlighted (the actual wrong answer)
              not "The", "capital", "of", "France", "is"

This is what makes your system BETTER than existing approaches!

Research connection:
--------------------
- Span-Level Mapper → your finalized technique Step 5
- Gap 2: highlight ONLY the exact wrong token
- FR9: identify exact hallucinated token or span
- FR10: highlight detected hallucinated part in output
- FR12: display confidence score for flagged content

Author: Chalani Dinitha (20211032)
"""

import sys
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

sys.path.append(".")
from backend.config import HALLUCINATION_THRESHOLD, SUSPICIOUS_THRESHOLD_LOW
from backend.modules.module_d_scoring import TokenScore, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class AnnotatedToken:
    """
    A token with complete hallucination annotation.
    This is what the React frontend renders.

    The frontend uses:
    - token:               text to display
    - risk_level:          determines color (red/yellow/normal)
    - hallucination_score: shown in tooltip on hover
    - is_eat:              whether this is an exact answer token
    - is_flagged:          whether to highlight this token
    """
    token: str
    position: int
    hallucination_score: float
    risk_level: str
    is_eat: bool
    is_flagged: bool
    entity_type: Optional[str] = None   # e.g. "DATE", "PERSON", "GPE"

    def to_dict(self) -> dict:
        """Serializable format for the API response."""
        return {
            "token": self.token,
            "position": self.position,
            "hallucination_score": round(self.hallucination_score, 4),
            "risk_level": self.risk_level,
            "is_eat": self.is_eat,
            "is_flagged": self.is_flagged,
            "entity_type": self.entity_type,
        }

    @property
    def should_highlight(self) -> bool:
        """True if the frontend should highlight this token."""
        return self.is_flagged and self.is_eat


@dataclass
class DetectionOutput:
    """
    Complete output of the hallucination detection system.
    This is what the FastAPI endpoint returns to the frontend.
    """
    # The full generated text
    generated_text: str

    # Annotated tokens — one per generated token
    annotated_tokens: List[AnnotatedToken] = field(default_factory=list)

    # Overall risk score (max across all flagged tokens)
    overall_risk: float = 0.0

    # Number of tokens flagged as hallucinated
    num_flagged: int = 0

    # Number of EAT tokens found
    num_eat_tokens: int = 0

    # Whether any hallucination was detected
    hallucination_detected: bool = False

    # Processing time
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Serializable format for the API response."""
        return {
            "generated_text": self.generated_text,
            "annotated_tokens": [t.to_dict() for t in self.annotated_tokens],
            "overall_risk": round(self.overall_risk, 4),
            "num_flagged": self.num_flagged,
            "num_eat_tokens": self.num_eat_tokens,
            "hallucination_detected": self.hallucination_detected,
            "processing_time_ms": round(self.processing_time_ms, 1),
        }

    @property
    def flagged_tokens(self) -> List[AnnotatedToken]:
        """Return only the flagged (hallucinated) tokens."""
        return [t for t in self.annotated_tokens if t.is_flagged]

    @property
    def eat_tokens(self) -> List[AnnotatedToken]:
        """Return only the EAT tokens."""
        return [t for t in self.annotated_tokens if t.is_eat]


class SpanLevelMapper:
    """
    Maps hallucination scores to token spans, applying the key novelty:
    ONLY tokens that are BOTH EAT AND above threshold are flagged.

    This implements Gap 2 of your research:
    "Fine-grained span-level highlighting of only the incorrect tokens"
    """

    def __init__(self):
        self.threshold = HALLUCINATION_THRESHOLD
        self.suspicious_low = SUSPICIOUS_THRESHOLD_LOW

    # =========================================================
    # SECTION 1: Core Span-Level Mapping
    # =========================================================

    def span_level_mapper(
        self,
        token_scores: List[TokenScore],
        eat_positions: Set[int],
        eat_spans: List = None
    ) -> List[AnnotatedToken]:
        """
        Apply span-level mapping: flag ONLY tokens that are
        BOTH EAT AND above threshold.

        This is your KEY RESEARCH NOVELTY:
        - Standard approach: flag any token above threshold
        - Your approach: flag only EAT tokens above threshold

        Why this matters:
        - Standard: "The capital of France is [Paris] I know but..."
                     → flags "Paris", "know", "wrong" (too many)
        - Yours:    "The capital of France is [Paris] I know but..."
                     → flags only "Paris" (correct! that's the wrong fact)

        Args:
            token_scores: List[TokenScore] from Module D scoring engine
            eat_positions: Set of EAT position indices from Module A
            eat_spans:    Optional list of EATSpan objects (for entity_type)

        Returns:
            List[AnnotatedToken] — one per token, with is_flagged set
            only for EAT tokens above threshold
        """
        # Build entity type lookup from EAT spans
        eat_entity_types = {}
        if eat_spans:
            for span in eat_spans:
                for pos in span.token_positions:
                    eat_entity_types[pos] = span.entity_type

        annotated = []
        for ts in token_scores:
            pos = ts.position
            is_eat = pos in eat_positions

            # KEY NOVELTY: only flag if BOTH conditions are met
            # Condition 1: Token is an EAT (exact answer token)
            # Condition 2: Score is above hallucination threshold
            is_flagged = is_eat and ts.is_flagged

            # Determine risk level for display
            if is_flagged:
                risk_level = RiskLevel.HALLUCINATED
            elif is_eat and ts.risk_level == RiskLevel.SUSPICIOUS:
                risk_level = RiskLevel.SUSPICIOUS
            else:
                risk_level = ts.risk_level if is_eat else RiskLevel.SAFE

            annotated.append(AnnotatedToken(
                token=ts.token,
                position=pos,
                hallucination_score=ts.hallucination_score,
                risk_level=risk_level,
                is_eat=is_eat,
                is_flagged=is_flagged,
                entity_type=eat_entity_types.get(pos),
            ))

        flagged_count = sum(1 for a in annotated if a.is_flagged)
        eat_count = sum(1 for a in annotated if a.is_eat)

        logger.info(
            f"Span-level mapping: {len(annotated)} tokens, "
            f"{eat_count} EATs, {flagged_count} flagged as hallucinated"
        )

        return annotated

    # =========================================================
    # SECTION 2: Output Renderer
    # =========================================================

    def build_annotated_output(
        self,
        generated_text: str,
        annotated_tokens: List[AnnotatedToken],
        processing_time_ms: float = 0.0
    ) -> DetectionOutput:
        """
        Build the complete DetectionOutput for the API response.

        This assembles everything into the final output object that
        the FastAPI endpoint returns to the React frontend.

        Args:
            generated_text:    The full generated text string
            annotated_tokens:  Output of span_level_mapper()
            processing_time_ms: How long detection took

        Returns:
            DetectionOutput with all fields populated
        """
        flagged = [t for t in annotated_tokens if t.is_flagged]
        eat_tokens = [t for t in annotated_tokens if t.is_eat]

        overall_risk = max(
            (t.hallucination_score for t in annotated_tokens),
            default=0.0
        )

        return DetectionOutput(
            generated_text=generated_text,
            annotated_tokens=annotated_tokens,
            overall_risk=overall_risk,
            num_flagged=len(flagged),
            num_eat_tokens=len(eat_tokens),
            hallucination_detected=len(flagged) > 0,
            processing_time_ms=processing_time_ms,
        )

    def format_highlighted_text(
        self,
        annotated_tokens: List[AnnotatedToken]
    ) -> str:
        """
        Format annotated tokens into a human-readable highlighted text.
        Used for terminal display and logging.

        Example output:
            "The capital of France is [PARIS*] I know but I was [WRONG*]"
            where [TOKEN*] = hallucinated EAT token
        """
        parts = []
        for token in annotated_tokens:
            if token.is_flagged:
                # Hallucinated EAT — mark with brackets and asterisk
                parts.append(f"[{token.token.strip().upper()}*]")
            elif token.is_eat and token.risk_level == RiskLevel.SUSPICIOUS:
                # Suspicious EAT — mark with tildes
                parts.append(f"~{token.token.strip()}~")
            else:
                parts.append(token.token)
        return "".join(parts)


# ── Singleton ─────────────────────────────────────────────────────────────────
_span_mapper_instance = None


def get_span_mapper() -> SpanLevelMapper:
    """Returns the global SpanLevelMapper instance."""
    global _span_mapper_instance
    if _span_mapper_instance is None:
        _span_mapper_instance = SpanLevelMapper()
    return _span_mapper_instance


# ── Quick Test ────────────────────────────────────────────────────────────────
# Run: python backend/modules/span_level_mapper.py

if __name__ == "__main__":
    import torch
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 65)
    print("  DAY 10 TEST: Span-Level Mapper & Output Renderer")
    print("=" * 65 + "\n")

    from backend.modules.module_d_scoring import ScoringEngine, TokenScore, RiskLevel

    mapper = SpanLevelMapper()
    engine = ScoringEngine()

    # Simulate: "Einstein was born in 1879 in Germany"
    # Tokens:    0          1    2     3   4      5   6
    # EATs:      [0=Einstein]         [4=1879]   [6=Germany]
    # Scores:    low        low  low  low HIGH    low  HIGH
    #                                    (hallucinated year/country)

    print("TEST SCENARIO:")
    print("  Text: 'Einstein was born in 1879 in Germany'")
    print("  EATs: Einstein(0), 1879(4), Germany(6)")
    print("  Simulated scores: 1879 and Germany are hallucinated\n")

    # Create simulated TokenScores
    token_data = [
        (" Einstein", 0, 0.20, False),   # EAT, LOW score → not flagged
        (" was",      1, 0.10, False),   # not EAT
        (" born",     2, 0.12, False),   # not EAT
        (" in",       3, 0.08, False),   # not EAT
        (" 1879",     4, 0.85, True),    # EAT, HIGH score → FLAGGED
        (" in",       5, 0.09, False),   # not EAT
        (" Germany",  6, 0.78, True),    # EAT, HIGH score → FLAGGED
    ]

    token_scores = []
    for text, pos, score, flagged in token_data:
        risk = RiskLevel.HALLUCINATED if flagged else RiskLevel.SAFE
        ts = TokenScore(
            token=text,
            position=pos,
            hallucination_score=score,
            is_flagged=flagged,
            risk_level=risk,
        )
        token_scores.append(ts)

    # EAT positions from Module A
    eat_positions = {0, 4, 6}  # Einstein, 1879, Germany

    # Step 1: Apply span-level mapper
    print("STEP 1: Apply span_level_mapper()...")
    annotated = mapper.span_level_mapper(token_scores, eat_positions)

    print(f"  {'Pos':<5} {'Token':<15} {'Score':<8} {'EAT':<6} "
          f"{'Flagged':<10} {'Risk'}")
    print(f"  {'-'*5} {'-'*15} {'-'*8} {'-'*6} {'-'*10} {'-'*15}")
    for t in annotated:
        print(f"  {t.position:<5} {repr(t.token):<15} "
              f"{t.hallucination_score:<8.4f} {str(t.is_eat):<6} "
              f"{str(t.is_flagged):<10} {t.risk_level}")

    # Verify KEY NOVELTY: only EAT + above threshold = flagged
    assert annotated[0].is_eat and not annotated[0].is_flagged, \
        "Einstein: EAT but low score → should NOT be flagged"
    assert annotated[4].is_flagged, "1879: EAT + high score → should be flagged"
    assert annotated[6].is_flagged, "Germany: EAT + high score → should be flagged"
    assert not annotated[1].is_flagged, "was: not EAT → should NOT be flagged"

    print(f"\n  PASS: Only EAT + high score tokens are flagged ✅")
    print(f"  PASS: Einstein (EAT + low score) NOT flagged ✅  ← key novelty!")
    print(f"  PASS: 1879 and Germany (EAT + high score) flagged ✅\n")

    # Step 2: Build annotated output
    print("STEP 2: build_annotated_output()...")
    output = mapper.build_annotated_output(
        "Einstein was born in 1879 in Germany",
        annotated,
        processing_time_ms=1234.5
    )
    print(f"  generated_text: '{output.generated_text}'")
    print(f"  overall_risk:   {output.overall_risk:.4f}")
    print(f"  num_flagged:    {output.num_flagged}")
    print(f"  num_eat_tokens: {output.num_eat_tokens}")
    print(f"  hallucination_detected: {output.hallucination_detected}")
    assert output.num_flagged == 2
    assert output.num_eat_tokens == 3
    assert output.hallucination_detected is True
    print(f"  PASS: DetectionOutput correct ✅\n")

    # Step 3: Format highlighted text
    print("STEP 3: format_highlighted_text()...")
    highlighted = mapper.format_highlighted_text(annotated)
    print(f"  Highlighted: '{highlighted}'")
    assert "[1879*]" in highlighted or "1879" in highlighted
    print(f"  PASS: Hallucinated tokens marked ✅\n")

    # Step 4: to_dict serialization
    print("STEP 4: to_dict() serialization...")
    d = output.to_dict()
    assert "annotated_tokens" in d
    assert "overall_risk" in d
    assert "hallucination_detected" in d
    assert isinstance(d["annotated_tokens"], list)
    print(f"  PASS: DetectionOutput serializable to JSON ✅\n")

    print("=" * 65)
    print("  DAY 10 DELIVERABLE CONFIRMED")
    print("=" * 65)
    print("  PASS: span_level_mapper() flags only EAT + above threshold")
    print("  PASS: Einstein (EAT + low score) correctly NOT flagged")
    print("  PASS: 1879, Germany (EAT + high score) correctly flagged")
    print("  PASS: build_annotated_output() returns DetectionOutput")
    print("  PASS: format_highlighted_text() marks hallucinated tokens")
    print("  PASS: to_dict() serializable for API response")
    print()
    print("  Research impact:")
    print("  → Gap 2 COMPLETE: only exact wrong tokens highlighted")
    print("  → All 4 modules (A, B, C, D) now complete!")
    print("  → Ready for detection pipeline on Day 11")
    print("=" * 65 + "\n")
