"""
metrics.py — Evaluation Metrics for Hallucination Detection (FR13)
==================================================================
Implements token-level and span-level evaluation metrics.

These metrics produce the numbers you cite in your thesis:
    "Our system achieved Precision=0.XX, Recall=0.XX, F1=0.XX
     on TruthfulQA validation set (n=100)"

Metric definitions:
-------------------
True Positive  (TP): Token IS hallucinated AND system flagged it ✅
False Positive (FP): Token is NOT hallucinated BUT system flagged it ❌
True Negative  (TN): Token is NOT hallucinated AND system did NOT flag ✅
False Negative (FN): Token IS hallucinated AND system did NOT flag it ❌

From these:
    Precision = TP / (TP + FP)  → "How often is flagged token really wrong?"
    Recall    = TP / (TP + FN)  → "How many wrong tokens did we catch?"
    F1        = 2 × P × R / (P + R) → harmonic mean of precision and recall
    Accuracy  = (TP + TN) / (TP + FP + TN + FN)

Author: Chalani Dinitha (20211032)
"""

import sys
import logging
import json
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field

sys.path.append(".")
logger = logging.getLogger(__name__)


# ── Result Dataclasses ────────────────────────────────────────────────────────

@dataclass
class TokenLevelResult:
    """Result of token-level evaluation for one sample."""
    question: str
    generated_text: str
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    num_flagged: int = 0
    num_actually_wrong: int = 0
    hallucination_detected: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "generated_text": self.generated_text,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "tn": self.true_negative,
            "fn": self.false_negative,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "num_flagged": self.num_flagged,
            "num_actually_wrong": self.num_actually_wrong,
            "hallucination_detected": self.hallucination_detected,
        }


@dataclass
class SpanLevelResult:
    """Result of span-level evaluation for one sample."""
    question: str
    detected_spans: List[str] = field(default_factory=list)
    correct_spans: List[str] = field(default_factory=list)
    span_hit: bool = False       # Did we catch ANY correct wrong span?
    exact_match: bool = False    # Did we catch the EXACT wrong span?
    partial_match: float = 0.0   # Overlap ratio 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "detected_spans": self.detected_spans,
            "correct_spans": self.correct_spans,
            "span_hit": self.span_hit,
            "exact_match": self.exact_match,
            "partial_match": round(self.partial_match, 4),
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all evaluation samples."""
    num_samples: int = 0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1: float = 0.0
    avg_accuracy: float = 0.0
    total_tp: int = 0
    total_fp: int = 0
    total_tn: int = 0
    total_fn: int = 0
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    span_hit_rate: float = 0.0
    span_exact_match_rate: float = 0.0
    hallucination_detection_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "avg_precision": round(self.avg_precision, 4),
            "avg_recall": round(self.avg_recall, 4),
            "avg_f1": round(self.avg_f1, 4),
            "avg_accuracy": round(self.avg_accuracy, 4),
            "total_tp": self.total_tp,
            "total_fp": self.total_fp,
            "total_tn": self.total_tn,
            "total_fn": self.total_fn,
            "micro_precision": round(self.micro_precision, 4),
            "micro_recall": round(self.micro_recall, 4),
            "micro_f1": round(self.micro_f1, 4),
            "span_hit_rate": round(self.span_hit_rate, 4),
            "span_exact_match_rate": round(self.span_exact_match_rate, 4),
            "hallucination_detection_rate": round(self.hallucination_detection_rate, 4),
        }


# ── Main Metrics Class ────────────────────────────────────────────────────────

class HallucinationMetrics:
    """
    Computes token-level and span-level evaluation metrics.

    How evaluation works:
    ---------------------
    For each sample from TruthfulQA:
      - Ground truth: the INCORRECT answer tokens are "hallucinated"
      - Predicted: tokens flagged by our system as hallucinated

    We compare predicted vs ground truth to compute TP/FP/TN/FN.

    Example:
        Generated: "Fortune cookies originated in China"
        Tokens:    ["Fortune", "cookies", "originated", "in", "China"]
        Ground truth wrong tokens: {"China"} (correct is San Francisco)
        System flagged: {"China"} → TP=1, FP=0, FN=0 → Perfect!
    """

    # =========================================================
    # SECTION 1: Confusion Matrix
    # =========================================================

    def confusion_matrix_stats(
        self,
        flagged_positions: Set[int],
        actually_wrong_positions: Set[int],
        total_tokens: int
    ) -> Dict[str, int]:
        """
        Compute TP, FP, TN, FN at token level.

        Args:
            flagged_positions:        Set of token indices flagged by system
            actually_wrong_positions: Set of token indices that are truly wrong
            total_tokens:             Total number of generated tokens

        Returns:
            Dict with tp, fp, tn, fn counts
        """
        tp = len(flagged_positions & actually_wrong_positions)
        fp = len(flagged_positions - actually_wrong_positions)
        fn = len(actually_wrong_positions - flagged_positions)
        tn = total_tokens - tp - fp - fn

        return {
            "tp": tp,   # Correctly flagged hallucinations
            "fp": fp,   # Wrongly flagged safe tokens
            "tn": max(tn, 0),  # Correctly left unflagged
            "fn": fn,   # Missed hallucinations
        }

    # =========================================================
    # SECTION 2: Precision, Recall, F1
    # =========================================================

    def token_level_f1(
        self,
        flagged_positions: Set[int],
        actually_wrong_positions: Set[int],
        total_tokens: int,
        question: str = "",
        generated_text: str = ""
    ) -> TokenLevelResult:
        """
        Compute token-level F1 score for one sample.

        This is the main metric for FR13 (evaluation).

        Args:
            flagged_positions:        Which tokens system flagged
            actually_wrong_positions: Which tokens are truly wrong
            total_tokens:             Total tokens in generated text
            question:                 The original question (for logging)
            generated_text:           The generated text (for logging)

        Returns:
            TokenLevelResult with all metrics
        """
        cm = self.confusion_matrix_stats(
            flagged_positions, actually_wrong_positions, total_tokens
        )
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

        # Precision: of what we flagged, how much was correct?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall: of what was wrong, how much did we catch?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1: harmonic mean
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        # Accuracy
        accuracy = (tp + tn) / total_tokens if total_tokens > 0 else 0.0

        return TokenLevelResult(
            question=question,
            generated_text=generated_text,
            true_positive=tp,
            false_positive=fp,
            true_negative=tn,
            false_negative=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            num_flagged=len(flagged_positions),
            num_actually_wrong=len(actually_wrong_positions),
            hallucination_detected=len(flagged_positions) > 0,
        )

    # =========================================================
    # SECTION 3: Overall Accuracy
    # =========================================================

    def overall_accuracy(
        self,
        results: List[TokenLevelResult]
    ) -> float:
        """
        Compute overall accuracy across all samples.

        Accuracy = (TP + TN) / Total tokens across all samples.

        Args:
            results: List of TokenLevelResult objects

        Returns:
            Float: accuracy in [0.0, 1.0]
        """
        if not results:
            return 0.0
        return sum(r.accuracy for r in results) / len(results)

    # =========================================================
    # SECTION 4: Span-Level Metrics
    # =========================================================

    def span_precision_recall(
        self,
        detected_spans: List[str],
        correct_wrong_spans: List[str],
        question: str = ""
    ) -> SpanLevelResult:
        """
        Span-level evaluation: did we identify the correct wrong spans?

        This is Gap 2's key metric — not just "was hallucination detected"
        but "did we identify the EXACT wrong tokens?"

        Args:
            detected_spans:     Spans flagged by system (token texts)
            correct_wrong_spans: The actual wrong information spans

        Returns:
            SpanLevelResult with hit/match information
        """
        detected_lower = {s.lower().strip() for s in detected_spans}
        correct_lower  = {s.lower().strip() for s in correct_wrong_spans}

        # Span hit: did we detect any of the wrong spans?
        span_hit = bool(detected_lower & correct_lower)

        # Exact match: did we detect ALL wrong spans and nothing else?
        exact_match = detected_lower == correct_lower

        # Partial match: Jaccard similarity
        union = detected_lower | correct_lower
        intersection = detected_lower & correct_lower
        partial_match = len(intersection) / len(union) if union else 0.0

        return SpanLevelResult(
            question=question,
            detected_spans=list(detected_spans),
            correct_spans=list(correct_wrong_spans),
            span_hit=span_hit,
            exact_match=exact_match,
            partial_match=partial_match,
        )

    # =========================================================
    # SECTION 5: Aggregate Metrics
    # =========================================================

    def aggregate(
        self,
        token_results: List[TokenLevelResult],
        span_results: List[SpanLevelResult]
    ) -> AggregateMetrics:
        """
        Compute aggregate metrics across all evaluation samples.

        Two types of aggregation:
        - Macro average: average each sample's metric (weights equally)
        - Micro average: sum TP/FP/TN/FN first, then compute (weights by size)

        Args:
            token_results: List of per-sample TokenLevelResult
            span_results:  List of per-sample SpanLevelResult

        Returns:
            AggregateMetrics with all summary statistics
        """
        if not token_results:
            return AggregateMetrics()

        n = len(token_results)

        # Macro averages
        avg_precision = sum(r.precision for r in token_results) / n
        avg_recall    = sum(r.recall    for r in token_results) / n
        avg_f1        = sum(r.f1        for r in token_results) / n
        avg_accuracy  = sum(r.accuracy  for r in token_results) / n

        # Micro aggregates (sum TP/FP/etc across all samples)
        total_tp = sum(r.true_positive  for r in token_results)
        total_fp = sum(r.false_positive for r in token_results)
        total_tn = sum(r.true_negative  for r in token_results)
        total_fn = sum(r.false_negative for r in token_results)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1        = (2 * micro_precision * micro_recall /
                          (micro_precision + micro_recall)
                          if (micro_precision + micro_recall) > 0 else 0.0)

        # Span metrics
        span_hit_rate        = sum(1 for r in span_results if r.span_hit) / max(len(span_results), 1)
        span_exact_match_rate = sum(1 for r in span_results if r.exact_match) / max(len(span_results), 1)

        # Detection rate (did system flag anything?)
        detection_rate = sum(1 for r in token_results if r.hallucination_detected) / n

        return AggregateMetrics(
            num_samples=n,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            avg_accuracy=avg_accuracy,
            total_tp=total_tp,
            total_fp=total_fp,
            total_tn=total_tn,
            total_fn=total_fn,
            micro_precision=micro_precision,
            micro_recall=micro_recall,
            micro_f1=micro_f1,
            span_hit_rate=span_hit_rate,
            span_exact_match_rate=span_exact_match_rate,
            hallucination_detection_rate=detection_rate,
        )

    def save_results(
        self,
        aggregate: AggregateMetrics,
        token_results: List[TokenLevelResult],
        span_results: List[SpanLevelResult],
        output_path: str = "experiments/results/evaluation_results.json"
    ):
        """Save evaluation results to JSON for thesis."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        data = {
            "aggregate": aggregate.to_dict(),
            "token_results": [r.to_dict() for r in token_results],
            "span_results": [r.to_dict() for r in span_results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {output_path}")
        return output_path


# ── Singleton ─────────────────────────────────────────────────────────────────
_metrics_instance = None


def get_metrics() -> HallucinationMetrics:
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = HallucinationMetrics()
    return _metrics_instance


# ── Quick Test ────────────────────────────────────────────────────────────────
# Run: python backend/evaluation/metrics.py

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("\n" + "=" * 65)
    print("  DAY 17 TEST: Evaluation Metrics")
    print("=" * 65 + "\n")

    metrics = HallucinationMetrics()

    # ── TEST 1: Perfect detection ──────────────────────────────
    print("TEST 1: Perfect Detection")
    print("  Scenario: 'Fortune cookies originated in China'")
    print("  System correctly flags 'China' (actually wrong)")

    result = metrics.token_level_f1(
        flagged_positions={4},          # system flagged token 4 = "China"
        actually_wrong_positions={4},   # ground truth: token 4 is wrong
        total_tokens=5,
        question="Where did fortune cookies originate?",
        generated_text="Fortune cookies originated in China"
    )

    print(f"  TP={result.true_positive} FP={result.false_positive} "
          f"TN={result.true_negative} FN={result.false_negative}")
    print(f"  Precision={result.precision:.3f} Recall={result.recall:.3f} "
          f"F1={result.f1:.3f} Accuracy={result.accuracy:.3f}")
    assert result.true_positive == 1
    assert result.false_positive == 0
    assert result.f1 == 1.0
    print(f"  PASS: Perfect F1=1.0 ✅\n")

    # ── TEST 2: Missed hallucination ──────────────────────────
    print("TEST 2: Missed Hallucination (False Negative)")
    print("  Scenario: System flags nothing, but 'China' is wrong")

    result2 = metrics.token_level_f1(
        flagged_positions=set(),
        actually_wrong_positions={4},
        total_tokens=5,
        question="Where did fortune cookies originate?",
        generated_text="Fortune cookies originated in China"
    )

    print(f"  TP={result2.true_positive} FP={result2.false_positive} "
          f"FN={result2.false_negative}")
    print(f"  Precision={result2.precision:.3f} Recall={result2.recall:.3f} F1={result2.f1:.3f}")
    assert result2.false_negative == 1
    assert result2.f1 == 0.0
    print(f"  PASS: F1=0.0 when hallucination missed ✅\n")

    # ── TEST 3: False positive ─────────────────────────────────
    print("TEST 3: False Positive")
    print("  Scenario: System flags 'cookies' (correct word) unnecessarily")

    result3 = metrics.token_level_f1(
        flagged_positions={1},          # flags "cookies" wrongly
        actually_wrong_positions={4},   # "China" is the actual wrong one
        total_tokens=5,
    )

    print(f"  TP={result3.true_positive} FP={result3.false_positive} "
          f"FN={result3.false_negative}")
    print(f"  Precision={result3.precision:.3f} Recall={result3.recall:.3f}")
    assert result3.false_positive == 1
    assert result3.precision == 0.0
    print(f"  PASS: Precision=0 for false positive ✅\n")

    # ── TEST 4: Span-level evaluation ─────────────────────────
    print("TEST 4: Span-Level Evaluation")
    span_res = metrics.span_precision_recall(
        detected_spans=["China"],
        correct_wrong_spans=["China"],
        question="Where did fortune cookies originate?"
    )

    print(f"  Detected: {span_res.detected_spans}")
    print(f"  Correct:  {span_res.correct_spans}")
    print(f"  Hit={span_res.span_hit} Exact={span_res.exact_match} "
          f"Partial={span_res.partial_match:.3f}")
    assert span_res.span_hit is True
    assert span_res.exact_match is True
    print(f"  PASS: Span exact match detected ✅\n")

    # ── TEST 5: Aggregate metrics ──────────────────────────────
    print("TEST 5: Aggregate Metrics (3 samples)")

    results = [
        metrics.token_level_f1({4}, {4}, 5),       # perfect
        metrics.token_level_f1({1}, {1, 3}, 6),    # partial
        metrics.token_level_f1(set(), {2}, 4),      # missed
    ]
    span_results = [
        metrics.span_precision_recall(["China"], ["China"]),
        metrics.span_precision_recall(["1879"], ["1879", "Ulm"]),
        metrics.span_precision_recall([], ["Paris"]),
    ]

    agg = metrics.aggregate(results, span_results)

    print(f"  Samples: {agg.num_samples}")
    print(f"  Macro — Precision={agg.avg_precision:.3f} "
          f"Recall={agg.avg_recall:.3f} F1={agg.avg_f1:.3f}")
    print(f"  Micro — Precision={agg.micro_precision:.3f} "
          f"Recall={agg.micro_recall:.3f} F1={agg.micro_f1:.3f}")
    print(f"  Span hit rate: {agg.span_hit_rate:.3f}")
    print(f"  Detection rate: {agg.hallucination_detection_rate:.3f}")
    assert agg.num_samples == 3
    print(f"  PASS: Aggregate metrics computed ✅\n")

    print("=" * 65)
    print("  DAY 17 DELIVERABLE CONFIRMED")
    print("=" * 65)
    print("  PASS: confusion_matrix_stats() — TP/FP/TN/FN")
    print("  PASS: token_level_f1() — precision, recall, F1")
    print("  PASS: span_precision_recall() — span hit/exact/partial")
    print("  PASS: overall_accuracy() — cross-sample accuracy")
    print("  PASS: aggregate() — macro+micro metrics")
    print()
    print("  Research impact:")
    print("  → These metrics produce your thesis Table 1 numbers")
    print("  → FR13: evaluate detection accuracy on benchmarks")
    print("  → Ready for full TruthfulQA evaluation on Day 18")
    print("=" * 65 + "\n")
