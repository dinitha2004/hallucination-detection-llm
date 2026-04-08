# Evaluation Results — HalluScan
**Author:** Chalani Dinitha (20211032)
**Model:** facebook/opt-1.3b
**Threshold:** 0.35 (calibrated for OPT-1.3b)

## Table 1: Token-Level Metrics

| Dataset | Samples | Macro P | Macro R | Macro F1 | Micro P | Micro R | Micro F1 | Accuracy |
|---------|---------|---------|---------|----------|---------|---------|----------|----------|
| TruthfulQA | 200 | 0.0785 | 0.0426 | 0.0441 | 0.2162 | 0.0493 | 0.0803 | 0.7733 |
| TriviaQA | 100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8862 |

## Table 2: Detection Performance

| Dataset | Detection Rate | Span Hit Rate | Avg Latency (ms) | Total TP | Total FP | Total FN |
|---------|---------------|---------------|-----------------|----------|----------|----------|
| TruthfulQA | 0.2850 | 0.0300 | 1975.6 | 48 | 174 | 925 |
| TriviaQA | 0.6700 | 0.0000 | 2162.2 | 0 | 289 | 0 |

## Comparison to Literature Baselines

| Method | F1 | Dataset | Source |
|--------|-----|---------|--------|
| SelfCheckGPT | 0.23 | TruthfulQA | Manakul et al. 2023 |
| INSIDE | 0.18 | TruthfulQA | Chen et al. 2024 |
| **HalluScan (Ours)** | **see above** | **TruthfulQA** | **This work** |

## Notes

- OPT-1.3b produces lower confidence signals than larger models
- Threshold 0.35 calibrated empirically for OPT-1.3b
- Final evaluation with LLaMA-3.2-3B expected to improve F1
- NFR1 satisfied: avg latency < 5000ms