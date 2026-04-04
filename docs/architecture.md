# System Architecture

## Overview

The hallucination detection system consists of 4 modules that run synchronously with the LLM's generation pipeline.

## Module A — EAT Detection
- Uses spaCy NER to identify Exact Answer Tokens (names, dates, numbers)
- Maps entity spans to token position indices
- Feeds EAT positions to Module D for localized scoring

## Module B — Hidden State Extraction
- Attaches PyTorch `register_forward_hook` to target layers
- Extracts hidden states at TBG (Token Before Generating) position
- Applies INSIDE Feature Clipping to suppress overconfidence

## Module C — Distribution Shift
- Computes Wasserstein distance between consecutive layer activations
- Computes cosine similarity across layer window (size=2)
- Trains and applies Truthfulness Separator Vector (TSV)

## Module D — Scoring & Localization
- Calculates semantic entropy from hidden state distributions
- Aggregates score: 0.4×entropy + 0.4×Wasserstein + 0.2×TSV
- Maps scores only to EAT positions (Span-Level Mapper)
- Flags tokens above HALLUCINATION_THRESHOLD

## Pipeline Flow
```
Prompt → Module A (EAT) → LLM Generation
                               │
                     Module B (Hidden States)
                               │
                     Module C (Shift + TSV)
                               │
                     Module D (Score + Localize)
                               │
                    Annotated Token Output
```
