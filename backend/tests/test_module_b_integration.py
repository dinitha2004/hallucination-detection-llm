"""
test_module_b_integration.py — Day 5: Module B Integration Tests
================================================================
Integration tests for Module B — these test the FULL pipeline:
Model → Hooks → Hidden States → Feature Clipping → Memory Bank

Unlike unit tests (which use dummy data), integration tests
use the REAL model to verify everything works end-to-end.

HOW TO RUN:
    pytest backend/tests/test_module_b_integration.py -v -s

NOTE: These tests load the actual model, so they take ~1-2 minutes.
Run unit tests (test_module_b.py) for fast checks during development.

Author: Chalani Dinitha (20211032)
Day 5 Deliverable: Module B fully tested and merged to main
"""

import sys
import pytest
import torch

sys.path.append(".")

# ── Shared model fixture ──────────────────────────────────────────────────────
# We load the model ONCE and share it across all tests in this file.
# This avoids reloading 2.5GB of weights for every test.

@pytest.fixture(scope="module")
def loaded_model():
    """
    Load the model once for all integration tests.
    scope="module" means it loads once, shared across all tests here.
    """
    from backend.llm.model_loader import ModelLoader
    loader = ModelLoader()
    success = loader.load()
    if not success:
        pytest.skip("Model could not be loaded — skipping integration tests")
    return loader


@pytest.fixture(scope="module")
def extractor():
    """Create a fresh HiddenStateExtractor for integration tests."""
    from backend.modules.module_b_hidden import HiddenStateExtractor
    return HiddenStateExtractor()


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestModuleBIntegration:
    """
    End-to-end integration tests for Module B.
    These verify the complete pipeline with the real model.
    """

    def test_hooks_attach_to_real_model(self, loaded_model, extractor):
        """
        Verify hooks attach successfully to the real OPT/LLaMA model.
        This confirms our _get_model_layers() correctly finds the
        transformer layers in the model architecture.
        """
        model = loaded_model.get_model()
        extractor.clear_captured()

        n_hooks = extractor.attach_hooks(model)

        assert n_hooks > 0, "No hooks attached — model architecture not recognized"
        assert n_hooks == len(extractor.target_layers), \
            f"Expected {len(extractor.target_layers)} hooks, got {n_hooks}"

        extractor.remove_hooks()
        print(f"\n  PASS: {n_hooks} hooks attached to layers {extractor.target_layers}")

    def test_hidden_states_captured_during_generation(self, loaded_model, extractor):
        """
        Core Gap 1 test: verify hidden states are captured DURING
        token generation, not after.

        This is the most important integration test for your research.
        """
        model = loaded_model.get_model()
        tokenizer = loaded_model.get_tokenizer()

        extractor.clear_captured()
        extractor.attach_hooks(model)

        # Run generation
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        n_new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        extractor.remove_hooks()

        # Verify captures
        activations = extractor.extract_layer_activations()

        assert len(activations) > 0, "No hidden states captured"

        for layer_idx in extractor.target_layers:
            assert layer_idx in activations, \
                f"Layer {layer_idx} not captured"
            assert len(activations[layer_idx]) == n_new_tokens, \
                f"Expected {n_new_tokens} vectors, got {len(activations[layer_idx])}"

        print(f"\n  PASS: {n_new_tokens} tokens captured at "
              f"{len(activations)} layers")

    def test_tbg_vector_shape_matches_hidden_size(self, loaded_model, extractor):
        """
        Verify TBG vector shape matches model's hidden_size.
        For OPT-1.3B: hidden_size = 2048, so shape should be (2048,)
        """
        model = loaded_model.get_model()
        tokenizer = loaded_model.get_tokenizer()
        info = loaded_model.get_model_info()
        expected_hidden_size = info["hidden_size"]

        extractor.clear_captured()
        extractor.attach_hooks(model)

        inputs = tokenizer("Einstein was born in", return_tensors="pt")
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        extractor.remove_hooks()

        first_layer = extractor.target_layers[0]
        tbg = extractor.extract_tbg_hidden_state(first_layer, 0)

        assert tbg is not None
        assert tbg.dim() == 1, f"Expected 1D tensor, got {tbg.dim()}D"
        assert tbg.shape[0] == expected_hidden_size, \
            f"Expected shape ({expected_hidden_size},), got {tbg.shape}"

        print(f"\n  PASS: TBG shape = {tbg.shape} "
              f"(matches hidden_size={expected_hidden_size})")

    def test_later_layers_have_higher_norms(self, loaded_model, extractor):
        """
        Research finding test: verify that later layers have higher
        hidden state norms than earlier layers.

        This is your KEY RESEARCH FINDING:
        'Mid-to-late layers contain stronger truthfulness signals'

        From your Day 4 results:
            Layer 12: 28.73  (lower)
            Layer 16: 42.76  (medium)
            Layer 20: 84.96  (highest)
        """
        model = loaded_model.get_model()
        tokenizer = loaded_model.get_tokenizer()

        extractor.clear_captured()
        extractor.attach_hooks(model)

        inputs = tokenizer("The largest planet is", return_tensors="pt")
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        extractor.remove_hooks()
        activations = extractor.extract_layer_activations()

        # Get norms for each target layer at first token
        norms = {}
        for layer_idx in extractor.target_layers:
            if layer_idx in activations and activations[layer_idx]:
                norms[layer_idx] = activations[layer_idx][0].norm().item()

        sorted_layers = sorted(extractor.target_layers)
        print(f"\n  Layer norms:")
        for layer in sorted_layers:
            if layer in norms:
                print(f"    Layer {layer}: {norms[layer]:.4f}")

        # Later layers should generally have higher norms
        if len(sorted_layers) >= 2:
            early = sorted_layers[0]
            late = sorted_layers[-1]
            if early in norms and late in norms:
                print(f"  Early layer ({early}) norm: {norms[early]:.4f}")
                print(f"  Late layer ({late}) norm:  {norms[late]:.4f}")
                assert norms[late] > norms[early], \
                    f"Expected later layer to have higher norm: " \
                    f"Layer {late}={norms[late]:.2f} vs Layer {early}={norms[early]:.2f}"
                print(f"  PASS: Layer {late} > Layer {early} (confirms research finding)")

    def test_feature_clipping_reduces_max_on_real_data(self, loaded_model, extractor):
        """
        Test INSIDE Feature Clipping on REAL hidden states (not dummy data).
        Verifies the mechanism works with actual model activations.
        """
        model = loaded_model.get_model()
        tokenizer = loaded_model.get_tokenizer()

        extractor.clear_captured()
        extractor.attach_hooks(model)

        inputs = tokenizer("Water boils at", return_tensors="pt")
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        extractor.remove_hooks()
        activations = extractor.extract_layer_activations()

        # Test clipping on first layer
        first_layer = extractor.target_layers[0]
        if first_layer in activations and activations[first_layer]:
            original = activations[first_layer][0]
            clipped = extractor.apply_inside_feature_clipping(original, first_layer)

            assert clipped.shape == original.shape
            assert clipped.max().item() <= original.max().item()

            print(f"\n  Original max: {original.max().item():.4f}")
            print(f"  Clipped max:  {clipped.max().item():.4f}")
            print(f"  PASS: Feature clipping reduces max on real model data")

    def test_memory_bank_updates_with_real_activations(self, loaded_model, extractor):
        """
        Test that the Memory Bank correctly stores real model activations
        and computes meaningful clip thresholds.
        """
        model = loaded_model.get_model()
        tokenizer = loaded_model.get_tokenizer()

        # Run a few truthful prompts to build memory bank
        truthful_prompts = [
            "The sun rises in the east",
            "Paris is in France",
            "Water is made of hydrogen and oxygen",
        ]

        for prompt in truthful_prompts:
            extractor.clear_captured()
            extractor.attach_hooks(model)
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            extractor.remove_hooks()
            activations = extractor.extract_layer_activations()
            extractor.update_memory_bank(activations, label="truthful")

        assert extractor.has_memory_bank
        assert len(extractor._clip_thresholds) > 0

        print(f"\n  Memory bank built from {len(truthful_prompts)} prompts")
        print(f"  Clip thresholds computed: {extractor._clip_thresholds}")
        print(f"  PASS: Memory bank operational with real model data")

    def test_hooks_removed_after_generation(self, loaded_model, extractor):
        """
        Verify hooks are properly removed after calling remove_hooks().
        Leaked hooks cause memory issues and incorrect captures.
        """
        model = loaded_model.get_model()
        extractor.attach_hooks(model)
        assert len(extractor._hooks) > 0

        extractor.remove_hooks()
        assert len(extractor._hooks) == 0

        print(f"\n  PASS: All hooks removed cleanly, no memory leaks")

    def test_repeated_generation_gives_consistent_norms(self, loaded_model, extractor):
        """
        NFR4 Reliability: same prompt must give same hidden state norms.
        This verifies the deterministic (do_sample=False) generation works.
        """
        model = loaded_model.get_model()
        tokenizer = loaded_model.get_tokenizer()
        prompt = "The capital of Germany is"

        norms_run1 = []
        norms_run2 = []
        first_layer = extractor.target_layers[0]

        for norms_list in [norms_run1, norms_run2]:
            extractor.clear_captured()
            extractor.attach_hooks(model)
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            extractor.remove_hooks()
            acts = extractor.extract_layer_activations()
            if first_layer in acts:
                for v in acts[first_layer]:
                    norms_list.append(round(v.norm().item(), 3))

        assert norms_run1 == norms_run2, \
            f"Norms differ between runs!\nRun1: {norms_run1}\nRun2: {norms_run2}"

        print(f"\n  Run 1 norms: {norms_run1}")
        print(f"  Run 2 norms: {norms_run2}")
        print(f"  PASS: NFR4 Reliability confirmed — same input = same output")


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
