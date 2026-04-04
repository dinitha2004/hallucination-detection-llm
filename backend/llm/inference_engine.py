"""
inference_engine.py — Token Generation + Hidden State Capture
==============================================================
This file runs the LLM to generate text AND captures hidden states.

IMPORTANT FOR BEGINNERS — What is "inference"?
-----------------------------------------------
"Inference" means using a trained model to make predictions.
We are NOT training the model (that would take weeks and huge computers).
We are just USING it to generate text, while secretly watching its internal
numbers (hidden states) as it generates each word.

How the LLM generates text:
---------------------------
1. You give it a prompt: "What year was Einstein born?"
2. It predicts the next most likely word/token: "Einstein"... "was"... "born"... "in"... "1879"
3. It keeps adding one word at a time until it decides to stop.
4. At each step, every layer inside the model produces a "hidden state" —
   a vector of numbers representing its internal "understanding" at that moment.

Why output_hidden_states=True matters:
--------------------------------------
Normally model.generate() only returns the text.
But with output_hidden_states=True and return_dict_in_generate=True,
it ALSO returns a tuple of hidden states for every generated token.
This is what we analyze to detect hallucinations (Gap 1).

Author: Chalani Dinitha (20211032)
"""

import sys
import logging
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

sys.path.append(".")
from backend.config import (
    MAX_NEW_TOKENS, TARGET_LAYERS, DEVICE
)
from backend.llm.model_loader import get_model_loader

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """
    Everything we get back from one inference run.

    Think of this as a structured container — instead of returning
    a messy mix of variables, we pack everything neatly into one object.
    """
    # The generated text (what the LLM actually wrote)
    generated_text: str = ""

    # The original prompt the user typed
    prompt: str = ""

    # List of individual generated tokens (words/subwords)
    # Example: ["The", " capital", " of", " France", " is", " Paris", "."]
    tokens: List[str] = field(default_factory=list)

    # Token IDs (the numbers the model uses internally)
    # Example: [464, 3139, 286, 4881, 318, 6342, 13]
    token_ids: List[int] = field(default_factory=list)

    # Hidden states for each generated token, organized by layer
    # Structure: { layer_index: [tensor_for_token_0, tensor_for_token_1, ...] }
    # Each tensor has shape: (1, hidden_size) e.g. (1, 2048)
    hidden_states_by_layer: Dict[int, List[torch.Tensor]] = field(default_factory=dict)

    # How long generation took (milliseconds)
    generation_time_ms: float = 0.0

    # Did something go wrong?
    error: Optional[str] = None

    @property
    def num_tokens(self) -> int:
        """How many tokens were generated."""
        return len(self.tokens)

    @property
    def success(self) -> bool:
        """Was generation successful?"""
        return self.error is None and len(self.tokens) > 0


class InferenceEngine:
    """
    Runs text generation and captures hidden states.

    This class is the bridge between your research and the LLM.
    It handles all the complexity of running the model so that
    the detection modules (B, C, D) receive clean, ready-to-use data.
    """

    def __init__(self):
        self._loader = get_model_loader()

    def generate(self, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> GenerationResult:
        """
        Generate text for a given prompt AND capture all hidden states.

        This is the main function you call when a user submits a question.

        Args:
            prompt: The question or text from the user.
            max_new_tokens: Maximum number of new words to generate.

        Returns:
            GenerationResult object containing:
            - generated_text: The full response
            - tokens: Each word/subword that was generated
            - hidden_states_by_layer: Internal model states (for hallucination detection)

        Example:
            engine = InferenceEngine()
            result = engine.generate("What year was Einstein born?")
            print(result.generated_text)   # "Einstein was born in 1879."
            print(result.tokens)           # ["Einstein", " was", ...]
            print(result.hidden_states_by_layer[20].shape)  # torch.Size([1, 2048])
        """
        import time

        result = GenerationResult(prompt=prompt)
        start_time = time.time()

        try:
            model = self._loader.get_model()
            tokenizer = self._loader.get_tokenizer()

            # ── Step 1: Tokenize the prompt ───────────────────────────
            # Convert the text string into numbers the model can read
            logger.info(f"Tokenizing prompt: '{prompt[:60]}...'")

            inputs = tokenizer(
                prompt,
                return_tensors="pt",  # "pt" means PyTorch tensors
                padding=True,
                truncation=True,
                max_length=512,       # limit prompt length
            ).to(self._loader.device)

            input_length = inputs["input_ids"].shape[1]
            logger.info(f"Prompt tokenized: {input_length} tokens")

            # ── Step 2: Run generation ────────────────────────────────
            # This is where the LLM actually generates the response.
            # We tell it to:
            #   - return_dict_in_generate=True  → give us a dict, not just text
            #   - output_hidden_states=True      → include hidden states in output
            logger.info(f"Running generation (max {max_new_tokens} new tokens)...")

            with torch.no_grad():
                # torch.no_grad() tells PyTorch we are NOT training
                # This saves memory and makes inference faster
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,   # ← gives us structured output
                    output_hidden_states=True,       # ← gives us hidden states!
                    do_sample=False,                 # deterministic (reproducible) output
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # ── Step 3: Extract generated tokens ─────────────────────
            # The model generated new tokens AFTER the prompt tokens
            # We only want the NEW tokens, not the prompt tokens again
            generated_ids = outputs.sequences[0][input_length:]
            result.token_ids = generated_ids.tolist()

            # Decode each token individually into text
            # This gives us per-token strings for span-level mapping later
            result.tokens = [
                tokenizer.decode([tid], skip_special_tokens=True)
                for tid in result.token_ids
            ]

            # Full generated text (everything after the prompt)
            result.generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

            logger.info(f"Generated {len(result.tokens)} tokens")
            logger.info(f"Output: '{result.generated_text[:100]}...'")

            # ── Step 4: Extract hidden states ─────────────────────────
            # outputs.hidden_states is a tuple of tuples:
            #   - Outer tuple: one entry per generated token (length = num_new_tokens)
            #   - Inner tuple: one tensor per layer (length = num_layers + 1)
            #   - Each tensor shape: (batch_size, sequence_length, hidden_size)
            #
            # We extract: for each target layer → list of hidden state vectors
            #             one vector per generated token

            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                result.hidden_states_by_layer = self._extract_hidden_states(
                    hidden_states_all_steps=outputs.hidden_states,
                    num_generated_tokens=len(result.token_ids)
                )
                logger.info(
                    f"✓ Hidden states extracted for layers: {list(result.hidden_states_by_layer.keys())}"
                )
            else:
                logger.warning("⚠ No hidden states in output — check output_hidden_states=True in model config")

        except Exception as e:
            result.error = str(e)
            logger.error(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()

        result.generation_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Total generation time: {result.generation_time_ms:.1f}ms")

        return result

    def _extract_hidden_states(
        self,
        hidden_states_all_steps: tuple,
        num_generated_tokens: int
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Reorganize hidden states from the model's output format into
        a cleaner structure organized by layer index.

        Model output format (messy):
            hidden_states_all_steps[step][layer][batch, seq, hidden]

        What we want (clean):
            {layer_index: [vector_for_token_0, vector_for_token_1, ...]}

        Each vector is the hidden state of ONE token in ONE layer.
        This is what Module B and Module C will analyze.

        Args:
            hidden_states_all_steps: Raw hidden state output from model.generate()
            num_generated_tokens: How many new tokens were generated

        Returns:
            Dict mapping layer_index → list of hidden state tensors
        """
        extracted = {layer: [] for layer in TARGET_LAYERS}

        for step_idx in range(num_generated_tokens):
            if step_idx >= len(hidden_states_all_steps):
                break

            # hidden_states for this generation step
            # This is a tuple: one tensor per layer
            step_hidden = hidden_states_all_steps[step_idx]

            for layer_idx in TARGET_LAYERS:
                # layer_idx + 1 because index 0 is the embedding layer, not a transformer layer
                actual_layer_idx = layer_idx + 1

                if actual_layer_idx < len(step_hidden):
                    # Shape: (batch, seq_len, hidden_size)
                    # We take the LAST position in the sequence — this is the
                    # TBG (Token Before Generating) position from your research
                    layer_tensor = step_hidden[actual_layer_idx]

                    # Extract last position, squeeze batch dimension
                    # Result shape: (hidden_size,) e.g. (2048,)
                    tbg_vector = layer_tensor[0, -1, :].detach().cpu()

                    extracted[layer_idx].append(tbg_vector)

        return extracted

    def get_tbg_hidden_state(
        self,
        result: GenerationResult,
        layer_idx: int,
        token_position: int
    ) -> Optional[torch.Tensor]:
        """
        Get the TBG (Token Before Generating) hidden state for a specific
        token at a specific layer.

        This is the key concept from your finalized technique:
        "Extract hidden activations at the TBG position — the last token
        of the input prompt in a single forward pass."

        Args:
            result: A completed GenerationResult
            layer_idx: Which layer to look at (e.g. 16, 20, 24)
            token_position: Which generated token position (0, 1, 2, ...)

        Returns:
            A tensor of shape (hidden_size,) or None if unavailable
        """
        if layer_idx not in result.hidden_states_by_layer:
            return None
        layer_states = result.hidden_states_by_layer[layer_idx]
        if token_position >= len(layer_states):
            return None
        return layer_states[token_position]


# ── Singleton Pattern ─────────────────────────────────────────────────────────
_inference_engine_instance = None


def get_inference_engine() -> InferenceEngine:
    """Returns the global InferenceEngine instance (created once)."""
    global _inference_engine_instance
    if _inference_engine_instance is None:
        _inference_engine_instance = InferenceEngine()
    return _inference_engine_instance


# ── Quick Test ────────────────────────────────────────────────────────────────
# Run this to test:
# Command: python backend/llm/inference_engine.py

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TEST: Inference Engine")
    print("=" * 60 + "\n")

    # First load the model
    from backend.llm.model_loader import get_model_loader
    loader = get_model_loader()
    if not loader.load():
        print("❌ Model loading failed. Exiting.")
        sys.exit(1)

    # Now test generation
    engine = InferenceEngine()
    test_prompt = "The capital of France is"

    print(f"Test prompt: '{test_prompt}'")
    print("Running inference...\n")

    result = engine.generate(test_prompt, max_new_tokens=20)

    if result.success:
        print("✅ GENERATION SUCCESSFUL")
        print(f"\nGenerated text: '{result.generated_text}'")
        print(f"Number of tokens: {result.num_tokens}")
        print(f"Tokens: {result.tokens}")
        print(f"Generation time: {result.generation_time_ms:.1f}ms")
        print(f"\nHidden states captured for layers: {list(result.hidden_states_by_layer.keys())}")

        for layer in TARGET_LAYERS:
            if layer in result.hidden_states_by_layer:
                states = result.hidden_states_by_layer[layer]
                print(f"  Layer {layer}: {len(states)} token vectors, each shape: {states[0].shape}")

        print("\n✅ Day 2 deliverable CONFIRMED:")
        print("   - Model loads without errors")
        print("   - Hidden states output flag working")
        print("   - Token-level hidden states extracted per layer")
    else:
        print(f"❌ Generation failed: {result.error}")
        sys.exit(1)
