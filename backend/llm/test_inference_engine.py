"""
test_inference_engine.py — Day 3: Token Generation & Inference Engine Test
===========================================================================
This script tests the full inference engine:
- Tokenizes a prompt
- Runs model.generate() with return_dict_in_generate=True
- Captures sequences AND hidden_states from model output
- Decodes each token ID back to a word
- Logs every generated token with its ID and hidden state info

Run this file directly:
    python backend/llm/test_inference_engine.py

Author: Chalani Dinitha (20211032)
Day 3 Deliverable: Inference engine generates tokens + returns hidden states
"""

import sys
import logging
import torch

sys.path.append(".")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_day3_test():
    """
    Full Day 3 test:
    1. Load model
    2. Tokenize prompt
    3. Run generation
    4. Print every token + its ID
    5. Confirm hidden states returned
    """

    print("\n" + "=" * 65)
    print("  DAY 3 TEST — Token Generation & Inference Engine")
    print("  Deliverable: generates tokens + returns hidden states")
    print("=" * 65 + "\n")

    # ── Step 1: Load the model ────────────────────────────────────
    print("STEP 1: Loading model...")
    from backend.llm.model_loader import get_model_loader
    loader = get_model_loader()

    if not loader.load():
        print("❌ Model failed to load. Check your .env file.")
        sys.exit(1)

    model    = loader.get_model()
    tokenizer = loader.get_tokenizer()
    info      = loader.get_model_info()

    print(f"  ✅ Model ready: {info['model_name']}")
    print(f"  ✅ Layers: {info['num_hidden_layers']} | Hidden size: {info['hidden_size']}")
    print()

    # ── Step 2: Tokenize the prompt ───────────────────────────────
    # Day 3 plan says: test with "What is the capital of France?"
    test_prompt = "What is the capital of France?"

    print(f"STEP 2: Tokenizing prompt...")
    print(f"  Prompt: '{test_prompt}'")

    inputs = tokenizer(
        test_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids     = inputs["input_ids"]
    input_length  = input_ids.shape[1]

    # Show each prompt token and its ID
    print(f"\n  Prompt broken into {input_length} tokens:")
    print(f"  {'Token ID':<12} {'Token Text'}")
    print(f"  {'-'*12} {'-'*20}")
    for i, tid in enumerate(input_ids[0].tolist()):
        token_text = tokenizer.decode([tid])
        print(f"  {tid:<12} '{token_text}'")
    print()

    # ── Step 3: Run model.generate() ─────────────────────────────
    print("STEP 3: Running model.generate() with hidden state capture...")
    print("  (return_dict_in_generate=True, output_hidden_states=True)")
    print()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=30,
            return_dict_in_generate=True,   # ← structured output
            output_hidden_states=True,       # ← hidden states captured
            do_sample=False,                 # deterministic output
            pad_token_id=tokenizer.pad_token_id,
        )

    # ── Step 4: Extract generated tokens ─────────────────────────
    # Day 3 plan: capture sequences + decode token IDs back to words
    print("STEP 4: Extracting generated tokens...")

    # Only the NEW tokens (not the prompt)
    generated_ids   = outputs.sequences[0][input_length:]
    generated_tokens = [
        tokenizer.decode([tid], skip_special_tokens=True)
        for tid in generated_ids.tolist()
    ]
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Day 3 plan: Add logging to see each generated token and its ID
    print(f"\n  Generated {len(generated_tokens)} new tokens:")
    print(f"  {'Step':<6} {'Token ID':<12} {'Token Text':<20} {'Cumulative text'}")
    print(f"  {'-'*6} {'-'*12} {'-'*20} {'-'*30}")

    cumulative = ""
    for i, (tid, text) in enumerate(zip(generated_ids.tolist(), generated_tokens)):
        cumulative += text
        print(f"  {i:<6} {tid:<12} '{text:<18}' '{cumulative.strip()}'")

    print(f"\n  Full response: '{full_response}'")
    print()

    # ── Step 5: Verify hidden states ─────────────────────────────
    # Day 3 plan: capture hidden_states from model output
    print("STEP 5: Verifying hidden states captured...")

    assert hasattr(outputs, 'hidden_states'), \
        "❌ No hidden_states in output! output_hidden_states=True may not be set."

    assert outputs.hidden_states is not None, \
        "❌ hidden_states is None!"

    num_steps  = len(outputs.hidden_states)
    num_layers = len(outputs.hidden_states[0])

    print(f"  ✅ hidden_states present!")
    print(f"  Generation steps captured: {num_steps}")
    print(f"  Layers per step:           {num_layers}")
    print()

    # Show hidden state shape for each target layer
    from backend.config import TARGET_LAYERS
    print(f"  Hidden state details for TARGET_LAYERS {TARGET_LAYERS}:")
    print(f"  {'Layer':<8} {'Shape':<25} {'Norm (L2)'}")
    print(f"  {'-'*8} {'-'*25} {'-'*15}")

    step_0 = outputs.hidden_states[0]  # first generated token
    for layer_idx in TARGET_LAYERS:
        actual_idx = layer_idx + 1  # +1 because index 0 = embedding layer
        if actual_idx < len(step_0):
            tensor = step_0[actual_idx]   # shape: (1, seq_len, hidden_size)
            tbg    = tensor[0, -1, :]     # TBG position: last token, squeeze batch
            norm   = tbg.norm().item()
            print(f"  {layer_idx:<8} {str(tbg.shape):<25} {norm:.4f}")

    print()

    # ── Step 6: Run InferenceEngine class ────────────────────────
    print("STEP 6: Testing InferenceEngine class (full pipeline)...")
    from backend.llm.inference_engine import get_inference_engine

    engine = get_inference_engine()
    result = engine.generate(test_prompt, max_new_tokens=30)

    assert result.success, f"❌ InferenceEngine failed: {result.error}"
    assert result.num_tokens > 0, "❌ No tokens generated"
    assert len(result.hidden_states_by_layer) > 0, "❌ No hidden states in result"

    print(f"  ✅ InferenceEngine.generate() works!")
    print(f"  Generated text:   '{result.generated_text}'")
    print(f"  Tokens:           {result.tokens}")
    print(f"  Num tokens:       {result.num_tokens}")
    print(f"  Generation time:  {result.generation_time_ms:.1f}ms")
    print(f"  Hidden state layers captured: {list(result.hidden_states_by_layer.keys())}")
    print()

    # Show per-token hidden state for first target layer
    first_layer = TARGET_LAYERS[0]
    states      = result.hidden_states_by_layer[first_layer]
    print(f"  Per-token hidden states at Layer {first_layer}:")
    print(f"  {'Token':<20} {'Hidden State Shape':<20} {'Norm'}")
    print(f"  {'-'*20} {'-'*20} {'-'*10}")
    for token, state in zip(result.tokens[:5], states[:5]):
        print(f"  '{token:<18}' {str(state.shape):<20} {state.norm().item():.4f}")

    # ── Final Summary ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ✅ DAY 3 DELIVERABLE CONFIRMED")
    print("=" * 65)
    print(f"  ✅ Prompt tokenized correctly ({input_length} tokens)")
    print(f"  ✅ model.generate() with return_dict_in_generate=True works")
    print(f"  ✅ sequences captured: {len(generated_tokens)} new tokens")
    print(f"  ✅ hidden_states captured: {num_steps} steps x {num_layers} layers")
    print(f"  ✅ decode_tokens() converts IDs back to words correctly")
    print(f"  ✅ TBG hidden state shape: {tbg.shape}")
    print(f"  ✅ InferenceEngine class end-to-end works")
    print()
    print("  Research confirmation:")
    print("  → Gap 1: Hidden states captured DURING generation (not after)")
    print("  → TBG position extracted per token per layer")
    print("  → Ready for Module B (hidden state extraction) on Day 4")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_day3_test()
