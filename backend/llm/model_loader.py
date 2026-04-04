"""
model_loader.py — LLM Model Loader
====================================
This file loads the Large Language Model (LLM) into memory.

IMPORTANT FOR BEGINNERS:
- A "model" is a huge file of numbers that represents what the AI has learned.
- We must load it with output_hidden_states=True so the model gives us its
  internal "thoughts" (hidden states) at every layer while generating text.
- This is the foundation of Gap 1 — we NEED hidden states to detect hallucinations.

Model choices:
  1. facebook/opt-1.3b  → Use this NOW. Free, no approval needed, 1.3 billion parameters.
  2. meta-llama/Llama-3.2-3B-Instruct → Use this after getting HuggingFace approval.
     Apply at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

Author: Chalani Dinitha (20211032)
"""

import sys
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import our central config (all settings live in config.py)
sys.path.append(".")
from backend.config import MODEL_NAME, HF_TOKEN, DEVICE, TARGET_LAYERS

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and manages the LLM and its tokenizer.

    What is a tokenizer?
    --------------------
    Before the model reads text, it converts words into numbers called "tokens".
    For example: "Hello world" → [15496, 995]
    The tokenizer does this conversion in both directions.

    What is output_hidden_states=True?
    ------------------------------------
    Normally the model only gives you the final generated text.
    With this flag=True, it ALSO gives you the internal numbers from every
    layer inside the model. These are the "hidden states" your research uses.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = MODEL_NAME
        self.device = DEVICE
        self._is_loaded = False

    def load(self) -> bool:
        """
        Load the model and tokenizer into memory.

        Returns:
            True if loading succeeded, False if it failed.

        This is the most important function in Day 2.
        When you call this, Python downloads the model (first time only)
        and loads it into your computer's RAM/VRAM.
        """
        logger.info("=" * 60)
        logger.info(f"  Loading model: {self.model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Target layers: {TARGET_LAYERS}")
        logger.info("=" * 60)

        try:
            # ── Step 1: Load the Tokenizer ────────────────────────────
            # The tokenizer converts text ↔ numbers
            logger.info("Loading tokenizer...")

            tokenizer_kwargs = {
                "trust_remote_code": True,  # needed for some models
            }

            # Only add token if we have one (needed for gated models like LLaMA)
            if HF_TOKEN:
                tokenizer_kwargs["token"] = HF_TOKEN

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tokenizer_kwargs
            )

            # Some models (like OPT) don't have a padding token by default
            # We set it to the end-of-sequence token as a safe fallback
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token = eos_token (model had none by default)")

            logger.info(f"✓ Tokenizer loaded. Vocabulary size: {self.tokenizer.vocab_size:,}")

            # ── Step 2: Load the Model ────────────────────────────────
            # output_hidden_states=True is CRITICAL for our research
            # Without this, we cannot access the hidden states
            logger.info("Loading model weights (this may take 1–5 minutes first time)...")

            model_kwargs = {
                "output_hidden_states": True,   # ← KEY: gives us hidden states
                "trust_remote_code": True,
                "torch_dtype": torch.float32,   # Use float32 for CPU compatibility
            }

            # Add GPU-specific settings if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16  # float16 saves GPU memory
                model_kwargs["device_map"] = "auto"          # auto-assigns GPU layers

            if HF_TOKEN:
                model_kwargs["token"] = HF_TOKEN

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Move model to device (if not already done by device_map)
            if self.device == "cpu" or not torch.cuda.is_available():
                self.model = self.model.to("cpu")

            # Put model in evaluation mode (NOT training mode)
            # This is important: we are only reading/analyzing, not training
            self.model.eval()

            self._is_loaded = True

            # ── Step 3: Print Confirmation ────────────────────────────
            num_layers = self.model.config.num_hidden_layers
            hidden_size = self.model.config.hidden_size
            total_params = sum(p.numel() for p in self.model.parameters())

            logger.info("=" * 60)
            logger.info("  ✅ Model loaded successfully!")
            logger.info(f"  Model name:        {self.model_name}")
            logger.info(f"  Number of layers:  {num_layers}")
            logger.info(f"  Hidden state size: {hidden_size}")
            logger.info(f"  Total parameters:  {total_params:,}")
            logger.info(f"  Running on:        {self.device}")
            logger.info("=" * 60)

            # Validate that our target layers are valid
            self._validate_target_layers(num_layers)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error("Common fixes:")
            logger.error("  1. Check your internet connection")
            logger.error("  2. Try MODEL_NAME=facebook/opt-1.3b in your .env file")
            logger.error("  3. For LLaMA: make sure HF_TOKEN is set and access is approved")
            return False

    def _validate_target_layers(self, total_layers: int):
        """
        Check that the layers we want to extract are valid for this model.
        Prints a warning if any layer index is out of range.
        """
        valid = [l for l in TARGET_LAYERS if 0 <= l < total_layers]
        invalid = [l for l in TARGET_LAYERS if l < 0 or l >= total_layers]

        if invalid:
            logger.warning(f"⚠ Some target layers are out of range for this model!")
            logger.warning(f"  Model has {total_layers} layers (indices 0 to {total_layers-1})")
            logger.warning(f"  Invalid layer indices: {invalid}")
            logger.warning(f"  Valid target layers: {valid}")
            logger.warning(f"  → Update TARGET_LAYERS in your .env file")
        else:
            logger.info(f"✓ All target layers valid: {TARGET_LAYERS} (model has {total_layers} layers)")

    def get_model(self):
        """
        Returns the loaded model.
        Raises an error if you forgot to call .load() first.
        """
        if not self._is_loaded:
            raise RuntimeError(
                "Model not loaded yet! Call model_loader.load() first."
            )
        return self.model

    def get_tokenizer(self):
        """
        Returns the loaded tokenizer.
        Raises an error if you forgot to call .load() first.
        """
        if not self._is_loaded:
            raise RuntimeError(
                "Tokenizer not loaded yet! Call model_loader.load() first."
            )
        return self.tokenizer

    def get_model_info(self) -> dict:
        """
        Returns a dictionary of model information.
        Used by the API's GET /api/config endpoint.
        """
        if not self._is_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_name": self.model_name,
            "num_hidden_layers": self.model.config.num_hidden_layers,
            "hidden_size": self.model.config.hidden_size,
            "vocab_size": self.model.config.vocab_size,
            "device": self.device,
            "target_layers": TARGET_LAYERS,
        }

    def is_loaded(self) -> bool:
        """Returns True if the model has been loaded successfully."""
        return self._is_loaded


# ── Singleton Pattern ─────────────────────────────────────────────────────────
# We create ONE global instance of ModelLoader.
# This means the model is only loaded into memory once,
# even if many parts of the code call get_model_loader().

_model_loader_instance = None


def get_model_loader() -> ModelLoader:
    """
    Returns the global ModelLoader instance.
    Creates it if it doesn't exist yet.

    This is called a "singleton" pattern — we only ever have one model in memory.
    Loading a model twice would waste gigabytes of RAM.
    """
    global _model_loader_instance
    if _model_loader_instance is None:
        _model_loader_instance = ModelLoader()
    return _model_loader_instance


# ── Quick Test ────────────────────────────────────────────────────────────────
# Run this file directly to test model loading:
# Command: python backend/llm/model_loader.py

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TEST: Model Loader")
    print("=" * 60 + "\n")

    loader = ModelLoader()
    success = loader.load()

    if success:
        info = loader.get_model_info()
        print("\n✅ TEST PASSED — Model loaded successfully!")
        print(f"\nModel Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Test tokenizer
        tokenizer = loader.get_tokenizer()
        test_text = "The capital of France is"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"\nTokenizer test:")
        print(f"  Input text:  '{test_text}'")
        print(f"  Token IDs:   {tokens['input_ids'].tolist()}")
        print(f"  Num tokens:  {tokens['input_ids'].shape[1]}")

        print("\n✅ Day 2 deliverable confirmed: model loads, hidden states enabled!")
    else:
        print("\n❌ TEST FAILED — Check error messages above")
        print("\nTroubleshooting:")
        print("  1. Make sure you ran: pip install -r requirements.txt")
        print("  2. Check your .env file has the correct MODEL_NAME")
        print("  3. For OPT-1.3B: no token needed")
        print("  4. For LLaMA: set HF_TOKEN in your .env file")
        sys.exit(1)
