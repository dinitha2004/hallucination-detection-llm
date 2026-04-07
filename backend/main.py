"""
main.py — FastAPI Application Entry Point
==========================================
This is the web server that connects the React frontend
to your hallucination detection pipeline.

Endpoints:
    POST /api/detect   → run detection pipeline (main endpoint)
    GET  /api/health   → server health check
    GET  /api/config   → get current configuration
    POST /api/config   → update thresholds (FR14)

How to run:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Then visit:
    http://localhost:8000/docs  → interactive API documentation
    http://localhost:8000       → API root

Author: Chalani Dinitha (20211032)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    PromptRequest, DetectionResponse,
    AnnotatedTokenResponse, HealthResponse,
    ConfigResponse, ConfigUpdateRequest, ConfigUpdateResponse
)
from backend.config import (
    TARGET_LAYERS, HALLUCINATION_THRESHOLD,
    SUSPICIOUS_THRESHOLD_LOW, WEIGHT_ENTROPY,
    WEIGHT_WASSERSTEIN, WEIGHT_TSV, MODEL_NAME
)
from backend.pipeline.detection_pipeline import get_detection_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Pipeline singleton ────────────────────────────────────────────────────────
pipeline = get_detection_pipeline()


# ── Lifespan: runs on startup and shutdown ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the pipeline when the server starts."""
    logger.info("Server starting — initializing detection pipeline...")
    success = pipeline.initialize()
    if success:
        logger.info("Pipeline ready!")
    else:
        logger.error("Pipeline initialization FAILED — check logs")
    yield
    logger.info("Server shutting down")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hallucination Detection API",
    description=(
        "Real-time hallucination detection using LLM hidden states. "
        "Detects hallucinations during token generation and highlights "
        "only the exact incorrect tokens (EAT-level localization)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# Allows the React frontend (localhost:3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # React dev server
        "http://localhost",         # Docker frontend
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name": "Hallucination Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


# ── POST /api/detect ─────────────────────────────────────────────────────────
@app.post("/api/detect", response_model=DetectionResponse)
async def detect_hallucination(request: PromptRequest):
    """
    Run hallucination detection on a user prompt.

    This is the main endpoint called by the React frontend.
    It runs the complete A→B→C→D pipeline and returns
    annotated tokens with hallucination scores.

    - **prompt**: The question or text to analyze
    - **max_new_tokens**: How many tokens to generate (10-500)
    """
    if not pipeline.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not ready. Server is still initializing."
        )

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty"
        )

    logger.info(f"Detection request: '{request.prompt[:60]}...'")

    # Run the detection pipeline
    output = pipeline.run(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens
    )

    # Convert annotated tokens to response format
    annotated = [
        AnnotatedTokenResponse(
            token=t.token,
            position=t.position,
            hallucination_score=t.hallucination_score,
            risk_level=t.risk_level,
            is_eat=t.is_eat,
            is_flagged=t.is_flagged,
            entity_type=t.entity_type,
        )
        for t in output.annotated_tokens
    ]

    model_info = pipeline._loader.get_model_info() if pipeline._loader else {}

    return DetectionResponse(
        generated_text=output.generated_text,
        annotated_tokens=annotated,
        overall_risk=output.overall_risk,
        num_flagged=output.num_flagged,
        num_eat_tokens=output.num_eat_tokens,
        hallucination_detected=output.hallucination_detected,
        processing_time_ms=output.processing_time_ms,
        model_name=model_info.get("model_name", MODEL_NAME),
    )


# ── GET /api/health ───────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the server and pipeline are ready.
    The React frontend calls this on startup to show loading state.
    """
    model_info = {}
    if pipeline._loader:
        model_info = pipeline._loader.get_model_info()

    return HealthResponse(
        status="ready" if pipeline.is_initialized else "initializing",
        model_loaded=model_info.get("loaded", False),
        pipeline_ready=pipeline.is_initialized,
        model_name=model_info.get("model_name", MODEL_NAME),
    )


# ── GET /api/config ───────────────────────────────────────────────────────────
@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """
    Get the current pipeline configuration.
    Used by the React ConfigPanel to show current settings.
    """
    scoring = pipeline._scoring_engine
    tsv = pipeline._tsv

    threshold = scoring.threshold if scoring else HALLUCINATION_THRESHOLD
    susp_low = scoring.suspicious_low if scoring else SUSPICIOUS_THRESHOLD_LOW
    w_e = scoring.w_entropy if scoring else WEIGHT_ENTROPY
    w_w = scoring.w_wasserstein if scoring else WEIGHT_WASSERSTEIN
    w_t = scoring.w_tsv if scoring else WEIGHT_TSV

    return ConfigResponse(
        model_name=MODEL_NAME,
        target_layers=TARGET_LAYERS,
        hallucination_threshold=threshold,
        suspicious_threshold_low=susp_low,
        weight_entropy=w_e,
        weight_wasserstein=w_w,
        weight_tsv=w_t,
        tsv_trained=tsv.is_trained if tsv else False,
    )


# ── POST /api/config ──────────────────────────────────────────────────────────
@app.post("/api/config", response_model=ConfigUpdateResponse)
async def update_config(request: ConfigUpdateRequest):
    """
    Update pipeline configuration at runtime.
    Implements FR14: researcher can configure thresholds.
    Used by the React ConfigPanel sliders.
    """
    if not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    messages = []

    if request.hallucination_threshold is not None:
        pipeline.update_threshold(request.hallucination_threshold)
        messages.append(
            f"Threshold updated to {request.hallucination_threshold}"
        )

    if request.suspicious_threshold_low is not None:
        if pipeline._scoring_engine:
            pipeline._scoring_engine.suspicious_low = \
                request.suspicious_threshold_low
            pipeline._span_mapper.suspicious_low = \
                request.suspicious_threshold_low
            messages.append(
                f"Suspicious threshold updated to "
                f"{request.suspicious_threshold_low}"
            )

    # Return updated config
    updated = await get_config()

    return ConfigUpdateResponse(
        success=True,
        message="; ".join(messages) if messages else "No changes made",
        updated_config=updated,
    )
