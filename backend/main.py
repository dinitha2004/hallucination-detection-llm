import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import (
    PromptRequest, DetectionResponse, AnnotatedTokenResponse,
    HealthResponse, ConfigResponse, ConfigUpdateRequest, ConfigUpdateResponse
)
from backend.config import (
    TARGET_LAYERS, HALLUCINATION_THRESHOLD, SUSPICIOUS_THRESHOLD_LOW,
    WEIGHT_ENTROPY, WEIGHT_WASSERSTEIN, WEIGHT_TSV, MODEL_NAME
)
from backend.pipeline.detection_pipeline import get_detection_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
pipeline = get_detection_pipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting — initializing detection pipeline...")
    success = pipeline.initialize()
    logger.info("Pipeline ready!" if success else "Pipeline FAILED")
    yield
    logger.info("Server shutting down")

app = FastAPI(title="Hallucination Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost","http://127.0.0.1:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"name":"Hallucination Detection API","version":"1.0.0","docs":"/docs","health":"/api/health"}

@app.post("/api/detect", response_model=DetectionResponse)
async def detect_hallucination(request: PromptRequest):
    if not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    output = pipeline.run(prompt=request.prompt, max_new_tokens=request.max_new_tokens)
    model_info = pipeline._loader.get_model_info() if pipeline._loader else {}
    return DetectionResponse(
        generated_text=output.generated_text,
        annotated_tokens=[AnnotatedTokenResponse(
            token=t.token, position=t.position,
            hallucination_score=t.hallucination_score,
            risk_level=t.risk_level, is_eat=t.is_eat,
            is_flagged=t.is_flagged, entity_type=t.entity_type
        ) for t in output.annotated_tokens],
        overall_risk=output.overall_risk,
        num_flagged=output.num_flagged,
        num_eat_tokens=output.num_eat_tokens,
        hallucination_detected=output.hallucination_detected,
        processing_time_ms=output.processing_time_ms,
        model_name=model_info.get("model_name", MODEL_NAME),
    )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    model_info = pipeline._loader.get_model_info() if pipeline._loader else {}
    return HealthResponse(
        status="ready" if pipeline.is_initialized else "initializing",
        model_loaded=model_info.get("loaded", False),
        pipeline_ready=pipeline.is_initialized,
        model_name=model_info.get("model_name", MODEL_NAME),
    )

@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    s = pipeline._scoring_engine
    t = pipeline._tsv
    return ConfigResponse(
        model_name=MODEL_NAME, target_layers=TARGET_LAYERS,
        hallucination_threshold=s.threshold if s else HALLUCINATION_THRESHOLD,
        suspicious_threshold_low=s.suspicious_low if s else SUSPICIOUS_THRESHOLD_LOW,
        weight_entropy=s.w_entropy if s else WEIGHT_ENTROPY,
        weight_wasserstein=s.w_wasserstein if s else WEIGHT_WASSERSTEIN,
        weight_tsv=s.w_tsv if s else WEIGHT_TSV,
        tsv_trained=t.is_trained if t else False,
    )

@app.post("/api/config", response_model=ConfigUpdateResponse)
async def update_config(request: ConfigUpdateRequest):
    if not pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    messages = []
    if request.hallucination_threshold is not None:
        pipeline.update_threshold(request.hallucination_threshold)
        messages.append(f"Threshold updated to {request.hallucination_threshold}")
    if request.suspicious_threshold_low is not None and pipeline._scoring_engine:
        pipeline._scoring_engine.suspicious_low = request.suspicious_threshold_low
        messages.append(f"Suspicious threshold updated to {request.suspicious_threshold_low}")
    updated = await get_config()
    return ConfigUpdateResponse(success=True, message="; ".join(messages) or "No changes", updated_config=updated)
