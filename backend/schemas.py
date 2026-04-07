from pydantic import BaseModel, Field
from typing import List, Optional

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_new_tokens: int = Field(default=50, ge=10, le=500)

class ConfigUpdateRequest(BaseModel):
    hallucination_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    suspicious_threshold_low: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class AnnotatedTokenResponse(BaseModel):
    token: str
    position: int
    hallucination_score: float
    risk_level: str
    is_eat: bool
    is_flagged: bool
    entity_type: Optional[str] = None

class DetectionResponse(BaseModel):
    generated_text: str
    annotated_tokens: List[AnnotatedTokenResponse]
    overall_risk: float
    num_flagged: int
    num_eat_tokens: int
    hallucination_detected: bool
    processing_time_ms: float
    model_name: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    pipeline_ready: bool
    model_name: str

class ConfigResponse(BaseModel):
    model_name: str
    target_layers: List[int]
    hallucination_threshold: float
    suspicious_threshold_low: float
    weight_entropy: float
    weight_wasserstein: float
    weight_tsv: float
    tsv_trained: bool

class ConfigUpdateResponse(BaseModel):
    success: bool
    message: str
    updated_config: ConfigResponse
