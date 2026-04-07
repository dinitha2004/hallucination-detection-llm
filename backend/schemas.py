"""
schemas.py — FastAPI Request & Response Models
===============================================
Pydantic models define the exact structure of data
that goes IN and OUT of the API.

Why Pydantic?
- Automatic validation: wrong data types are rejected
- Auto-generates API docs at /docs
- Type safety between frontend and backend

Author: Chalani Dinitha (20211032)
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ── Request Models (what the frontend SENDS) ──────────────────────────────────

class PromptRequest(BaseModel):
    """
    Request body for POST /api/detect
    Frontend sends: { "prompt": "What year was Einstein born?" }
    """
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question or prompt to analyze",
        example="What year was Einstein born?"
    )
    max_new_tokens: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Maximum number of tokens to generate"
    )


class ConfigUpdateRequest(BaseModel):
    """
    Request body for POST /api/config
    Allows researcher to update thresholds at runtime (FR14).
    """
    hallucination_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Score above which token is flagged as hallucinated"
    )
    suspicious_threshold_low: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Score above which token is shown as suspicious"
    )


# ── Response Models (what the API RETURNS) ────────────────────────────────────

class AnnotatedTokenResponse(BaseModel):
    """One token with its hallucination annotation."""
    token: str
    position: int
    hallucination_score: float
    risk_level: str           # "safe", "suspicious", "hallucinated"
    is_eat: bool              # Exact Answer Token
    is_flagged: bool          # Above hallucination threshold
    entity_type: Optional[str] = None  # e.g. "DATE", "PERSON", "GPE"


class DetectionResponse(BaseModel):
    """
    Response body for POST /api/detect
    Contains the full annotated result sent to the React frontend.
    """
    generated_text: str
    annotated_tokens: List[AnnotatedTokenResponse]
    overall_risk: float
    num_flagged: int
    num_eat_tokens: int
    hallucination_detected: bool
    processing_time_ms: float
    model_name: str


class HealthResponse(BaseModel):
    """Response for GET /api/health"""
    status: str
    model_loaded: bool
    pipeline_ready: bool
    model_name: str


class ConfigResponse(BaseModel):
    """Response for GET /api/config"""
    model_name: str
    target_layers: List[int]
    hallucination_threshold: float
    suspicious_threshold_low: float
    weight_entropy: float
    weight_wasserstein: float
    weight_tsv: float
    tsv_trained: bool


class ConfigUpdateResponse(BaseModel):
    """Response for POST /api/config"""
    success: bool
    message: str
    updated_config: ConfigResponse
