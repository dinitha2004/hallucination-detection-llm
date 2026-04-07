# API Reference — HalluScan Backend

**Base URL:** `http://localhost:8000`  
**Interactive Docs:** `http://localhost:8000/docs`  
**Author:** Chalani Dinitha (20211032)

---

## Endpoints

### GET /
API root — returns version info.

**Response:**
```json
{
  "name": "Hallucination Detection API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/api/health"
}
```

---

### GET /api/health
Check if server and pipeline are ready.

**Response:**
```json
{
  "status": "ready",
  "model_loaded": true,
  "pipeline_ready": true,
  "model_name": "facebook/opt-1.3b"
}
```

---

### POST /api/detect
Run hallucination detection on a prompt.

**Request:**
```json
{
  "prompt": "Einstein was born in 1879 in Germany",
  "max_new_tokens": 20
}
```

**Response:**
```json
{
  "generated_text": " He was a brilliant physicist...",
  "annotated_tokens": [
    {
      "token": " He",
      "position": 0,
      "hallucination_score": 0.2197,
      "risk_level": "safe",
      "is_eat": false,
      "is_flagged": false,
      "entity_type": null
    },
    {
      "token": " Einstein",
      "position": 5,
      "hallucination_score": 0.4555,
      "risk_level": "suspicious",
      "is_eat": true,
      "is_flagged": false,
      "entity_type": "PERSON"
    }
  ],
  "overall_risk": 0.4693,
  "num_flagged": 0,
  "num_eat_tokens": 3,
  "hallucination_detected": false,
  "processing_time_ms": 942.4,
  "model_name": "facebook/opt-1.3b"
}
```

**Risk levels:** `safe` | `suspicious` | `hallucinated`

---

### GET /api/config
Get current pipeline configuration.

**Response:**
```json
{
  "model_name": "facebook/opt-1.3b",
  "target_layers": [18, 20, 22],
  "hallucination_threshold": 0.65,
  "suspicious_threshold_low": 0.45,
  "weight_entropy": 0.4,
  "weight_wasserstein": 0.4,
  "weight_tsv": 0.2,
  "tsv_trained": true
}
```

---

### POST /api/config
Update thresholds at runtime (FR14).

**Request:**
```json
{
  "hallucination_threshold": 0.5,
  "suspicious_threshold_low": 0.35
}
```

**Response:**
```json
{
  "success": true,
  "message": "Threshold updated to 0.5",
  "updated_config": { ... }
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (empty prompt) |
| 422 | Validation error (invalid field) |
| 503 | Pipeline not ready yet |
