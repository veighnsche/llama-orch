# 📋 OpenAPI Spec Plan for llm-worker-rbee Worker

**Date**: 2025-10-09  
**Target**: llm-worker-rbee worker HTTP API  
**Spec Version**: OpenAPI 3.1.0

---

## 🎯 Goals

1. **Document all HTTP endpoints** exposed by the worker
2. **Define request/response schemas** with validation rules
3. **Document SSE event types** for streaming responses
4. **Enable API client generation** (TypeScript, Python, Rust)
5. **Provide interactive API docs** (Swagger UI)

---

## 📊 Worker HTTP API Endpoints

### Summary

| Method | Path | Purpose | Response Type |
|--------|------|---------|---------------|
| GET | `/health` | Health check | JSON |
| POST | `/execute` | Execute inference | SSE stream |
| POST | `/cancel` | Cancel inference | JSON |
| POST | `/shutdown` | Graceful shutdown | JSON (202) |
| GET | `/metrics` | Prometheus metrics | text/plain |

---

## 📝 Endpoint Details

### 1. GET /health

**Purpose**: Check worker health and capabilities

**Request**: None (GET)

**Response**: 200 OK
```json
{
  "status": "healthy",
  "model": "llama-7b",
  "resident": true,
  "quant_kind": "Q4_K_M",
  "vram_bytes_used": 7000000000,
  "tokenizer_kind": "gguf-bpe",
  "vocab_size": 32000,
  "context_length": 2048,
  "uptime_seconds": 3600,
  "sm": 86,
  "cuda_runtime_version": "12.1"
}
```

**Status Codes**:
- 200 OK - Worker is healthy
- 503 Service Unavailable - Worker is unhealthy

---

### 2. POST /execute

**Purpose**: Execute inference request

**Request Headers**:
- `Content-Type: application/json`
- `X-Correlation-Id: <uuid>` (optional, generated if missing)

**Request Body**:
```json
{
  "job_id": "job-123",
  "prompt": "Write a haiku about GPUs",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "min_p": 0.05,
  "stop": ["\n\n", "END"],
  "seed": 42
}
```

**Validation Rules**:
- `job_id`: Required, non-empty string, max 256 chars
- `prompt`: Required, non-empty string, max 32768 chars
- `max_tokens`: Required, integer, range 1-4096
- `temperature`: Optional, float, range 0.0-2.0, default 1.0
- `top_p`: Optional, float, range 0.0-1.0, default 1.0
- `top_k`: Optional, integer, range 0-vocab_size, default 0 (disabled)
- `repetition_penalty`: Optional, float, range 0.0-2.0, default 1.0
- `min_p`: Optional, float, range 0.0-1.0, default 0.0
- `stop`: Optional, array of strings, max 4 sequences, each max 32 chars
- `seed`: Optional, unsigned 64-bit integer

**Response**: 200 OK (SSE stream)
- `Content-Type: text/event-stream`
- `X-Correlation-Id: <uuid>`

**SSE Event Types**:

1. **started**
```json
{
  "type": "started",
  "job_id": "job-123",
  "model": "llama-7b",
  "started_at": "2025-10-09T13:27:00Z"
}
```

2. **narration** (NEW)
```json
{
  "type": "narration",
  "actor": "candle-backend",
  "action": "inference_start",
  "target": "job-123",
  "human": "Starting inference (prompt: 28 chars, max_tokens: 100)",
  "cute": "Time to generate 100 tokens! Let's go! 🚀",
  "correlation_id": "req-abc123",
  "job_id": "job-123"
}
```

3. **token**
```json
{
  "type": "token",
  "t": "Hello",
  "i": 0
}
```

4. **metrics** (optional)
```json
{
  "type": "metrics",
  "tokens_per_sec": 42.5,
  "vram_bytes": 7000000000
}
```

5. **end**
```json
{
  "type": "end",
  "tokens_out": 42,
  "decode_time_ms": 1234,
  "stop_reason": "MAX_TOKENS",
  "stop_sequence_matched": null
}
```

6. **error**
```json
{
  "type": "error",
  "code": "VRAM_OOM",
  "message": "Out of VRAM during inference"
}
```

**Event Ordering**:
```
started → narration* → token* → narration* → (end | error)
```

**Error Codes**:
- `VRAM_OOM` - Out of VRAM
- `CANCELLED` - Job cancelled
- `TIMEOUT` - Job timed out
- `INVALID_REQUEST` - Invalid parameters
- `INFERENCE_FAILED` - Inference error

**Status Codes**:
- 200 OK - SSE stream started
- 400 Bad Request - Validation failed
- 503 Service Unavailable - Worker unhealthy

---

### 3. POST /cancel

**Purpose**: Cancel running inference

**Request**:
```json
{
  "job_id": "job-123"
}
```

**Response**: 202 Accepted
```json
{
  "job_id": "job-123",
  "status": "cancelling"
}
```

**Behavior**:
- Idempotent (repeated cancels are safe)
- Stops decoding within 100ms
- Frees VRAM buffers
- Emits SSE `error` event with code `CANCELLED`

**Status Codes**:
- 202 Accepted - Cancellation initiated
- 404 Not Found - Job not found
- 400 Bad Request - Invalid job_id

---

### 4. POST /shutdown

**Purpose**: Graceful shutdown

**Request**: Empty body or `{}`

**Response**: 202 Accepted
```json
{
  "status": "shutting_down",
  "timeout_seconds": 30
}
```

**Behavior**:
- Stops accepting new requests (returns 503)
- Finishes active inference job
- Frees VRAM
- Exits with code 0
- Timeout: 30 seconds

**Status Codes**:
- 202 Accepted - Shutdown initiated
- 503 Service Unavailable - Already shutting down

---

### 5. GET /metrics

**Purpose**: Prometheus metrics

**Response**: 200 OK (text/plain)
```
# HELP worker_vram_bytes Current VRAM usage
# TYPE worker_vram_bytes gauge
worker_vram_bytes 7000000000

# HELP worker_requests_total Request count by outcome
# TYPE worker_requests_total counter
worker_requests_total{outcome="success"} 42
worker_requests_total{outcome="error"} 3

# HELP worker_tokens_generated_total Total output tokens
# TYPE worker_tokens_generated_total counter
worker_tokens_generated_total 12345

# HELP worker_inference_duration_ms Inference latency
# TYPE worker_inference_duration_ms histogram
worker_inference_duration_ms_bucket{le="100"} 10
worker_inference_duration_ms_bucket{le="500"} 35
worker_inference_duration_ms_bucket{le="+Inf"} 42
worker_inference_duration_ms_sum 8500
worker_inference_duration_ms_count 42

# HELP worker_uptime_seconds Worker uptime
# TYPE worker_uptime_seconds counter
worker_uptime_seconds 3600
```

**Status Codes**:
- 200 OK - Metrics returned

---

## 📋 OpenAPI Spec Structure

```yaml
openapi: 3.1.0
info:
  title: llm-worker-rbee Worker API
  version: 0.1.0
  description: |
    Candle-based Llama inference worker for llama-orch.
    
    This worker provides:
    - Text generation via SSE streaming
    - Real-time narration events (what's happening behind the scenes)
    - Health monitoring
    - Graceful cancellation and shutdown
    
  contact:
    name: llama-orch Team
  license:
    name: GPL-3.0-or-later

servers:
  - url: http://localhost:{port}
    description: Worker HTTP server
    variables:
      port:
        default: '8080'
        description: Port assigned by pool-manager

paths:
  /health:
    get:
      summary: Health check
      operationId: getHealth
      tags: [Health]
      responses:
        '200':
          description: Worker is healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'
        '503':
          description: Worker is unhealthy
          
  /execute:
    post:
      summary: Execute inference
      operationId: executeInference
      tags: [Inference]
      parameters:
        - name: X-Correlation-Id
          in: header
          schema:
            type: string
            format: uuid
          description: Correlation ID for request tracing
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ExecuteRequest'
      responses:
        '200':
          description: SSE stream of inference results
          headers:
            X-Correlation-Id:
              schema:
                type: string
                format: uuid
          content:
            text/event-stream:
              schema:
                $ref: '#/components/schemas/InferenceEventStream'
        '400':
          description: Validation failed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ValidationError'
                
  /cancel:
    post:
      summary: Cancel inference
      operationId: cancelInference
      tags: [Inference]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CancelRequest'
      responses:
        '202':
          description: Cancellation initiated
          
  /shutdown:
    post:
      summary: Graceful shutdown
      operationId: shutdown
      tags: [Lifecycle]
      responses:
        '202':
          description: Shutdown initiated
          
  /metrics:
    get:
      summary: Prometheus metrics
      operationId: getMetrics
      tags: [Monitoring]
      responses:
        '200':
          description: Prometheus metrics
          content:
            text/plain:
              schema:
                type: string

components:
  schemas:
    HealthResponse:
      type: object
      required: [status, model, resident, vram_bytes_used]
      properties:
        status:
          type: string
          enum: [healthy, unhealthy]
        model:
          type: string
        resident:
          type: boolean
        quant_kind:
          type: string
          enum: [Q4_K_M, MXFP4, Q4_0]
        vram_bytes_used:
          type: integer
          format: int64
        tokenizer_kind:
          type: string
          enum: [gguf-bpe, hf-json]
        vocab_size:
          type: integer
        context_length:
          type: integer
          nullable: true
        uptime_seconds:
          type: integer
        sm:
          type: integer
        cuda_runtime_version:
          type: string
          
    ExecuteRequest:
      type: object
      required: [job_id, prompt, max_tokens]
      properties:
        job_id:
          type: string
          maxLength: 256
        prompt:
          type: string
          minLength: 1
          maxLength: 32768
        max_tokens:
          type: integer
          minimum: 1
          maximum: 4096
        temperature:
          type: number
          format: float
          minimum: 0.0
          maximum: 2.0
          default: 1.0
        top_p:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          default: 1.0
        top_k:
          type: integer
          minimum: 0
          default: 0
        repetition_penalty:
          type: number
          format: float
          minimum: 0.0
          maximum: 2.0
          default: 1.0
        min_p:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          default: 0.0
        stop:
          type: array
          items:
            type: string
            maxLength: 32
          maxItems: 4
        seed:
          type: integer
          format: int64
          minimum: 0
          
    InferenceEventStream:
      oneOf:
        - $ref: '#/components/schemas/StartedEvent'
        - $ref: '#/components/schemas/NarrationEvent'
        - $ref: '#/components/schemas/TokenEvent'
        - $ref: '#/components/schemas/MetricsEvent'
        - $ref: '#/components/schemas/EndEvent'
        - $ref: '#/components/schemas/ErrorEvent'
        
    StartedEvent:
      type: object
      required: [type, job_id, model, started_at]
      properties:
        type:
          type: string
          enum: [started]
        job_id:
          type: string
        model:
          type: string
        started_at:
          type: string
          format: date-time
          
    NarrationEvent:
      type: object
      required: [type, actor, action, target, human]
      properties:
        type:
          type: string
          enum: [narration]
        actor:
          type: string
        action:
          type: string
        target:
          type: string
        human:
          type: string
          maxLength: 100
        cute:
          type: string
          maxLength: 150
        story:
          type: string
          maxLength: 200
        correlation_id:
          type: string
        job_id:
          type: string
          
    TokenEvent:
      type: object
      required: [type, t, i]
      properties:
        type:
          type: string
          enum: [token]
        t:
          type: string
        i:
          type: integer
          format: uint32
          
    MetricsEvent:
      type: object
      required: [type, tokens_per_sec, vram_bytes]
      properties:
        type:
          type: string
          enum: [metrics]
        tokens_per_sec:
          type: number
          format: float
        vram_bytes:
          type: integer
          format: int64
          
    EndEvent:
      type: object
      required: [type, tokens_out, decode_time_ms, stop_reason]
      properties:
        type:
          type: string
          enum: [end]
        tokens_out:
          type: integer
          format: uint32
        decode_time_ms:
          type: integer
          format: uint64
        stop_reason:
          type: string
          enum: [MAX_TOKENS, STOP_SEQUENCE, EOS, CANCELLED, ERROR]
        stop_sequence_matched:
          type: string
          nullable: true
          
    ErrorEvent:
      type: object
      required: [type, code, message]
      properties:
        type:
          type: string
          enum: [error]
        code:
          type: string
          enum: [VRAM_OOM, CANCELLED, TIMEOUT, INVALID_REQUEST, INFERENCE_FAILED]
        message:
          type: string
          
    CancelRequest:
      type: object
      required: [job_id]
      properties:
        job_id:
          type: string
          
    ValidationError:
      type: object
      required: [errors]
      properties:
        errors:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
              message:
                type: string
```

---

## 🔧 Implementation Steps

### Step 1: Create OpenAPI Spec File

**Location**: `bin/llm-worker-rbee/openapi.yaml`

**Tools**:
- Use [Swagger Editor](https://editor.swagger.io/) for validation
- Use `openapi-generator` for client generation

### Step 2: Add Swagger UI

**Option A: Embed in worker (optional)**
```rust
// Add to Cargo.toml
utoipa = "4.0"
utoipa-swagger-ui = { version = "4.0", features = ["axum"] }

// Add to routes.rs
use utoipa_swagger_ui::SwaggerUi;

Router::new()
    .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
```

**Option B: Serve statically**
- Host `openapi.yaml` at `/openapi.yaml`
- Use external Swagger UI pointing to worker URL

### Step 3: Generate Clients

**TypeScript**:
```bash
openapi-generator generate \
  -i openapi.yaml \
  -g typescript-fetch \
  -o clients/typescript
```

**Python**:
```bash
openapi-generator generate \
  -i openapi.yaml \
  -g python \
  -o clients/python
```

**Rust**:
```bash
openapi-generator generate \
  -i openapi.yaml \
  -g rust \
  -o clients/rust
```

### Step 4: Validate Against Implementation

**Use `utoipa` macros** to generate spec from code:

```rust
use utoipa::{OpenApi, ToSchema};

#[derive(ToSchema)]
struct ExecuteRequest {
    job_id: String,
    prompt: String,
    max_tokens: u32,
    // ...
}

#[utoipa::path(
    post,
    path = "/execute",
    request_body = ExecuteRequest,
    responses(
        (status = 200, description = "SSE stream"),
        (status = 400, description = "Validation failed")
    )
)]
async fn handle_execute() { /* ... */ }
```

---

## 📊 Benefits

1. **Auto-generated clients** - TypeScript, Python, Rust
2. **Interactive docs** - Swagger UI for testing
3. **Type safety** - Schema validation
4. **Contract testing** - Verify implementation matches spec
5. **Documentation** - Single source of truth

---

## ✅ Deliverables

- [ ] `openapi.yaml` - Complete OpenAPI 3.1 spec
- [ ] Swagger UI integration (optional)
- [ ] Generated TypeScript client
- [ ] Generated Python client
- [ ] Validation tests (spec vs implementation)
- [ ] Documentation with examples

---

*Planned by the Narration Core Team 🎀*  
*May your APIs be well-documented and your schemas be valid! 💝*
