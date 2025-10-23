# LLM-WORKER-RBEE BEHAVIOR INVENTORY

**Team:** TEAM-219  
**Component:** `bin/30_llm_worker_rbee` - Worker daemon (runs models)  
**Date:** Oct 22, 2025  
**Lines of Code:** ~6,500 LOC (51 source files)

// TEAM-219: Investigated - Oct 22, 2025

---

## 1. Public API Surface

### HTTP Endpoints

**Public (No Auth):**
- `GET /health` → `{"status": "healthy"}` (routes.rs:76-78)

**Protected (Auth Required in Network Mode):**
- `POST /v1/inference` → Create job, returns `{job_id, sse_url}` (execute.rs:40-139)
- `GET /v1/inference/{job_id}/stream` → SSE stream of tokens (stream.rs:30-156)

### Request Schema (validation.rs:20-54)
```rust
ExecuteRequest {
    prompt: String,           // 1-32768 chars
    max_tokens: u32,          // 1-2048
    temperature: f32,         // 0.0-2.0 (default: 1.0)
    seed: Option<u64>,        // Optional
    top_p: f32,               // 0.0-1.0 (default: 1.0)
    top_k: u32,               // 0+ (default: 0 = disabled)
    repetition_penalty: f32,  // 0.0-2.0 (default: 1.0)
    stop: Vec<String>,        // Max 4 sequences, each ≤100 chars
    min_p: f32,               // 0.0-1.0 (default: 0.0)
}
```

### Response Schema (sse.rs:23-98)
```rust
InferenceEvent::Started { job_id, model, started_at }
InferenceEvent::Token { t: String, i: u32 }
InferenceEvent::Metrics { tokens_per_sec, vram_bytes }
InferenceEvent::Narration { actor, action, target, human, cute, ... }
InferenceEvent::End { tokens_out, decode_time_ms, stop_reason, stop_sequence_matched }
InferenceEvent::Error { code, message }
```

### CLI Arguments (main.rs:38-70)
```bash
llm-worker-rbee \
  --worker-id <UUID> \
  --model <path.gguf> \
  --model-ref <hf:model-name> \
  --backend <cpu|cuda|metal> \
  --device <gpu_id> \
  --port <port> \
  --hive-url <http://...> \
  --local-mode <true|false>
```

### Authentication (middleware/auth.rs:40-87)
- **Network Mode:** Requires `LLORCH_API_TOKEN` env var, Bearer token validation
- **Local Mode:** No auth, binds to 127.0.0.1 only (main.rs:236-254)
- **Timing-safe comparison:** Uses `auth_min::timing_safe_eq()` (CWE-208 protection)
- **Token fingerprinting:** Logs `token_fp6()` (never raw tokens)

---

## 2. State Machine Behaviors

### Daemon Lifecycle (main.rs:87-277)

**States:**
1. **Startup** → Parse args → Load model → Create queue → Start engine → Start heartbeat → Start HTTP server
2. **Running** → Process requests via queue, stream tokens via SSE
3. **Shutdown** → Graceful (SIGTERM) or force kill (SIGKILL) by hive

**Model Loading (inference.rs:47-85):**
- Detects architecture from config.json or GGUF metadata (models/mod.rs:232-301)
- Supports: Llama, Mistral, Phi, Qwen (SafeTensors + GGUF quantized)
- Loads tokenizer (HuggingFace format or GGUF-embedded)
- Calculates model size for VRAM tracking
- **Failure:** Narrates error, exits with error code

**Device Initialization (device.rs:16-71):**
- **CPU:** `Device::Cpu` (feature: cpu)
- **CUDA:** `Device::new_cuda(gpu_id)` (feature: cuda)
- **Metal:** `Device::new_metal(gpu_id)` (feature: metal)
- Verifies device with smoke test (create tensor, sum operation)

### Request Processing Flow (TEAM-154 Dual-Call Pattern)

**POST /v1/inference (execute.rs:40-139):**
1. Validate request parameters (validation.rs:148-247)
2. Generate `job_id` (server-side, not client-provided)
3. Create `tokio::mpsc::unbounded_channel()` for token streaming
4. Store receiver in `JobRegistry`
5. Add `GenerationRequest` to `RequestQueue`
6. Return JSON: `{job_id, sse_url: "/v1/inference/{job_id}/stream"}`

**GET /v1/inference/{job_id}/stream (stream.rs:30-156):**
1. Check job exists in registry
2. Check job state (not failed)
3. Take token receiver from registry (consumes it - one stream per job)
4. Build SSE stream: Started → Token* → [DONE]
5. Stream closes when `TokenResponse::Done` received

**Generation Engine (generation_engine.rs:51-214):**
- Runs in `tokio::spawn_blocking` (CPU-intensive work off async runtime)
- Pulls requests from queue sequentially
- Locks backend only during generation (not while waiting)
- **Token-by-token streaming:** Sends `TokenResponse::Token(String)` immediately after decode
- **Client disconnect:** Detects via channel close, stops generation gracefully

### Inference Pipeline (inference.rs:165-398)

**Steps:**
1. **Tokenize** prompt (tokenizers crate)
2. **Reset KV cache** (clears warmup pollution - TEAM-021 fix)
3. **Generate tokens** (loop 0..max_tokens):
   - First iteration: Forward pass with full prompt
   - Subsequent: Forward pass with last token only
   - Sample next token (LogitsProcessor with temperature/top_p/top_k/min_p/repetition_penalty)
   - Check EOS (tokenizer or model eos_token_id)
   - Decode token (TokenOutputStream for proper UTF-8 + space handling)
   - Send token immediately via channel
4. **Decode rest** (flush TokenOutputStream buffer)
5. **Send Done** signal

**Stopping Conditions:**
- EOS token detected (inference.rs:316-323)
- Max tokens reached (loop limit)
- Client disconnects (channel send fails - generation_engine.rs:184-188)

### Heartbeat System (heartbeat.rs:59-79)

**Flow:**
- Spawns tokio task at startup
- Sends heartbeat every 30 seconds
- Payload: `{worker_id, timestamp_ms, health_status}`
- Target: `POST {hive_url}/v1/heartbeat`
- **Failure:** Logs error, continues (non-fatal)
- **Note:** HTTP POST is a future feature (line 45-47 marked TODO)

---

## 3. Data Flows

### Inputs

**Configuration:**
- CLI arguments (main.rs:38-70)
- Environment variables:
  - `LLORCH_API_TOKEN` (network mode auth)
  - `LLORCH_LOG_FORMAT` (json|pretty, default: pretty)

**HTTP Requests:**
- `POST /v1/inference` with JSON body (ExecuteRequest)
- `GET /v1/inference/{job_id}/stream` (SSE connection)
- `Authorization: Bearer <token>` header (network mode)

**Model Files:**
- SafeTensors: `model.safetensors` + `config.json` + `tokenizer.json`
- GGUF: `model.gguf` (self-contained, includes tokenizer)

### Outputs

**HTTP Responses:**
- JSON: `{job_id, sse_url}` (POST /v1/inference)
- SSE stream: `data: {type, ...}\n\n` (GET /v1/inference/{job_id}/stream)
- Error JSON: `{errors: [{field, constraint, message, value?}]}`

**Logging:**
- Structured tracing (tracing crate)
- JSON format (production) or pretty format (dev)
- Narration events (observability-narration-core)

**Heartbeats:**
- `POST {hive_url}/v1/heartbeat` every 30s (future feature - HTTP POST not yet implemented)

### Internal Data Flow

**Request Queue Pattern (TEAM-149):**
```
HTTP Handler → RequestQueue → GenerationEngine (spawn_blocking)
                                      ↓
                              TokenResponse channel
                                      ↓
                              SSE Stream (GET endpoint)
```

**Job Registry (TEAM-154):**
- Maps `job_id` → `TokenReceiver`
- Enables dual-call pattern (POST creates, GET streams)
- Thread-safe (Arc<JobRegistry<TokenResponse>>)

---

## 4. Error Handling

### Error Types

**Validation Errors (validation.rs:68-128):**
- `FieldError {field, constraint, message, value?}`
- `ValidationErrorResponse {errors: Vec<FieldError>}`
- HTTP 400 Bad Request

**Inference Errors (error.rs:9-29):**
- `ModelError` - Model loading/architecture detection failed
- `TensorError` - Tensor operations failed
- `GgufError` - GGUF parsing failed
- `CheckpointError` - Checkpoint validation failed
- `CudaError` - CUDA runtime error (feature-gated)
- `IoError` - File I/O failed

**SSE Error Codes (sse.rs:101-116):**
- `VRAM_OOM` - Out of VRAM
- `CANCELLED` - Client cancelled
- `TIMEOUT` - Request timeout
- `INVALID_REQUEST` - Bad parameters
- `INFERENCE_FAILED` - Generation error

### Error Propagation

**Model Loading (main.rs:146-188):**
- Narrates `model_load_failed` with error details
- Logs error with tracing
- Returns `Err(e)` → daemon exits

**Request Validation (execute.rs:44-59):**
- Returns `ValidationErrorResponse` (HTTP 400)
- Narrates validation failure

**Generation Errors (generation_engine.rs:79-90):**
- Logs error with tracing
- Sends `TokenResponse::Error(String)` to SSE stream
- SSE converts to `InferenceEvent::Error`

**Authentication Errors (middleware/auth.rs:52-79):**
- HTTP 401 Unauthorized
- JSON: `{error: {code, message}}`
- Logs with token fingerprint (never raw token)

### Edge Cases

**Empty Prompt:** Validation error "must not be empty" (validation.rs:151-156)

**Very Long Prompt:** Validation error if >32768 chars (validation.rs:157-162)

**Invalid Sampling Params:** Validation catches out-of-range values (validation.rs:178-222)

**Model File Missing:** `anyhow::Error` → narration → daemon exits (inference.rs:146-188)

**VRAM Exhausted:** CUDA error → `InferenceEvent::Error {code: "VRAM_OOM"}` (sse.rs:103)

**Client Disconnect:** Channel send fails → generation stops gracefully (generation_engine.rs:184-188)

**Context Length Exceeded:** Tensor shape error → `InferenceEvent::Error` (not explicitly handled)

**EOS Token Mismatch:** Checks tokenizer first, falls back to model (inference.rs:300-304)

---

## 5. Integration Points

### Dependencies

**Upstream (Spawned By):**
- `rbee-hive` - Spawns worker with CLI args, manages lifecycle

**Shared Crates:**
- `observability-narration-core` - Narration events (23 imports)
- `job-registry` - Dual-call pattern (6 imports)
- `rbee-heartbeat` - Heartbeat types (4 imports)
- `auth-min` - Authentication (1 import)
- `timeout-enforcer` - Not used (no imports found)

**External:**
- `candle-core` - Tensor operations, Device abstraction
- `candle-nn` - Neural network functions (rms_norm, silu, softmax)
- `candle-transformers` - Model implementations (Llama, Mistral, Phi, Qwen)
- `tokenizers` - HuggingFace tokenizers
- `axum` - HTTP server
- `tokio` - Async runtime (single-threaded: `flavor = "current_thread"`)

### Dependents

**Downstream (Calls Worker):**
- `rbee-hive` - Sends inference requests, receives SSE streams
- `rbee-keeper` (indirect) - CLI client → queen → hive → worker

### Contracts

**HTTP API:**
- OpenAI-compatible parameter names (temperature, top_p, top_k, etc.)
- SSE event format (JSON with `type` field)
- Dual-call pattern (POST creates, GET streams)

**Heartbeat:**
- Payload: `WorkerHeartbeatPayload {worker_id, timestamp_ms, health_status}`
- Frequency: 30 seconds
- Target: `POST {hive_url}/v1/heartbeat` (future feature - HTTP POST not yet implemented)

**Narration:**
- All narration includes `.worker_id()` for tracing
- Uses standard actors: `ACTOR_LLM_WORKER_RBEE`, `ACTOR_MODEL_LOADER`, `ACTOR_CANDLE_BACKEND`
- Uses standard actions: `ACTION_STARTUP`, `ACTION_MODEL_LOAD`, `ACTION_INFERENCE_START`, etc.

---

## 6. Critical Invariants

### Safety Guarantees

**Memory Safety:**
- All unsafe code must have `// SAFETY:` documentation (clippy: `undocumented_unsafe_blocks = "deny"`)
- No unsafe blocks found in current codebase

**Thread Safety:**
- Backend wrapped in `Arc<Mutex<CandleInferenceBackend>>` (main.rs:201)
- Request queue uses `tokio::sync::mpsc::unbounded_channel` (thread-safe)
- Job registry uses `Arc<JobRegistry<TokenResponse>>` (thread-safe)

**UTF-8 Safety:**
- `TokenOutputStream` handles UTF-8 decoding with proper space handling (token_output_stream.rs)
- SSE events are JSON (UTF-8 by default)

### Performance Characteristics

**Single-Threaded Runtime:**
- `#[tokio::main(flavor = "current_thread")]` (main.rs:86)
- Optimal for CPU-bound inference (no context switching)
- Generation runs in `spawn_blocking` to avoid blocking HTTP handlers

**Real-Time Streaming:**
- Tokens sent immediately after decode (generation_engine.rs:179-189)
- No batching or buffering
- Latency: ~decode_time_per_token (typically <100ms)

**Sequential Processing:**
- One request at a time (generation engine processes queue sequentially)
- No concurrent inference (backend locked during generation)

### Correctness Invariants

**KV Cache Reset:**
- Cache MUST be reset before inference (clears warmup pollution - inference.rs:208-221)
- Failure to reset causes mask broadcasting errors (TEAM-021 bug)

**Job ID Uniqueness:**
- Server generates `job_id` (client doesn't provide)
- UUID v4 format (uuid crate with v4 feature)

**Token Receiver Consumption:**
- `take_token_receiver()` consumes receiver (stream.rs:77)
- Can only stream once per job (prevents duplicate streams)

**EOS Detection:**
- Checks tokenizer EOS first, falls back to model (inference.rs:300-304)
- Stops generation immediately on EOS (no extra tokens)

---

## 7. Existing Test Coverage

### Unit Tests

**Validation (validation.rs:410-889):**
- 48 test cases covering all parameter validation
- Edge cases: empty prompt, max length, out-of-range values
- Advanced parameters: top_p, top_k, repetition_penalty, min_p
- Stop sequences: empty, too many, too long
- Backward compatibility with old request format

**SSE Events (sse.rs:138-397):**
- 20 test cases for event serialization
- Terminal event detection
- Event name mapping
- Unicode handling (emoji, CJK)
- Stop reason serialization

**Sampling Config (sampling_config.rs:142-386):**
- 25 test cases for sampling configuration
- Advanced sampling detection
- Greedy vs stochastic modes
- Consistency validation (conflicting params)

**Inference Result (inference_result.rs:148-395):**
- 28 test cases for result tracking
- Stop reason classification
- Success/failure detection
- Partial generation on error/cancellation

**Authentication (middleware/auth.rs:89-179):**
- 4 test cases for auth middleware
- Success, missing header, invalid token, invalid format

**Device (device.rs:86-114):**
- 3 test cases for device initialization (CPU, CUDA, Metal)
- Smoke test (create tensor, verify operations)

### Integration Tests (tests/)

**team_009_smoke.rs:**
- Model loading smoke tests
- Basic inference validation

**team_011_integration.rs:**
- End-to-end inference tests
- Multi-token generation
- Temperature variations

**multi_model_support.rs:**
- Llama, Mistral, Phi, Qwen architecture tests
- SafeTensors and GGUF format tests

**team_013_cuda_integration.rs:**
- CUDA-specific tests (feature-gated)
- GPU memory tracking

**test_question_mark_tokenization.rs:**
- Edge case: question mark tokenization

### BDD Tests (bdd/)

**Status:** Placeholder only (bdd/tests/features/placeholder.feature)
- No real BDD scenarios implemented
- Feature file is generic template

### Coverage Gaps

**Missing Unit Tests:**
- `generation_engine.rs` - No unit tests (complex spawn_blocking logic)
- `request_queue.rs` - No unit tests (channel-based queue)
- `heartbeat.rs` - No unit tests (periodic task)
- `http/execute.rs` - No unit tests (HTTP handler)
- `http/stream.rs` - No unit tests (SSE streaming)
- `backend/models/*` - Limited model-specific tests

**Missing Integration Tests:**
- Dual-call pattern (POST + GET flow)
- SSE streaming with real tokens
- Client disconnect during generation
- Authentication flow (network vs local mode)
- Model warmup behavior
- KV cache reset verification

**Future Feature (Not a Test Gap):**
- Heartbeat integration with hive (HTTP POST not yet implemented)

**Missing E2E Tests:**
- Full daemon lifecycle (startup → inference → shutdown)
- Multiple concurrent requests (queue behavior)
- Long-running generation (>1000 tokens)
- Error recovery (model load failure, VRAM OOM)
- Graceful shutdown (SIGTERM handling)

---

## 8. Behavior Checklist

### HTTP API
- [x] All endpoints documented (health, inference, stream)
- [x] Request/response schemas defined
- [x] OpenAI compatibility (parameter names)
- [x] Error responses (validation, auth, inference)

### Model Loading
- [x] Path resolution (SafeTensors, GGUF)
- [x] GPU allocation (CPU, CUDA, Metal)
- [x] VRAM allocation (model size tracking)
- [x] Validation (architecture detection)
- [x] Failure handling (narration + exit)

### Inference
- [x] Request handling (validation, queue, generation)
- [x] Token generation (sampling, EOS detection)
- [x] Sampling (temperature, top_p, top_k, min_p, repetition_penalty)
- [x] Stopping conditions (EOS, max_tokens, client disconnect)
- [x] Response formatting (SSE events)

### Streaming
- [x] SSE setup (dual-call pattern)
- [x] Token streaming (real-time, no batching)
- [x] Stream completion ([DONE] marker)
- [x] Stream errors (InferenceEvent::Error)

### Heartbeat
- [x] Heartbeat to hive (30s interval)
- [x] Frequency (configurable via interval)
- [x] Payload (worker_id, timestamp, health_status)
- [ ] HTTP POST implementation (future feature - not yet implemented)

### Resource Management
- [x] GPU memory tracking (model_size_bytes)
- [ ] Slot management (not implemented - single request at a time)
- [x] Request queuing (RequestQueue + GenerationEngine)
- [x] Cleanup (graceful shutdown via SIGTERM/SIGKILL)

### Configuration
- [x] Config loading (CLI args, env vars)
- [x] Validation (parameter ranges)
- [x] Defaults (temperature=1.0, top_p=1.0, etc.)
- [x] Model configs (architecture-specific)

### Daemon Lifecycle
- [x] Startup sequence (parse → load → queue → engine → heartbeat → HTTP)
- [x] Model loading (at startup)
- [x] Graceful shutdown (SIGTERM/SIGKILL by hive)
- [x] Cleanup (backend drop, channel close)

---

## Critical Focus Areas for Testing

### 1. Dual-Call Pattern (TEAM-154)
**Complexity:** High  
**Risk:** Medium (new pattern, channel lifecycle)  
**Gaps:**
- No integration tests for POST → GET flow
- No tests for multiple GET attempts (should fail - receiver consumed)
- No tests for GET before POST (job not found)

### 2. Real-Time Streaming (TEAM-149)
**Complexity:** High  
**Risk:** High (spawn_blocking, channels, SSE)  
**Gaps:**
- No tests for token-by-token latency
- No tests for client disconnect during generation
- No tests for channel backpressure

### 3. KV Cache Reset (TEAM-021)
**Complexity:** Medium  
**Risk:** High (correctness bug if not reset)  
**Gaps:**
- No tests verifying cache is reset before inference
- No tests for warmup pollution (mask broadcasting error)

### 4. Model Loading (Multi-Architecture)
**Complexity:** High  
**Risk:** Medium (many architectures, two formats)  
**Gaps:**
- Limited tests for GGUF quantized models
- No tests for architecture detection failures
- No tests for corrupted model files

### 5. Authentication (Network vs Local Mode)
**Complexity:** Medium  
**Risk:** High (security critical)  
**Gaps:**
- No integration tests for auth flow
- No tests for token rotation
- No tests for timing attack resistance (timing_safe_eq)

---

**Status:** ✅ COMPLETE  
**Next:** TEAM-242 (test planning) can use this inventory to create comprehensive test plans  
**Code Signatures:** All investigated files marked with `// TEAM-219: Investigated`
