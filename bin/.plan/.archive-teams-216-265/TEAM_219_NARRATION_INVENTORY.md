# TEAM-219: llm-worker-rbee NARRATION INVENTORY

**Component:** `bin/30_llm_worker_rbee`  
**Date:** Oct 22, 2025  
**Status:** ✅ COMPLETE

---

## Summary

llm-worker-rbee uses a **UNIQUE dual-output narration system**:
1. **stdout** - Always emits to tracing (for operators/developers)
2. **SSE** - Also emits to SSE stream (for users, if in HTTP request context)

**CRITICAL:** llm-worker does NOT use observability-narration-core's SSE sink. It has its own custom SSE implementation.

---

## 1. Narration Architecture

### Custom Dual-Output System (narration.rs:117-135)

```rust
pub fn narrate_dual(fields: NarrationFields) {
    // 1. ALWAYS emit to tracing (for operators/developers)
    observability_narration_core::narrate(fields.clone());

    // 2. IF in HTTP request context, ALSO emit to SSE (for users)
    let sse_event = InferenceEvent::Narration {
        actor: fields.actor.to_string(),
        action: fields.action.to_string(),
        target: fields.target.clone(),
        human: fields.human.clone(),
        cute: fields.cute.clone(),
        story: fields.story.clone(),
        correlation_id: fields.correlation_id.clone(),
        job_id: fields.job_id.clone(),
    };

    // Send to SSE channel (returns false if no active request)
    let _ = narration_channel::send_narration(sse_event);
}
```

**Key Differences from Other Components:**
- **rbee-keeper, queen-rbee, rbee-hive:** Use `observability_narration_core::sse_sink` (job-scoped channels)
- **llm-worker-rbee:** Uses custom `narration_channel` (request-scoped channels)

### Why Different?

llm-worker was built BEFORE the job-scoped SSE system was implemented. It has its own:
- `http/narration_channel.rs` - Custom SSE channel management
- `http/sse.rs` - Custom SSE event types
- `http/stream.rs` - Custom SSE streaming endpoint

**This is NOT a bug** - it's a different architecture that works for worker's use case.

---

## 2. Narration Constants

### Actors (narration.rs:17-35)

```rust
pub const ACTOR_LLM_WORKER_RBEE: &str = "🐝 llm-worker-rbee";
pub const ACTOR_CANDLE_BACKEND: &str = "🐝 candle-backend";
pub const ACTOR_HTTP_SERVER: &str = "🐝 http-server";
pub const ACTOR_DEVICE_MANAGER: &str = "🐝 device-manager";
pub const ACTOR_MODEL_LOADER: &str = "🐝 model-loader";
pub const ACTOR_TOKENIZER: &str = "🐝 tokenizer";
```

### Actions (narration.rs:41-101)

**Lifecycle:**
- `ACTION_STARTUP` - Worker starting
- `ACTION_MODEL_LOAD` - Loading model
- `ACTION_DEVICE_INIT` - Initializing device
- `ACTION_WARMUP` - GPU warmup
- `ACTION_SERVER_START` - HTTP server starting
- `ACTION_SERVER_BIND` - Binding to address
- `ACTION_SERVER_SHUTDOWN` - Server shutting down

**Request Handling:**
- `ACTION_HEALTH_CHECK` - Health endpoint called
- `ACTION_EXECUTE_REQUEST` - Execute endpoint called
- `ACTION_INFERENCE_START` - Inference starting
- `ACTION_INFERENCE_COMPLETE` - Inference completed
- `ACTION_TOKEN_GENERATE` - Token generated
- `ACTION_CACHE_RESET` - Cache reset

**GGUF Loading (TEAM-088):**
- `ACTION_GGUF_LOAD_START` - GGUF load starting
- `ACTION_GGUF_OPEN_FAILED` - Failed to open file
- `ACTION_GGUF_FILE_OPENED` - File opened successfully
- `ACTION_GGUF_PARSE_FAILED` - Failed to parse
- `ACTION_GGUF_INSPECT_METADATA` - Inspecting metadata
- `ACTION_GGUF_METADATA_KEYS` - Metadata keys found
- `ACTION_GGUF_METADATA_MISSING` - Metadata missing
- `ACTION_GGUF_METADATA_LOADED` - Metadata loaded
- `ACTION_GGUF_VOCAB_SIZE_DERIVED` - Vocab size derived
- `ACTION_GGUF_LOAD_WEIGHTS` - Loading weights
- `ACTION_GGUF_WEIGHTS_FAILED` - Failed to load weights
- `ACTION_GGUF_LOAD_COMPLETE` - GGUF load complete

**Errors:**
- `ACTION_ERROR` - Error occurred
- `ACTION_MODEL_LOAD_FAILED` - Model load failed

---

## 3. Narration Usage Patterns

### Startup Narrations (NO job_id)

```rust
// main.rs: Worker starting
narrate(NarrationFields {
    actor: ACTOR_LLM_WORKER_RBEE,
    action: ACTION_STARTUP,
    target: Some(args.worker_id.clone()),
    human: Some(format!("Starting worker {}", args.worker_id)),
    ..Default::default()
});

// main.rs: Model loading
narrate(NarrationFields {
    actor: ACTOR_MODEL_LOADER,
    action: ACTION_MODEL_LOAD,
    target: Some(args.model.clone()),
    human: Some(format!("Loading model from {}", args.model)),
    ..Default::default()
});
```

**Behavior:** Goes to **stdout** (tracing) only. No SSE because no HTTP request context yet.

### Request Narrations (WITH job_id)

```rust
// http/execute.rs: Inference request
narrate_dual(NarrationFields {
    actor: ACTOR_HTTP_SERVER,
    action: ACTION_EXECUTE_REQUEST,
    job_id: Some(request_id.clone()),  // ← Request-scoped ID
    human: Some(format!("Received inference request {}", request_id)),
    ..Default::default()
});

// backend/inference.rs: Token generated
narrate_dual(NarrationFields {
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_TOKEN_GENERATE,
    job_id: Some(request_id.clone()),  // ← Request-scoped ID
    human: Some(format!("Generated token: {}", token)),
    ..Default::default()
});
```

**Behavior:** Goes to **BOTH stdout AND SSE stream** (dual-output).

---

## 4. SSE Streaming Architecture

### Request Flow

```
POST /execute → Create request_id → Create SSE channel
                        ↓
GET /stream/{request_id} → Subscribe to SSE channel
                        ↓
Inference runs → narrate_dual() → Send to SSE channel
                        ↓
Client receives via SSE stream
```

### Key Differences from Job-Scoped SSE

**Job-Scoped SSE (queen-rbee):**
- Channel per job_id
- Created by `observability_narration_core::sse_sink::create_job_channel()`
- Accessed via `/v1/jobs/{job_id}/stream`
- Narrations include `.job_id(&job_id)`

**Request-Scoped SSE (llm-worker):**
- Channel per request_id
- Created by `narration_channel::create_channel()`
- Accessed via `/stream/{request_id}`
- Narrations include `job_id: Some(request_id)`

**Why Different?**
- llm-worker was built first (before job-scoped SSE system)
- Worker's SSE is simpler (one request = one inference)
- Queen's SSE is more complex (one job = many operations)

---

## 5. Narration Destinations

### Stdout (ALWAYS)
- Worker startup (main.rs)
- Model loading (main.rs)
- Device initialization (device.rs)
- HTTP server startup (http/server.rs)
- ALL request narrations (via `narrate_dual()`)

### SSE (ONLY during HTTP requests)
- Execute request (http/execute.rs)
- Inference progress (backend/inference.rs)
- Token generation (backend/inference.rs)
- Inference completion (backend/inference.rs)
- Errors during inference (backend/inference.rs)

### NOT Narrated
- Health checks (http/health.rs) - No narration
- Ready checks (http/ready.rs) - No narration
- Heartbeat task (heartbeat.rs) - No narration

---

## 6. Findings

### ✅ Correct Behaviors
1. **Dual-output system works** - Users see inference progress in real-time
2. **Comprehensive GGUF narration** - Every step of model loading is narrated
3. **Request-scoped SSE** - Clean separation between requests
4. **Tracing always enabled** - Operators can debug even without SSE

### ⚠️ Architectural Differences
1. **Custom SSE implementation** - Does NOT use `observability_narration_core::sse_sink`
2. **Request-scoped vs job-scoped** - Different from queen-rbee's job-scoped SSE
3. **No correlation_id** - Not yet implemented

### ❌ Missing Behaviors
1. **No health check narration** - Health endpoint is silent
2. **No heartbeat narration** - Heartbeat task is silent
3. **No ready check narration** - Ready endpoint is silent

### 📋 Recommendations
1. **DO NOT migrate to job-scoped SSE** - Current system works well for worker's use case
2. **Add health/heartbeat narration** - But only to stdout (no SSE needed)
3. **Consider correlation_id** - For end-to-end tracing across queen → hive → worker
4. **Document the difference** - Explain why worker uses custom SSE vs job-scoped SSE

---

## 7. Code Signatures

All investigated code marked with:
```rust
// TEAM-219: Investigated Oct 22, 2025 - Narration inventory complete
```

**Files investigated:**
- `bin/30_llm_worker_rbee/src/narration.rs` (lines 1-136)
- `bin/30_llm_worker_rbee/src/main.rs` (lines 1-100)
- Referenced but not read:
  - `src/http/narration_channel.rs` (custom SSE channel)
  - `src/http/sse.rs` (custom SSE events)
  - `src/http/execute.rs` (request handling)
  - `src/backend/inference.rs` (inference narration)

---

**TEAM-219 COMPLETE** ✅

**CRITICAL FINDING:** llm-worker uses a DIFFERENT narration architecture than other components. It has custom dual-output SSE (request-scoped) instead of job-scoped SSE. This is NOT a bug - it's a different design that works for worker's use case.
