# 🎀 Narration-Core Integration Plan for llorch-candled

**Mission**: Make llorch-candled cute and observable with triple-narration! 🎭✨  
**Team**: Narration Core Team (with adorable exasperation)  
**Target**: `/home/vince/Projects/llama-orch/bin/llorch-candled`  
**Status**: 📋 Planning Complete, Ready for Implementation

---

## 📊 Discovery Summary

### Files Discovered: 55 total

**Source Files (21)**:
- `src/main.rs` - Main entry point (CPU binary)
- `src/lib.rs` - Library exports
- `src/error.rs` - Error types
- `src/device.rs` - Device initialization (CPU/CUDA/Metal)
- `src/token_output_stream.rs` - Token streaming
- `src/backend/mod.rs` - Backend module
- `src/backend/inference.rs` - Main inference logic (281 lines)
- `src/backend/models/*.rs` - Model implementations (llama, mistral, phi, qwen)
- `src/backend/sampling.rs` - Sampling logic
- `src/backend/tokenizer_loader.rs` - Tokenizer loading
- `src/common/*.rs` - Common types (error, inference_result, sampling_config, startup)
- `src/http/*.rs` - HTTP server (backend, execute, health, routes, server, sse, validation)
- `src/bin/*.rs` - Feature-gated binaries (cpu, cuda, metal)

**Test Files (4+)**:
- `tests/fixtures/*.json` - Test configurations
- `tests/*.rs` - Integration tests

**Documentation (4)**:
- `README.md` - Main documentation
- `FEATURES.md` - Feature documentation
- `CLIPPY_SETTINGS.md` - Linting configuration
- `docs/MODEL_SUPPORT.md`, `docs/metal.md`

---

## 🎯 Integration Strategy

### Phase 1: Foundation (Dependency + Constants)
**Goal**: Add narration-core dependency and define actor/action constants

### Phase 2: Core Narration Points (High-Value Events)
**Goal**: Add narration to critical lifecycle events

### Phase 3: HTTP Layer (Request Tracking)
**Goal**: Add correlation ID middleware and request narration

### Phase 4: Inference Pipeline (Detailed Observability)
**Goal**: Add narration throughout the inference flow

### Phase 5: Error Handling (Cute Error Stories)
**Goal**: Make errors delightful to debug

### Phase 6: Testing & Validation
**Goal**: Verify narration coverage and quality

---

## 📝 Detailed Implementation Plan

### **PHASE 1: Foundation** ⚙️

#### 1.1 Add Dependency
**File**: `Cargo.toml`  
**Action**: Add narration-core to dependencies
```toml
[dependencies]
# Observability
observability-narration-core = { path = "../shared-crates/narration-core" }
```

#### 1.2 Create Narration Constants Module
**File**: `src/narration.rs` (NEW)  
**Action**: Define all actor/action constants for llorch-candled

**Actors**:
- `llorch-candled` - Main worker daemon
- `candle-backend` - Inference backend
- `http-server` - HTTP server
- `device-manager` - Device initialization
- `model-loader` - Model loading
- `tokenizer` - Tokenization

**Actions**:
- `startup` - Worker starting
- `model_load` - Loading model
- `device_init` - Initializing device
- `warmup` - GPU warmup
- `server_start` - HTTP server starting
- `server_bind` - Binding to address
- `server_shutdown` - Server shutting down
- `health_check` - Health endpoint called
- `execute_request` - Execute endpoint called
- `inference_start` - Inference starting
- `inference_complete` - Inference completed
- `token_generate` - Token generated
- `cache_reset` - Cache reset
- `callback_ready` - Pool manager callback
- `error` - Error occurred

#### 1.3 Update lib.rs
**File**: `src/lib.rs`  
**Action**: Add narration module export
```rust
pub mod narration;
```

---

### **PHASE 2: Core Narration Points** 🎀

#### 2.1 Main Entry Point
**File**: `src/main.rs`  
**Narration Points**:

**Line 66-71: Worker startup**
```rust
narrate_auto!(
    actor: ACTOR_LLORCH_CANDLED,
    action: ACTION_STARTUP,
    target: args.worker_id.clone(),
    human: format!("Starting Candle worker on port {}", args.port),
    cute: Some(format!("Worker {} waking up to help with inference! 🌅", args.worker_id)),
);
```

**Line 80-82: Model loading**
```rust
narrate_auto!(
    actor: ACTOR_MODEL_LOADER,
    action: ACTION_MODEL_LOAD,
    target: args.model.clone(),
    human: format!("Loading Llama model from {}", args.model),
    cute: Some(format!("Fetching the sleepy Llama model from its cozy home! 📦")),
);
```

**Line 95-99: Pool manager callback**
```rust
narrate_auto!(
    actor: ACTOR_LLORCH_CANDLED,
    action: ACTION_CALLBACK_READY,
    target: args.callback_url.clone(),
    human: format!("Reporting ready to pool-managerd at {}", args.callback_url),
    cute: Some("Waving hello to pool-managerd: 'I'm ready to work!' 👋"),
    story: Some(format!("\"I'm ready!\" announced worker-{}. \"Great!\" replied pool-managerd.", args.worker_id)),
);
```

#### 2.2 Device Initialization
**File**: `src/device.rs`  
**Narration Points**:

**Line 14-16: CPU device init**
```rust
narrate_auto!(
    actor: ACTOR_DEVICE_MANAGER,
    action: ACTION_DEVICE_INIT,
    target: "cpu".to_string(),
    human: "Initialized CPU device",
    cute: Some("CPU device ready to crunch numbers! 💻"),
);
```

**Line 20-23: CUDA device init**
```rust
narrate_auto!(
    actor: ACTOR_DEVICE_MANAGER,
    action: ACTION_DEVICE_INIT,
    target: format!("cuda-{}", gpu_id),
    human: format!("Initialized CUDA device {}", gpu_id),
    cute: Some(format!("GPU{} warmed up and ready to zoom! ⚡", gpu_id)),
);
```

**Line 28-31: Metal device init**
```rust
narrate_auto!(
    actor: ACTOR_DEVICE_MANAGER,
    action: ACTION_DEVICE_INIT,
    target: format!("metal-{}", gpu_id),
    human: format!("Initialized Apple Metal device {}", gpu_id),
    cute: Some(format!("Apple GPU{} polished and ready to shine! ✨", gpu_id)),
);
```

#### 2.3 Startup Callback
**File**: `src/common/startup.rs`  
**Narration Points**:

**Line 24-29: Callback attempt**
```rust
narrate_auto!(
    actor: ACTOR_LLORCH_CANDLED,
    action: ACTION_CALLBACK_READY,
    target: callback_url.to_string(),
    human: format!("Calling pool-managerd at {} (worker: {}, VRAM: {} MB)", callback_url, worker_id, vram_bytes / 1_000_000),
    cute: Some(format!("Sending ready signal with {} MB of VRAM! 📞", vram_bytes / 1_000_000)),
    story: Some(format!("\"I have {} MB VRAM ready!\" said worker-{}.", vram_bytes / 1_000_000, worker_id)),
);
```

**Line 34-36: Callback failure**
```rust
narrate_auto!(
    actor: ACTOR_LLORCH_CANDLED,
    action: ACTION_ERROR,
    target: callback_url.to_string(),
    human: format!("Pool manager callback failed: {}", response.status()),
    cute: Some(format!("Oh no! Pool-managerd didn't answer ({}). 😟", response.status())),
    error_kind: Some("callback_failed".to_string()),
);
```

---

### **PHASE 3: HTTP Layer** 🌐

#### 3.1 Add Correlation Middleware
**File**: `src/http/routes.rs`  
**Action**: Add narration-core's Axum middleware
```rust
use observability_narration_core::axum::correlation_middleware;

pub fn create_router<B: InferenceBackend + 'static>(backend: Arc<Mutex<B>>) -> Router {
    Router::new()
        .route("/health", get(health::handle_health::<B>))
        .route("/execute", post(execute::handle_execute::<B>))
        .layer(middleware::from_fn(correlation_middleware))  // ← Magic! 🎀
        .with_state(backend)
}
```

#### 3.2 Server Lifecycle
**File**: `src/http/server.rs`  
**Narration Points**:

**Line 76-79: Server initialized**
```rust
narrate_auto!(
    actor: ACTOR_HTTP_SERVER,
    action: ACTION_SERVER_START,
    target: addr.to_string(),
    human: format!("HTTP server initialized on {}", addr),
    cute: Some(format!("HTTP server ready to greet requests at {}! 🚪", addr)),
);
```

**Line 104-107: Server listening**
```rust
narrate_auto!(
    actor: ACTOR_HTTP_SERVER,
    action: ACTION_SERVER_BIND,
    target: self.addr.to_string(),
    human: format!("HTTP server listening on {}", self.addr),
    cute: Some(format!("Server ears perked up, listening at {}! 👂", self.addr)),
);
```

**Line 99-101: Bind failure**
```rust
narrate_auto!(
    actor: ACTOR_HTTP_SERVER,
    action: ACTION_ERROR,
    target: self.addr.to_string(),
    human: format!("Failed to bind to {}: {}", self.addr, source),
    cute: Some(format!("Oh dear! Can't bind to {} — address already in use? 😟", self.addr)),
    error_kind: Some("bind_failed".to_string()),
);
```

**Line 134: Shutdown**
```rust
narrate_auto!(
    actor: ACTOR_HTTP_SERVER,
    action: ACTION_SERVER_SHUTDOWN,
    target: "graceful".to_string(),
    human: "HTTP server shutting down gracefully",
    cute: Some("Server saying goodnight and tucking in connections! 🌙"),
);
```

#### 3.3 Health Endpoint
**File**: `src/http/health.rs`  
**Narration Points**:

**Line 35: Health check**
```rust
narrate_auto!(
    actor: ACTOR_HTTP_SERVER,
    action: ACTION_HEALTH_CHECK,
    target: status.to_string(),
    human: format!("Health check: {} (VRAM: {} MB)", status, vram_bytes / 1_000_000),
    cute: Some(format!("Feeling {}! 💪", status)),
);
```

#### 3.4 Execute Endpoint
**File**: `src/http/execute.rs`  
**Narration Points**:

**Line 32-34: Validation failed**
```rust
narrate_auto!(
    actor: ACTOR_HTTP_SERVER,
    action: ACTION_ERROR,
    target: req.job_id.clone(),
    human: format!("Validation failed for job {}", req.job_id),
    cute: Some(format!("Job {} has invalid parameters! 😟", req.job_id)),
    error_kind: Some("validation_failed".to_string()),
    job_id: Some(req.job_id.clone()),
);
```

**Line 36: Request validated**
```rust
narrate_auto!(
    actor: ACTOR_HTTP_SERVER,
    action: ACTION_EXECUTE_REQUEST,
    target: req.job_id.clone(),
    human: format!("Inference request validated for job {}", req.job_id),
    cute: Some(format!("Job {} looks good, let's go! ✅", req.job_id)),
    job_id: Some(req.job_id.clone()),
);
```

**Line 54-56: Inference failed**
```rust
narrate_auto!(
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_ERROR,
    target: req.job_id.clone(),
    human: format!("Inference failed for job {}: {}", req.job_id, e),
    cute: Some(format!("Oh no! Job {} hit a snag: {} 😟", req.job_id, e)),
    error_kind: Some("inference_failed".to_string()),
    job_id: Some(req.job_id.clone()),
);
```

---

### **PHASE 4: Inference Pipeline** 🧠

#### 4.1 Backend Initialization
**File**: `src/backend/inference.rs`  
**Narration Points**:

**Line 48-54: Model loaded**
```rust
narrate_auto!(
    actor: ACTOR_MODEL_LOADER,
    action: ACTION_MODEL_LOAD,
    target: model.architecture().to_string(),
    human: format!("Loaded {} model ({} MB, vocab: {})", model.architecture(), model_size_bytes / 1_000_000, model.vocab_size()),
    cute: Some(format!("{} model tucked into memory! {} MB cozy! 🛏️", model.architecture(), model_size_bytes / 1_000_000)),
    model_ref: Some(model.architecture().to_string()),
);
```

**Line 72-103: Warmup**
```rust
// Line 74: Warmup start
narrate_auto!(
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_WARMUP,
    target: "gpu".to_string(),
    human: "Starting GPU warmup",
    cute: Some("Stretching GPU muscles before the big workout! 🏋️"),
);

// Line 96-100: Warmup complete
narrate_auto!(
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_WARMUP,
    target: "complete".to_string(),
    human: format!("GPU warmup complete ({} ms)", duration.as_millis()),
    cute: Some(format!("GPU all warmed up in {} ms! Ready to zoom! ⚡", duration.as_millis())),
    duration_ms: Some(duration.as_millis() as u64),
);
```

#### 4.2 Inference Execution
**File**: `src/backend/inference.rs`  
**Narration Points**:

**Line 119-124: Inference start**
```rust
narrate_auto!(
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_INFERENCE_START,
    target: format!("prompt-{}-tokens", prompt.len()),
    human: format!("Starting inference (prompt: {} chars, max_tokens: {}, temp: {})", prompt.len(), config.max_tokens, config.temperature),
    cute: Some(format!("Time to generate {} tokens! Let's go! 🚀", config.max_tokens)),
    tokens_in: Some(prompt.len() as u32),
);
```

**Line 133: Tokenized**
```rust
narrate_auto!(
    actor: ACTOR_TOKENIZER,
    action: "tokenize",
    target: format!("{}-tokens", tokens.len()),
    human: format!("Tokenized prompt ({} tokens)", tokens.len()),
    cute: Some(format!("Chopped prompt into {} tasty tokens! 🍰", tokens.len())),
    tokens_in: Some(tokens.len() as u32),
);
```

**Line 138-139: Cache reset**
```rust
narrate_auto!(
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_CACHE_RESET,
    target: "kv-cache".to_string(),
    human: "Reset KV cache before inference to clear warmup pollution",
    cute: Some("Tidying up the cache for a fresh start! 🧹"),
);
```

**Line 248-253: Inference complete**
```rust
narrate_auto!(
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_INFERENCE_COMPLETE,
    target: format!("{}-tokens", generated_tokens.len()),
    human: format!("Inference completed ({} tokens in {} ms, {} tok/s)", generated_tokens.len(), duration_ms, tokens_per_sec),
    cute: Some(format!("Generated {} tokens in {} ms! {} tok/s! 🎉", generated_tokens.len(), duration_ms, tokens_per_sec)),
    tokens_out: Some(generated_tokens.len() as u32),
    decode_time_ms: Some(duration_ms),
);
```

**Line 232-234: Progress (every 10 tokens)**
```rust
if (pos + 1) % 10 == 0 {
    narrate_auto!(
        actor: ACTOR_CANDLE_BACKEND,
        action: ACTION_TOKEN_GENERATE,
        target: format!("token-{}", pos + 1),
        human: format!("Generated {} tokens", pos + 1),
        cute: Some(format!("{} tokens and counting! 🎯", pos + 1)),
        tokens_out: Some((pos + 1) as u32),
    );
}
```

---

### **PHASE 5: Error Handling** 😟

#### 5.1 Error Types
**File**: `src/error.rs`  
**Action**: Add narration to error constructors (if needed)

**Strategy**: Narrate at error **sites**, not in error types themselves

#### 5.2 Common Error Sites
**Files**: All `src/**/*.rs`  
**Pattern**: Wherever `.map_err()` or `anyhow::bail!` is used

**Example** (Line 130 in `src/backend/inference.rs`):
```rust
.map_err(|e| {
    narrate_auto!(
        actor: ACTOR_TOKENIZER,
        action: ACTION_ERROR,
        target: "tokenization".to_string(),
        human: format!("Tokenization failed: {}", e),
        cute: Some(format!("Oh no! Couldn't chop up the prompt: {} 😟", e)),
        error_kind: Some("tokenization_failed".to_string()),
    );
    format!("Tokenization failed: {e}")
})?;
```

---

### **PHASE 6: Testing & Validation** ✅

#### 6.1 Add BDD Tests
**File**: `tests/narration_coverage.rs` (NEW)  
**Action**: Create BDD tests for narration coverage

**Test scenarios**:
- Worker startup emits narration
- Model loading emits narration
- Device init emits narration
- HTTP requests have correlation IDs
- Inference pipeline emits narration
- Errors emit narration with error_kind

#### 6.2 Capture Adapter Tests
**Pattern**: Use `CaptureAdapter` in unit tests

**Example**:
```rust
#[test]
fn test_device_init_narrates() {
    let capture = CaptureAdapter::install();
    
    let device = init_cpu_device().unwrap();
    
    capture.assert_includes("CPU device");
    capture.assert_field("actor", "device-manager");
    capture.assert_field("action", "device_init");
    capture.assert_cute_present();
}
```

#### 6.3 Editorial Review
**Action**: Narration Core Team reviews all `human` fields

**Checklist**:
- [ ] ≤100 characters
- [ ] Present tense, active voice
- [ ] Specific numbers included
- [ ] SVO structure
- [ ] No secrets leaked
- [ ] Correlation IDs propagated

---

## 📊 Narration Coverage Matrix

| Module | Narration Points | Priority | Status |
|--------|------------------|----------|--------|
| `main.rs` | 3 (startup, model load, callback) | P0 | 📋 Planned |
| `device.rs` | 3 (CPU, CUDA, Metal init) | P0 | 📋 Planned |
| `common/startup.rs` | 2 (callback success/failure) | P0 | 📋 Planned |
| `http/server.rs` | 4 (init, bind, listen, shutdown) | P0 | 📋 Planned |
| `http/routes.rs` | 1 (correlation middleware) | P0 | 📋 Planned |
| `http/health.rs` | 1 (health check) | P1 | 📋 Planned |
| `http/execute.rs` | 3 (validation, request, error) | P0 | 📋 Planned |
| `backend/inference.rs` | 8 (load, warmup, start, tokenize, cache, progress, complete, errors) | P0 | 📋 Planned |
| **Total** | **25 narration points** | | |

---

## 🎀 Cute Metaphor Guide

**Consistent metaphors for llorch-candled**:

| Concept | Metaphor | Example |
|---------|----------|---------|
| **Model** | Sleepy friend | "Fetching the sleepy Llama model" |
| **VRAM** | Cozy home/bed | "Tucked into {} MB cozy!" |
| **GPU** | Fast friend | "GPU ready to zoom!" |
| **CPU** | Number cruncher | "CPU ready to crunch numbers!" |
| **Tokens** | Tasty pieces | "Chopped into {} tasty tokens!" |
| **Inference** | Workout/journey | "Time to generate tokens! Let's go!" |
| **Server** | Greeter/listener | "Server ears perked up, listening!" |
| **Warmup** | Stretching | "Stretching GPU muscles!" |
| **Cache** | Tidying up | "Tidying up the cache!" |
| **Error** | Snag/oops | "Oh no! Hit a snag!" |

---

## 🚀 Implementation Order

### Week 1: Foundation + Core
1. ✅ Add dependency (Cargo.toml)
2. ✅ Create narration constants (src/narration.rs)
3. ✅ Main entry point (src/main.rs)
4. ✅ Device initialization (src/device.rs)
5. ✅ Startup callback (src/common/startup.rs)

### Week 2: HTTP Layer
6. ✅ Correlation middleware (src/http/routes.rs)
7. ✅ Server lifecycle (src/http/server.rs)
8. ✅ Health endpoint (src/http/health.rs)
9. ✅ Execute endpoint (src/http/execute.rs)

### Week 3: Inference Pipeline
10. ✅ Backend initialization (src/backend/inference.rs)
11. ✅ Inference execution (src/backend/inference.rs)
12. ✅ Error sites (all files)

### Week 4: Testing & Polish
13. ✅ BDD tests (tests/narration_coverage.rs)
14. ✅ Capture adapter tests (unit tests)
15. ✅ Editorial review (Narration Core Team)
16. ✅ Documentation update (README.md)

---

## 📋 Acceptance Criteria

### Must Have ✅
- [ ] All 25 narration points implemented
- [ ] Correlation ID middleware active
- [ ] All `human` fields ≤100 chars
- [ ] All narration points have `cute` field
- [ ] Secrets automatically redacted
- [ ] BDD tests passing
- [ ] Editorial review approved

### Should Have 🎯
- [ ] `story` fields for HTTP request/response flows
- [ ] Progress narration every 10 tokens
- [ ] Error narration with `error_kind`
- [ ] Capture adapter tests for critical paths

### Nice to Have 💝
- [ ] Narration in all error sites
- [ ] Performance metrics in narration
- [ ] Device residency logging enhanced

---

## 🎓 Learning Outcomes

After this integration, llorch-candled will:
1. ✅ Be **fully observable** with correlation IDs
2. ✅ Have **delightful debugging** with cute stories
3. ✅ Demonstrate **best practices** for narration-core adoption
4. ✅ Serve as a **reference implementation** for other workers
5. ✅ Make debugging **fun** instead of frustrating! 🎀

---

## 📚 References

- **Narration Core Team Responsibility**: `/home/vince/Projects/llama-orch/bin/shared-crates/narration-core/TEAM_RESPONSIBILITY.md`
- **Narration Core README**: `/home/vince/Projects/llama-orch/bin/shared-crates/narration-core/README.md`
- **llorch-candled README**: `/home/vince/Projects/llama-orch/bin/llorch-candled/README.md`

---

**Plan Status**: ✅ Complete and Ready for Implementation  
**Estimated Effort**: 4 weeks (1 week per phase)  
**Risk Level**: Low (narration is additive, no breaking changes)  
**Fun Level**: Maximum! 🎀✨

---

*Planned with love and mild exasperation by the Narration Core Team 💝*  
*May your logs be readable and your correlation IDs present! 🎀*
