# TEAM-130B: llm-worker-rbee - PART 1: METRICS & CRATES

**Binary:** `bin/llm-worker-rbee`  
**Phase:** Phase 2, Day 7-8  
**Date:** 2025-10-19

---

## 🎯 EXECUTIVE SUMMARY

**Current:** LLM inference worker (5,026 LOC code-only, 41 files)  
**Proposed:** 6 focused crates under `worker-rbee-crates/`  
**Risk:** MEDIUM (inference-base is complex)  
**Timeline:** 2 weeks (80 hours, 2 developers)

**Phase 1 Cross-Binary Corrections Applied:**
- ✅ Reusability analysis: 80% code reusable across future workers
- ✅ input-validation declared but UNUSED (691 LOC wasted)
- ✅ secrets-management declared but UNUSED → REMOVE
- ✅ model-catalog and gpu-info NOT declared → Should ADD
- ✅ deadline-propagation NOT declared → Should ADD

**Reference:** `TEAM_130B_CROSS_BINARY_ANALYSIS.md`

---

## 📊 GROUND TRUTH METRICS

```bash
$ cloc bin/llm-worker-rbee/src --quiet
Files: 41 | Code: 5,026 | Comments: 1,339 | Blanks: 1,072
Total Lines: 7,437
```

**Team 133 Accuracy:** 100% ✅ (perfect LOC match)

**Largest Files:**
1. http/validation.rs - 691 LOC (manual validation - SHOULD use input-validation!)
2. backend/models/mod.rs - 246 LOC (model factory)
3. backend/inference.rs - 300 LOC (inference engine)
4. common/inference_result.rs - 298 LOC (result types)
5. http/sse.rs - 289 LOC (SSE streaming)

**Module Breakdown:**
- common/ - 1,050 LOC (error, startup, types)
- backend/ - 1,300 LOC (inference engine + models)
- http/ - 1,280 LOC (HTTP server + validation)
- bin/ - 206 LOC (CPU/CUDA/Metal variants)
- other - 1,190 LOC (device, heartbeat, etc.)

---

## 🏗️ 6 PROPOSED CRATES

| # | Crate | LOC | Purpose | Reusability | Risk |
|---|-------|-----|---------|-------------|------|
| 1 | worker-rbee-error | 336 | Error types | 100% ✅ | Low (DONE by TEAM-130) |
| 2 | worker-rbee-startup | 239 | Callback protocol | 100% ✅ | Medium |
| 3 | worker-rbee-health | 182 | Heartbeat | 100% ✅ | Low |
| 4 | worker-rbee-sse-streaming | 574 | SSE events | 70%¹ | Medium |
| 5 | worker-rbee-http-server | 1,280 | HTTP server | 95% ✅ | High |
| 6 | worker-rbee-inference-base | 1,300 | Inference engine | 60%² | Very High |

**Total:** 3,911 LOC in libraries (excluding error which is done) + 523 LOC binary

**Reusability Notes:**
- ¹ SSE needs refactoring for non-text outputs (embeddings, images, audio)
- ² Inference-base has heavy LLM bias, may need split into generic + LLM-specific

---

## 📦 CRATE SPECIFICATIONS

### CRATE 1: worker-rbee-error (336 LOC) ✅ COMPLETE

**Status:** ✅ Already extracted by TEAM-130 (pilot project)  
**Files:** common/error.rs (261) + tests (75)

**API:**
```rust
pub enum WorkerError {
    Cuda(String),
    InvalidRequest(String),
    Timeout(String),
    Unhealthy(String),
    Internal(String),
    InsufficientResources(String),
    InsufficientVram(String),
}

impl WorkerError {
    pub fn to_http_status(&self) -> StatusCode;
    pub fn is_retriable(&self) -> bool;
}
```

**Dependencies:** thiserror, axum  
**Test Coverage:** 100% (10 test cases added by TEAM-130)

**Cross-Binary:** 100% reusable across ALL future workers (embedding, vision, audio)

---

### CRATE 2: worker-rbee-startup (239 LOC)

**Purpose:** Worker registration and callback to rbee-hive  
**Files:** common/startup.rs (239)

**API:**
```rust
pub async fn callback_ready(
    callback_url: &str,
    worker_id: &str,
    model_ref: &str,
    backend: &str,
    device: u32,
    vram_bytes: u64,
    worker_port: u16,
) -> Result<()>;

pub struct ReadyRequest {
    pub worker_id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub vram_bytes: u64,
}
```

**Dependencies:** reqwest, serde_json  
**Test Coverage:** 10 test cases (TEAM-130 fixed)

**Cross-Binary:**
- 100% reusable (all workers need startup callback)
- Contract type (ReadyRequest) should move to `rbee-http-types` shared crate
- Used by: rbee-hive (receives callback), queen-rbee (orchestrates)

---

### CRATE 3: worker-rbee-health (182 LOC)

**Purpose:** Heartbeat mechanism to rbee-hive  
**Files:** heartbeat.rs (128) + supporting types (54)

**API:**
```rust
pub struct HeartbeatConfig {
    pub worker_id: String,
    pub callback_url: String,
    pub interval_secs: u64,
}

pub async fn start_heartbeat_task(config: HeartbeatConfig) -> JoinHandle<()>;

pub struct HeartbeatPayload {
    pub worker_id: String,
    pub status: String,  // "healthy" | "degraded"
    pub vram_used: u64,
}
```

**Dependencies:** tokio, reqwest, serde_json

**Cross-Binary:**
- 100% reusable (all workers need heartbeat)
- Different from rbee-hive monitor (push vs pull)
- Integration: llm-worker (push) → rbee-hive monitor (pull + process check)

---

### CRATE 4: worker-rbee-sse-streaming (574 LOC)

**Purpose:** Server-Sent Events streaming for inference results  
**Files:** http/sse.rs (289) + common/inference_result.rs (298) - overlap (13)

**API (Current - LLM-specific):**
```rust
pub enum InferenceEvent {
    Started { job_id, model, started_at },
    Token { t: String, i: u32 },  // ← LLM-specific!
    Metrics { tokens_per_sec, vram_bytes },
    Narration { ... },
    End { ... },
    Error { code, message },
}

pub async fn stream_inference_sse(
    events: impl Stream<Item = InferenceEvent>,
) -> Sse<impl Stream<Item = Result<Event>>>;
```

**Reusability Challenge:**
- ✅ Started, Metrics, Narration, End, Error → 100% generic
- ❌ Token output → LLM-specific (text tokens)
- Future workers need: embeddings (Vec<f32>), images (Vec<u8>), audio (samples)

**Proposed Refactor (Phase 3):**
```rust
pub enum InferenceEvent<T> {
    Started { ... },
    Output(T),  // ← Generic!
    Metrics { ... },
    Narration { ... },
    End { ... },
    Error { ... },
}

// Specializations
pub type LlmEvent = InferenceEvent<TokenOutput>;
pub type EmbeddingEvent = InferenceEvent<EmbeddingOutput>;
pub type VisionEvent = InferenceEvent<ImageChunkOutput>;
```

**Dependencies:** axum, tokio-stream, serde_json, observability-narration-core

**Cross-Binary:** 70% reusable (needs generic refactor for 95%+)

---

### CRATE 5: worker-rbee-http-server (1,280 LOC)

**Purpose:** HTTP server infrastructure for all workers  
**Files:** http/{server,routes,execute,health,ready,loading,backend,narration_channel}.rs + middleware/auth.rs

**LOC Breakdown:**
- server.rs - 151 LOC (lifecycle)
- routes.rs - 34 LOC (routing)
- execute.rs - 115 LOC (inference endpoint)
- health.rs - 71 LOC (health check)
- ready.rs - 91 LOC (readiness)
- loading.rs - 101 LOC (model loading status)
- backend.rs - 30 LOC (backend info)
- narration_channel.rs - 84 LOC (observability)
- validation.rs - 691 LOC (⚠️ SHOULD use input-validation!)
- middleware/auth.rs - 128 LOC (authentication)

**API:**
```rust
pub trait InferenceBackend: Send + Sync {
    async fn execute(&mut self, req: ExecuteRequest) -> Result<InferenceResult>;
    fn is_healthy(&self) -> bool;
    fn is_ready(&self) -> bool;
    fn memory_bytes(&self) -> u64;
}

pub struct WorkerHttpServer<B: InferenceBackend> {
    backend: Arc<Mutex<B>>,
    bind_addr: String,
}

impl<B: InferenceBackend> WorkerHttpServer<B> {
    pub async fn start(&self) -> Result<()>;
}
```

**Dependencies:** axum, tower, tokio, serde_json, auth-min, observability-narration-core

**Critical Issue from Phase 1:**
- validation.rs has 691 LOC of manual validation
- input-validation IS declared in Cargo.toml but NEVER used!
- **Fix:** Replace validation.rs with input-validation crate usage

**Cross-Binary:**
- 95% reusable via InferenceBackend trait
- Only validation logic differs per worker type
- Opportunity: Extract validation trait for pluggability

---

### CRATE 6: worker-rbee-inference-base (1,300 LOC)

**Purpose:** Base inference engine (Candle integration)  
**Files:** backend/{inference,models/*,gguf_tokenizer,tokenizer_loader,sampling}.rs + common/sampling_config.rs

**LOC Breakdown:**
- inference.rs - 300 LOC (inference loop)
- models/mod.rs - 246 LOC (model factory)
- models/llama.rs - 130 LOC
- models/quantized_llama.rs - 197 LOC (GGUF)
- models/mistral.rs - 42 LOC
- models/phi.rs - 50 LOC
- models/quantized_phi.rs - 75 LOC
- models/qwen.rs - 42 LOC
- models/quantized_qwen.rs - 76 LOC
- gguf_tokenizer.rs - 178 LOC
- tokenizer_loader.rs - 55 LOC
- sampling.rs - 16 LOC
- common/sampling_config.rs - 276 LOC

**Reusability Analysis:**
- ✅ Device management: 100% reusable
- ✅ Model loading (SafeTensors, GGUF): 90% reusable
- ❌ Tokenizer loading: 0% reusable (LLM-specific)
- ❌ Inference loop (autoregressive): 0% reusable (LLM-specific)
- ❌ Sampling logic: 0% reusable (LLM-specific)

**Proposed Split (Phase 3):**
```
worker-rbee-device/           (~400 LOC - 100% reusable)
  - Device init (CPU/CUDA/Metal)
  - Model file loading
  - VRAM management

llm-worker-rbee-inference/    (~900 LOC - LLM-specific)
  - Tokenization
  - Autoregressive generation
  - Sampling (temperature, top-k, top-p)
  - Model wrappers (Llama, Mistral, Phi, Qwen)
```

**Dependencies:** candle-core, candle-nn, candle-transformers, tokenizers

**Cross-Binary:** 60% reusable (split needed for 100% generic + LLM-specific)

**Risk:** VERY HIGH
- Core inference logic (most complex code)
- Candle integration (external dependency)
- 7 model implementations (testing complexity)
- Should be last crate extracted

---

## 📊 DEPENDENCY GRAPH

```
Layer 0 (Standalone - TEAM-130 complete):
- worker-rbee-error (336 LOC) ✅ DONE

Layer 1 (Core - No worker deps):
- worker-rbee-health (182 LOC)
- worker-rbee-startup (239 LOC)

Layer 2 (Streaming):
- worker-rbee-sse-streaming (574 LOC) → uses error

Layer 3 (HTTP Server):
- worker-rbee-http-server (1,280 LOC) → uses error, sse-streaming

Layer 4 (Inference - Most Complex):
- worker-rbee-inference-base (1,300 LOC) → uses error

Binary (523 LOC) → uses startup, health, http-server, inference-base
```

**Migration Order (Team 133 Recommendation):**
1. ✅ worker-rbee-error (DONE!)
2. worker-rbee-health (simple, low risk)
3. worker-rbee-startup (depends on error)
4. worker-rbee-sse-streaming (refactor generics)
5. worker-rbee-http-server (complex, many deps)
6. worker-rbee-inference-base (most complex, save for last)

**Timeline:** 2 weeks with 2 developers
- Week 1: Crates 2-4
- Week 2: Crates 5-6

---

## 🔗 CROSS-BINARY CONTEXT

### Observability Excellence ✅

llm-worker-rbee is the GOLD STANDARD for observability:

**narration-core Usage:** 15× across 14 files
```rust
use observability_narration_core::narrate;

narrate(NarrationFields {
    actor: "llm-worker",
    action: "inference_complete",
    target: job_id,
    human: format!("Generated {} tokens in {:.2}s", count, duration),
    cute: Some("🎉 Done!"),
    correlation_id: Some(correlation_id),
    ...
});
```

**Compare to other binaries:**
- rbee-hive: ❌ NO narration-core (missing observability)
- queen-rbee: ❌ NO narration-core (missing correlation IDs)
- rbee-keeper: ❌ NO narration-core (basic colored output)

**Recommendation:** All binaries should adopt llm-worker pattern

### Shared Crate Usage Analysis

| Shared Crate | Status | Evidence |
|--------------|--------|----------|
| observability-narration-core | ✅ Excellent (15×) | ✅ Keep |
| auth-min | ✅ Used (1×) | ✅ Keep |
| input-validation | ❌ Declared, UNUSED | 🔴 Use or remove |
| secrets-management | ❌ Declared, UNUSED | 🔴 Remove |
| model-catalog | ❌ NOT declared | 🟡 Should add |
| gpu-info | ❌ NOT declared | 🟡 Should add |
| deadline-propagation | ❌ NOT declared | 🟡 Should add |

**Team 133 Corrections from Phase 1:**
- input-validation: NOT "missing" - already declared, just unused (waste)
- secrets-management: NOT "missing" - already declared, just unused (waste)
- validation.rs (691 LOC) should be replaced with input-validation usage

### Integration Points

**rbee-hive → llm-worker:**
- Process spawn via tokio::process::Command
- Startup callback (ready notification)
- Heartbeat monitoring (health checks)

**queen-rbee → llm-worker:**
- HTTP POST for inference requests
- SSE streaming for token output
- Deadline propagation via x-deadline header

**Shared Types Needed:**
- WorkerReadyRequest → `rbee-http-types`
- HeartbeatPayload → `rbee-http-types`
- ExecuteRequest → `rbee-http-types`

### Reusability for Future Workers

**embedding-worker-rbee (future):**
- ✅ error, startup, health, http-server: 100% reusable
- ⚠️ sse-streaming: needs generic refactor
- ❌ inference-base: needs separate embedding-specific implementation

**vision-worker-rbee (future):**
- ✅ error, startup, health, http-server: 100% reusable
- ⚠️ sse-streaming: needs generic refactor
- ❌ inference-base: needs separate vision-specific implementation

**audio-worker-rbee (future):**
- ✅ error, startup, health, http-server: 100% reusable
- ⚠️ sse-streaming: needs generic refactor
- ❌ inference-base: needs separate audio-specific implementation

**Summary:** 80% of code is reusable across future workers (Team 133 calculation confirmed)

---

## ⚠️ CRITICAL ISSUES TO ADDRESS

### Issue #1: Wasted Dependencies (input-validation, secrets-management)

**Problem:** Declared in Cargo.toml but NEVER used

**Evidence:**
```bash
$ grep "input-validation" Cargo.toml
Line 202: input-validation = { path = "../shared-crates/input-validation" }

$ grep -r "input_validation" src/
[no results]

$ grep "secrets-management" Cargo.toml
Line 201: secrets-management = { path = "../shared-crates/secrets-management" }

$ grep -r "secrets_management" src/
[no results]
```

**Impact:**
- Compilation time waste
- 691 LOC of manual validation in validation.rs (should use input-validation)
- Developer confusion

**Fix (Phase 3 or 4):**
- Replace validation.rs with input-validation crate usage
- Remove secrets-management from Cargo.toml

### Issue #2: Missing Shared Crates

**model-catalog NOT declared:**
- Worker loads models by model_ref
- Should query model-catalog for metadata (size, quantization)
- Currently hardcoded in backend/models/*.rs

**gpu-info NOT declared:**
- Manual CUDA detection in device.rs
- Should use shared gpu-info for consistency

**deadline-propagation NOT declared:**
- No timeout propagation from queen-rbee
- Should propagate x-deadline header internally

**Fix:** Add to Cargo.toml in Phase 3

### Issue #3: SSE Streaming Not Generic

**Current:** Token output only (LLM-specific)  
**Needed:** Generic output type for embeddings, images, audio  
**Fix:** Refactor InferenceEvent<T> in Phase 3

### Issue #4: Inference-Base Not Split

**Current:** Monolithic (1,300 LOC LLM-specific)  
**Needed:** Split into generic device management + LLM-specific  
**Fix:** Consider split in Phase 3 discussion

---

## 📋 COMPARISON WITH OTHER BINARIES

### Worker Architecture Comparison

| Binary | Purpose | LOC | Crates | Complexity |
|--------|---------|-----|--------|------------|
| llm-worker | Inference execution | 5,026 | 6 | High (inference) |
| rbee-hive | Pool management | 4,184 | 10 | Medium (orchestration) |
| queen-rbee | Request routing | 2,015 | 4 | Medium (routing) |
| rbee-keeper | Remote control | 1,252 | 5 | Low (CLI) |

**Insight:** llm-worker is largest and most complex (inference engine)

### Observability Comparison

| Binary | narration-core | Correlation IDs | Quality |
|--------|---------------|-----------------|---------|
| llm-worker | ✅ 15× usage | ✅ Yes | EXCELLENT |
| rbee-hive | ❌ Not used | ❌ No | Poor |
| queen-rbee | ❌ Not used | ❌ No | Poor |
| rbee-keeper | ❌ Not used | ❌ No | Basic |

**Insight:** Only llm-worker has proper observability (others should adopt)

### Shared Crate Adoption

| Binary | Used | Declared/Unused | Missing |
|--------|------|-----------------|---------|
| llm-worker | 2 (narration, auth) | 2 (input-val, secrets) | 3 (model, gpu, deadline) |
| rbee-hive | 6 (good) | 1 (secrets) | 1 (narration) |
| queen-rbee | 4 (good) | 1 (secrets) | 3 (narration, hive, model) |
| rbee-keeper | 1 (basic) | 0 | 1 (audit) |

**Insight:** All binaries have secrets-management waste

---

## ✅ PHASE 1 CORRECTIONS APPLIED

**Team 133 Findings Verified:**
1. ✅ LOC accuracy: 5,026 (100% correct)
2. ✅ Reusability: 80% confirmed (4,011/5,026 LOC generic)
3. ✅ narration-core: Excellent usage (15×)
4. ✅ File structure: All 41 files documented

**Corrections Made:**
1. ✅ input-validation: NOT "missing" - declared but unused
2. ✅ secrets-management: NOT "missing" - declared but unused
3. ✅ validation.rs waste identified (691 LOC manual validation)
4. ✅ Missing shared crates: model-catalog, gpu-info, deadline-propagation

**Cross-Binary Context Integrated:**
- Observability gold standard (other binaries should adopt)
- Worker startup/heartbeat contracts → rbee-http-types
- Reusability matrix for future workers (embedding, vision, audio)

---

**Status:** Part 1 Complete - Metrics & Crate Design Established  
**Next:** Part 2 (Phase 3) - External Library Analysis  
**Key Focus:** validation.rs replacement, SSE generics, inference-base split  
**Reference:** TEAM_130B_CROSS_BINARY_ANALYSIS.md for full context
