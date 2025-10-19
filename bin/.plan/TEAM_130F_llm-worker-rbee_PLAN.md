# TEAM-130F: llm-worker-rbee BINARY + CRATES PLAN

**Phase:** Phase 3 Implementation Planning  
**Date:** 2025-10-19  
**Team:** TEAM-130F  
**Status:** 📋 PLAN (Future Architecture)

---

## 🎯 MISSION

Define **PLANNED** architecture for llm-worker-rbee after Phase 3 consolidation.

**Key Changes:**
- ✅ Replace manual validation (691 LOC) with input-validation crate
- ✅ Use shared crates (rbee-types, rbee-http-client)
- ✅ Remove unused dependencies (secrets-management)
- ✅ Keep inference-base in binary (NOT reusable)

---

## 📊 METRICS (PLANNED)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total LOC** | 5,026 | ~4,435 | **-591 LOC** |
| **Files** | 41 | 40 | -1 file |
| **Shared crate deps** | 3 | 4 | +1 dep (input-validation) |

**LOC Breakdown:**
- Remove manual validation: -691 LOC (validation.rs)
- Add input-validation usage: +50 LOC
- Add rbee-types usage: +30 LOC
- Add rbee-http-client usage: +20 LOC
- **Net savings: -591 LOC**

---

## 📦 INTERNAL CRATES (Within Binary)

### 1. inference-engine (~2,500 LOC)
**Location:** `src/backend/`  
**Purpose:** Model loading, tokenization, inference execution  
**Why NOT shared:** Worker-specific, tightly coupled to Candle

```
src/backend/
├─ mod.rs                    (~100 LOC) - Backend abstraction
├─ inference.rs              (~400 LOC) - Inference execution
├─ sampling.rs               (~200 LOC) - Sampling logic
├─ tokenizer_loader.rs       (~150 LOC) - Tokenizer loading
├─ gguf_tokenizer.rs         (~300 LOC) - GGUF tokenizer
└─ models/
   ├─ mod.rs                 (~50 LOC)
   ├─ llama.rs               (~400 LOC) - Llama model
   ├─ mistral.rs             (~300 LOC) - Mistral model
   ├─ phi.rs                 (~200 LOC) - Phi model
   ├─ qwen.rs                (~200 LOC) - Qwen model
   └─ quantized_*.rs         (~400 LOC) - Quantized variants
```

**Dependencies:** `candle-core`, `candle-nn`, `candle-transformers`, `tokenizers`

---

### 2. http-server (~900 LOC)
**Location:** `src/http/`  
**Purpose:** Axum HTTP server with inference endpoints  
**Why NOT shared:** Worker-specific API

```
src/http/
├─ mod.rs                    (~30 LOC)
├─ server.rs                 (~150 LOC) - Axum server
├─ routes.rs                 (~80 LOC) - Route definitions
├─ health.rs                 (~40 LOC) - Health endpoint
├─ execute.rs                (~200 LOC) - Inference execution
├─ loading.rs                (~100 LOC) - Loading progress SSE
├─ ready.rs                  (~80 LOC) - Ready callback
├─ backend.rs                (~60 LOC) - Backend info
├─ sse.rs                    (~100 LOC) - SSE streaming
├─ narration_channel.rs      (~80 LOC) - Narration SSE
└─ middleware/
   ├─ mod.rs                 (~5 LOC)
   └─ auth.rs                (~50 LOC) - Auth middleware
```

**Changes:**
- Use `rbee-http-client` for callback to rbee-hive
- Use `rbee-types` for request/response types
- Use `input-validation` for request validation

---

### 3. validation (~50 LOC) **REPLACED**
**Location:** `src/http/validation.rs` (DELETE 691 LOC, ADD 50 LOC)  
**Purpose:** Request validation using input-validation crate  
**Why NOT shared:** Worker-specific validation rules

**BEFORE (691 LOC):**
```rust
pub fn validate_execute_request(req: &ExecuteRequest) -> Result<(), Vec<FieldError>> {
    let mut errors = Vec::new();
    
    // Job ID validation (manual - 50 LOC)
    if req.job_id.is_empty() {
        errors.push(FieldError { ... });
    }
    
    // Prompt validation (manual - 80 LOC)
    if req.prompt.is_empty() || req.prompt.len() > 32768 {
        errors.push(FieldError { ... });
    }
    
    // ... 561 more lines of manual validation
}
```

**AFTER (50 LOC):**
```rust
use input_validation::{validate_identifier, validate_prompt, validate_range};

pub fn validate_execute_request(req: &ExecuteRequest) -> Result<(), Vec<FieldError>> {
    let mut errors = Vec::new();
    
    // Job ID validation
    if let Err(e) = validate_identifier(&req.job_id, 64) {
        errors.push(FieldError::from(e));
    }
    
    // Prompt validation
    if let Err(e) = validate_prompt(&req.prompt) {
        errors.push(FieldError::from(e));
    }
    
    // Max tokens validation
    if let Err(e) = validate_range(req.max_tokens, 1, 2048, "max_tokens") {
        errors.push(FieldError::from(e));
    }
    
    // Temperature validation
    if let Err(e) = validate_range(req.temperature, 0.0, 2.0, "temperature") {
        errors.push(FieldError::from(e));
    }
    
    // ... ~40 LOC total
    
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
```

**Savings:** 691 - 50 = **641 LOC**

---

### 4. device-detection (~100 LOC)
**Location:** `src/device.rs`  
**Purpose:** Device detection (CPU/CUDA/Metal)  
**Why NOT shared:** Worker-specific, backend-dependent

---

### 5. error-types (~336 LOC)
**Location:** `src/error.rs` + `src/common/error.rs`  
**Purpose:** Worker-specific error types  
**Why NOT shared:** Worker-specific error contexts

**Note:** Already extracted as `worker-rbee-error` in TEAM-130D analysis

---

### 6. heartbeat (~80 LOC)
**Location:** `src/heartbeat.rs`  
**Purpose:** Heartbeat to rbee-hive  
**Why NOT shared:** Worker-specific

**Changes:**
- Use `rbee-http-client` for heartbeat HTTP calls

---

### 7. narration (~150 LOC)
**Location:** `src/narration.rs` + `src/http/narration_channel.rs`  
**Purpose:** Narration integration (uses narration-core)  
**Why NOT shared:** Worker-specific narration points

---

### 8. startup (~100 LOC)
**Location:** `src/common/startup.rs`  
**Purpose:** Worker startup and ready callback  
**Why NOT shared:** Worker-specific startup sequence

**Changes:**
- Use `rbee-http-client` for ready callback

---

## 🔗 DEPENDENCIES (PLANNED)

```toml
[dependencies]
# Phase 3 NEW: Shared crates
input-validation = { path = "../shared-crates/input-validation" }
rbee-http-client = { path = "../shared-crates/rbee-http-client" }
rbee-types = { path = "../shared-crates/rbee-types" }

# Existing: Shared crates
auth-min = { path = "../shared-crates/auth-min" }
observability-narration-core = { path = "../shared-crates/narration-core", features = ["axum", "cute-mode"] }

# REMOVED: secrets-management (unused)

# HTTP server
axum = "0.8"
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "cors"] }
futures = "0.3"
async-stream = "0.3"

# Candle integration
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"
candle-kernels = { version = "0.9", optional = true }
cudarc = { version = "0.11", optional = true, features = ["cuda-12050"] }
tokenizers = "0.15"
ndarray = "0.15"

# Async runtime
tokio = { version = "1", features = ["rt", "rt-multi-thread", "macros", "sync", "time", "signal"] }
tokio-stream = "0.1"
async-trait = "0.1"

# Utilities
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
clap = { version = "4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = "0.4"

[features]
default = ["cpu"]
cpu = []
cuda = ["candle-kernels", "cudarc", "candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
```

---

## 📋 BINARY STRUCTURE (PLANNED)

```
bin/llm-worker-rbee/
├─ src/
│  ├─ main.rs                    (~50 LOC) - Entry point
│  ├─ lib.rs                     (~80 LOC) - Library exports
│  ├─ bin/
│  │  ├─ cpu.rs                  (~30 LOC) - CPU binary
│  │  ├─ cuda.rs                 (~30 LOC) - CUDA binary
│  │  └─ metal.rs                (~30 LOC) - Metal binary
│  ├─ backend/
│  │  ├─ mod.rs                  (~100 LOC) - Backend abstraction
│  │  ├─ inference.rs            (~400 LOC) - Inference execution
│  │  ├─ sampling.rs             (~200 LOC) - Sampling logic
│  │  ├─ tokenizer_loader.rs    (~150 LOC) - Tokenizer loading
│  │  ├─ gguf_tokenizer.rs      (~300 LOC) - GGUF tokenizer
│  │  └─ models/
│  │     ├─ mod.rs               (~50 LOC)
│  │     ├─ llama.rs             (~400 LOC) - Llama model
│  │     ├─ mistral.rs           (~300 LOC) - Mistral model
│  │     ├─ phi.rs               (~200 LOC) - Phi model
│  │     ├─ qwen.rs              (~200 LOC) - Qwen model
│  │     ├─ quantized_llama.rs   (~150 LOC) - Quantized Llama
│  │     ├─ quantized_phi.rs     (~100 LOC) - Quantized Phi
│  │     └─ quantized_qwen.rs    (~150 LOC) - Quantized Qwen
│  ├─ http/
│  │  ├─ mod.rs                  (~30 LOC)
│  │  ├─ server.rs               (~150 LOC) - Axum server
│  │  ├─ routes.rs               (~80 LOC) - Route definitions
│  │  ├─ health.rs               (~40 LOC) - Health endpoint
│  │  ├─ execute.rs              (~200 LOC) - Inference execution
│  │  ├─ loading.rs              (~100 LOC) - Loading progress SSE
│  │  ├─ ready.rs                (~80 LOC) - Ready callback
│  │  ├─ backend.rs              (~60 LOC) - Backend info
│  │  ├─ validation.rs           (~50 LOC) - REPLACED: Use input-validation
│  │  ├─ sse.rs                  (~100 LOC) - SSE streaming
│  │  ├─ narration_channel.rs    (~80 LOC) - Narration SSE
│  │  └─ middleware/
│  │     ├─ mod.rs               (~5 LOC)
│  │     └─ auth.rs              (~50 LOC) - Auth middleware
│  ├─ common/
│  │  ├─ mod.rs                  (~20 LOC)
│  │  ├─ error.rs                (~200 LOC) - Error types
│  │  ├─ inference_result.rs     (~80 LOC) - Inference result
│  │  ├─ sampling_config.rs      (~100 LOC) - Sampling config
│  │  └─ startup.rs              (~100 LOC) - Startup logic
│  ├─ device.rs                  (~100 LOC) - Device detection
│  ├─ error.rs                   (~136 LOC) - Top-level error
│  ├─ heartbeat.rs               (~80 LOC) - Heartbeat
│  ├─ narration.rs               (~70 LOC) - Narration integration
│  └─ token_output_stream.rs     (~150 LOC) - Token streaming
├─ Cargo.toml
└─ README.md
```

**Removed Files:**
- ❌ None (validation.rs stays but is rewritten)

**Modified Files:**
- ⚠️ `http/validation.rs` (691 → 50 LOC)
- ⚠️ `common/startup.rs` (use rbee-http-client)
- ⚠️ `heartbeat.rs` (use rbee-http-client)

---

## 🔧 IMPLEMENTATION PLAN

### Day 1: Fix Validation (CRITICAL)

**Replace validation.rs:**
```rust
// DELETE 691 LOC of manual validation
// ADD 50 LOC using input-validation

use input_validation::{validate_identifier, validate_prompt, validate_range};

pub fn validate_execute_request(req: &ExecuteRequest) -> Result<(), Vec<FieldError>> {
    let mut errors = Vec::new();
    
    // Job ID validation
    if let Err(e) = validate_identifier(&req.job_id, 64) {
        errors.push(FieldError {
            field: "job_id".to_string(),
            message: e.to_string(),
        });
    }
    
    // Prompt validation
    if let Err(e) = validate_prompt(&req.prompt) {
        errors.push(FieldError {
            field: "prompt".to_string(),
            message: e.to_string(),
        });
    }
    
    // Max tokens validation
    if let Err(e) = validate_range(req.max_tokens, 1, 2048, "max_tokens") {
        errors.push(FieldError {
            field: "max_tokens".to_string(),
            message: e.to_string(),
        });
    }
    
    // Temperature validation
    if let Err(e) = validate_range(req.temperature, 0.0, 2.0, "temperature") {
        errors.push(FieldError {
            field: "temperature".to_string(),
            message: e.to_string(),
        });
    }
    
    // Top-p validation
    if let Err(e) = validate_range(req.top_p, 0.0, 1.0, "top_p") {
        errors.push(FieldError {
            field: "top_p".to_string(),
            message: e.to_string(),
        });
    }
    
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
```

**Update Cargo.toml:**
```toml
[dependencies]
# NEW: Phase 3 shared crates
input-validation = { path = "../shared-crates/input-validation" }

# REMOVE
# secrets-management = { path = "../shared-crates/secrets-management" }
```

**Test:**
```bash
cargo test --bin llm-worker-rbee -- validation
```

---

### Day 2: Integrate HTTP Client

**Update startup.rs (ready callback):**
```rust
// BEFORE
let client = reqwest::Client::new();
let response = client.post(callback_url).json(&payload).send().await?;

// AFTER
use rbee_http_client::RbeeHttpClient;
use rbee_types::ReadyRequest;

let client = RbeeHttpClient::new();
let payload = ReadyRequest {
    worker_id: worker_id.clone(),
    url: worker_url.clone(),
    model_ref: model_ref.clone(),
    backend: backend.clone(),
    device: device_id,
};
client.post_json(callback_url, &payload).await?;
```

**Update heartbeat.rs:**
```rust
// BEFORE
let client = reqwest::Client::new();
let response = client.post(&heartbeat_url).json(&heartbeat).send().await?;

// AFTER
use rbee_http_client::RbeeHttpClient;

let client = RbeeHttpClient::new();
client.post_json(&heartbeat_url, &heartbeat).await?;
```

---

### Day 3: Add Type Safety

**Update Cargo.toml:**
```toml
[dependencies]
rbee-types = { path = "../shared-crates/rbee-types" }
rbee-http-client = { path = "../shared-crates/rbee-http-client" }
```

**Update HTTP types:**
```rust
// Use shared types where applicable
use rbee_types::{WorkerState, ReadyRequest};

// Keep worker-specific types local
pub struct ExecuteRequest {
    pub job_id: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    // ... worker-specific fields
}
```

---

### Day 4: Testing

**Unit tests:**
```bash
cargo test --bin llm-worker-rbee
```

**Integration tests:**
```bash
# Test validation
curl -X POST http://localhost:8081/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"job_id":"","prompt":"test"}' # Should fail validation

# Test inference
curl -X POST http://localhost:8081/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test-123","prompt":"Hello","max_tokens":50}'
```

---

## ✅ ACCEPTANCE CRITERIA

1. ✅ validation.rs uses input-validation crate (691 → 50 LOC)
2. ✅ Uses `rbee-http-client` for callbacks and heartbeat
3. ✅ Uses `rbee-types` for shared types (WorkerState, ReadyRequest)
4. ✅ secrets-management dependency removed
5. ✅ All tests pass
6. ✅ Binary compiles without warnings
7. ✅ Validation works correctly (same behavior, less code)
8. ✅ Ready callback works (uses rbee-http-client)
9. ✅ Heartbeat works (uses rbee-http-client)

---

## 📝 CRITICAL NOTES

### inference-base Stays in Binary!

**From TEAM-130D:**
> "inference-base stays in BINARY (NOT reusable)"

**Why:**
- Tightly coupled to Candle
- Worker-specific inference logic
- Not generic enough for reuse

### Validation is the BIGGEST Win

**641 LOC savings** from replacing manual validation with input-validation crate.

This is the LARGEST single consolidation opportunity in the entire codebase.

### WorkerState is Shared

Only the `WorkerState` enum is shared (Loading/Idle/Busy).

Worker-specific types (ExecuteRequest, etc.) stay in the binary.

---

**Status:** 📋 PLAN COMPLETE  
**LOC Impact:** -591 LOC (5,026 → 4,435)  
**Critical Fix:** Replace manual validation with input-validation
