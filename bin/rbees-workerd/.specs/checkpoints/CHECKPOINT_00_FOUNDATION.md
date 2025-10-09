# CHECKPOINT 0: Foundation Setup (HTTP Server + Project Structure)

**Phase:** 0 - Foundation  
**Component:** Project Setup, HTTP Server, Worker Crates Integration  
**Tolerance:** N/A (Setup validation)  
**Critical Level:** 🔴 CRITICAL - Must be correct before model implementation

**Created by:** TEAM-000

---

## Purpose

Validate that the foundational infrastructure is correctly set up before implementing the Llama-2 model. This checkpoint ensures:
1. HTTP server works (using worker-http)
2. Project structure is correct
3. All worker crates are integrated
4. Stub endpoints respond correctly

**This is NOT about the model - it's about the infrastructure around it.**

---

## When to Check

- **Location:** Before implementing any model code
- **Input:** HTTP requests to stub server
- **Timing:** Week 1, Day 1 - Before Checkpoint 1

---

## Project Structure

### Directory Layout

```
llorch-candled/
├── Cargo.toml                          # Dependencies (worker-crates + ndarray + candle-kernels)
├── src/
│   ├── main.rs                         # HTTP server entry point
│   │                                   # IMPORTS: worker-http, worker-common
│   ├── lib.rs                          # Library exports
│   ├── backend/
│   │   ├── mod.rs                      # Backend module
│   │   └── candle_backend.rs           # CandleInferenceBackend (implements InferenceBackend)
│   │                                   # IMPORTS: worker-http, worker-common, worker-tokenizer
│   ├── model/
│   │   ├── mod.rs                      # Model module
│   │   └── llama2.rs                   # Llama2Model struct (stub for now)
│   ├── layers/
│   │   ├── mod.rs                      # Layers module
│   │   ├── rms_norm.rs                 # RMSNorm (Checkpoint 1)
│   │   ├── rope.rs                     # RoPE (Checkpoint 1B)
│   │   ├── embedding.rs                # Embeddings
│   │   ├── attention/                  # Attention module (Checkpoints 2, 4, 5)
│   │   │   ├── mod.rs                  # Attention orchestration
│   │   │   ├── qkv.rs                  # Separate Q, K, V projection (Checkpoint 2)
│   │   │   ├── scores.rs               # Attention scores (Checkpoint 4)
│   │   │   └── output.rs               # Attention output (Checkpoint 5)
│   │   ├── swiglu.rs                   # SwiGLU FFN (Checkpoint 6)
│   │   └── transformer.rs              # TransformerBlock (Checkpoint 7)
│   ├── cache/                          # KV Cache module (Checkpoint 3)
│   │   ├── mod.rs                      # Cache module exports
│   │   └── kv_cache.rs                 # Simple KV cache for Llama-2
│   ├── tensor/
│   │   ├── mod.rs                      # Tensor module
│   │   └── ops.rs                      # CPU tensor operations (ndarray)
│   └── error.rs                        # Error types
└── tests/
    ├── checkpoint_00_foundation.rs     # This checkpoint
    ├── checkpoint_01_rms_norm.rs
    ├── checkpoint_01b_rope.rs
    ├── checkpoint_02_qkv.rs
    └── ...
```

---

## Worker Crates Import Map

### main.rs
```rust
use worker_http::{create_router, HttpServer};
use worker_common::startup;
use llorch_candled::backend::CandleInferenceBackend;
```

### backend/candle_backend.rs
```rust
use worker_http::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};
use worker_tokenizer::Tokenizer;  // After model implementation
```

### model/llama2.rs
```rust
use worker_models::LlamaConfig;  // If available
use worker_common::SamplingConfig;
```

### layers/attention/mod.rs
```rust
use crate::cache::KVCache;  // Internal import, not worker-crate
```

---

## Validation Checklist

### ✓ Cargo.toml Setup

- [ ] Package name: `llorch-candled`
- [ ] Edition: 2021
- [ ] All worker-crates dependencies added:
  - [ ] worker-common
  - [ ] worker-http
  - [ ] worker-tokenizer
  - [ ] worker-models
  - [ ] worker-gguf
- [ ] Candle integration (optional):
  - [ ] candle-kernels (feature-gated)
  - [ ] cudarc (feature-gated)
- [ ] CPU tensor dependencies:
  - [ ] ndarray
- [ ] Async runtime:
  - [ ] tokio (with minimal features)
  - [ ] async-trait
- [ ] Utilities:
  - [ ] anyhow, thiserror
  - [ ] tracing, tracing-subscriber
  - [ ] clap, serde

### ✓ Project Structure Created

- [ ] All directories exist (src/backend, src/model, src/layers, src/cache, src/tensor)
- [ ] Attention subdirectory created (src/layers/attention/)
- [ ] Cache directory created (src/cache/) - **top-level for future growth**
- [ ] All mod.rs files created
- [ ] Stub files created for future implementation
- [ ] tests/ directory exists

**Notes:**
- Cache is top-level to signal future optimization work
- **Single-threaded is CRITICAL** for performance - use `flavor = "current_thread"` in tokio

### ✓ HTTP Server Wiring (CRITICAL)

**Purpose:** Wire up HTTP server BEFORE model implementation

This is NOT a CLI tool. This is an HTTP server that:
1. Loads model on startup
2. Starts HTTP server
3. Waits for inference requests
4. Processes requests via HTTP endpoints

**Architecture:**
```
main.rs
  ↓
Load CandleInferenceBackend (stub for now)
  ↓
Call pool manager callback (worker ready)
  ↓
Start HTTP server (worker-http)
  ↓
Listen for requests on /health and /execute
  ↓
Process requests via InferenceBackend trait
```

**Orchestration Flow:**
```
pool-managerd
  ↓ (spawns process)
llorch-candled --worker-id=... --model=... --port=8080 --callback-url=...
  ↓ (loads model)
CandleInferenceBackend::load()
  ↓ (calls back)
POST http://pool-managerd/workers/{worker_id}/ready
  ↓ (starts server)
HTTP server listening on :8080
  ↓ (waits for requests)
orchestratord → POST http://worker:8080/execute
  ↓ (processes)
backend.execute(prompt, config)
  ↓ (returns)
SSE stream with tokens
```

---

## HTTP Endpoint Testing

### Test 1: Health Endpoint

**Request:**
```bash
curl http://localhost:8080/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vram_bytes": 0,
  "memory_type": "cpu"
}
```

**Validation:**
- [ ] Endpoint responds
- [ ] Status code: 200
- [ ] JSON format correct
- [ ] vram_bytes is 0 for CPU (or >0 for CUDA)

### Test 2: Execute Endpoint (Stub)

**Request:**
```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-123",
    "prompt": "Hello",
    "max_tokens": 10,
    "temperature": 0.7
  }'
```

**Expected Response:** SSE stream
```
event: started
data: {"job_id":"test-123","model":"model","started_at":"0"}

event: token
data: {"t":"STUB","i":0}

event: token
data: {"t":"LLAMA2","i":1}

event: token
data: {"t":"RESPONSE","i":2}

event: end
data: {"tokens_out":3,"decode_time_ms":0,"stop_reason":"MAX_TOKENS","stop_sequence_matched":null}
```

**Validation:**
- [ ] Endpoint responds
- [ ] SSE stream format correct
- [ ] Returns stub tokens ("STUB", "LLAMA2", "RESPONSE")
- [ ] All event types present (started, token, end)

### Test 3: Server Startup

**Command:**
```bash
cargo run -- \
  --worker-id test-worker \
  --model test.gguf \
  --port 8080 \
  --callback-url http://localhost:9999
```

**Expected Output:**
```json
{"level":"INFO","message":"Candle worker starting","worker_id":"test-worker","model":"test.gguf","port":8080}
{"level":"INFO","message":"Loading Llama-2 model...","model":"test.gguf"}
{"level":"INFO","message":"Model loaded successfully"}
{"level":"INFO","message":"Test mode: skipping pool manager callback"}
{"level":"INFO","message":"Worker ready, starting HTTP server"}
```

**Validation:**
- [ ] Server starts without errors
- [ ] Logs in JSON format
- [ ] Listens on specified port
- [ ] Test mode detected (localhost:9999)

---

## Compilation Tests

### Test 1: Build Succeeds

```bash
cargo build
```

**Validation:**
- [ ] No compilation errors
- [ ] All dependencies resolved
- [ ] Binary created

### Test 2: Build with CUDA Feature

```bash
cargo build --features cuda
```

**Validation:**
- [ ] Compiles with CUDA feature
- [ ] candle-kernels dependency resolved
- [ ] cudarc dependency resolved

### Test 3: Tests Compile

```bash
cargo test --no-run
```

**Validation:**
- [ ] Test files compile
- [ ] No missing test dependencies

### Test 4: Clippy Clean

```bash
cargo clippy -- -D warnings
```

**Validation:**
- [ ] No clippy warnings
- [ ] Code follows Rust idioms

---

## Integration Test

**File:** `tests/checkpoint_00_foundation.rs`

```rust
use llorch_candled::backend::CandleInferenceBackend;
use worker_common::SamplingConfig;
use worker_http::InferenceBackend;

#[tokio::test]
async fn test_backend_stub_works() {
    // Create backend
    let backend = CandleInferenceBackend::load("test.gguf").unwrap();
    
    // Test execute returns stub data
    let config = SamplingConfig::default();
    let result = backend.execute("Hello", &config).await.unwrap();
    
    // Validate stub response
    assert_eq!(result.tokens.len(), 3);
    assert_eq!(result.tokens[0], "STUB");
    assert_eq!(result.tokens[1], "LLAMA2");
    assert_eq!(result.tokens[2], "RESPONSE");
}

#[test]
fn test_backend_is_healthy() {
    let backend = CandleInferenceBackend::load("test.gguf").unwrap();
    assert!(backend.is_healthy());
}

#[test]
fn test_backend_memory_type() {
    let backend = CandleInferenceBackend::load("test.gguf").unwrap();
    
    #[cfg(feature = "cuda")]
    assert_eq!(backend.memory_architecture(), "cuda");
    
    #[cfg(not(feature = "cuda"))]
    assert_eq!(backend.memory_architecture(), "cpu");
}
```

**Validation:**
- [ ] All tests pass
- [ ] Stub backend works
- [ ] InferenceBackend trait implemented correctly

---

## Success Criteria

### Minimum Requirements

- ✅ Project compiles without errors
- ✅ All worker-crates dependencies integrated
- ✅ HTTP server starts and responds
- ✅ GET /health returns correct response
- ✅ POST /execute returns stub response
- ✅ All tests pass
- ✅ Compiles with and without CUDA feature

### Code Quality

- ✅ No clippy warnings
- ✅ Proper module structure
- ✅ Clean separation of concerns
- ✅ Logging configured
- ✅ TEAM-000 signatures in place

### Documentation

- ✅ Cargo.toml documented
- ✅ Module purposes clear
- ✅ Checkpoint format followed

---

## Common Failures

### ❌ Compilation Errors

**Symptom:** cargo build fails  
**Cause:** Missing dependencies or wrong versions  
**Fix:** Check Cargo.toml matches this checkpoint spec

### ❌ HTTP Server Won't Start

**Symptom:** Server panics or port in use  
**Cause:** Port already bound or missing async runtime  
**Fix:** Check port availability, verify tokio runtime

### ❌ Worker-crates Not Found

**Symptom:** "crate not found" errors  
**Cause:** Wrong path to worker-crates  
**Fix:** Verify path = "../worker-crates/worker-*"

### ❌ Candle Kernels Not Found

**Symptom:** "crate not found" with cuda feature  
**Cause:** Wrong path to candle reference  
**Fix:** Verify path = "../../reference/candle/candle-kernels"

---

## Next Steps

If this checkpoint **PASSES**:
- ✅ Foundation is solid
- ✅ HTTP server works
- ✅ Ready to implement model
- ✅ Proceed to Checkpoint 1 (RMSNorm)

If this checkpoint **FAILS**:
- ❌ Fix foundation issues first
- ❌ Do not proceed to model implementation
- ❌ HTTP server must work before model work

---

## Notes

- This checkpoint is about **infrastructure**, not the model
- HTTP server is **already done** (worker-http) - just integrate it
- Stub responses are expected - real inference comes later
- Focus: Project structure + HTTP integration + Worker crates + Candle setup
- Candle kernels are OPTIONAL - CPU path must work first

---

**Status:** ⬜ Not Started  
**Estimated Time:** 4-6 hours  
**Blocking:** Must pass before Checkpoint 1

---

Built by TEAM-000 🌊 (Foundation of them all)
