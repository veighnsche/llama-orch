# CHECKPOINT 0: Foundation Setup (HTTP Server + Project Structure)

**Phase:** 0 - Foundation  
**Component:** Project Setup, HTTP Server, Worker Crates Integration  
**Tolerance:** N/A (Setup validation)  
**Critical Level:** 🔴 CRITICAL - Must be correct before model implementation

---

## Purpose

Validate that the foundational infrastructure is correctly set up before implementing the GPT-2 model. This checkpoint ensures:
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
llorch-cpud/
├── Cargo.toml                          # Dependencies (worker-crates + ndarray)
├── src/
│   ├── main.rs                         # HTTP server entry point
│   │                                   # IMPORTS: worker-http, worker-common
│   ├── lib.rs                          # Library exports
│   ├── backend/
│   │   ├── mod.rs                      # Backend module
│   │   └── cpu_backend.rs              # CpuInferenceBackend (implements InferenceBackend)
│   │                                   # IMPORTS: worker-http, worker-common, worker-tokenizer
│   ├── model/
│   │   ├── mod.rs                      # Model module
│   │   ├── gpt2.rs                     # GPT2Model struct (stub for now)
│   │   │                               # IMPORTS: worker-models (GPTConfig)
│   │   └── config.rs                   # Model configuration
│   │                                   # IMPORTS: worker-models
│   ├── layers/
│   │   ├── mod.rs                      # Layers module
│   │   ├── layer_norm.rs               # LayerNorm (Checkpoint 1)
│   │   ├── embedding.rs                # Embeddings
│   │   ├── attention/                  # Attention module (Checkpoints 2, 4, 5)
│   │   │   ├── mod.rs                  # Attention orchestration
│   │   │   │                           # IMPORTS: crate::cache::KVCache
│   │   │   ├── qkv.rs                  # QKV projection (Checkpoint 2)
│   │   │   ├── scores.rs               # Attention scores (Checkpoint 4)
│   │   │   └── output.rs               # Attention output (Checkpoint 5)
│   │   ├── ffn.rs                      # FFN (Checkpoint 6)
│   │   └── transformer.rs              # TransformerBlock (Checkpoint 7)
│   ├── cache/                          # KV Cache module (Checkpoint 3)
│   │   ├── mod.rs                      # Cache module exports
│   │   └── kv_cache.rs                 # Simple KV cache for GPT-2 (room to grow)
│   ├── tensor/
│   │   ├── mod.rs                      # Tensor module
│   │   └── ops.rs                      # CPU tensor operations (ndarray)
│   └── error.rs                        # Error types
└── tests/
    ├── checkpoint_00_foundation.rs     # This checkpoint
    ├── checkpoint_01_layer_norm.rs
    ├── checkpoint_02_qkv.rs
    └── ...
```

---

## Worker Crates Import Map

### main.rs
```rust
use worker_http::{create_router, HttpServer};
use worker_common::startup;
use llorch_cpud::backend::CpuInferenceBackend;
```

### backend/cpu_backend.rs
```rust
use worker_http::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};
use worker_tokenizer::Tokenizer;  // After model implementation
```

### model/gpt2.rs
```rust
use worker_models::GPTConfig;
use worker_common::SamplingConfig;
```

### layers/attention/mod.rs
```rust
use crate::cache::KVCache;  // Internal import, not worker-crate
```

---

---

## Validation Checklist

### ✓ Cargo.toml Setup

- [ ] Package name: `llorch-cpud`
- [ ] Edition: 2021
- [ ] All worker-crates dependencies added:
  - [ ] worker-common
  - [ ] worker-http
  - [ ] worker-tokenizer
  - [ ] worker-models
- [ ] CPU tensor dependencies:
  - [ ] ndarray
  - [ ] ndarray-linalg
- [ ] Async runtime:
  - [ ] tokio (with "full" features)
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
- See `ATTENTION_MODULE_STRUCTURE.md` for detailed explanation of attention/ subdirectory structure.
- See `KV_CACHE_MODULE_ANALYSIS.md` for rationale on cache as top-level module.
- See `SINGLE_THREADED_ARCHITECTURE.md` for CRITICAL single-threaded requirement.
- **Cache is top-level** to signal future optimization work, but implementation stays simple for MVP.
- **Single-threaded is CRITICAL** for performance - use `flavor = "current_thread"` in tokio.

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
Load CpuInferenceBackend (stub for now)
  ↓
Call pool manager callback (worker ready)
  ↓
Start HTTP server (worker-http)
  ↓
Listen for requests on /health and /execute
  ↓
Process requests via InferenceBackend trait
```

**Key Point:** The HTTP server runs CONTINUOUSLY. It's not a one-shot CLI.

**Orchestration Flow:**
```
pool-managerd
  ↓ (spawns process)
llorch-cpud --worker-id=... --model=... --port=8080 --callback-url=...
  ↓ (loads model)
CpuInferenceBackend::load()
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

**Why This Matters:**
- ✅ Worker is spawned by pool-managerd (not run manually)
- ✅ CLI args are for orchestration, not user interaction
- ✅ Server runs forever until killed by pool-managerd
- ✅ Multiple workers can run on different ports
- ✅ Each worker handles one model, many requests

---

### ✓ CpuInferenceBackend Stub

**File:** `src/backend/cpu_backend.rs`

```rust
use async_trait::async_trait;
use worker_http::InferenceBackend;
use worker_common::{InferenceResult, SamplingConfig};
use anyhow::Result;

/// CPU inference backend
/// 
/// This implements the InferenceBackend trait from worker-http,
/// allowing the HTTP server to call our inference code.
/// 
/// For Checkpoint 0: Returns stub data
/// After Checkpoint 12: Returns real inference results
pub struct CpuInferenceBackend {
    model_path: String,
    // model: GPT2Model,  // Will be added later
    // tokenizer: Tokenizer,  // Will be added later
}

impl CpuInferenceBackend {
    /// Load model from disk
    /// 
    /// For Checkpoint 0: Just stores path
    /// After model implementation: Actually loads weights
    pub fn load(model_path: &str) -> Result<Self> {
        Ok(Self {
            model_path: model_path.to_string(),
        })
    }
    
    /// Get memory usage in bytes
    /// 
    /// For Checkpoint 0: Returns 0
    /// After model implementation: Returns actual memory
    pub fn memory_bytes(&self) -> u64 {
        0  // Stub: will calculate actual memory later
    }
    
    /// Get memory architecture type
    pub fn memory_architecture(&self) -> &str {
        "cpu"  // CPU worker uses CPU memory
    }
    
    /// Get worker type
    pub fn worker_type(&self) -> &str {
        "cpu"  // CPU worker type
    }
    
    /// Get worker capabilities
    pub fn capabilities(&self) -> Vec<&str> {
        vec!["text-gen"]  // Text generation capability
    }
}

#[async_trait]
impl InferenceBackend for CpuInferenceBackend {
    /// Execute inference
    /// 
    /// This is called by worker-http when POST /execute is hit.
    /// 
    /// For Checkpoint 0: Returns stub data
    /// After Checkpoint 12: Returns real inference
    async fn execute(
        &self,
        prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        // STUB: Return dummy response
        // TODO: Replace with real inference after Checkpoint 12
        Ok(InferenceResult::max_tokens(
            vec!["STUB".to_string(), "RESPONSE".to_string()],
            vec![1, 2],
            config.seed,
            0,
        ))
    }
    
    /// Cancel inference (not implemented for CPU)
    async fn cancel(&self, _job_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())  // CPU is fast, no cancellation needed
    }
    
    /// Get VRAM usage (always 0 for CPU)
    fn vram_usage(&self) -> u64 {
        0  // CPU worker, no VRAM
    }
    
    /// Check if backend is healthy
    fn is_healthy(&self) -> bool {
        true  // Always healthy for stub
    }
}
```

**Validation:**
- [ ] File compiles without errors
- [ ] Implements all InferenceBackend methods
- [ ] Returns stub data (not real inference yet)
- [ ] Comments explain stub vs real implementation

### ✓ Main.rs HTTP Server

**File:** `src/main.rs`

**Purpose:** HTTP server entry point - runs continuously, NOT a one-shot CLI

```rust
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use worker_common::startup;
use worker_http::{create_router, HttpServer};
use llorch_cpud::backend::CpuInferenceBackend;

/// CLI arguments for worker daemon
/// 
/// These are provided by pool-managerd when spawning the worker.
/// This is NOT a user-facing CLI - it's for orchestration.
#[derive(Parser, Debug)]
#[command(name = "llorch-cpud")]
#[command(about = "CPU worker daemon for llama-orch")]
struct Args {
    /// Worker ID (UUID) - assigned by pool-managerd
    #[arg(long)]
    worker_id: String,

    /// Model file path (GGUF format)
    #[arg(long)]
    model: String,

    /// HTTP server port - assigned by pool-managerd
    #[arg(long)]
    port: u16,

    /// Pool manager callback URL - where to report ready status
    #[arg(long)]
    callback_url: String,
}

/// Main entry point
/// 
/// CRITICAL: Uses single-threaded tokio runtime for SPEED
/// - flavor = "current_thread" ensures NO thread pool
/// - All async operations run on ONE thread
/// - No context switching overhead
/// - Optimal for CPU-bound inference
/// 
/// Flow:
/// 1. Parse args (from pool-managerd)
/// 2. Load model to memory
/// 3. Call back to pool-managerd (worker ready)
/// 4. Start HTTP server
/// 5. Run forever (until killed by pool-managerd)
#[tokio::main(flavor = "current_thread")]  // CRITICAL: Single-threaded!
async fn main() -> anyhow::Result<()> {
    // Initialize tracing (JSON format for structured logging)
    tracing_subscriber::fmt().with_target(false).json().init();

    // Parse CLI arguments (from pool-managerd)
    let args = Args::parse();

    tracing::info!(
        worker_id = %args.worker_id,
        model = %args.model,
        port = args.port,
        "CPU worker starting"
    );

    // ============================================================
    // STEP 1: Load model to memory
    // ============================================================
    tracing::info!(model = %args.model, "Loading model...");
    let backend = CpuInferenceBackend::load(&args.model)?;
    tracing::info!("Model loaded (stub)");

    // ============================================================
    // STEP 2: Call back to pool-managerd (worker ready)
    // ============================================================
    // This tells pool-managerd:
    // - Worker is ready to accept requests
    // - Worker is listening on args.port
    // - Worker has loaded X bytes of memory
    // - Worker type and capabilities
    if !args.callback_url.contains("localhost:9999") {
        // TODO: Update worker-common to accept extended fields
        // For now, use basic callback
        startup::callback_ready(
            &args.callback_url,
            &args.worker_id,
            backend.memory_bytes(),
            args.port,
        )
        .await?;
        
        // TODO: After worker-common update, use:
        // startup::callback_ready_extended(
        //     &args.callback_url,
        //     &args.worker_id,
        //     backend.memory_bytes(),
        //     args.port,
        //     backend.memory_architecture(),  // "cpu"
        //     backend.worker_type(),          // "cpu"
        //     &backend.capabilities(),        // ["text-gen"]
        // ).await?;
        
        tracing::info!("Callback sent to pool-managerd");
    } else {
        tracing::info!("Test mode: skipping pool manager callback");
    }

    // ============================================================
    // STEP 3: Start HTTP server (runs forever)
    // ============================================================
    tracing::info!("Worker ready, starting HTTP server");

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let backend = Arc::new(backend);
    
    // Create router with our backend (worker-http)
    // This wires up:
    // - GET /health -> backend.is_healthy()
    // - POST /execute -> backend.execute()
    let router = create_router(backend);
    
    // Start HTTP server (worker-http)
    let server = HttpServer::new(addr, router).await?;

    // ============================================================
    // STEP 4: Run forever (until killed)
    // ============================================================
    // This blocks forever, processing HTTP requests
    // Pool-managerd will kill this process when:
    // - Worker needs to be shut down
    // - Model needs to be unloaded
    // - System is shutting down
    server.run().await?;

    Ok(())
}
```

**Validation:**
- [ ] File compiles without errors
- [ ] Uses worker-http for HTTP server
- [ ] Uses worker-common for startup callback
- [ ] CLI arguments parsed correctly
- [ ] Logging configured
- [ ] Comments explain flow clearly
- [ ] Server runs continuously (not one-shot)

### ✓ Lib.rs Exports

**File:** `src/lib.rs`

```rust
pub mod backend;
pub mod cache;    // Top-level cache module
pub mod error;
pub mod layers;
pub mod model;
pub mod tensor;

// Re-export commonly used types
pub use backend::CpuInferenceBackend;
pub use cache::KVCache;
pub use error::LlorchError;
```

**Validation:**
- [ ] All modules declared
- [ ] Key types re-exported
- [ ] File compiles

### ✓ Module Stubs

**File:** `src/backend/mod.rs`
```rust
mod cpu_backend;
pub use cpu_backend::CpuInferenceBackend;
```

**File:** `src/model/mod.rs`
```rust
// Stub: will implement GPT2Model via checkpoints
```

**File:** `src/layers/mod.rs`
```rust
pub mod layer_norm;
pub mod embedding;
pub mod attention;
pub mod ffn;
pub mod transformer;
```

**File:** `src/layers/attention/mod.rs`
```rust
// Attention module - split into focused files
// See: ATTENTION_MODULE_STRUCTURE.md

mod qkv;      // Checkpoint 2
mod scores;   // Checkpoint 4
mod output;   // Checkpoint 5

pub use qkv::QKVProjection;
pub use scores::AttentionScores;
pub use output::AttentionOutput;

// Main Attention struct will orchestrate all components
// Note: KVCache is in src/cache/ (top-level module)
// pub struct Attention { ... }
```

**File:** `src/cache/mod.rs`
```rust
// KV Cache module - top-level for future growth
// See: KV_CACHE_MODULE_ANALYSIS.md
//
// This is a TOP-LEVEL module to signal that cache will need
// significant optimization work in the future (paged attention,
// memory pooling, etc.). For MVP, keep implementation simple.

mod kv_cache;

pub use kv_cache::KVCache;

// Future: Add more cache strategies here
// mod paged_cache;
// mod rotating_cache;
// mod memory_pool;
```

**File:** `src/cache/kv_cache.rs`
```rust
// Simple KV cache for GPT-2
// Checkpoint 3 validation
//
// IMPLEMENTATION: Keep simple for MVP
// STRUCTURE: Room to grow (this is why cache/ is top-level)

use ndarray::Array3;

pub struct KVCache {
    k_cache: Option<Array3<f32>>,
    v_cache: Option<Array3<f32>>,
    max_seq_len: usize,
}

impl KVCache {
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        // Simple implementation - will be validated in Checkpoint 3
        todo!("Implement in Week 2, Day 2")
    }
    
    pub fn update(
        &mut self,
        k: Array3<f32>,
        v: Array3<f32>,
        start_pos: usize,
    ) -> (Array3<f32>, Array3<f32>) {
        // Simple update logic - will be validated in Checkpoint 3
        todo!("Implement in Week 2, Day 2")
    }
}
```

**File:** `src/tensor/mod.rs`
```rust
// Stub: will implement CPU tensor ops
```

**File:** `src/error.rs`
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlorchError {
    #[error("Model error: {0}")]
    ModelError(String),
    
    #[error("Tensor error: {0}")]
    TensorError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
```

**Validation:**
- [ ] All module files exist
- [ ] All files compile
- [ ] No missing dependencies

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
- [ ] vram_bytes is 0 (CPU worker)

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
data: {"t":"RESPONSE","i":1}

event: end
data: {"tokens_out":2,"decode_time_ms":0,"stop_reason":"MAX_TOKENS","stop_sequence_matched":null}
```

**Validation:**
- [ ] Endpoint responds
- [ ] SSE stream format correct
- [ ] Returns stub tokens ("STUB", "RESPONSE")
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
{"level":"INFO","message":"CPU worker starting","worker_id":"test-worker","model":"test.gguf","port":8080}
{"level":"INFO","message":"Loading model...","model":"test.gguf"}
{"level":"INFO","message":"Model loaded (stub)"}
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

### Test 2: Tests Compile

```bash
cargo test --no-run
```

**Validation:**
- [ ] Test files compile
- [ ] No missing test dependencies

### Test 3: Clippy Clean

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
use llorch_cpud::backend::CpuInferenceBackend;
use worker_common::SamplingConfig;
use worker_http::InferenceBackend;

#[tokio::test]
async fn test_backend_stub_works() {
    // Create backend
    let backend = CpuInferenceBackend::load("test.gguf").unwrap();
    
    // Test execute returns stub data
    let config = SamplingConfig::default();
    let result = backend.execute("Hello", &config).await.unwrap();
    
    // Validate stub response
    assert_eq!(result.tokens.len(), 2);
    assert_eq!(result.tokens[0], "STUB");
    assert_eq!(result.tokens[1], "RESPONSE");
}

#[test]
fn test_backend_is_healthy() {
    let backend = CpuInferenceBackend::load("test.gguf").unwrap();
    assert!(backend.is_healthy());
}

#[test]
fn test_backend_vram_is_zero() {
    let backend = CpuInferenceBackend::load("test.gguf").unwrap();
    assert_eq!(backend.vram_usage(), 0);
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

### Code Quality

- ✅ No clippy warnings
- ✅ Proper module structure
- ✅ Clean separation of concerns
- ✅ Logging configured

### Documentation

- ✅ README.md exists (optional)
- ✅ Cargo.toml documented
- ✅ Module purposes clear

---

## Common Failures

### ❌ Compilation Errors

**Symptom:** cargo build fails  
**Cause:** Missing dependencies or wrong versions  
**Fix:** Check Cargo.toml matches IMPLEMENTATION_ROADMAP.md

### ❌ HTTP Server Won't Start

**Symptom:** Server panics or port in use  
**Cause:** Port already bound or missing async runtime  
**Fix:** Check port availability, verify tokio runtime

### ❌ Worker-crates Not Found

**Symptom:** "crate not found" errors  
**Cause:** Wrong path to worker-crates  
**Fix:** Verify path = "../worker-crates/worker-*"

### ❌ InferenceBackend Trait Not Implemented

**Symptom:** "trait not implemented" error  
**Cause:** Missing methods or wrong signature  
**Fix:** Check all 4 methods: execute, cancel, vram_usage, is_healthy

---

## Next Steps

If this checkpoint **PASSES**:
- ✅ Foundation is solid
- ✅ HTTP server works
- ✅ Ready to implement model
- ✅ Proceed to Checkpoint 1 (LayerNorm)

If this checkpoint **FAILS**:
- ❌ Fix foundation issues first
- ❌ Do not proceed to model implementation
- ❌ HTTP server must work before model work

---

## Notes

- This checkpoint is about **infrastructure**, not the model
- The GPT-2 spec (01_GPT2_PIPELINE_COMPLETE_BEHAVIORS.md) starts at Checkpoint 1
- HTTP server is **already done** (worker-http) - just integrate it
- Stub responses are expected - real inference comes later
- Focus: Project structure + HTTP integration + Worker crates

---

**Status:** ⬜ Not Started  
**Estimated Time:** 4-6 hours  
**Blocking:** Must pass before Checkpoint 1

---

Built by TEAM CASCADE 🌊
