# llorch-cpud API Integration with worker-crates

**Date:** 2025-10-08  
**Purpose:** Show how llorch-cpud uses existing worker-crates HTTP API  
**Status:** Ready to integrate

---

## Overview

**Good news:** The HTTP API is already extracted and ready to use!

The `worker-crates/worker-http` crate provides a complete, platform-agnostic HTTP server that llorch-cpud can use directly.

---

## Existing API (Already Extracted) ✅

### worker-http Crate

**Location:** `/home/vince/Projects/llama-orch/bin/worker-crates/worker-http/`

**Provides:**
- ✅ HTTP server lifecycle (`HttpServer`)
- ✅ Route configuration (`create_router`)
- ✅ Health endpoint (`GET /health`)
- ✅ Execute endpoint (`POST /execute`)
- ✅ SSE streaming
- ✅ Request validation
- ✅ Platform abstraction (`InferenceBackend` trait)

**Status:** Fully implemented, tested, ready to use

---

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vram_bytes": 0,
  "memory_type": "cpu"
}
```

**Implementation:** `worker-http/src/health.rs`

---

### 2. Execute Inference

**Endpoint:** `POST /execute`

**Request:**
```json
{
  "job_id": "uuid-here",
  "prompt": "Hello, world!",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.0,
  "min_p": 0.0,
  "stop": ["</s>"],
  "seed": 42
}
```

**Response:** Server-Sent Events (SSE) stream

```
event: started
data: {"job_id":"uuid","model":"gpt2-medium","started_at":"0"}

event: token
data: {"t":"Hello","i":0}

event: token
data: {"t":" world","i":1}

event: end
data: {"tokens_out":2,"decode_time_ms":100,"stop_reason":"max_tokens","stop_sequence_matched":null}
```

**Implementation:** `worker-http/src/execute.rs`

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│ worker-http (SHARED - 100% reusable)                    │
│ • HTTP server (Axum)                                    │
│ • Routes (/health, /execute)                            │
│ • SSE streaming                                         │
│ • Request validation                                    │
│ • InferenceBackend trait                                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌────────────────────────────────────────────────────────┐
│ llorch-cpud (NEW - CPU implementation)                 │
│ • CpuInferenceBackend (implements InferenceBackend)   │
│ • CPU tensor operations                                │
│ • GPT-2 model implementation                           │
│ • Checkpoint validation                                │
└────────────────────────────────────────────────────────┘
```

---

## InferenceBackend Trait

**Location:** `worker-http/src/backend.rs`

```rust
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Execute inference on the given prompt
    async fn execute(
        &self,
        prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>>;

    /// Get model information
    fn model_info(&self) -> ModelInfo;

    /// Check if model is loaded
    fn is_loaded(&self) -> bool;
}
```

**What llorch-cpud needs to implement:** Just this trait!

---

## llorch-cpud Implementation

### Directory Structure

```
llorch-cpud/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── main.rs                    # NEW: Uses worker-http
│   ├── cpu_backend.rs             # NEW: Implements InferenceBackend
│   ├── tensor.rs                  # NEW: CPU tensor ops
│   ├── layers/
│   │   ├── mod.rs
│   │   ├── layer_norm.rs          # CHECKPOINT 1
│   │   ├── embedding.rs
│   │   ├── attention.rs           # CHECKPOINTS 2-5
│   │   ├── ffn.rs                 # CHECKPOINT 6
│   │   └── transformer.rs         # CHECKPOINT 7
│   └── model.rs                   # GPT-2 model
└── tests/
    └── checkpoint_*.rs
```

### Cargo.toml

```toml
[package]
name = "llorch-cpud"
version = "0.1.0"
edition = "2021"

[dependencies]
# Existing worker crates (100% reusable)
worker-http = { path = "../worker-crates/worker-http" }
worker-common = { path = "../worker-crates/worker-common" }
worker-tokenizer = { path = "../worker-crates/worker-tokenizer" }
worker-models = { path = "../worker-crates/worker-models" }

# CPU tensor operations
ndarray = "0.15"
ndarray-linalg = "0.16"

# Async runtime
tokio = { version = "1", features = ["full"] }
axum = "0.7"

# Utilities
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json"] }
clap = { version = "4", features = ["derive"] }
```

### main.rs (Using worker-http)

```rust
//! llorch-cpud — CPU worker daemon
//!
//! GPT-2 inference on CPU using checkpoint-validated implementation.

use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use worker_common::startup;
use worker_http::{create_router, HttpServer};
use llorch_cpud::CpuInferenceBackend;

#[derive(Parser, Debug)]
#[command(name = "llorch-cpud")]
#[command(about = "CPU worker daemon for llama-orch")]
struct Args {
    /// Worker ID (UUID)
    #[arg(long)]
    worker_id: String,

    /// Model file path (GGUF or PyTorch format)
    #[arg(long)]
    model: String,

    /// HTTP server port
    #[arg(long)]
    port: u16,

    /// Pool manager callback URL
    #[arg(long)]
    callback_url: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_target(false).json().init();

    // Parse CLI arguments
    let args = Args::parse();

    tracing::info!(
        worker_id = %args.worker_id,
        model = %args.model,
        port = args.port,
        "CPU worker starting"
    );

    // Load model to CPU memory
    tracing::info!(model = %args.model, "Loading model to CPU memory...");
    let cpu_backend = CpuInferenceBackend::load(&args.model)?;
    tracing::info!(
        memory_bytes = cpu_backend.memory_bytes(),
        "Model loaded to CPU memory"
    );

    // Call back to pool manager
    if !args.callback_url.contains("localhost:9999") {
        startup::callback_ready(
            &args.callback_url,
            &args.worker_id,
            cpu_backend.memory_bytes(),
            args.port,
        )
        .await?;
    } else {
        tracing::info!("Test mode: skipping pool manager callback");
    }

    tracing::info!("Worker ready, starting HTTP server");

    // Start HTTP server (using worker-http!)
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let backend = Arc::new(cpu_backend);
    let router = create_router(backend);  // ← Uses worker-http
    let server = HttpServer::new(addr, router).await?;  // ← Uses worker-http

    server.run().await?;

    Ok(())
}
```

### cpu_backend.rs (Implements InferenceBackend)

```rust
//! CPU inference backend for llorch-cpud

use async_trait::async_trait;
use worker_common::{InferenceResult, SamplingConfig};
use worker_http::{backend::InferenceBackend, ModelInfo};
use worker_tokenizer::Tokenizer;
use crate::model::GPT2Model;

pub struct CpuInferenceBackend {
    model: GPT2Model,
    tokenizer: Tokenizer,
    model_path: String,
}

impl CpuInferenceBackend {
    /// Load model from disk
    pub fn load(model_path: &str) -> anyhow::Result<Self> {
        // Load tokenizer
        let tokenizer = Tokenizer::from_gguf(model_path)?;
        
        // Load model weights
        let model = GPT2Model::load(model_path)?;
        
        Ok(Self {
            model,
            tokenizer,
            model_path: model_path.to_string(),
        })
    }
    
    /// Get memory usage
    pub fn memory_bytes(&self) -> u64 {
        self.model.memory_bytes()
    }
}

#[async_trait]
impl InferenceBackend for CpuInferenceBackend {
    async fn execute(
        &self,
        prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        // Tokenize prompt
        let tokens = self.tokenizer.encode(prompt)?;
        
        // Run inference (validated via checkpoints!)
        let output_tokens = self.model.generate(&tokens, config)?;
        
        // Decode tokens
        let output_text = self.tokenizer.decode(&output_tokens)?;
        
        Ok(InferenceResult {
            tokens: output_tokens.iter().map(|t| t.to_string()).collect(),
            text: output_text,
            decode_time_ms: 0, // TODO: Add timing
            stop_reason: "max_tokens".to_string(),
            stop_sequence_matched: None,
        })
    }

    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: "gpt2-medium".to_string(),
            path: self.model_path.clone(),
            memory_bytes: self.memory_bytes(),
            memory_type: "cpu".to_string(),
        }
    }

    fn is_loaded(&self) -> bool {
        true
    }
}
```

---

## What's Already Done ✅

From `worker-crates/worker-http/`:

1. **HTTP Server** ✅
   - Axum-based server
   - Lifecycle management
   - Graceful shutdown

2. **Routes** ✅
   - `GET /health`
   - `POST /execute`
   - Middleware support

3. **SSE Streaming** ✅
   - Token-by-token streaming
   - Event types (started, token, end, error)
   - Proper SSE formatting

4. **Request Validation** ✅
   - Field validation
   - Error responses
   - Type safety

5. **Platform Abstraction** ✅
   - `InferenceBackend` trait
   - Clean separation of concerns
   - Easy to implement

---

## What llorch-cpud Needs to Implement

### Only 3 Things:

1. **CpuInferenceBackend** (implements `InferenceBackend` trait)
   - `execute()` method
   - `model_info()` method
   - `is_loaded()` method

2. **GPT2Model** (validated via checkpoints)
   - LayerNorm (Checkpoint 1)
   - Attention (Checkpoints 2-5)
   - FFN (Checkpoint 6)
   - Transformer blocks (Checkpoint 7)
   - LM head (Checkpoints 8-9)
   - Sampling (Checkpoints 10-11)
   - Generation (Checkpoint 12)

3. **CPU Tensor Operations**
   - Using ndarray
   - Basic ops (matmul, add, etc.)
   - LayerNorm, GELU, etc.

---

## Code Reuse Statistics

| Component | Lines | Source | Status |
|-----------|-------|--------|--------|
| HTTP Server | ~500 | worker-http | ✅ Ready |
| Routes | ~200 | worker-http | ✅ Ready |
| SSE Streaming | ~100 | worker-http | ✅ Ready |
| Validation | ~300 | worker-http | ✅ Ready |
| Tokenizer | ~1200 | worker-tokenizer | ✅ Ready |
| Common Types | ~300 | worker-common | ✅ Ready |
| **Subtotal (Reusable)** | **~2600** | **worker-crates** | **✅ Ready** |
| CPU Backend | ~200 | llorch-cpud | ⬜ TODO |
| GPT-2 Model | ~1500 | llorch-cpud | ⬜ TODO |
| CPU Tensor Ops | ~500 | llorch-cpud | ⬜ TODO |
| **Subtotal (New)** | **~2200** | **llorch-cpud** | **⬜ TODO** |
| **TOTAL** | **~4800** | | |

**Reuse: 54% of code is already done!**

---

## API Compatibility

### Same API as worker-orcd

llorch-cpud will have **identical API** to worker-orcd:

```bash
# worker-orcd (CUDA)
./worker-orcd --worker-id uuid --model model.gguf --gpu-device 0 --port 8080 --callback-url http://...

# llorch-cpud (CPU)
./llorch-cpud --worker-id uuid --model model.gguf --port 8080 --callback-url http://...
```

**Same endpoints:**
- `GET /health` → Same response format
- `POST /execute` → Same request/response format

**Same SSE events:**
- `started`, `token`, `end`, `error`

**Orchestratord doesn't need to know the difference!**

---

## Testing Strategy

### 1. Unit Tests (Checkpoints)

```rust
#[test]
fn checkpoint_01_layer_norm() {
    let backend = CpuInferenceBackend::load("test_model.gguf").unwrap();
    // Test LayerNorm matches reference
}
```

### 2. Integration Tests (HTTP API)

```rust
#[tokio::test]
async fn test_execute_endpoint() {
    let backend = Arc::new(CpuInferenceBackend::load("test_model.gguf").unwrap());
    let router = create_router(backend);
    
    // Test POST /execute
    let response = router
        .oneshot(Request::builder()
            .uri("/execute")
            .method("POST")
            .body(json!({
                "job_id": "test",
                "prompt": "Hello",
                "max_tokens": 10
            }))
            .unwrap())
        .await
        .unwrap();
    
    assert_eq!(response.status(), 200);
}
```

### 3. End-to-End Tests

```bash
# Start server
./llorch-cpud --worker-id test --model gpt2-medium.gguf --port 8080 --callback-url http://localhost:9999

# Test health
curl http://localhost:8080/health

# Test inference
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test","prompt":"Hello","max_tokens":10}'
```

---

## Implementation Timeline

### Week 1: Foundation
- [ ] Set up llorch-cpud crate
- [ ] Integrate worker-http
- [ ] Implement basic CpuInferenceBackend stub
- [ ] Test HTTP endpoints work

### Week 2-5: Model Implementation
- [ ] Implement GPT-2 model (using checkpoints)
- [ ] Validate each checkpoint
- [ ] Integrate with CpuInferenceBackend

### Week 6: Integration
- [ ] Complete end-to-end testing
- [ ] Performance optimization
- [ ] Documentation

---

## Benefits of Using worker-http

1. **No HTTP code to write** ✅
2. **No SSE streaming to implement** ✅
3. **No validation to write** ✅
4. **API compatibility guaranteed** ✅
5. **Tested and working** ✅
6. **Focus on model implementation** ✅

---

## Next Steps

1. **Create llorch-cpud crate** with worker-http dependency
2. **Implement CpuInferenceBackend stub** (returns dummy data)
3. **Test HTTP API works** (health + execute endpoints)
4. **Implement GPT-2 model** using checkpoints
5. **Connect model to backend** (replace stub with real inference)
6. **Test end-to-end** with Checkpoint 12

---

## Summary

**The API is already extracted and ready to use!**

- ✅ HTTP server: `worker-http`
- ✅ Endpoints: `/health`, `/execute`
- ✅ SSE streaming: Built-in
- ✅ Validation: Built-in
- ✅ Platform abstraction: `InferenceBackend` trait

**llorch-cpud just needs to:**
1. Implement `InferenceBackend` trait
2. Implement GPT-2 model (using checkpoints)
3. Connect them together

**That's it!** 54% of the code is already done.

---

Built by TEAM CASCADE 🌊
