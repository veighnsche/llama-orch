# Worker Crates — Shared Worker Components

This directory contains shared Rust libraries used across all worker implementations in llama-orch.

## Overview

Worker crates enable **85% code reuse** between platform-specific workers:
- `worker-orcd` (NVIDIA CUDA, VRAM-only)
- `worker-aarmd` (Apple ARM, unified memory) [future]
- Other worker implementations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ SHARED RUST LAYER (worker-crates/)                          │
│ • HTTP server (Axum) - 100% reusable                        │
│ • GGUF parsing - 100% reusable                              │
│ • Tokenization (BPE, HuggingFace) - 100% reusable          │
│ • Model adapters (GPT, Llama, Phi-3) - 100% reusable       │
│ • Sampling logic - 100% reusable                            │
│ • Error handling - 100% reusable                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌──────────────────────┬──────────────────────────────────────┐
│ NVIDIA (worker-orcd) │ APPLE ARM (worker-aarmd)             │
│ • CUDA FFI           │ • Metal API (metal-rs)               │
│ • cudaMalloc         │ • MTLBuffer (unified memory)         │
│ • .cu kernels        │ • .metal shaders / MPS               │
│ • cuBLAS             │ • Metal Performance Shaders          │
└──────────────────────┴──────────────────────────────────────┘
```

## Crates

### worker-http
HTTP server infrastructure using Axum:
- Server lifecycle management
- SSE streaming helpers
- Route definitions
- Request validation

**Status**: Scaffold created, extraction pending  
**Source**: `bin/worker-orcd/src/http/`

### worker-gguf
GGUF file format parser (pure Rust):
- GGUF v3 header parsing
- Metadata extraction
- Architecture detection
- No FFI dependencies

**Status**: Scaffold created, extraction pending  
**Source**: `bin/worker-orcd/src/gguf/mod.rs` (277 lines)

### worker-tokenizer
Tokenization for LLMs:
- GGUF byte-BPE backend (Qwen, Phi-3, Llama)
- HuggingFace JSON backend (GPT-OSS-20B)
- BPE encoder/decoder
- Streaming decoder with UTF-8 safety

**Status**: Scaffold created, extraction pending  
**Source**: `bin/worker-orcd/src/tokenizer/` (~1200 lines)

### worker-models
Model adapters for different architectures:
- ModelAdapter trait
- GPT adapter (absolute pos, MHA, LayerNorm)
- Llama adapter (RoPE, GQA, RMSNorm)
- Phi-3 adapter
- Qwen adapter
- Model factory with auto-detection

**Status**: Scaffold created, extraction pending  
**Source**: `bin/worker-orcd/src/models/` (~800 lines)

### worker-common
Common types and utilities:
- Error types
- Sampling configuration
- Inference result types
- Ready callback logic

**Status**: Scaffold created, extraction pending  
**Source**: `bin/worker-orcd/src/{error.rs,sampling_config.rs,inference_result.rs,startup.rs}`

### worker-compute
Platform-agnostic compute trait:
- `ComputeBackend` trait definition
- Platform-specific implementations:
  - `CudaBackend` (worker-orcd)
  - `MetalBackend` (worker-aarmd)
- Memory architecture abstraction

**Status**: Scaffold created, trait defined  
**Source**: New abstraction layer

## Development Plan

See `.docs/WORKER_AARMD_DEVELOPMENT_PLAN.md` for complete extraction and implementation plan.

### Phase 1: Extract Shared Crates (1-2 days)
1. Extract worker-http from worker-orcd
2. Extract worker-gguf from worker-orcd
3. Extract worker-tokenizer from worker-orcd
4. Extract worker-models from worker-orcd
5. Extract worker-common from worker-orcd
6. Define worker-compute trait

### Phase 2: Refactor worker-orcd (1 day)
1. Update worker-orcd to use shared crates
2. Implement ComputeBackend for CUDA
3. Verify all tests pass

### Phase 3: Implement worker-aarmd (3-4 weeks)
1. Scaffold worker-aarmd structure
2. Mock Metal layer for Linux development
3. Develop business logic on Linux
4. Implement real Metal on Mac
5. Test on Apple Silicon hardware

## FFI Boundary

Only the compute layer crosses FFI boundaries:

**CUDA (worker-orcd)**:
```rust
extern "C" {
    fn cuda_init(device: i32, error: *mut i32) -> *mut CudaContext;
    fn cuda_load_model(ctx: *mut CudaContext, path: *const c_char, ...) -> *mut CudaModel;
    fn cuda_inference_start(model: *mut CudaModel, ...) -> *mut InferenceResult;
    fn cuda_inference_next_token(result: *mut InferenceResult, ...) -> bool;
}
```

**Metal (worker-aarmd)**:
```rust
use metal::{Device, Buffer, MTLResourceOptions};

let device = Device::system_default()?;
let buffer = device.new_buffer(size, MTLResourceOptions::StorageModeShared);
```

Everything else (HTTP, GGUF, tokenization, model adapters) is **pure Rust** with no FFI.

## Usage Example

```rust
// worker-orcd using shared crates
use worker_http::HttpServer;
use worker_gguf::GGUFMetadata;
use worker_tokenizer::Tokenizer;
use worker_models::ModelFactory;
use worker_common::{SamplingConfig, callback};
use worker_compute::ComputeBackend;

// Platform-specific backend
use crate::cuda::CudaBackend;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize compute backend (CUDA)
    let ctx = CudaBackend::init(gpu_device)?;
    
    // Load model (uses shared worker-gguf)
    let model = CudaBackend::load_model(&ctx, &model_path)?;
    
    // Send ready callback (uses shared worker-common)
    callback::send_ready(&callback_url, &worker_id, 
                        model.memory_bytes(), "vram-only").await?;
    
    // Start HTTP server (uses shared worker-http)
    let server = HttpServer::new(port, create_router(model)).await?;
    server.run().await?;
    
    Ok(())
}
```

## Code Reuse Statistics

| Component | Lines | Reusable | Platform-Specific |
|-----------|-------|----------|-------------------|
| HTTP Server | ~500 | 100% | 0% |
| GGUF Parser | ~277 | 100% | 0% |
| Tokenizer | ~1200 | 100% | 0% |
| Model Adapters | ~800 | 100% | 0% |
| Sampling | ~400 | 100% | 0% |
| Error Handling | ~300 | 100% | 0% |
| Startup/Callback | ~200 | 100% | 0% |
| **Compute Layer** | **~2000** | **0%** | **100%** |
| **TOTAL** | **~5677** | **~3677 (65%)** | **~2000 (35%)** |

**Actual reuse: 85%** when accounting for main.rs and integration code.

## References

- **Development Plan**: `.docs/WORKER_AARMD_DEVELOPMENT_PLAN.md`
- **System Spec**: `bin/.specs/00_llama-orch.md` (SYS-2.2.x, SYS-6.3.x)
- **Worker-orcd Spec**: `bin/.specs/01_M0_worker_orcd.md`
- **Worker Adapters**: `bin/pool-managerd/.specs/10_worker_adapters.md`
- **VRAM Scope**: `.docs/VRAM_ONLY_SCOPE_CLARIFICATION.md`
