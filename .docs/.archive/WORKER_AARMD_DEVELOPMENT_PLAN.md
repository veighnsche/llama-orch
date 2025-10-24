# Worker-aarmd Development Plan — Apple ARM Worker Implementation

**Status**: Planning  
**Target**: M3.5 (Post-M2)  
**Team**: Team AARM  
**Timeline**: 3-4 weeks  
**Last Updated**: 2025-10-05

---

## Executive Summary

This plan outlines the development of `worker-aarmd`, the Apple Silicon worker using Metal unified memory architecture. The strategy maximizes code reuse from `worker-orcd` by extracting shared Rust components into `worker-crates/`, enabling 85% code reuse between NVIDIA and Apple workers.

**Key Insight**: GGUF parsing, tokenization, HTTP server, and model adapters are **already pure Rust** with no FFI dependencies. Only the compute layer (CUDA vs Metal) is platform-specific.

---

## Architecture Overview

### FFI Boundary Analysis

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

### Code Reuse Breakdown

| Component | Lines of Code | Reusable | Platform-Specific |
|-----------|---------------|----------|-------------------|
| HTTP Server | ~500 | 100% | 0% |
| GGUF Parser | ~277 | 100% | 0% |
| Tokenizer | ~1200 | 100% | 0% |
| Model Adapters | ~800 | 100% | 0% |
| Sampling | ~400 | 100% | 0% |
| Error Handling | ~300 | 100% | 0% |
| Startup/Callback | ~200 | 100% | 0% |
| **Compute Layer** | **~2000** | **0%** | **100%** |
| **TOTAL** | **~5677** | **~3677 (65%)** | **~2000 (35%)** |

**Actual reuse**: 85% when accounting for main.rs and integration code.

---

## Phase 1: Extract Shared Crates (1-2 days)

**Goal**: Extract pure Rust code from `worker-orcd` into reusable crates.

### 1.1 Create Worker-Crates Structure

```bash
mkdir -p bin/worker-crates/{worker-http,worker-gguf,worker-tokenizer,worker-models,worker-common,worker-compute}
```

### 1.2 Extract worker-http (4 hours)

**Source files** (from `bin/worker-orcd/src/http/`):
- `server.rs` (243 lines) → `bin/worker-crates/worker-http/src/server.rs`
- `sse.rs` → `bin/worker-crates/worker-http/src/sse.rs`
- `routes.rs` → `bin/worker-crates/worker-http/src/routes.rs`
- `validation.rs` → `bin/worker-crates/worker-http/src/validation.rs`

**Dependencies**:
```toml
[dependencies]
axum = "0.7"
tokio = { version = "1", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["trace"] }
tracing = "0.1"
observability-narration-core = { path = "../../../libs/observability/narration-core" }
```

**Verification**: `cargo build -p worker-http`

### 1.3 Extract worker-gguf (2 hours)

**Source files** (from `bin/worker-orcd/src/gguf/`):
- `mod.rs` (277 lines) → `bin/worker-crates/worker-gguf/src/lib.rs`

**Dependencies**:
```toml
[dependencies]
thiserror = "1.0"
```

**Note**: Already pure Rust! No FFI dependencies.

**Verification**: `cargo test -p worker-gguf`

### 1.4 Extract worker-tokenizer (6 hours)

**Source files** (from `bin/worker-orcd/src/tokenizer/`):
- `mod.rs` (26 lines) → `bin/worker-crates/worker-tokenizer/src/lib.rs`
- `backend.rs` → `bin/worker-crates/worker-tokenizer/src/backend.rs`
- `decoder.rs` → `bin/worker-crates/worker-tokenizer/src/decoder.rs`
- `encoder.rs` → `bin/worker-crates/worker-tokenizer/src/encoder.rs`
- `hf_json.rs` → `bin/worker-crates/worker-tokenizer/src/hf_json.rs`
- `streaming.rs` → `bin/worker-crates/worker-tokenizer/src/streaming.rs`
- `vocab.rs` → `bin/worker-crates/worker-tokenizer/src/vocab.rs`
- `merges.rs` → `bin/worker-crates/worker-tokenizer/src/merges.rs`
- `metadata.rs` → `bin/worker-crates/worker-tokenizer/src/metadata.rs`
- `error.rs` → `bin/worker-crates/worker-tokenizer/src/error.rs`
- `discovery.rs` → `bin/worker-crates/worker-tokenizer/src/discovery.rs`

**Dependencies**:
```toml
[dependencies]
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

**Verification**: `cargo test -p worker-tokenizer`

### 1.5 Extract worker-models (4 hours)

**Source files** (from `bin/worker-orcd/src/models/`):
- `mod.rs` → `bin/worker-crates/worker-models/src/lib.rs`
- `adapter.rs` → `bin/worker-crates/worker-models/src/adapter.rs`
- `factory.rs` → `bin/worker-crates/worker-models/src/factory.rs`
- `gpt.rs` → `bin/worker-crates/worker-models/src/gpt.rs`
- `phi3.rs` → `bin/worker-crates/worker-models/src/phi3.rs`
- `qwen.rs` → `bin/worker-crates/worker-models/src/qwen.rs`

**Dependencies**:
```toml
[dependencies]
worker-gguf = { path = "../worker-gguf" }
thiserror = "1.0"
```

**Verification**: `cargo test -p worker-models`

### 1.6 Extract worker-common (3 hours)

**Source files** (from `bin/worker-orcd/src/`):
- `error.rs` (2071 bytes) → `bin/worker-crates/worker-common/src/error.rs`
- `sampling_config.rs` (11308 bytes) → `bin/worker-crates/worker-common/src/sampling.rs`
- `inference_result.rs` (5880 bytes) → `bin/worker-crates/worker-common/src/inference.rs`
- `startup.rs` (989 bytes) → `bin/worker-crates/worker-common/src/callback.rs`

**Dependencies**:
```toml
[dependencies]
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
reqwest = { version = "0.11", features = ["json"] }
tokio = { version = "1", features = ["full"] }
```

**Verification**: `cargo build -p worker-common`

### 1.7 Create worker-compute (2 hours)

**New trait-based abstraction** for platform-agnostic compute:

```rust
// bin/worker-crates/worker-compute/src/backend.rs
pub trait ComputeBackend {
    type Context;
    type Model;
    type InferenceResult;
    
    fn init(device_id: i32) -> Result<Self::Context, ComputeError>;
    fn load_model(ctx: &Self::Context, path: &str) -> Result<Self::Model, ComputeError>;
    fn inference_start(model: &Self::Model, prompt: &str, config: SamplingConfig) 
        -> Result<Self::InferenceResult, ComputeError>;
    fn inference_next_token(result: &mut Self::InferenceResult) 
        -> Result<Option<String>, ComputeError>;
}
```

**Verification**: `cargo build -p worker-compute`

---

## Phase 2: Refactor worker-orcd (1 day)

**Goal**: Prove shared crates work by refactoring existing NVIDIA worker.

### 2.1 Update worker-orcd Cargo.toml (30 minutes)

```toml
[dependencies]
# Shared worker crates
worker-http = { path = "../worker-crates/worker-http" }
worker-gguf = { path = "../worker-crates/worker-gguf" }
worker-tokenizer = { path = "../worker-crates/worker-tokenizer" }
worker-models = { path = "../worker-crates/worker-models" }
worker-common = { path = "../worker-crates/worker-common" }
worker-compute = { path = "../worker-crates/worker-compute" }

# CUDA-specific (keep these)
# ... existing CUDA dependencies
```

### 2.2 Update Imports (2 hours)

Replace local imports with crate imports:

```rust
// Before
use crate::http::server::HttpServer;
use crate::gguf::GGUFMetadata;
use crate::tokenizer::Tokenizer;

// After
use worker_http::HttpServer;
use worker_gguf::GGUFMetadata;
use worker_tokenizer::Tokenizer;
```

### 2.3 Implement ComputeBackend for CUDA (3 hours)

```rust
// bin/worker-orcd/src/cuda/backend.rs
use worker_compute::ComputeBackend;

pub struct CudaBackend;

impl ComputeBackend for CudaBackend {
    type Context = CudaContext;
    type Model = CudaModel;
    type InferenceResult = CudaInferenceResult;
    
    fn init(device_id: i32) -> Result<Self::Context, ComputeError> {
        // Existing cuda_init logic
    }
    
    // ... implement other trait methods
}
```

### 2.4 Verification (2 hours)

```bash
# Build worker-orcd with shared crates
cargo build -p worker-orcd

# Run existing tests
cargo test -p worker-orcd

# Run integration tests
cd bin/worker-orcd
cargo test --test integration
```

**Success Criteria**:
- ✅ All tests pass
- ✅ No compilation errors
- ✅ Binary size similar to before
- ✅ Runtime behavior unchanged

---

## Phase 3: Scaffold worker-aarmd (1 hour)

**Goal**: Create Apple ARM worker structure with mocked Metal layer.

### 3.1 Create Directory Structure

```bash
mkdir -p bin/worker-aarmd/{src/{metal,http},tests,benches}
```

### 3.2 Copy Shared Structure from worker-orcd

```bash
# Copy main.rs (will need minor edits)
cp bin/worker-orcd/src/main.rs bin/worker-aarmd/src/main.rs

# Copy test structure
cp -r bin/worker-orcd/tests/common bin/worker-aarmd/tests/

# Copy Cargo.toml (will need edits)
cp bin/worker-orcd/Cargo.toml bin/worker-aarmd/Cargo.toml
```

### 3.3 Create Cargo.toml

```toml
[package]
name = "worker-aarmd"
version = "0.1.0"
edition = "2021"

[dependencies]
# Shared worker crates (SAME AS worker-orcd!)
worker-http = { path = "../worker-crates/worker-http" }
worker-gguf = { path = "../worker-crates/worker-gguf" }
worker-tokenizer = { path = "../worker-crates/worker-tokenizer" }
worker-models = { path = "../worker-crates/worker-models" }
worker-common = { path = "../worker-crates/worker-common" }
worker-compute = { path = "../worker-crates/worker-compute" }

# Metal-specific (DIFFERENT from worker-orcd)
metal = "0.27"
metal-performance-shaders = "0.1"
objc = "0.2"

# Common dependencies
clap = { version = "4.5", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
axum = "0.7"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json"] }
anyhow = "1.0"
thiserror = "1.0"
```

---

## Phase 4: Mock Metal Layer (2 hours)

**Goal**: Enable Linux development with mocked Metal API.

### 4.1 Create Metal Module Structure

```rust
// bin/worker-aarmd/src/metal/mod.rs

#[cfg(target_os = "macos")]
mod real;
#[cfg(target_os = "macos")]
pub use real::*;

#[cfg(not(target_os = "macos"))]
mod mock;
#[cfg(not(target_os = "macos"))]
pub use mock::*;

// Common interface (works on both Linux and macOS)
pub use backend::MetalBackend;
```

### 4.2 Create Mock Implementation

```rust
// bin/worker-aarmd/src/metal/mock.rs

use worker_compute::{ComputeBackend, ComputeError};

pub struct MetalContext;
pub struct MetalModel;
pub struct MetalInferenceResult;

pub struct MetalBackend;

impl ComputeBackend for MetalBackend {
    type Context = MetalContext;
    type Model = MetalModel;
    type InferenceResult = MetalInferenceResult;
    
    fn init(device_id: i32) -> Result<Self::Context, ComputeError> {
        eprintln!("MOCK: Metal not available on Linux");
        Err(ComputeError::DeviceNotFound)
    }
    
    // ... mock other methods
}
```

### 4.3 Verification on Linux

```bash
# Should compile on Linux (with mocks)
cargo build -p worker-aarmd --target x86_64-unknown-linux-gnu

# Tests should compile (but skip Metal tests)
cargo test -p worker-aarmd --target x86_64-unknown-linux-gnu
```

---

## Phase 5: Develop Business Logic on Linux (1-2 weeks)

**Goal**: Implement all non-Metal logic on Linux development machine.

### 5.1 Update main.rs (2 hours)

```rust
// bin/worker-aarmd/src/main.rs

use worker_http::HttpServer;
use worker_common::callback;
use crate::metal::MetalBackend;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse CLI (same as worker-orcd)
    let args = Args::parse();
    
    // Initialize Metal context
    let ctx = MetalBackend::init(args.device)?;
    
    // Load model (uses shared worker-gguf!)
    let model = MetalBackend::load_model(&ctx, &args.model)?;
    
    // Send ready callback (uses shared worker-common!)
    callback::send_ready(&args.callback_url, &args.worker_id, 
                        model.memory_bytes(), "unified").await?;
    
    // Start HTTP server (uses shared worker-http!)
    let server = HttpServer::new(args.port, create_router(model)).await?;
    server.run().await?;
    
    Ok(())
}
```

### 5.2 Implement HTTP Handlers (4 hours)

```rust
// bin/worker-aarmd/src/http/execute.rs

use worker_http::{Handler, Request, Response};
use worker_common::SamplingConfig;

pub async fn execute_handler(
    State(model): State<MetalModel>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<impl Stream>, StatusCode> {
    // Same logic as worker-orcd, different backend
    let result = MetalBackend::inference_start(&model, &req.prompt, req.config)?;
    
    let stream = async_stream::stream! {
        while let Some(token) = MetalBackend::inference_next_token(&mut result)? {
            yield Event::default().data(token);
        }
    };
    
    Ok(Sse::new(stream))
}
```

### 5.3 Write Unit Tests (3 days)

```rust
// bin/worker-aarmd/tests/http_tests.rs

#[tokio::test]
async fn test_execute_endpoint() {
    // Uses mock Metal on Linux
    let app = create_test_app();
    let response = app
        .oneshot(Request::post("/execute")
            .json(&ExecuteRequest { prompt: "test", max_tokens: 10 })
            .unwrap())
        .await
        .unwrap();
    
    assert_eq!(response.status(), StatusCode::OK);
}
```

### 5.4 Integration Tests (2 days)

```rust
// bin/worker-aarmd/tests/integration/mod.rs

#[tokio::test]
#[cfg(target_os = "macos")]  // Only run on Mac
async fn test_full_inference_flow() {
    // Real Metal test
    let worker = spawn_worker("test-model.gguf").await;
    let response = worker.execute("Hello").await;
    assert!(response.tokens.len() > 0);
}
```

---

## Phase 6: Implement Real Metal (3-5 days)

**Goal**: Implement actual Metal backend on macOS.

### 6.1 Metal Context Management (1 day)

```rust
// bin/worker-aarmd/src/metal/real/context.rs

use metal::{Device, CommandQueue};
use worker_compute::{ComputeBackend, ComputeError};

pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
}

impl MetalContext {
    pub fn new(device_id: i32) -> Result<Self, ComputeError> {
        let device = Device::system_default()
            .ok_or(ComputeError::DeviceNotFound)?;
        
        let command_queue = device.new_command_queue();
        
        Ok(Self { device, command_queue })
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }
}
```

### 6.2 Model Loading with Unified Memory (1 day)

```rust
// bin/worker-aarmd/src/metal/real/model.rs

use metal::{Buffer, MTLResourceOptions};
use worker_gguf::GGUFMetadata;

pub struct MetalModel {
    weights: Buffer,  // Unified memory buffer
    config: ModelConfig,
    memory_bytes: u64,
}

impl MetalModel {
    pub fn load(ctx: &MetalContext, path: &str) -> Result<Self, ComputeError> {
        // Parse GGUF (uses shared worker-gguf!)
        let gguf = GGUFMetadata::from_file(path)?;
        
        // Allocate unified memory
        let weights = ctx.device().new_buffer(
            gguf.total_size as u64,
            MTLResourceOptions::StorageModeShared,  // Unified memory!
        );
        
        // Copy weights (zero-copy, CPU and GPU can access!)
        unsafe {
            std::ptr::copy_nonoverlapping(
                gguf.weights.as_ptr(),
                weights.contents() as *mut u8,
                gguf.total_size,
            );
        }
        
        Ok(Self {
            weights,
            config: gguf.config,
            memory_bytes: gguf.total_size as u64,
        })
    }
}
```

### 6.3 Inference with Metal Performance Shaders (2 days)

```rust
// bin/worker-aarmd/src/metal/real/inference.rs

use metal_performance_shaders as mps;

pub struct MetalInference {
    context: MetalContext,
    model: MetalModel,
}

impl MetalInference {
    pub fn execute(&self, tokens: &[u32]) -> Result<Vec<f32>, ComputeError> {
        // Use Metal Performance Shaders for matmul
        let matmul = mps::MPSMatrixMultiplication::new(
            self.context.device(),
            /* dimensions */
        );
        
        let command_buffer = self.context.command_queue.new_command_buffer();
        matmul.encode_to_command_buffer(
            command_buffer,
            &self.model.weights,
            &input_buffer,
            &output_buffer,
        );
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(output)
    }
}
```

### 6.4 Implement ComputeBackend Trait (1 day)

```rust
// bin/worker-aarmd/src/metal/real/backend.rs

impl ComputeBackend for MetalBackend {
    type Context = MetalContext;
    type Model = MetalModel;
    type InferenceResult = MetalInferenceResult;
    
    fn init(device_id: i32) -> Result<Self::Context, ComputeError> {
        MetalContext::new(device_id)
    }
    
    fn load_model(ctx: &Self::Context, path: &str) -> Result<Self::Model, ComputeError> {
        MetalModel::load(ctx, path)
    }
    
    fn inference_start(
        model: &Self::Model,
        prompt: &str,
        config: SamplingConfig,
    ) -> Result<Self::InferenceResult, ComputeError> {
        MetalInference::start(model, prompt, config)
    }
    
    fn inference_next_token(
        result: &mut Self::InferenceResult,
    ) -> Result<Option<String>, ComputeError> {
        result.next_token()
    }
}
```

---

## Phase 7: Test on Mac Hardware (2-3 days)

**Goal**: Validate unified memory behavior and performance.

### 7.1 Functional Tests (1 day)

```bash
# On Mac:
cd bin/worker-aarmd

# Build with real Metal
cargo build --release --target aarch64-apple-darwin

# Run unit tests
cargo test

# Run integration tests
cargo test --test integration
```

### 7.2 Performance Benchmarks (1 day)

```rust
// bin/worker-aarmd/benches/inference_benchmark.rs

#[bench]
fn bench_qwen_inference(b: &mut Bencher) {
    let model = load_test_model("qwen-0.5b.gguf");
    
    b.iter(|| {
        model.inference("Hello world", 10)
    });
}
```

### 7.3 Memory Validation (1 day)

```rust
// bin/worker-aarmd/tests/memory_tests.rs

#[test]
#[cfg(target_os = "macos")]
fn test_unified_memory_access() {
    let model = MetalModel::load("test.gguf").unwrap();
    
    // Verify CPU can read GPU memory
    let weights_ptr = model.weights.contents() as *const f32;
    let first_weight = unsafe { *weights_ptr };
    assert!(first_weight.is_finite());
    
    // Verify GPU can access same memory
    let gpu_result = model.inference("test", 1).unwrap();
    assert!(gpu_result.len() > 0);
}
```

---

## Development Workflow

### Linux Development (90% of work)

```bash
# On Linux machine:
cd /home/vince/Projects/llama-orch

# 1. Extract shared crates
# ... (Phase 1)

# 2. Refactor worker-orcd
cargo build -p worker-orcd
cargo test -p worker-orcd

# 3. Develop worker-aarmd business logic
cargo build -p worker-aarmd --target x86_64-unknown-linux-gnu
cargo test -p worker-aarmd --target x86_64-unknown-linux-gnu

# 4. Push to Git
git add .
git commit -m "feat: Add worker-aarmd with mocked Metal"
git push
```

### Mac Development (10% of work)

```bash
# On Mac:
git pull

cd bin/worker-aarmd

# 1. Implement real Metal
# ... (Phase 6)

# 2. Build with real Metal
cargo build --release --target aarch64-apple-darwin

# 3. Test on actual hardware
cargo test
./target/release/worker-aarmd --model test.gguf --device 0 --port 8001

# 4. Push Metal implementation
git add .
git commit -m "feat: Implement real Metal backend for worker-aarmd"
git push
```

---

## Success Criteria

### Phase 1-2 (Shared Crates)
- ✅ All shared crates compile independently
- ✅ worker-orcd builds with shared crates
- ✅ All existing worker-orcd tests pass
- ✅ No performance regression

### Phase 3-5 (Linux Development)
- ✅ worker-aarmd compiles on Linux (with mocks)
- ✅ All unit tests pass on Linux
- ✅ HTTP handlers work with mock backend
- ✅ Integration tests compile (skip Metal tests)

### Phase 6-7 (Mac Implementation)
- ✅ worker-aarmd compiles on macOS with real Metal
- ✅ Model loads into unified memory successfully
- ✅ Inference produces valid tokens
- ✅ Performance comparable to llama.cpp on Apple Silicon
- ✅ Memory usage reported correctly
- ✅ All integration tests pass

---

## Risk Mitigation

### Risk 1: Metal API Complexity
**Mitigation**: Use Metal Performance Shaders (MPS) instead of writing custom shaders. MPS provides optimized matmul, attention, and other operations.

### Risk 2: Unified Memory Bugs
**Mitigation**: Add extensive memory validation tests. Verify CPU and GPU see same data.

### Risk 3: Performance Issues
**Mitigation**: Benchmark early against llama.cpp. Use Instruments.app for profiling.

### Risk 4: Cross-Platform Build Issues
**Mitigation**: Use `#[cfg(target_os = "macos")]` extensively. Mock Metal on Linux.

---

## Timeline Estimate

| Phase | Duration | Dependencies | Blocker Risk |
|-------|----------|--------------|--------------|
| 1. Extract Shared Crates | 1-2 days | None | Low |
| 2. Refactor worker-orcd | 1 day | Phase 1 | Low |
| 3. Scaffold worker-aarmd | 1 hour | Phase 2 | Low |
| 4. Mock Metal Layer | 2 hours | Phase 3 | Low |
| 5. Business Logic (Linux) | 1-2 weeks | Phase 4 | Medium |
| 6. Real Metal (Mac) | 3-5 days | Phase 5 | High |
| 7. Mac Testing | 2-3 days | Phase 6 | Medium |
| **TOTAL** | **3-4 weeks** | Sequential | - |

---

## Team Assignments

### Team AARM Roles

**Linux Development** (Phases 1-5):
- Extract shared crates
- Refactor worker-orcd
- Implement business logic
- Write tests with mocks

**Mac Development** (Phases 6-7):
- Implement Metal backend
- Test on Apple Silicon
- Performance optimization
- Memory validation

---

## Verification Checklist

### Phase 1: Shared Crates
- [ ] `cargo build -p worker-http` succeeds
- [ ] `cargo build -p worker-gguf` succeeds
- [ ] `cargo build -p worker-tokenizer` succeeds
- [ ] `cargo build -p worker-models` succeeds
- [ ] `cargo build -p worker-common` succeeds
- [ ] `cargo build -p worker-compute` succeeds
- [ ] All crate tests pass

### Phase 2: Refactored worker-orcd
- [ ] `cargo build -p worker-orcd` succeeds
- [ ] `cargo test -p worker-orcd` passes
- [ ] Integration tests pass
- [ ] Binary runs successfully
- [ ] No performance regression

### Phase 3-5: worker-aarmd on Linux
- [ ] `cargo build -p worker-aarmd` succeeds on Linux
- [ ] `cargo test -p worker-aarmd` passes on Linux
- [ ] HTTP handlers compile
- [ ] Mock Metal returns appropriate errors

### Phase 6-7: worker-aarmd on Mac
- [ ] `cargo build -p worker-aarmd` succeeds on macOS
- [ ] `cargo test -p worker-aarmd` passes on macOS
- [ ] Model loads into unified memory
- [ ] Inference produces valid output
- [ ] Performance benchmarks meet targets
- [ ] Memory tests pass

---

## References

- **System Spec**: `bin/.specs/00_llama-orch.md` (SYS-2.2.x, SYS-6.3.x)
- **Worker-orcd Spec**: `bin/.specs/01_M0_worker_orcd.md`
- **Worker Adapters**: `bin/pool-managerd/.specs/10_worker_adapters.md` (POOL-1030-1034)
- **VRAM Scope Clarification**: `.docs/VRAM_ONLY_SCOPE_CLARIFICATION.md`

---

**Document Owner**: Team AARM  
**Review Cadence**: Weekly during development  
**Status Updates**: Daily standups during Phases 6-7
