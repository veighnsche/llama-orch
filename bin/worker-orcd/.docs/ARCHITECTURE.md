# Worker-orcd Architecture

**Status**: Authoritative  
**Date**: 2025-10-03

---

## Executive Summary

Worker-orcd is a **hybrid Rust + C++/CUDA binary** that executes LLM inference on a single GPU. It is NOT a collection of Rust crates - it is a single binary with embedded CUDA code.

**Key Principles**:
1. **CUDA context is per-process** - All GPU operations happen in C++/CUDA within a single context. Rust provides HTTP/orchestration layer.
2. **Single-threaded execution** (M0-W-1301) - Worker processes requests sequentially using tokio's `current_thread` runtime. No concurrent inference, no thread pool overhead.
3. **Async for I/O, not compute** - Tokio provides non-blocking HTTP handling via event loop. CUDA operations remain synchronous and sequential.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ RUST LAYER (src/*.rs)                                        │
│                                                               │
│  main.rs         → Entry point, CLI parsing                  │
│  http/           → Axum HTTP server                          │
│    ├─ execute.rs → POST /execute (SSE streaming)            │
│    └─ health.rs  → GET /health                              │
│  startup.rs      → Pool manager callback                     │
│  error.rs        → Error type conversions                    │
│  cuda/mod.rs     → FFI bindings (unsafe extern "C")         │
│                                                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ FFI Boundary (C API)
                         │
┌────────────────────────▼────────────────────────────────────┐
│ C++/CUDA LAYER (cuda/*.cpp, *.cu)                           │
│                                                               │
│  context.cpp     → CUDA context, device init                │
│  model.cpp       → GGUF parsing, model loading              │
│  inference.cu    → Inference loop, token generation         │
│  health.cpp      → VRAM residency checks                    │
│  errors.cpp      → Error code definitions                   │
│  kernels/        → CUDA kernels                             │
│    ├─ attention.cu                                          │
│    ├─ matmul.cu                                             │
│    ├─ sampling.cu                                           │
│    └─ rope.cu                                               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Why This Architecture?

### Problem 1: CUDA Context is Per-Process

CUDA allocations and operations are tied to a **single CUDA context** per process. You cannot:
- ❌ Have multiple Rust crates independently calling CUDA
- ❌ Share CUDA context across crate boundaries
- ❌ Mix Rust and C++ CUDA calls safely

**Solution**: Single Binary with Embedded CUDA

All CUDA operations happen in **C++/CUDA code** within the binary:
- ✅ Single CUDA context for entire worker lifetime
- ✅ All GPU operations in C++ (context, model, inference, health)
- ✅ Rust calls into C++ via thin FFI layer
- ✅ Clear ownership: C++ owns GPU memory, Rust owns handles

### Problem 2: M0 Requires Sequential Execution (No Concurrency)

**Spec M0-W-1301**: "Worker-orcd MUST process inference requests sequentially (one at a time)."

Initial implementation violated this by using multi-threaded tokio runtime by default.

**Solution**: Single-Threaded Tokio Runtime

```rust
// src/main.rs
#[tokio::main(flavor = "current_thread")]  // ← Single-threaded!
async fn main() -> anyhow::Result<()> {
    // ...
}
```

**Why this works**:
- ✅ Tokio event loop handles HTTP I/O without blocking
- ✅ CUDA operations run sequentially on same thread
- ✅ No thread pool, no work-stealing, no context switching
- ✅ Matches spec requirement exactly
- ✅ Simpler logging (no mutex needed, like llama.cpp)

**What we get**:
- HTTP server remains responsive (non-blocking I/O via event loop)
- CUDA inference runs sequentially (one request at a time)
- Zero threading overhead
- Spec compliant

---

## What This Eliminates

### ❌ No Separate Crates for CUDA Operations

These are **NOT** separate crates:
- ~~`model-lifecycle`~~ → Just `cuda/src/model.cpp`
- ~~`vram-policy`~~ → Just `cuda/src/context.cpp` (enforcement flags)
- ~~`health-monitor`~~ → Just `cuda/src/health.cpp`
- ~~`inference-api`~~ → Just `src/http/execute.rs` (calls CUDA via FFI)

### Why?
- These operations are tightly coupled to CUDA context
- Separating them into crates creates artificial boundaries
- CUDA context cannot be shared across crate boundaries
- Simpler to maintain as a single binary

---

## Directory Structure

```
bin/worker-orcd/
├── .specs/
│   ├── 00_worker-orcd.md           # Worker spec (WORK-3xxx)
│   └── 01_cuda_ffi_boundary.md     # FFI boundary spec (CUDA-4xxx)
├── .docs/
│   ├── ARCHITECTURE.md             # This file
│   └── CUDA_INTEGRATION.md         # FFI integration guide
├── Cargo.toml                      # Rust package config
├── build.rs                        # Build script (compiles CUDA)
├── src/                            # Rust implementation
│   ├── main.rs                     # Entry point
│   ├── cuda/
│   │   └── mod.rs                  # FFI bindings + safe wrappers
│   ├── http/
│   │   ├── mod.rs
│   │   ├── execute.rs              # POST /execute
│   │   └── health.rs               # GET /health
│   ├── startup.rs                  # Startup + callback
│   └── error.rs                    # Error types
├── cuda/                           # C++/CUDA implementation
│   ├── .specs/
│   │   ├── 00_cuda_overview.md     # CUDA architecture (CUDA-5xxx)
│   │   ├── 01_context.md           # Context module (CUDA-5100)
│   │   ├── 02_model.md             # Model module (CUDA-5200)
│   │   ├── 03_inference.md         # Inference module (CUDA-5300)
│   │   └── 04_health.md            # Health module (CUDA-5400)
│   ├── CMakeLists.txt              # CMake build config
│   ├── README.md                   # CUDA-specific docs
│   ├── include/
│   │   ├── worker_cuda.h           # C API (FFI boundary)
│   │   ├── context.hpp             # C++ headers
│   │   ├── model.hpp
│   │   ├── inference.hpp
│   │   ├── health.hpp
│   │   ├── errors.hpp
│   │   └── types.hpp
│   ├── src/
│   │   ├── ffi.cpp                 # C API implementation
│   │   ├── context.cpp             # Context management
│   │   ├── model.cpp               # Model loading
│   │   ├── inference.cu            # Inference execution
│   │   ├── health.cpp              # Health checks
│   │   ├── errors.cpp              # Error handling
│   │   └── utils.cpp               # Utilities
│   ├── kernels/
│   │   ├── attention.cu            # Attention kernels
│   │   ├── matmul.cu               # Matrix multiplication
│   │   ├── sampling.cu             # Token sampling
│   │   ├── rope.cu                 # RoPE embeddings
│   │   └── common.cuh              # Kernel utilities
│   └── tests/
│       ├── test_context.cpp        # Context tests
│       ├── test_model.cpp          # Model tests
│       ├── test_inference.cpp      # Inference tests
│       └── test_health.cpp         # Health tests
└── tests/
    ├── integration_test.rs         # Rust integration tests
    └── ffi_smoke_test.rs           # FFI boundary tests
```

---

## Data Flow

### Startup
```
1. main.rs parses CLI args
   ↓
2. cuda::safe::ContextHandle::new(gpu_device)
   ↓ FFI
3. cuda/context.cpp: Initialize CUDA, set device, disable UMA
   ↓ FFI
4. cuda::safe::ModelHandle::load(ctx, path)
   ↓ FFI
5. cuda/model.cpp: Parse GGUF, cudaMalloc, cudaMemcpy
   ↓ FFI
6. startup::callback_ready() → POST to pool manager
   ↓
7. http::create_router() → Start Axum server
```

### Inference Request
```
1. POST /execute → http/execute.rs
   ↓
2. cuda::safe::InferenceHandle::start(model, prompt, ...)
   ↓ FFI
3. cuda/inference.cu: Tokenize, allocate KV cache
   ↓ FFI
4. Loop: cuda::safe::InferenceHandle::next_token()
   ↓ FFI
5. cuda/inference.cu: Run kernels, sample token
   ↓ FFI
6. http/execute.rs: Send SSE event with token
   ↓
7. Repeat until done
```

### Health Check
```
1. GET /health → http/health.rs
   ↓
2. cuda::safe::ModelHandle::check_vram_residency()
   ↓ FFI
3. cuda/health.cpp: Query CUDA memory info, verify pointers
   ↓ FFI
4. http/health.rs: Return JSON response
```

---

## Build Process

### Prerequisites
- CUDA Toolkit 11.8+ or 12.x
- CMake 3.18+
- C++ compiler (GCC 9+, Clang 10+)

### Build Commands
```bash
# Build worker (compiles CUDA automatically)
cargo build --release

# Test CUDA layer
cd cuda/build
cmake ..
make
ctest

# Test Rust layer
cargo test

# Run worker
./target/release/worker-orcd \
  --worker-id worker-abc \
  --model /models/llama-7b.gguf \
  --gpu-device 0 \
  --port 8001 \
  --callback-url http://localhost:9200/v2/internal/workers/ready
```

---

## Comparison with Other Binaries

### Orchestratord
- **Pure Rust** binary
- No GPU dependencies
- Multiple crates for modularity

### Pool-managerd
- **Pure Rust** binary
- Uses NVML (read-only GPU queries)
- Multiple crates for modularity

### Worker-orcd
- **Hybrid Rust + C++/CUDA** binary
- Uses CUDA (VRAM allocation, compute)
- **Single binary** (no separate crates for CUDA)

**Why different?**
- CUDA context is per-process and single-threaded
- Cannot split CUDA operations across crates
- Simpler to maintain as monolithic binary with embedded CUDA

---

## Migration from Old Design

### Old (Incorrect)
```
worker-orcd (Rust binary)
├── vram-policy (Rust crate) → calls CUDA
├── model-lifecycle (Rust crate) → calls CUDA
├── health-monitor (Rust crate) → calls CUDA
└── inference-api (Rust crate) → calls CUDA
```

**Problem**: Multiple Rust crates trying to coordinate CUDA operations leads to context conflicts.

### New (Correct)
```
worker-orcd (Hybrid binary)
├── src/ (Rust)
│   ├── main.rs → orchestration
│   ├── cuda/mod.rs → FFI bindings
│   └── http/ → HTTP handlers
└── cuda/ (C++/CUDA)
    ├── src/context.cpp → CUDA context
    ├── src/model.cpp → Model loading
    ├── src/inference.cu → Inference
    └── src/health.cpp → Health checks
```

**Solution**: All CUDA operations in C++, Rust calls via FFI. Single CUDA context, clear ownership.

---

## Specs

### Worker Specs
- `00_worker-orcd.md` — Worker responsibilities (WORK-3xxx)
- `01_cuda_ffi_boundary.md` — Rust/CUDA boundary (CUDA-4xxx)

### CUDA Specs
- `cuda/.specs/00_cuda_overview.md` — CUDA architecture (CUDA-5xxx)
- `cuda/.specs/01_context.md` — Context management (CUDA-5100)
- `cuda/.specs/02_model.md` — Model loading (CUDA-5200)
- `cuda/.specs/03_inference.md` — Inference execution (CUDA-5300)
- `cuda/.specs/04_health.md` — Health monitoring (CUDA-5400)

---

## Status

- ✅ Architecture defined
- ✅ Specs written
- ✅ Directory structure created
- ✅ FFI boundary defined
- ⏳ C++/CUDA implementation pending
- ⏳ Rust integration pending

---

**End of Architecture Document**
