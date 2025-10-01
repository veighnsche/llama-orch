# worker-orcd

**The GPU worker daemon that loads models and executes inference jobs.**

---

## Overview

`worker-orcd` is the inference worker process that:

1. **Loads LLM models** onto GPU VRAM (single model, single GPU/device-mask)
2. **Notifies pool-managerd** via HTTP callback when ready
3. **Executes inference jobs** via internal RPC protocol
4. **Reports status** and metrics back to pool-managerd

**Architecture**: One worker per model per GPU. Workers are spawned and managed by `pool-managerd`.

---

## Status

⚠️ **Early development** — Worker daemon and crates are being planned and implemented.

**Current phase**: Development planning and initial implementation

See:
- **Design doc**: `.docs/WORKER_READINESS_CALLBACK_DESIGN.md`
- **Test models**: `.docs/testing/TEST_MODELS.md`
- **Specs**: `.specs/35-worker-adapters.md`, `.specs/40-worker-adapters-llamacpp-http.md`

---

## Responsibilities

### Model Loading
- Load GGUF models via llama.cpp bindings
- Fail fast on insufficient VRAM
- Report actual VRAM usage to pool-managerd

### Readiness Callback
- POST to pool-managerd's `/v2/internal/workers/ready` endpoint
- Include: worker_id, pool_id, endpoint, VRAM usage, capabilities
- Retry on transient failures, fail fast on 4xx rejections

### Inference Execution
- Accept jobs via internal RPC (HTTP or gRPC, TBD)
- Generate tokens with deterministic seeding
- Stream results back to orchestratord

### Status Reporting
- Heartbeat to pool-managerd
- Report slot utilization, queue depth
- Emit Prometheus metrics

---

## Architecture

### Language Split: Rust + C++/CUDA

`worker-orcd` is a **hybrid binary**:

- **Rust** (`src/`): HTTP server, RPC protocol, lifecycle management, metrics
- **C++/CUDA** (`cuda/`): Model loading, inference kernels, GPU memory management

**FFI boundary** (`src/ffi.rs`): Rust calls into CUDA via C-compatible interface.

**Build**: `build.rs` invokes `nvcc` or CMake to compile `cuda/` → static library → linked into Rust binary.

### Directory Structure

```
bin/worker-orcd/
├── src/               ← Rust implementation
│   ├── main.rs
│   ├── server.rs
│   └── ffi.rs         ← FFI bindings to CUDA
├── cuda/              ← C++/CUDA implementation
│   ├── CMakeLists.txt
│   ├── include/
│   └── src/
│       └── *.cu
└── build.rs           ← Compiles CUDA code
```

### Crates

Worker functionality is split into modular crates under `bin/worker-orcd-crates/`:

- **TBD** — Crates structure being planned

**Note**: The `handoff-watcher` crate is deprecated and will be removed in favor of direct HTTP callbacks.

---

## Configuration

Workers are configured via command-line arguments or JSON config passed by pool-managerd:

```json
{
  "worker_id": "gpu-0",
  "pool_id": "pool-0",
  "model_path": "/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
  "rpc_port": 8001,
  "readiness_callback_url": "http://localhost:9200/v2/internal/workers/ready",
  "gpu_device": 0,
  "slots_total": 4,
  "capabilities": {
    "engine_version": "llama.cpp-b1234",
    "model_ref": "qwen2.5-0.5b-instruct"
  }
}
```

---

## Testing

### Test Models

Test models are stored in `.test-models/` at workspace root:

- **Qwen2.5-0.5B-Instruct** (352MB) — Primary test model
- **TinyLlama-1.1B-Chat** (600MB) — Secondary for E2E tests

See `.docs/testing/TEST_MODELS.md` for download instructions.

### Running Tests

```bash
# Ensure test model is downloaded first
cd ../../.test-models/qwen
wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

# Run worker tests
cd ../../bin/worker-orcd
cargo test -- --nocapture
```

---

## Development Plan

📋 **Planning in progress** — See `bin/worker-orcd/.specs/` (to be created)

Key milestones:
1. ✅ Test model infrastructure (`.test-models/`)
2. 🔄 Worker daemon skeleton and crate structure
3. ⏳ Model loading with llama.cpp bindings
4. ⏳ HTTP readiness callback implementation
5. ⏳ RPC server for inference jobs
6. ⏳ Integration tests with pool-managerd

---

## Related

- **Pool Manager**: `bin/pool-managerd/`
- **Orchestrator**: `bin/orchestratord/`
- **Worker Adapters (legacy)**: `bin/orchestratord-crates/agentic-api/` (HTTP adapters for external engines)
- **Test Harness**: `test-harness/e2e-haiku/`

---

**License**: GPL-3.0-or-later  
**Maintainers**: @llama-orch-maintainers