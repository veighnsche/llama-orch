# worker-orcd CUDA Backend

**C++/CUDA implementation for model loading and inference.**

---

## Overview

This directory contains the **GPU-accelerated inference engine** for `worker-orcd`:

- **Model loading**: Parse GGUF, allocate VRAM, load weights
- **Inference kernels**: Attention, MLP, sampling, KV cache
- **Memory management**: cudaMalloc/cudaFree, VRAM tracking
- **Deterministic sampling**: Seeded RNG for reproducibility

**Language**: C++17 + CUDA  
**Build system**: CMake (invoked by Rust `build.rs`)  
**FFI**: C-compatible API exposed to Rust

---

## Directory Structure

```
cuda/
├── CMakeLists.txt          ← Build configuration
├── README.md               ← You are here
├── include/                ← Public C API headers
│   ├── inference.h         ← Main inference API
│   ├── model_loader.h      ← Model loading
│   └── types.h             ← Shared types
└── src/                    ← Implementation
    ├── inference.cu        ← Inference orchestration
    ├── model_loader.cu     ← GGUF parsing, weight loading
    ├── memory.cu           ← GPU memory management
    └── kernels/            ← CUDA kernels
        ├── attention.cu    ← Attention mechanism
        ├── sampling.cu     ← Token sampling
        └── mlp.cu          ← MLP layers
```

---

## API Contract

### Public C API (include/inference.h)

```c
// Load model onto GPU
WorkerError worker_model_load(
    const WorkerModelConfig* config,
    WorkerModel** out_model,
    uint64_t* out_vram_bytes
);

// Run inference (single token)
WorkerError worker_model_infer(
    WorkerModel* model,
    const WorkerInferenceRequest* request,
    WorkerInferenceResponse* response
);

// Free model resources
WorkerError worker_model_free(WorkerModel* model);
```

**Rust calls these functions** via FFI (`../src/ffi.rs`).

---

## Build

### Standalone (for testing)

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Via Rust (normal workflow)

```bash
cd ..  # Back to worker-orcd/
cargo build --release
```

Rust's `build.rs` automatically invokes CMake to compile this directory.

---

## Requirements

- **CUDA Toolkit**: 11.8+ (12.x recommended)
- **CMake**: 3.18+
- **Compiler**: GCC 11+ or Clang 14+
- **GPU**: NVIDIA with compute capability 7.5+ (Turing, Ampere, Ada)

---

## CUDA Architecture Targets

Currently building for:
- **sm_75**: Turing (RTX 20xx, GTX 16xx)
- **sm_80**: Ampere (RTX 30xx, A100)
- **sm_86**: Ampere (RTX 30xx mobile)
- **sm_89**: Ada Lovelace (RTX 40xx)

Adjust in `CMakeLists.txt` if targeting different GPUs.

---

## Development

### Code Style

- **C++17 standard**: Modern C++, avoid legacy patterns
- **CUDA best practices**: Coalesced memory access, shared memory, streams
- **Error handling**: Return `WorkerError` enum, never throw exceptions
- **Memory safety**: RAII wrappers, no raw `new`/`delete`
- **Naming**: `snake_case` for C API, `PascalCase` for C++ classes

### Testing

CUDA code is tested via:
1. **Rust FFI tests** (`../tests/ffi_smoke.rs`)
2. **Integration tests** (`../tests/integration.rs`)
3. **Standalone C++ tests** (optional, in `tests/` subdirectory)

---

## Performance Notes

- **Model loading**: ~2-3 seconds for Qwen-0.5B (cold start)
- **Inference**: ~100 tok/s on RTX 3090 (Qwen-0.5B, batch=1)
- **VRAM usage**: ~500MB for Qwen-0.5B (Q4_K_M, 2K context)

---

## References

- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **GGUF Format**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **FFI Integration**: `../.docs/CUDA_INTEGRATION.md`

---

## Related

- **Worker README**: `../README.md`
- **FFI Bindings**: `../src/ffi.rs`
- **Build Script**: `../build.rs`
- **Integration Guide**: `../.docs/CUDA_INTEGRATION.md`

---

**Status**: Early development  
**License**: GPL-3.0-or-later
