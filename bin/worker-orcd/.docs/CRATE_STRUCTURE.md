# Worker-orcd Crate Structure

Final crate organization aligned with `.specs/00_worker-orcd.md`.

## Crate Layout

```
bin/worker-orcd/
├── src/
│   ├── main.rs                    # Main binary entry point
│   ├── cuda_ffi/                  # Safe CUDA FFI boundary (WORKER-4400-4413)
│   │   └── mod.rs                 # SafeCudaPtr, CudaContext, bounds checking
│   └── inference_engine/          # Inference orchestration (WORKER-4700-4722)
│       └── (to be implemented)    # Forward pass, KV cache, token generation
├── cuda/
│   └── kernels/                   # CUDA C++ kernels (WORKER-4700-4703)
│       ├── README.md              # Kernel documentation
│       ├── gemm.cu                # cuBLAS GEMM wrapper
│       ├── rope.cu                # Rotary Position Embedding
│       ├── attention.cu           # Attention (prefill + decode, GQA)
│       ├── rmsnorm.cu             # RMSNorm normalization
│       └── sampling.cu            # Token sampling (greedy, top-k, temperature)
└── bdd/                           # BDD test scenarios

bin/worker-orcd-crates/
├── api/                           # HTTP endpoints (WORKER-4200-4253)
│   └── src/lib.rs                 # Plan, Commit, Ready, Execute endpoints
├── vram-residency/                # VRAM-only policy (WORKER-4100-4122)
│   └── src/lib.rs                 # ModelShardHandle, seal verification
├── model-loader/                  # Model validation (WORKER-4310-4314)
│   └── src/lib.rs                 # GGUF parsing, hash verification
├── capability-matcher/            # MCD/ECP matching (WORKER-4600-4623)
│   └── src/lib.rs                 # Compatibility checking
├── scheduler/                     # Single-slot scheduler (M0)
│   └── src/lib.rs                 # Job state tracking
└── error-handler/                 # (To be removed - distribute errors)

libs/shared-crates/
└── input-validation/              # Request validation (WORKER-4300-4305)
    └── (shared across binaries)   # Prompt, params, model_ref validation
```

## Responsibility Matrix

| Crate | Spec Section | Responsibility |
|-------|--------------|----------------|
| `api` | WORKER-4200-4253 | HTTP endpoints, auth, SSE streaming |
| `vram-residency` | WORKER-4100-4122 | VRAM-only policy, sealed shards, seal integrity |
| `model-loader` | WORKER-4310-4314 | Model validation, GGUF parsing, hash verification |
| `capability-matcher` | WORKER-4600-4623 | MCD/ECP matching, compatibility checks |
| `scheduler` | M0 scope | Single-slot job scheduling |
| `cuda_ffi` | WORKER-4400-4413 | Safe CUDA FFI, bounds checking, error mapping |
| `cuda/kernels` | WORKER-4700-4703 | CUDA kernels (GEMM, RoPE, attention, RMSNorm, sampling) |
| `inference_engine` | WORKER-4700-4722 | Forward pass orchestration, KV cache, token generation |
| `input-validation` | WORKER-4300-4305 | Request validation (shared) |

## Security Tiers

**Tier 1 (Critical)**: `vram-residency`, `cuda_ffi`
- Deny all unsafe patterns
- Maximum security enforcement

**Tier 2 (High)**: `api`, `model-loader`, `capability-matcher`
- Deny unwrap/panic, warn on arithmetic
- Strict enforcement

**Tier 3 (Medium)**: `scheduler`, `error-handler`
- Warn on unwrap/panic
- Moderate enforcement

## Changes from Initial Speculation

### ✅ Kept & Enhanced
- `api` — HTTP endpoints (added auth requirements)
- `vram-residency` — VRAM-only policy (added seal crypto)
- `model-loader` — Model validation (added bounds checking)

### ✅ Added
- `capability-matcher` — MCD/ECP matching (new requirement)
- `cuda_ffi` — Safe CUDA FFI boundary (security requirement)
- `cuda/kernels` — CUDA C++ kernels (architecture requirement)

### ✅ Renamed
- `execution-planner` → `scheduler` (M0-appropriate scope)

### ✅ Split
- `inference` → `cuda/kernels` + `inference_engine` (separation of concerns)

### ⏳ To Remove
- `error-handler` — Distribute errors to each crate (use `thiserror`)

### 📍 Location Decisions
- **CUDA FFI**: `bin/worker-orcd/src/cuda_ffi/` (Rust side of FFI boundary)
- **CUDA kernels**: `bin/worker-orcd/cuda/kernels/` (C++ side, compiled via build.rs)
- **Inference engine**: `bin/worker-orcd/src/inference_engine/` (Rust orchestration)
- **Input validation**: `libs/shared-crates/input-validation/` (shared across binaries)

## Build Process

1. **CUDA kernels** compiled via `build.rs`:
   - Uses `nvcc` or CMake
   - Produces static library
   - Linked into Rust binary

2. **Rust crates** compiled via Cargo:
   - FFI boundary wraps CUDA library
   - Inference engine calls FFI
   - API layer exposes HTTP endpoints

## Testing Strategy

- **Unit tests**: Each crate has `#[cfg(test)]` modules
- **Integration tests**: `bin/worker-orcd/tests/`
- **BDD scenarios**: `bin/worker-orcd/bdd/`
- **Proof bundles**: Per `.specs/00_proof-bundle.md`

## Next Steps

1. ✅ Spec created (`.specs/00_worker-orcd.md`)
2. ✅ Crate scaffolding complete
3. ⏳ Implement CUDA kernels (Phase 3, Task Group 3)
4. ⏳ Implement FFI boundary (Phase 3, Task Group 2)
5. ⏳ Implement inference engine (Phase 3, Task Group 4)
6. ⏳ Wire up HTTP endpoints (Phase 3, Task Group 1)
7. ⏳ Integration testing (Phase 4)

---

**Status**: Scaffolding complete, ready for M0 implementation  
**Date**: 2025-10-01  
**Spec**: `.specs/00_worker-orcd.md`
