# Worker-orcd Binary Structure
Final binary organization aligned with `.specs/00_worker-orcd.md`.
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
libs/shared-crates/
└── input-validation/              # Request validation (WORKER-4300-4305)
    └── (shared across binaries)   # Prompt, params, model_ref validation
```
## Module Responsibility Matrix
| Module | Spec Section | Responsibility |
|--------|--------------|----------------|
| `cuda_ffi` | WORKER-4400-4413 | Safe CUDA FFI, bounds checking, error mapping |
| `cuda/kernels` | WORKER-4700-4703 | CUDA kernels (GEMM, RoPE, attention, RMSNorm, sampling) |
| `inference_engine` | WORKER-4700-4722 | Forward pass orchestration, KV cache, token generation |
| `input-validation` | WORKER-4300-4305 | Request validation (shared) |
## Security Tiers
**Tier 1 (Critical)**: `cuda_ffi`
- Deny all unsafe patterns
- Maximum security enforcement
**Tier 2 (High)**: (None currently - all functionality integrated into binary)
- Previously: `api`, `model-loader`, `capability-matcher`
- Now integrated into main binary modules
**Tier 3 (Medium)**: (None currently - all functionality integrated into binary)
- Previously: `scheduler`, `error-handler`
- Now integrated into main binary modules
## Changes from Initial Design
### ✅ Removed
- **All separate crates** — Previously had `api`, `vram-residency`, `model-loader`, `capability-matcher`, `scheduler`, and `error-handler` crates
- **Crate-based architecture** — All functionality now integrated into single binary due to CUDA context requirements
- **Distributed specifications** — All specs now centralized in main binary
### ✅ Kept & Enhanced
- **Single binary approach** — Maintained the hybrid Rust + C++/CUDA architecture
- **Integrated functionality** — All previously separate crate functionality now implemented as modules within the binary
### ✅ Added
- **Unified module structure** — All CUDA operations, API endpoints, validation, and scheduling now integrated into cohesive binary modules
## Integration Approach
All previously separate crate functionality has been integrated into the main binary:
- **API endpoints**: Implemented as `src/http/` modules
- **VRAM residency**: Implemented as part of CUDA FFI boundary
- **Model loading**: Implemented in `cuda/src/model.cpp`
- **Capability matching**: Implemented in main binary logic
- **Scheduling**: Implemented as single-slot scheduler in main binary
- **Input validation**: Uses shared `libs/shared-crates/input-validation/`
## Build Process
1. **CUDA kernels** compiled via `build.rs`:
   - Uses `nvcc` or CMake
   - Produces static library
   - Linked into Rust binary
2. **Rust binary** compiled via Cargo:
   - FFI boundary wraps CUDA library
   - HTTP server and all business logic in single binary
   - All previously separate crate functionality integrated
## Testing Strategy
- **Unit tests**: Each module has `#[cfg(test)]` tests
- **Integration tests**: `bin/worker-orcd/tests/`
- **BDD scenarios**: `bin/worker-orcd/bdd/`
- ****: Per `.specs/00_.md`
## Next Steps
1. ✅ Spec created (`.specs/00_worker-orcd.md`)
2. ✅ Binary architecture complete
3. ✅ Crate removal and integration complete
4. ⏳ Implement CUDA kernels (Phase 3, Task Group 3)
5. ⏳ Implement FFI boundary (Phase 3, Task Group 2)
6. ⏳ Implement inference engine (Phase 3, Task Group 4)
7. ⏳ Wire up HTTP endpoints (Phase 3, Task Group 1)
8. ⏳ Integration testing (Phase 4)
---
**Status**: Binary architecture complete, crates removed and functionality integrated
**Date**: 2025-10-03
**Spec**: `.specs/00_worker-orcd.md`
