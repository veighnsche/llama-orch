# CUDA Kernels

CUDA C++ kernels for worker-orcd inference engine.

## M0 Kernel Set

Required for M0 pilot (Phase 3 of ARCHITECTURE_CHANGE_PLAN.md):

- **`gemm.cu`** — cuBLAS matrix multiplication (SGEMM wrapper)
- **`rope.cu`** — Rotary Position Embedding (RoPE) for Llama
- **`attention.cu`** — Naive attention (prefill + decode, GQA support)
- **`rmsnorm.cu`** — RMSNorm layer normalization
- **`sampling.cu`** — Token sampling (greedy, top-k, temperature)

## Post-M0 Optimizations

- **FlashAttention** — Fused attention kernel for throughput
- **PagedAttention** — vLLM-style KV cache management
- **Quantization kernels** — Q5_1, Q8_0, AWQ, GPTQ support
- **Fused kernels** — Combined operations to reduce memory bandwidth

## Build System

Kernels are compiled via `build.rs` in the worker-orcd binary:
- Uses `nvcc` or CMake to compile `.cu` files
- Links as static library into Rust binary
- Exposes C-compatible interface for FFI

## Security Requirements

Per SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #11:
- All kernel launches must have bounds checking
- Tensor dimensions validated before launch
- No buffer overflows in shared memory
- Error handling for all CUDA API calls

## Testing

Each kernel should have:
- Unit tests (compare against reference implementation)
- Bounds checking tests (invalid dimensions rejected)
- Performance benchmarks
- Determinism tests (same input → same output)

## References

- ARCHITECTURE_CHANGE_PLAN.md — Phase 3, Task Group 3
- SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md — Issue #11 (unsafe CUDA FFI)
- .specs/00_worker-orcd.md — WORKER-4400-4413 (FFI safety)
