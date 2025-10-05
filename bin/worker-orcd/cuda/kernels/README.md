# CUDA Kernels

CUDA C++ kernels for worker-orcd inference engine.

## M0 Kernel Set

Required for M0 pilot (Phase 3 of ARCHITECTURE_CHANGE_PLAN.md):

- **`embedding.cu`** ✅ — Token embedding lookup (FP16/FP32, coalesced access)
- **`cublas_wrapper`** ✅ — cuBLAS GEMM wrapper (FP16, deterministic, FP32 accumulation)
- **`sampling.cu`** ✅ — Complete sampling suite:
  - Temperature scaling (FP16/FP32, 0.0-2.0 range)
  - Greedy sampling (deterministic argmax)
  - Stochastic sampling (softmax + CDF, numerically stable)
- **`gemm.cu`** — Custom GEMM kernels (fallback, not needed for M0)
- **`rope.cu`** — Rotary Position Embedding (RoPE) for Llama
- **`attention.cu`** — Naive attention (prefill + decode, GQA support)
- **`rmsnorm.cu`** — RMSNorm layer normalization
- **Advanced sampling** — TODO (FT-019-extended: Top-P, Top-K, repetition penalty)

## GGUF Quantization Kernels

**Dequantization kernels for GGML quantization formats:**

- **`q6_k_dequant.cu`** ✅ — Q6_K dequantization (256 elements/block, 210 bytes)
- **`q5_0_dequant.cu`** ✅ — Q5_0 dequantization (32 elements/block, 22 bytes)
- **`q8_0_dequant.cu`** ✅ — Q8_0 dequantization (32 elements/block, 34 bytes)
- **`gguf_dequant.cuh`** ✅ — FFI header for Rust integration

These kernels move dequantization from CPU (Rust) to GPU (CUDA) for 100× performance improvement.
Each kernel uses coalesced memory access with one thread per element.

## Post-M0 Optimizations

- **FlashAttention** — Fused attention kernel for throughput
- **PagedAttention** — vLLM-style KV cache management
- **Quantization kernels** — Q4_K, Q5_1, AWQ, GPTQ support
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
