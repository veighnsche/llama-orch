# CUDA Kernels SPEC — Inference Compute Kernels (WORKER-4xxx)

**Status**: Draft  
**Applies to**: `bin/worker-orcd/cuda/kernels/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

This directory contains CUDA C++ kernels for inference execution. M0 implements naive kernels; post-M0 adds optimized fused kernels.

**Parent spec**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

## 1. M0 Kernel Set

### 1.1 Required Kernels

- [WORKER-4700] M0 workers MUST implement: cuBLAS GEMM, RoPE (llama variant), naive attention (prefill + decode), RMSNorm, greedy sampling.
- [WORKER-4701] Workers MUST support GGUF model loading with Q4_0 quantization (dequantize on load for M0).
- [WORKER-4702] Workers MUST implement KV cache management in VRAM with bounded size.
- [WORKER-4703] Workers MUST support single-GPU inference only for M0 (no tensor-parallel).

### 1.2 Kernel Files

| File | Purpose | Status |
|------|---------|--------|
| `gemm.cu` | cuBLAS GEMM wrapper | Stub |
| `rope.cu` | Rotary Position Embedding | Stub |
| `attention.cu` | Attention (prefill + decode) | Stub |
| `rmsnorm.cu` | RMSNorm normalization | Stub |
| `sampling.cu` | Token sampling | Stub |

---

## 2. GEMM Kernel (`gemm.cu`)

### 2.1 Requirements

- [KERNEL-GEMM-001] MUST wrap cuBLAS `cublasSgemm` for single-precision matrix multiplication.
- [KERNEL-GEMM-002] MUST support batched GEMM via `cublasSgemmBatched`.
- [KERNEL-GEMM-003] MUST validate dimensions (M, N, K > 0, no overflow).
- [KERNEL-GEMM-004] MUST handle cuBLAS errors and map to error codes.
- [KERNEL-GEMM-005] MUST optimize for row-major layout (Llama/Transformer format).

### 2.2 API

```c
extern "C" {
int cuda_gemm(
    int M, int N, int K,
    const float* A,
    const float* B,
    float* C
);
}
```

**Parameters**:
- `M, N, K` — Matrix dimensions
- `A` — Input matrix [M x K]
- `B` — Input matrix [K x N]
- `C` — Output matrix [M x N]

**Returns**: 0 on success, error code on failure

---

## 3. RoPE Kernel (`rope.cu`)

### 3.1 Requirements

- [KERNEL-ROPE-001] MUST implement RoPE (Rotary Position Embedding) for Llama models.
- [KERNEL-ROPE-002] MUST support `rope_llama` variant (freq_base=10000).
- [KERNEL-ROPE-003] MUST support `rope_neox` variant for compatibility.
- [KERNEL-ROPE-004] MUST validate dimensions (seq_len, head_dim, num_heads).
- [KERNEL-ROPE-005] MUST apply rotation to Q and K tensors.

### 3.2 RoPE Formula

```
theta_i = freq_base^(-2i/d) for i in [0, d/2)
x_rotated[2i]   = x[2i] * cos(m*theta_i) - x[2i+1] * sin(m*theta_i)
x_rotated[2i+1] = x[2i] * sin(m*theta_i) + x[2i+1] * cos(m*theta_i)
```

Where `m` is the position index, `d` is head_dim.

### 3.3 API

```c
extern "C" {
int cuda_rope(
    float* q,           // [batch, seq_len, num_heads, head_dim]
    float* k,           // [batch, seq_len, num_kv_heads, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float freq_base
);
}
```

---

## 4. Attention Kernel (`attention.cu`)

### 4.1 Requirements

- [KERNEL-ATTN-001] MUST implement prefill attention (full Q·K^T, softmax, ·V).
- [KERNEL-ATTN-002] MUST implement decode attention (single query, cached K/V).
- [KERNEL-ATTN-003] MUST support GQA (Grouped Query Attention) with configurable num_kv_heads.
- [KERNEL-ATTN-004] MUST apply causal masking for autoregressive generation.
- [KERNEL-ATTN-005] MUST use naive implementation for M0 (no FlashAttention fusion).

### 4.2 Prefill Attention

```
scores = Q @ K^T / sqrt(head_dim)  # [batch, num_heads, seq_q, seq_k]
attn = softmax(scores, dim=-1)
output = attn @ V  # [batch, num_heads, seq_q, head_dim]
```

### 4.3 Decode Attention

```
q_new = Q[:, -1, :, :]  # [batch, num_heads, 1, head_dim]
scores = q_new @ K_cache^T / sqrt(head_dim)  # [batch, num_heads, 1, cache_len]
attn = softmax(scores, dim=-1)
output = attn @ V_cache  # [batch, num_heads, 1, head_dim]
```

### 4.4 API

```c
extern "C" {
int cuda_attention_prefill(
    const float* q, const float* k, const float* v,
    float* output,
    int batch_size, int seq_q, int seq_k,
    int num_heads, int num_kv_heads, int head_dim
);

int cuda_attention_decode(
    const float* q,
    const float* k_cache, const float* v_cache,
    float* output,
    int batch_size, int cache_len,
    int num_heads, int num_kv_heads, int head_dim
);
}
```

---

## 5. RMSNorm Kernel (`rmsnorm.cu`)

### 5.1 Requirements

- [KERNEL-NORM-001] MUST implement RMSNorm layer normalization.
- [KERNEL-NORM-002] MUST fuse with weight multiplication where possible.
- [KERNEL-NORM-003] MUST handle epsilon for numerical stability.
- [KERNEL-NORM-004] MUST validate dimensions.

### 5.2 RMSNorm Formula

```
rms = sqrt(mean(x^2) + eps)
output = (x / rms) * weight
```

### 5.3 API

```c
extern "C" {
int cuda_rmsnorm(
    const float* input,     // [batch, seq_len, hidden_dim]
    const float* weight,    // [hidden_dim]
    float* output,          // [batch, seq_len, hidden_dim]
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps
);
}
```

---

## 6. Sampling Kernel (`sampling.cu`)

### 6.1 Requirements

- [KERNEL-SAMPLE-001] MUST implement greedy sampling (argmax).
- [KERNEL-SAMPLE-002] SHOULD implement top-k sampling (optional for M0).
- [KERNEL-SAMPLE-003] MUST implement temperature scaling.
- [KERNEL-SAMPLE-004] MUST ensure determinism with seeded RNG.

### 6.2 Greedy Sampling

```
token = argmax(logits)
```

### 6.3 Temperature Scaling

```
logits_scaled = logits / temperature
```

(temperature < 1 = more deterministic, > 1 = more random)

### 6.4 API

```c
extern "C" {
int cuda_greedy_sampling(
    const float* logits,    // [batch, vocab_size]
    int* output_tokens,     // [batch]
    int batch_size,
    int vocab_size
);

int cuda_temperature_scaling(
    float* logits,          // [batch, vocab_size] (in-place)
    int batch_size,
    int vocab_size,
    float temperature
);
}
```

---

## 7. Determinism

- [WORKER-4710] Workers MUST produce deterministic token streams for identical `{prompt, seed, params, model_digest}` (per ORCH-3045).
- [WORKER-4711] Workers MUST use deterministic CUDA kernels (disable non-deterministic cuBLAS algorithms).
- [WORKER-4712] Workers MUST seed RNG with provided `seed` parameter before inference.
- [WORKER-4713] Workers MUST document any sources of non-determinism (e.g., floating-point rounding).

---

## 8. Memory Safety & Bounds Checking

### 8.1 Security Requirements

Per SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #11:
- [KERNEL-SAFE-001] All kernel launches MUST have bounds checking.
- [KERNEL-SAFE-002] Tensor dimensions MUST be validated before launch.
- [KERNEL-SAFE-003] No buffer overflows in shared memory.
- [KERNEL-SAFE-004] Error handling for all CUDA API calls.

### 8.2 Validation Pattern

```c
// Validate dimensions
if (M <= 0 || N <= 0 || K <= 0) {
    return ERROR_INVALID_DIMENSIONS;
}

// Check for overflow
if (M > INT_MAX / N || K > INT_MAX / N) {
    return ERROR_OVERFLOW;
}

// Launch kernel with validated dimensions
kernel<<<grid, block>>>(params);

// Check for errors
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    return map_cuda_error(err);
}
```

---

## 9. Post-M0 Optimizations

- [WORKER-4720] Post-M0 workers SHOULD implement: FlashAttention, continuous batching, fused kernels, additional quant formats (Q5_1, Q8_0).
- [WORKER-4721] Post-M0 workers SHOULD support tensor-parallel execution with NCCL.
- [WORKER-4722] Post-M0 workers SHOULD optimize for vLLM-class throughput with PagedAttention.

---

## 10. Build System

Kernels are compiled via `build.rs` in worker-orcd binary:
- Uses `nvcc` or CMake to compile `.cu` files
- Links as static library into Rust binary
- Exposes C-compatible interface for FFI

---

## 11. Testing

Each kernel MUST have:
- Unit tests (compare against reference implementation)
- Bounds checking tests (invalid dimensions rejected)
- Performance benchmarks
- Determinism tests (same input → same output)

---

## 12. Traceability

**Code**: `bin/worker-orcd/cuda/kernels/*.cu`  
**FFI**: `bin/worker-orcd/src/cuda_ffi/mod.rs`  
**Tests**: `bin/worker-orcd/cuda/kernels/tests/`  
**Parent**: `bin/worker-orcd/.specs/00_worker-orcd.md` §8
