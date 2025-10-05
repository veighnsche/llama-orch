# FT-052: FP16 Attention Kernels

**Team**: Foundation-Alpha  
**Sprint**: Sprint 6 - FP16 Optimization (Post-M0)  
**Size**: M (2 days)  
**Priority**: High (Post-M0)  
**Spec Ref**: M0-W-1214, M0-W-1430

---

## ⚠️ Prerequisites

**Requires M0 completion:**
- Working FP32 attention kernels in `cuda/kernels/gqa_attention.cu`
- Functional prefill/decode implementation
- Performance baseline measurements

**Current State**: `gqa_attention.cu` has stub implementations. Implement FP32 baseline first.

---

## Story Description

Optimize attention kernels (GQA prefill/decode) to use FP16 throughout the pipeline: Q·K^T computation, softmax, attention·V. This reduces memory bandwidth and improves throughput while maintaining numerical accuracy.

---

## Acceptance Criteria

- [ ] FP16 attention prefill kernel (full sequence)
- [ ] FP16 attention decode kernel (single token)
- [ ] FP16 softmax with numerical stability (log-sum-exp trick)
- [ ] FP16 attention score scaling (1/sqrt(head_dim))
- [ ] Causal masking support (autoregressive generation)
- [ ] Numerical accuracy validation vs FP32 (tolerance: 1e-2)
- [ ] Unit tests for all attention variants
- [ ] Performance benchmarks (expect 1.4-1.8x speedup)
- [ ] Memory bandwidth profiling

---

## Dependencies

**Upstream**: FT-051 (FP16 GEMM, Day 2)  
**Downstream**: FT-053 (KV cache optimization)

---

## Technical Details

### Current State

`cuda/kernels/gqa_attention.cu` has stub implementations using FP16 types but no actual computation. Attention is currently naive (no optimization).

### Implementation Plan

**Phase 1: FP16 Softmax** (Day 1)

```cpp
// cuda/kernels/attention.cu

/**
 * FP16 Softmax with numerical stability
 * 
 * Uses log-sum-exp trick: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 * Operates on attention scores [batch, num_heads, seq_q, seq_k]
 */
__global__ void softmax_fp16_attention(
    half* scores,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k
) {
    // Each block processes one attention head for one query position
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int q_pos = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch >= batch_size || head >= num_heads || q_pos >= seq_q) {
        return;
    }
    
    // Pointer to this query's scores [seq_k]
    half* query_scores = scores + 
        (batch * num_heads * seq_q * seq_k) +
        (head * seq_q * seq_k) +
        (q_pos * seq_k);
    
    // Shared memory for reduction
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];
    
    // Phase 1: Find max (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < seq_k; i += blockDim.x) {
        float val = __half2float(query_scores[i]);
        local_max = fmaxf(local_max, val);
    }
    
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    float global_max = shared_max[0];
    __syncthreads();
    
    // Phase 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_k; i += blockDim.x) {
        float val = __half2float(query_scores[i]);
        float exp_val = expf(val - global_max);
        query_scores[i] = __float2half(exp_val);  // Store exp values
        local_sum += exp_val;
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce to find global sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    float global_sum = shared_sum[0];
    __syncthreads();
    
    // Phase 3: Normalize
    for (int i = tid; i < seq_k; i += blockDim.x) {
        float val = __half2float(query_scores[i]);
        query_scores[i] = __float2half(val / global_sum);
    }
}
```

**Phase 2: FP16 Attention Prefill** (Day 1-2)

```cpp
/**
 * GQA Attention Prefill (FP16 optimized)
 * 
 * Algorithm:
 * 1. Compute Q @ K^T using FP16 GEMM (from FT-051)
 * 2. Scale by 1/sqrt(head_dim)
 * 3. Apply causal mask (upper triangular)
 * 4. Apply softmax
 * 5. Compute attention @ V using FP16 GEMM
 * 6. Write K, V to cache
 */
int cuda_gqa_attention_prefill_fp16(
    half* output,
    const half* q,
    const half* k,
    const half* v,
    half* kv_cache_k,
    half* kv_cache_v,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cublasHandle_t cublas_handle
) {
    // Validate dimensions
    if (num_q_heads % num_kv_heads != 0) {
        fprintf(stderr, "GQA: num_q_heads must be divisible by num_kv_heads\n");
        return -1;
    }
    
    // Allocate temporary storage for attention scores
    half* d_scores;
    size_t scores_size = batch_size * num_q_heads * seq_len * seq_len * sizeof(half);
    cudaMalloc(&d_scores, scores_size);
    
    // Scale factor: 1/sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // For each batch and head
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_q_heads; ++h) {
            int kv_head = h / (num_q_heads / num_kv_heads);
            
            // Pointers to Q, K for this head
            const half* q_head = q + (b * seq_len * num_q_heads * head_dim) +
                                      (h * head_dim);
            const half* k_head = k + (b * seq_len * num_kv_heads * head_dim) +
                                      (kv_head * head_dim);
            
            half* scores_head = d_scores + (b * num_q_heads * seq_len * seq_len) +
                                            (h * seq_len * seq_len);
            
            // 1. Compute Q @ K^T (FP16 GEMM)
            // scores = Q @ K^T * scale
            // Q: [seq_len, head_dim], K^T: [head_dim, seq_len]
            // scores: [seq_len, seq_len]
            cuda_gemm_fp16(
                cublas_handle,
                seq_len, seq_len, head_dim,
                q_head, k_head,  // Note: K needs transpose
                scores_head,
                scale, 0.0f,
                true  // Use Tensor Cores
            );
        }
    }
    
    // 2. Apply causal mask (set upper triangle to -inf)
    launch_causal_mask_fp16(d_scores, batch_size, num_q_heads, seq_len);
    
    // 3. Apply softmax
    dim3 grid(seq_len, num_q_heads, batch_size);
    dim3 block(256);
    softmax_fp16_attention<<<grid, block>>>(
        d_scores, batch_size, num_q_heads, seq_len, seq_len
    );
    
    // 4. Compute attention @ V (FP16 GEMM)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_q_heads; ++h) {
            int kv_head = h / (num_q_heads / num_kv_heads);
            
            half* scores_head = d_scores + (b * num_q_heads * seq_len * seq_len) +
                                            (h * seq_len * seq_len);
            const half* v_head = v + (b * seq_len * num_kv_heads * head_dim) +
                                      (kv_head * head_dim);
            half* output_head = output + (b * seq_len * num_q_heads * head_dim) +
                                          (h * head_dim);
            
            // output = attention @ V
            // attention: [seq_len, seq_len], V: [seq_len, head_dim]
            // output: [seq_len, head_dim]
            cuda_gemm_fp16(
                cublas_handle,
                seq_len, head_dim, seq_len,
                scores_head, v_head,
                output_head,
                1.0f, 0.0f,
                true
            );
        }
    }
    
    // 5. Write K, V to cache
    cudaMemcpy(kv_cache_k, k, 
               batch_size * seq_len * num_kv_heads * head_dim * sizeof(half),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(kv_cache_v, v,
               batch_size * seq_len * num_kv_heads * head_dim * sizeof(half),
               cudaMemcpyDeviceToDevice);
    
    cudaFree(d_scores);
    return 0;
}
```

**Phase 3: FP16 Attention Decode** (Day 2)

```cpp
/**
 * GQA Attention Decode (FP16 optimized)
 * 
 * Single token generation with KV cache.
 */
int cuda_gqa_attention_decode_fp16(
    half* output,
    const half* q,
    const half* k_current,
    const half* v_current,
    half* kv_cache_k,
    half* kv_cache_v,
    int batch_size,
    int cache_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cublasHandle_t cublas_handle
) {
    // Append current K, V to cache
    // Then compute attention over full cache (cache_len + 1 positions)
    
    // Similar to prefill but:
    // - Q is [batch, 1, num_q_heads, head_dim]
    // - K_cache is [batch, cache_len+1, num_kv_heads, head_dim]
    // - scores is [batch, num_q_heads, 1, cache_len+1]
    
    // Implementation follows prefill pattern with adjusted dimensions
    return 0;
}
```

**Phase 4: Causal Masking** (Day 2)

```cpp
/**
 * Apply causal mask to attention scores (FP16)
 * 
 * Sets upper triangle to -inf for autoregressive generation.
 */
__global__ void causal_mask_fp16(
    half* scores,
    int batch_size,
    int num_heads,
    int seq_len
) {
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int q_pos = blockIdx.x;
    int k_pos = threadIdx.x;
    
    if (batch >= batch_size || head >= num_heads || 
        q_pos >= seq_len || k_pos >= seq_len) {
        return;
    }
    
    // Causal mask: can only attend to positions <= q_pos
    if (k_pos > q_pos) {
        int idx = (batch * num_heads * seq_len * seq_len) +
                  (head * seq_len * seq_len) +
                  (q_pos * seq_len) + k_pos;
        scores[idx] = __float2half(-INFINITY);
    }
}
```

---

## Files to Create/Modify

**Create**:
- `cuda/tests/test_attention_fp16.cu` - FP16 attention unit tests
- `cuda/tests/benchmark_attention.cu` - Performance benchmarks

**Modify**:
- `cuda/kernels/gqa_attention.cu` - Add FP16 implementations
- `cuda/kernels/attention.cu` - Add FP16 softmax
- `cuda/include/attention.h` - Add FP16 function declarations

---

## Testing Strategy

### Unit Tests (6 tests)

1. **test_softmax_fp16_numerical** - Compare vs FP32 softmax
2. **test_attention_prefill_fp16** - Full sequence attention
3. **test_attention_decode_fp16** - Single token with cache
4. **test_causal_mask_fp16** - Validate masking
5. **test_gqa_grouping** - Verify KV head grouping
6. **test_attention_fp16_accuracy** - End-to-end vs FP32

### Integration Tests (2 tests)

1. **test_transformer_layer_fp16** - Full layer with FP16
2. **test_inference_fp16** - Multi-token generation

### Performance Tests (3 benchmarks)

1. **bench_attention_prefill** - Seq lengths: 32, 128, 512
2. **bench_attention_decode** - Cache lengths: 32, 128, 512
3. **bench_attention_memory_bandwidth** - Measure BW reduction

---

## Performance Targets

| Operation | FP32 (ms) | FP16 (ms) | Speedup | Memory BW |
|-----------|-----------|-----------|---------|-----------|
| Prefill (seq=32) | 0.8 | 0.5 | 1.6x | -50% |
| Prefill (seq=128) | 8.0 | 5.0 | 1.6x | -50% |
| Decode (cache=128) | 0.3 | 0.2 | 1.5x | -50% |

---

## Numerical Accuracy

**Tolerance**: Max absolute error < 1e-2 (0.01)

Attention is less sensitive to precision than GEMM due to softmax normalization. FP16 is sufficient for production use.

**Validation**:
- Compare FP16 vs FP32 attention outputs
- Test on real prompts (Qwen, Phi-3)
- Verify generation quality (haiku test)

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing (6 tests)
- [ ] Integration tests passing (2 tests)
- [ ] Performance benchmarks complete
- [ ] Numerical accuracy validated
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-05

---

## References

- Flash Attention paper: https://arxiv.org/abs/2205.14135
- GQA paper: https://arxiv.org/abs/2305.13245
- CUDA FP16 programming guide

---
Built by Foundation-Alpha 🏗️
