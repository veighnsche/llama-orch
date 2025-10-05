# FT-055: Fused Kernel Optimization

**Team**: Foundation-Alpha  
**Sprint**: Sprint 6 - FP16 Optimization (Post-M0)  
**Size**: L (3 days)  
**Priority**: Medium (Stretch Goal, Post-M0)  
**Spec Ref**: M0-W-1430

---

## ⚠️ Prerequisites

**Requires M0 completion + FP16 baseline:**
- Working FP16 GEMM (FT-051)
- Working FP16 attention (FT-052)
- Performance profiling infrastructure (FT-054)

**Note**: This is a stretch goal even after M0. Only pursue if FT-051 through FT-054 show insufficient gains.

---

## Story Description

Implement fused CUDA kernels to reduce memory traffic by combining multiple operations into single kernel launches. Focus on high-impact fusions: RMSNorm+GEMM, GEMM+SwiGLU, and attention score computation.

---

## Acceptance Criteria

- [ ] Fused RMSNorm + GEMM kernel (attention Q/K/V projection)
- [ ] Fused GEMM + SwiGLU kernel (FFN activation)
- [ ] Fused attention score + softmax kernel
- [ ] Numerical accuracy validation vs unfused baseline
- [ ] Performance benchmarks (expect 1.2-1.5x speedup)
- [ ] Unit tests for all fused kernels
- [ ] Memory bandwidth reduction measurement
- [ ] Integration with transformer forward pass

---

## Dependencies

**Upstream**: FT-054 (Memory bandwidth profiling, Day 6)  
**Downstream**: FT-056 (Performance validation)

---

## Technical Details

### Motivation

**Unfused pipeline** (current):
1. RMSNorm: Read activations → Write normalized
2. GEMM: Read normalized → Write projected
3. Total: 2 kernel launches, 2x memory traffic

**Fused pipeline**:
1. RMSNorm+GEMM: Read activations → Write projected
2. Total: 1 kernel launch, 1.5x memory traffic

**Savings**: ~25-30% memory bandwidth reduction

### Implementation Plan

**Phase 1: Fused RMSNorm + GEMM** (Day 1-2)

```cpp
// cuda/kernels/fused_rmsnorm_gemm.cu

/**
 * Fused RMSNorm + GEMM kernel (FP16)
 * 
 * Combines layer normalization with matrix multiplication.
 * Each block processes one token, computes RMSNorm, then GEMM row.
 * 
 * Algorithm:
 * 1. Load input activations [hidden_dim]
 * 2. Compute RMSNorm in shared memory
 * 3. Multiply normalized values with weight matrix
 * 4. Write output [output_dim]
 * 
 * Memory savings:
 * - Unfused: Read input + Write norm + Read norm + Write output = 4x
 * - Fused: Read input + Read weights + Write output = 3x
 * - Reduction: 25%
 */
__global__ void fused_rmsnorm_gemm_fp16(
    half* output,              // [batch, seq_len, output_dim]
    const half* input,         // [batch, seq_len, hidden_dim]
    const half* norm_weight,   // [hidden_dim]
    const half* gemm_weight,   // [output_dim, hidden_dim]
    int batch_size,
    int seq_len,
    int hidden_dim,
    int output_dim,
    float eps
) {
    // Each block processes one token
    int token_idx = blockIdx.x;
    int output_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    
    if (token_idx >= batch_size * seq_len || output_idx >= output_dim) {
        return;
    }
    
    // Shared memory for input and normalized values
    extern __shared__ half shared_mem[];
    half* shared_input = shared_mem;
    half* shared_norm = shared_mem + hidden_dim;
    
    const half* token_input = input + token_idx * hidden_dim;
    
    // Phase 1: Load input to shared memory
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        shared_input[i] = token_input[i];
    }
    __syncthreads();
    
    // Phase 2: Compute RMSNorm
    // (Only thread 0 computes RMS, others wait)
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            float val = __half2float(shared_input[i]);
            sum_sq += val * val;
        }
        float rms = sqrtf(sum_sq / hidden_dim + eps);
        
        // Normalize and scale
        for (int i = 0; i < hidden_dim; ++i) {
            float val = __half2float(shared_input[i]);
            float w = __half2float(norm_weight[i]);
            shared_norm[i] = __float2half((val / rms) * w);
        }
    }
    __syncthreads();
    
    // Phase 3: GEMM - compute one output element
    if (output_idx < output_dim) {
        const half* weight_row = gemm_weight + output_idx * hidden_dim;
        
        float acc = 0.0f;
        for (int i = tid; i < hidden_dim; i += blockDim.x) {
            float norm_val = __half2float(shared_norm[i]);
            float weight_val = __half2float(weight_row[i]);
            acc += norm_val * weight_val;
        }
        
        // Reduce across threads
        __shared__ float shared_acc[256];
        shared_acc[tid] = acc;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_acc[tid] += shared_acc[tid + s];
            }
            __syncthreads();
        }
        
        // Write output
        if (tid == 0) {
            output[token_idx * output_dim + output_idx] = __float2half(shared_acc[0]);
        }
    }
}

extern "C" {

/**
 * Launch fused RMSNorm + GEMM
 * 
 * Typical use: Attention Q/K/V projection
 * - Input: [batch, seq_len, hidden_dim]
 * - Output: [batch, seq_len, output_dim]
 */
int cuda_fused_rmsnorm_gemm_fp16(
    half* output,
    const half* input,
    const half* norm_weight,
    const half* gemm_weight,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int output_dim,
    float eps
) {
    // Validate dimensions
    if (batch_size <= 0 || seq_len <= 0 || 
        hidden_dim <= 0 || output_dim <= 0) {
        fprintf(stderr, "Fused kernel: Invalid dimensions\n");
        return -1;
    }
    
    int num_tokens = batch_size * seq_len;
    
    // Kernel configuration
    // Block: (256 threads for reduction, 1 output element)
    // Grid: (num_tokens, output_dim)
    dim3 block(256, 1);
    dim3 grid(num_tokens, output_dim);
    
    // Shared memory: input + normalized
    size_t shared_mem_size = 2 * hidden_dim * sizeof(half);
    
    fused_rmsnorm_gemm_fp16<<<grid, block, shared_mem_size>>>(
        output, input, norm_weight, gemm_weight,
        batch_size, seq_len, hidden_dim, output_dim, eps
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Fused RMSNorm+GEMM launch failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}

} // extern "C"
```

**Phase 2: Fused GEMM + SwiGLU** (Day 2)

```cpp
// cuda/kernels/fused_gemm_swiglu.cu

/**
 * Fused GEMM + SwiGLU activation (FP16)
 * 
 * SwiGLU: output = (W_gate @ x) * swish(W_up @ x)
 * where swish(x) = x * sigmoid(x)
 * 
 * Fuses two GEMMs and activation into single kernel.
 */
__global__ void fused_gemm_swiglu_fp16(
    half* output,              // [batch, seq_len, ffn_dim]
    const half* input,         // [batch, seq_len, hidden_dim]
    const half* gate_weight,   // [ffn_dim, hidden_dim]
    const half* up_weight,     // [ffn_dim, hidden_dim]
    int batch_size,
    int seq_len,
    int hidden_dim,
    int ffn_dim
) {
    int token_idx = blockIdx.x;
    int ffn_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    
    if (token_idx >= batch_size * seq_len || ffn_idx >= ffn_dim) {
        return;
    }
    
    const half* token_input = input + token_idx * hidden_dim;
    
    // Compute gate projection
    const half* gate_row = gate_weight + ffn_idx * hidden_dim;
    float gate_acc = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        gate_acc += __half2float(token_input[i]) * 
                    __half2float(gate_row[i]);
    }
    
    // Compute up projection
    const half* up_row = up_weight + ffn_idx * hidden_dim;
    float up_acc = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        up_acc += __half2float(token_input[i]) * 
                  __half2float(up_row[i]);
    }
    
    // Reduce
    __shared__ float shared_gate[256];
    __shared__ float shared_up[256];
    shared_gate[tid] = gate_acc;
    shared_up[tid] = up_acc;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_gate[tid] += shared_gate[tid + s];
            shared_up[tid] += shared_up[tid + s];
        }
        __syncthreads();
    }
    
    // Apply SwiGLU activation
    if (tid == 0) {
        float gate_val = shared_gate[0];
        float up_val = shared_up[0];
        
        // swish(gate) = gate * sigmoid(gate)
        float sigmoid = 1.0f / (1.0f + expf(-gate_val));
        float swish = gate_val * sigmoid;
        
        // SwiGLU = swish(gate) * up
        float result = swish * up_val;
        
        output[token_idx * ffn_dim + ffn_idx] = __float2half(result);
    }
}
```

**Phase 3: Fused Attention Score + Softmax** (Day 3)

```cpp
// cuda/kernels/fused_attention_score_softmax.cu

/**
 * Fused attention score computation + softmax (FP16)
 * 
 * Combines Q @ K^T, scaling, causal mask, and softmax.
 * Reduces intermediate memory allocation.
 */
__global__ void fused_attention_score_softmax_fp16(
    half* attention_weights,   // [batch, num_heads, seq_q, seq_k]
    const half* q,             // [batch, seq_q, num_heads, head_dim]
    const half* k,             // [batch, seq_k, num_heads, head_dim]
    int batch_size,
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim,
    float scale,
    bool causal_mask
) {
    // Each block processes one query position for one head
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int q_pos = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch >= batch_size || head >= num_heads || q_pos >= seq_q) {
        return;
    }
    
    // Load query vector to shared memory
    extern __shared__ half shared_q[];
    const half* q_vec = q + (batch * seq_q * num_heads * head_dim) +
                             (q_pos * num_heads * head_dim) +
                             (head * head_dim);
    
    for (int i = tid; i < head_dim; i += blockDim.x) {
        shared_q[i] = q_vec[i];
    }
    __syncthreads();
    
    // Compute scores for all key positions
    __shared__ float shared_scores[512];  // Assume seq_k <= 512
    
    for (int k_pos = tid; k_pos < seq_k; k_pos += blockDim.x) {
        // Causal mask check
        if (causal_mask && k_pos > q_pos) {
            shared_scores[k_pos] = -INFINITY;
            continue;
        }
        
        const half* k_vec = k + (batch * seq_k * num_heads * head_dim) +
                                 (k_pos * num_heads * head_dim) +
                                 (head * head_dim);
        
        // Dot product Q · K
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += __half2float(shared_q[i]) * __half2float(k_vec[i]);
        }
        
        // Apply scale
        shared_scores[k_pos] = score * scale;
    }
    __syncthreads();
    
    // Softmax (same as standalone softmax kernel)
    // ... (implementation from FT-052) ...
    
    // Write attention weights
    half* output_row = attention_weights + 
        (batch * num_heads * seq_q * seq_k) +
        (head * seq_q * seq_k) +
        (q_pos * seq_k);
    
    for (int k_pos = tid; k_pos < seq_k; k_pos += blockDim.x) {
        output_row[k_pos] = __float2half(shared_scores[k_pos]);
    }
}
```

---

## Files to Create/Modify

**Create**:
- `cuda/kernels/fused_rmsnorm_gemm.cu` - Fused RMSNorm+GEMM
- `cuda/kernels/fused_gemm_swiglu.cu` - Fused GEMM+SwiGLU
- `cuda/kernels/fused_attention_score_softmax.cu` - Fused attention
- `cuda/tests/test_fused_kernels.cu` - Unit tests

**Modify**:
- `cuda/src/inference.cu` - Integrate fused kernels
- `src/cuda/inference.rs` - Add fused kernel API

---

## Testing Strategy

### Unit Tests (6 tests)

1. **test_fused_rmsnorm_gemm_accuracy** - Compare vs unfused
2. **test_fused_gemm_swiglu_accuracy** - Compare vs unfused
3. **test_fused_attention_accuracy** - Compare vs unfused
4. **test_fused_rmsnorm_gemm_edge_cases** - Small/large dims
5. **test_fused_gemm_swiglu_edge_cases** - Various FFN sizes
6. **test_fused_attention_causal_mask** - Validate masking

### Performance Tests (3 benchmarks)

1. **bench_fused_vs_unfused_rmsnorm_gemm** - Measure speedup
2. **bench_fused_vs_unfused_gemm_swiglu** - Measure speedup
3. **bench_fused_vs_unfused_attention** - Measure speedup

---

## Performance Targets

| Fused Operation | Unfused (ms) | Fused (ms) | Speedup | BW Reduction |
|-----------------|--------------|------------|---------|--------------|
| RMSNorm+GEMM | 0.8 | 0.6 | 1.33x | 25% |
| GEMM+SwiGLU | 2.0 | 1.5 | 1.33x | 25% |
| Attention Score+Softmax | 1.5 | 1.1 | 1.36x | 30% |

**Overall impact**: 1.2-1.5x speedup for full transformer layer.

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing (6 tests)
- [ ] Performance benchmarks complete
- [ ] Numerical accuracy validated (< 1e-2 error)
- [ ] Memory bandwidth reduction measured
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Story marked complete

---

**Status**: 📋 Ready (Stretch Goal)  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-05

---

## References

- Kernel fusion techniques: https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- Flash Attention (fused attention): https://arxiv.org/abs/2205.14135

---
Built by Foundation-Alpha 🏗️
