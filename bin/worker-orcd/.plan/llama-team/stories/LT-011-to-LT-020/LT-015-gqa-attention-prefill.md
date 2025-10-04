# LT-015: GQA Attention Kernel (Prefill)

**Team**: Llama-Beta  
**Sprint**: Sprint 4 - GQA Attention + Integration  
**Size**: L (4 days)  
**Days**: 42-45  
**Spec Ref**: M0-W-1214, M0-W-1430

---

## Story Description

Implement Grouped Query Attention (GQA) CUDA kernel for prefill phase. Compute attention scores and weighted values for initial prompt processing, supporting variable KV head counts for efficient memory usage in Llama models.

---

## Acceptance Criteria

- [ ] Implement GQA attention kernel for prefill (full sequence)
- [ ] Support variable Q heads and KV heads (e.g., 14 Q heads, 2 KV heads)
- [ ] Compute Q @ K^T scaled by sqrt(head_dim)
- [ ] Apply causal mask (upper triangular)
- [ ] Apply softmax to attention scores
- [ ] Compute attention @ V to get output
- [ ] Integrate with KV cache (write K, V to cache)
- [ ] Support flash attention optimization (optional)
- [ ] Unit tests validate attention computation
- [ ] Unit tests validate GQA head grouping (7 Q heads per KV head for Qwen)
- [ ] Benchmark kernel performance (TFLOPS)
- [ ] Error handling for invalid dimensions
- [ ] Log kernel launch parameters at DEBUG level

---

## Dependencies

### Upstream (Blocks This Story)
- LT-012: RoPE Kernel (needs rotary embeddings)
- FT-021: KV Cache Allocation (needs cache storage)
- FT-022: KV Cache Management (needs cache interface)

### Downstream (This Story Blocks)
- LT-016: GQA Attention Decode (needs attention logic)
- LT-024: Qwen Forward Pass (needs prefill attention)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/gqa_attention_prefill.cu` - GQA prefill kernel
- `bin/worker-orcd/cuda/kernels/gqa_attention_prefill.h` - Prefill interface
- `bin/worker-orcd/src/kernels/gqa_attention.rs` - Rust FFI wrapper

### Key Interfaces
```cpp
// GQA attention configuration
struct GQAAttentionConfig {
    int batch_size;        // Batch size
    int seq_len;           // Sequence length (prefill)
    int num_q_heads;       // Number of query heads (e.g., 14)
    int num_kv_heads;      // Number of KV heads (e.g., 2)
    int head_dim;          // Dimension per head (e.g., 64)
    float scale;           // 1.0 / sqrt(head_dim)
};

// GQA prefill attention
void gqa_attention_prefill(
    half* output,          // [batch, seq_len, num_q_heads * head_dim]
    const half* q,         // [batch, seq_len, num_q_heads, head_dim]
    const half* k,         // [batch, seq_len, num_kv_heads, head_dim]
    const half* v,         // [batch, seq_len, num_kv_heads, head_dim]
    half* kv_cache_k,      // KV cache for K
    half* kv_cache_v,      // KV cache for V
    const GQAAttentionConfig& config,
    cudaStream_t stream = nullptr
);
```

```rust
#[repr(C)]
pub struct GQAAttentionConfig {
    pub batch_size: i32,
    pub seq_len: i32,
    pub num_q_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

extern "C" {
    pub fn gqa_attention_prefill(
        output: *mut f16,
        q: *const f16,
        k: *const f16,
        v: *const f16,
        kv_cache_k: *mut f16,
        kv_cache_v: *mut f16,
        config: *const GQAAttentionConfig,
        stream: cudaStream_t,
    );
}
```

### Implementation Notes

**GQA Algorithm**:
```
1. Group Q heads: num_q_heads / num_kv_heads groups (e.g., 14/2 = 7)
2. For each Q head group:
   a. Compute attention scores: scores = (Q @ K^T) * scale
   b. Apply causal mask: scores[i,j] = -inf if j > i
   c. Apply softmax: attn_weights = softmax(scores)
   d. Compute output: output = attn_weights @ V
3. Write K, V to KV cache for decode phase
```

**CUDA Implementation**:
```cuda
__global__ void gqa_prefill_kernel(
    half* output,
    const half* q,
    const half* k,
    const half* v,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int batch = blockIdx.z;
    int q_head = blockIdx.y;
    int pos = blockIdx.x;
    
    // Determine which KV head this Q head uses
    int kv_head = q_head / (num_q_heads / num_kv_heads);
    
    // Compute attention scores for this position
    extern __shared__ float scores[];
    
    for (int j = threadIdx.x; j <= pos; j += blockDim.x) {
        // Q @ K^T
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            int q_idx = batch * seq_len * num_q_heads * head_dim + 
                        pos * num_q_heads * head_dim + 
                        q_head * head_dim + d;
            int k_idx = batch * seq_len * num_kv_heads * head_dim + 
                        j * num_kv_heads * head_dim + 
                        kv_head * head_dim + d;
            score += __half2float(q[q_idx]) * __half2float(k[k_idx]);
        }
        scores[j] = score * scale;
    }
    
    // Causal mask (positions > pos are masked)
    for (int j = pos + 1 + threadIdx.x; j < seq_len; j += blockDim.x) {
        scores[j] = -INFINITY;
    }
    __syncthreads();
    
    // Softmax
    // ... (parallel reduction for max, exp, sum)
    
    // Compute output: attn_weights @ V
    // ...
}
```

**Optimization Strategies**:
- Use cuBLAS for Q @ K^T (batched GEMM)
- Use flash attention for memory efficiency
- Tile computation to fit in shared memory
- Fuse softmax with attention computation

---

## Testing Strategy

### Unit Tests
- Test GQA with Qwen config (14 Q heads, 2 KV heads)
- Test GQA with Phi-3 config (32 Q heads, 32 KV heads = MHA)
- Test causal masking (upper triangle is masked)
- Test attention score computation (Q @ K^T)
- Test softmax normalization (sum = 1)
- Test output computation (attn @ V)
- Test KV cache writing

### Numerical Validation
- Compare against reference GQA implementation (PyTorch)
- Tolerance: ±0.05 (attention has accumulation errors)
- Test with random Q, K, V tensors

### Performance Tests
- Benchmark TFLOPS (compare with theoretical peak)
- Measure memory bandwidth utilization
- Compare with flash attention (if implemented)

### Manual Verification
1. Run GQA prefill with Qwen config
2. Verify output shape [batch, seq_len, num_q_heads * head_dim]
3. Verify KV cache populated
4. Check logs show kernel launch parameters

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (7+ tests)
- [ ] Numerical validation passing (±0.05 tolerance)
- [ ] Performance benchmarks recorded
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.5 (Inference Kernels)
- GQA Paper: https://arxiv.org/abs/2305.13245
- Flash Attention: https://arxiv.org/abs/2205.14135
- Llama GQA: https://github.com/facebookresearch/llama/blob/main/llama/model.py
- Related Stories: LT-012, LT-016, LT-024

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
