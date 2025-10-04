# LT-016: GQA Attention Kernel (Decode)

**Team**: Llama-Beta  
**Sprint**: Sprint 4 - GQA Attention + Integration  
**Size**: M (2 days)  
**Days**: 46-47  
**Spec Ref**: M0-W-1214

---

## Story Description

Implement Grouped Query Attention (GQA) CUDA kernel for decode phase. Compute attention for single token generation using cached K/V values, optimized for low-latency autoregressive decoding.

---

## Acceptance Criteria

- [ ] Implement GQA attention kernel for decode (single token)
- [ ] Read K, V from KV cache (all previous positions)
- [ ] Compute Q @ K_cache^T for current token
- [ ] Apply softmax to attention scores (no causal mask needed)
- [ ] Compute attention @ V_cache to get output
- [ ] Append current K, V to KV cache
- [ ] Support variable cache lengths (1 to max_seq_len)
- [ ] Optimize for low latency (single token)
- [ ] Unit tests validate decode attention
- [ ] Unit tests validate KV cache reading/writing
- [ ] Benchmark kernel latency (microseconds)
- [ ] Error handling for cache overflow
- [ ] Log kernel launch parameters at DEBUG level

---

## Dependencies

### Upstream (Blocks This Story)
- LT-015: GQA Attention Prefill (needs attention logic)
- FT-022: KV Cache Management (needs cache interface)

### Downstream (This Story Blocks)
- LT-024: Qwen Forward Pass (needs decode attention)
- LT-031: Phi-3 Forward Pass (needs decode attention)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/gqa_attention_decode.cu` - GQA decode kernel
- `bin/worker-orcd/cuda/kernels/gqa_attention_decode.h` - Decode interface
- `bin/worker-orcd/src/kernels/gqa_attention.rs` - Rust FFI wrapper (extend)

### Key Interfaces
```cpp
// GQA decode configuration
struct GQADecodeConfig {
    int batch_size;        // Batch size (usually 1 for decode)
    int cache_len;         // Current cache length (1 to max_seq_len)
    int num_q_heads;       // Number of query heads
    int num_kv_heads;      // Number of KV heads
    int head_dim;          // Dimension per head
    float scale;           // 1.0 / sqrt(head_dim)
};

// GQA decode attention
void gqa_attention_decode(
    half* output,          // [batch, 1, num_q_heads * head_dim]
    const half* q,         // [batch, 1, num_q_heads, head_dim]
    const half* k,         // [batch, 1, num_kv_heads, head_dim] (current token)
    const half* v,         // [batch, 1, num_kv_heads, head_dim] (current token)
    half* kv_cache_k,      // [batch, max_seq_len, num_kv_heads, head_dim]
    half* kv_cache_v,      // [batch, max_seq_len, num_kv_heads, head_dim]
    const GQADecodeConfig& config,
    cudaStream_t stream = nullptr
);
```

```rust
#[repr(C)]
pub struct GQADecodeConfig {
    pub batch_size: i32,
    pub cache_len: i32,
    pub num_q_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

extern "C" {
    pub fn gqa_attention_decode(
        output: *mut f16,
        q: *const f16,
        k: *const f16,
        v: *const f16,
        kv_cache_k: *mut f16,
        kv_cache_v: *mut f16,
        config: *const GQADecodeConfig,
        stream: cudaStream_t,
    );
}
```

### Implementation Notes

**GQA Decode Algorithm**:
```
1. Append current K, V to cache at position cache_len
2. For each Q head:
   a. Determine KV head: kv_head = q_head / (num_q_heads / num_kv_heads)
   b. Compute scores: scores[j] = Q @ K_cache[j]^T * scale (j = 0..cache_len)
   c. Apply softmax: attn_weights = softmax(scores)
   d. Compute output: output = sum(attn_weights[j] * V_cache[j])
```

**CUDA Implementation**:
```cuda
__global__ void gqa_decode_kernel(
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
    float scale
) {
    int batch = blockIdx.z;
    int q_head = blockIdx.y;
    int kv_head = q_head / (num_q_heads / num_kv_heads);
    
    // 1. Append current K, V to cache
    if (threadIdx.x == 0 && q_head == 0) {
        for (int d = 0; d < head_dim; ++d) {
            int cache_idx = batch * max_seq_len * num_kv_heads * head_dim + 
                            cache_len * num_kv_heads * head_dim + 
                            kv_head * head_dim + d;
            int current_idx = batch * num_kv_heads * head_dim + 
                              kv_head * head_dim + d;
            kv_cache_k[cache_idx] = k_current[current_idx];
            kv_cache_v[cache_idx] = v_current[current_idx];
        }
    }
    __syncthreads();
    
    // 2. Compute attention scores
    extern __shared__ float scores[];
    for (int pos = threadIdx.x; pos <= cache_len; pos += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            int q_idx = batch * num_q_heads * head_dim + q_head * head_dim + d;
            int k_idx = batch * max_seq_len * num_kv_heads * head_dim + 
                        pos * num_kv_heads * head_dim + 
                        kv_head * head_dim + d;
            score += __half2float(q[q_idx]) * __half2float(kv_cache_k[k_idx]);
        }
        scores[pos] = score * scale;
    }
    __syncthreads();
    
    // 3. Softmax
    // ... (parallel reduction)
    
    // 4. Compute output
    // ...
}
```

**Optimization**:
- Use shared memory for Q (reused across all cache positions)
- Vectorize K/V cache reads (half2 or float4)
- Fuse cache append with attention computation
- Minimize synchronization overhead

---

## Testing Strategy

### Unit Tests
- Test decode with cache_len=1 (first decode step)
- Test decode with cache_len=100 (mid-sequence)
- Test decode with cache_len=max_seq_len-1 (near end)
- Test KV cache append (verify correct position)
- Test attention score computation (Q @ K_cache^T)
- Test softmax normalization
- Test output computation (attn @ V_cache)

### Numerical Validation
- Compare against reference GQA decode (PyTorch)
- Tolerance: ±0.05 (attention accumulation errors)
- Test with random Q, K, V and populated cache

### Performance Tests
- Benchmark decode latency (microseconds per token)
- Measure cache read bandwidth
- Compare with prefill performance (should be faster)

### Manual Verification
1. Run decode step with cache_len=10
2. Verify output shape [batch, 1, num_q_heads * head_dim]
3. Verify KV cache updated at position 10
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
- KV Cache: https://arxiv.org/abs/2211.05102
- Related Stories: LT-015, LT-024, LT-031

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
