# LT-012: RoPE Kernel

**Team**: Llama-Beta  
**Sprint**: Sprint 3 - UTF-8 Safety + Llama Kernels  
**Size**: M (2 days)  
**Days**: 38-39  
**Spec Ref**: M0-W-1214, M0-W-1430

---

## Story Description

Implement Rotary Position Embedding (RoPE) CUDA kernel for Llama models. Apply rotary embeddings to query and key tensors to encode positional information, supporting both standard RoPE (base=10000) and extended context RoPE (base=1000000).

---

## Acceptance Criteria

- [ ] Implement RoPE CUDA kernel for Q and K tensors
- [ ] Support configurable frequency base (10000.0 or 1000000.0)
- [ ] Apply rotation to pairs of dimensions (d_i, d_{i+1})
- [ ] Handle variable sequence lengths (prefill and decode)
- [ ] Support GQA (different Q and K head counts)
- [ ] Optimize for memory bandwidth (coalesced access)
- [ ] Unit tests validate RoPE computation against reference
- [ ] Unit tests validate different frequency bases
- [ ] Benchmark kernel performance (TFLOPS, memory bandwidth)
- [ ] Error handling for invalid dimensions
- [ ] Log kernel launch parameters at DEBUG level

---

## Dependencies

### Upstream (Blocks This Story)
- FT-010: CUDA Context Init (needs CUDA runtime)
- FT-013: Device Memory RAII (needs VRAM allocation)

### Downstream (This Story Blocks)
- LT-015: GQA Attention Kernel (needs RoPE for Q/K)
- LT-024: Qwen Forward Pass (needs RoPE)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/rope.cu` - RoPE CUDA kernel
- `bin/worker-orcd/cuda/kernels/rope.h` - RoPE interface
- `bin/worker-orcd/src/kernels/rope.rs` - Rust FFI wrapper

### Key Interfaces
```cpp
// RoPE kernel configuration
struct RoPEConfig {
    int seq_len;           // Sequence length
    int num_heads;         // Number of attention heads
    int head_dim;          // Dimension per head
    float freq_base;       // Frequency base (10000.0 or 1000000.0)
    int rope_dim;          // RoPE dimensions (usually head_dim)
};

// Apply RoPE to query and key tensors
void rope_forward(
    half* q_out,           // [batch, seq_len, num_heads, head_dim]
    half* k_out,           // [batch, seq_len, num_kv_heads, head_dim]
    const half* q_in,      // Input query
    const half* k_in,      // Input key
    const RoPEConfig& config,
    cudaStream_t stream = nullptr
);
```

```rust
#[repr(C)]
pub struct RoPEConfig {
    pub seq_len: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub freq_base: f32,
    pub rope_dim: i32,
}

extern "C" {
    pub fn rope_forward(
        q_out: *mut f16,
        k_out: *mut f16,
        q_in: *const f16,
        k_in: *const f16,
        config: *const RoPEConfig,
        stream: cudaStream_t,
    );
}
```

### Implementation Notes

**RoPE Algorithm**:
```
For each position m and dimension pair (2i, 2i+1):
  θ_i = m / (freq_base^(2i / rope_dim))
  
  q[2i]   = q_in[2i]   * cos(θ_i) - q_in[2i+1] * sin(θ_i)
  q[2i+1] = q_in[2i]   * sin(θ_i) + q_in[2i+1] * cos(θ_i)
  
  k[2i]   = k_in[2i]   * cos(θ_i) - k_in[2i+1] * sin(θ_i)
  k[2i+1] = k_in[2i]   * sin(θ_i) + k_in[2i+1] * cos(θ_i)
```

**CUDA Kernel**:
```cuda
__global__ void rope_kernel(
    half* q_out,
    half* k_out,
    const half* q_in,
    const half* k_in,
    int seq_len,
    int num_heads,
    int head_dim,
    float freq_base,
    int rope_dim
) {
    int pos = blockIdx.x;  // Position in sequence
    int head = blockIdx.y; // Head index
    int dim = threadIdx.x * 2;  // Dimension pair
    
    if (pos >= seq_len || dim >= rope_dim) return;
    
    // Calculate rotation angle
    float theta = pos / powf(freq_base, (float)(dim) / rope_dim);
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    
    // Apply rotation to Q
    int q_idx = pos * num_heads * head_dim + head * head_dim + dim;
    half q0 = q_in[q_idx];
    half q1 = q_in[q_idx + 1];
    q_out[q_idx]     = __float2half(__half2float(q0) * cos_theta - __half2float(q1) * sin_theta);
    q_out[q_idx + 1] = __float2half(__half2float(q0) * sin_theta + __half2float(q1) * cos_theta);
    
    // Apply rotation to K (similar logic)
    // ...
}
```

**Optimization**:
- Use `__sincosf()` for simultaneous sin/cos computation
- Precompute theta values for all positions
- Coalesce memory access (warp-aligned)
- Use shared memory for theta values (reduce recomputation)

---

## Testing Strategy

### Unit Tests
- Test RoPE with seq_len=1 (single position)
- Test RoPE with seq_len=128 (prefill)
- Test RoPE with different frequency bases (10000, 1000000)
- Test RoPE with different head dimensions (64, 128)
- Test rotation correctness (compare with reference implementation)
- Test GQA support (num_heads != num_kv_heads)

### Numerical Validation
- Compare against reference RoPE implementation (PyTorch)
- Tolerance: ±0.01 (FP16 precision)
- Test positions: 0, 1, 100, 1000, 32000

### Performance Tests
- Benchmark kernel launch overhead
- Measure memory bandwidth utilization
- Compare with cuBLAS performance (if applicable)

### Manual Verification
1. Apply RoPE to random Q/K tensors
2. Verify rotation preserves magnitude (||q_out|| ≈ ||q_in||)
3. Check logs show kernel launch parameters

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (6+ tests)
- [ ] Numerical validation passing (±0.01 tolerance)
- [ ] Performance benchmarks recorded
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.5 (Inference Kernels)
- RoPE Paper: https://arxiv.org/abs/2104.09864
- Llama RoPE: https://github.com/facebookresearch/llama/blob/main/llama/model.py
- Related Stories: LT-015, LT-024

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
