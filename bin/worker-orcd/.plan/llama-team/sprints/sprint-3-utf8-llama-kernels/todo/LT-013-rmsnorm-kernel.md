# LT-013: RMSNorm Kernel

**Team**: Llama-Beta  
**Sprint**: Sprint 3 - UTF-8 Safety + Llama Kernels  
**Size**: S (1 day)  
**Days**: 40  
**Spec Ref**: M0-W-1214, M0-W-1430

---

## Story Description

Implement Root Mean Square Normalization (RMSNorm) CUDA kernel for Llama models. Normalize activations using RMS instead of LayerNorm to reduce computation while maintaining model quality.

---

## Acceptance Criteria

- [ ] Implement RMSNorm CUDA kernel
- [ ] Compute RMS: sqrt(mean(x^2) + eps)
- [ ] Normalize: x_out = x_in * weight / RMS
- [ ] Support configurable epsilon (default 1e-6)
- [ ] Handle variable hidden dimensions (896, 3072, etc.)
- [ ] Optimize for memory bandwidth (fused kernel)
- [ ] Unit tests validate RMSNorm computation against reference
- [ ] Unit tests validate numerical stability (small/large values)
- [ ] Benchmark kernel performance (GB/s)
- [ ] Error handling for invalid dimensions
- [ ] Log kernel launch parameters at DEBUG level

---

## Dependencies

### Upstream (Blocks This Story)
- FT-010: CUDA Context Init (needs CUDA runtime)
- FT-013: Device Memory RAII (needs VRAM allocation)

### Downstream (This Story Blocks)
- LT-024: Qwen Forward Pass (needs RMSNorm)
- LT-031: Phi-3 Forward Pass (needs RMSNorm)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/rmsnorm.cu` - RMSNorm CUDA kernel
- `bin/worker-orcd/cuda/kernels/rmsnorm.h` - RMSNorm interface
- `bin/worker-orcd/src/kernels/rmsnorm.rs` - Rust FFI wrapper

### Key Interfaces
```cpp
// RMSNorm kernel configuration
struct RMSNormConfig {
    int batch_size;        // Batch size
    int seq_len;           // Sequence length
    int hidden_dim;        // Hidden dimension
    float eps;             // Epsilon for numerical stability (1e-6)
};

// Apply RMSNorm
void rmsnorm_forward(
    half* output,          // [batch, seq_len, hidden_dim]
    const half* input,     // [batch, seq_len, hidden_dim]
    const half* weight,    // [hidden_dim]
    const RMSNormConfig& config,
    cudaStream_t stream = nullptr
);
```

```rust
#[repr(C)]
pub struct RMSNormConfig {
    pub batch_size: i32,
    pub seq_len: i32,
    pub hidden_dim: i32,
    pub eps: f32,
}

extern "C" {
    pub fn rmsnorm_forward(
        output: *mut f16,
        input: *const f16,
        weight: *const f16,
        config: *const RMSNormConfig,
        stream: cudaStream_t,
    );
}
```

### Implementation Notes

**RMSNorm Algorithm**:
```
1. Compute RMS: rms = sqrt(mean(x^2) + eps)
2. Normalize: x_norm = x / rms
3. Scale: output = x_norm * weight
```

**CUDA Kernel (Fused)**:
```cuda
__global__ void rmsnorm_kernel(
    half* output,
    const half* input,
    const half* weight,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float eps
) {
    int token_idx = blockIdx.x;  // Token index (batch * seq_len)
    int tid = threadIdx.x;       // Thread index
    
    if (token_idx >= batch_size * seq_len) return;
    
    const half* x = input + token_idx * hidden_dim;
    half* y = output + token_idx * hidden_dim;
    
    // 1. Compute sum of squares (parallel reduction)
    __shared__ float sum_sq[256];
    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        thread_sum += val * val;
    }
    sum_sq[tid] = thread_sum;
    __syncthreads();
    
    // 2. Reduce sum_sq to single value
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_sq[tid] += sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    // 3. Compute RMS
    float rms = sqrtf(sum_sq[0] / hidden_dim + eps);
    
    // 4. Normalize and scale
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        float w = __half2float(weight[i]);
        y[i] = __float2half((val / rms) * w);
    }
}
```

**Optimization**:
- Fuse RMS computation and normalization in single kernel
- Use shared memory for parallel reduction
- Coalesce memory access (warp-aligned)
- Use `__ldg()` for read-only input (texture cache)

---

## Testing Strategy

### Unit Tests
- Test RMSNorm with hidden_dim=896 (Qwen)
- Test RMSNorm with hidden_dim=3072 (Phi-3)
- Test numerical stability (very small values, eps=1e-6)
- Test numerical stability (very large values)
- Test weight scaling (different weight values)
- Test batch processing (batch_size > 1)

### Numerical Validation
- Compare against reference RMSNorm (PyTorch)
- Tolerance: ±0.01 (FP16 precision)
- Test with random inputs (normal distribution)

### Performance Tests
- Benchmark kernel throughput (GB/s)
- Compare with unfused implementation (separate RMS + scale)
- Measure reduction overhead

### Manual Verification
1. Apply RMSNorm to random tensor
2. Verify output has unit RMS (mean(output^2) ≈ 1)
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
- RMSNorm Paper: https://arxiv.org/abs/1910.07467
- Llama Implementation: https://github.com/facebookresearch/llama/blob/main/llama/model.py
- Related Stories: LT-024, LT-031

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
