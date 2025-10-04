# GT-009: LayerNorm Mean Reduction

**Team**: GPT-Gamma  
**Sprint**: Sprint 2 (GPT Kernels)  
**Size**: M (1.5 days)  
**Days**: 29-30 (partial)  
**Spec Ref**: M0-W-1432

---

## Story Description

Implement the mean reduction component of LayerNorm for GPT architecture. This is the first step of LayerNorm: computing the mean of activations across the embedding dimension. GPT uses LayerNorm (not RMSNorm like Llama).

---

## Acceptance Criteria

- [ ] CUDA kernel computes mean across embedding dimension
- [ ] Kernel uses Welford's online algorithm for numerical stability
- [ ] Kernel supports FP16 input with FP32 accumulation
- [ ] Kernel handles d_model up to 8192
- [ ] Unit test validates mean calculation correctness
- [ ] Unit test validates numerical stability
- [ ] Performance: <0.05ms per layer for d_model=2048
- [ ] Error handling for invalid dimensions
- [ ] Documentation explains LayerNorm vs RMSNorm

---

## Dependencies

### Upstream (Blocks This Story)
- GT-008: Absolute Positional Embedding (needs embeddings ready)

### Downstream (This Story Blocks)
- GT-010: LayerNorm Variance + Normalize (needs mean)
- GT-011: LayerNorm Unit Tests (needs complete LayerNorm)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/kernels/layernorm.cu` - LayerNorm kernels
- `bin/worker-orcd/cuda/kernels/layernorm.h` - LayerNorm interface
- `bin/worker-orcd/cuda/src/kernels/layernorm_test.cu` - Unit tests

### Key Interfaces
```cpp
// Compute mean of activations across embedding dimension
void layernorm_compute_mean(
    const half* input,    // [batch, seq_len, d_model]
    float* mean_out,      // [batch, seq_len] - output means
    int batch_size,
    int seq_len,
    int d_model,
    cudaStream_t stream
);
```

### CUDA Kernel Implementation
```cuda
__global__ void compute_mean_kernel(
    const half* input,    // [batch, seq_len, d_model]
    float* mean_out,      // [batch, seq_len]
    int d_model
) {
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for reduction
    __shared__ float shared_sum[256];
    
    // Accumulate sum using FP32
    float sum = 0.0f;
    for (int i = tid; i < d_model; i += blockDim.x) {
        int idx = (batch_idx * gridDim.x + seq_idx) * d_model + i;
        sum += __half2float(input[idx]);
    }
    
    // Reduce within block
    shared_sum[tid] = sum;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Write mean
    if (tid == 0) {
        mean_out[batch_idx * gridDim.x + seq_idx] = shared_sum[0] / d_model;
    }
}
```

### Implementation Notes
- LayerNorm: normalize across embedding dimension (GPT)
- RMSNorm: root mean square normalization (Llama)
- Use FP32 accumulation for numerical stability
- Use shared memory reduction for efficiency
- Launch with grid(seq_len, batch) blocks, block(256) threads
- Welford's algorithm optional for improved stability

---

## Testing Strategy

### Unit Tests
- Test mean calculation for known inputs
- Test numerical stability with large values
- Test FP16 to FP32 conversion accuracy
- Test reduction correctness
- Test edge cases (d_model = 1, very large d_model)

### Integration Tests
- Test with GPT-OSS-20B layer dimensions
- Test full sequence mean computation
- Compare with reference implementation

### Manual Verification
1. Run kernel with known input
2. Verify mean matches CPU reference
3. Check numerical stability
4. Profile kernel performance

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7.3 (GPT Kernels)
- LayerNorm Paper: https://arxiv.org/abs/1607.06450
- Related Stories: GT-010 (variance), GT-011 (tests)

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
