# FT-021: FP16 Sampling Support

**Team**: Foundation-Alpha  
**Sprint**: Sprint 5 - FP16 Optimization  
**Size**: S (1 day)  
**Days**: 1 - 1  
**Spec Ref**: M0-W-1421

---

## Story Description

Add FP16 support to all sampling kernels (greedy, stochastic, advanced). FP16 reduces memory bandwidth and improves performance for large vocabularies.

---

## Acceptance Criteria

- [ ] FP16 greedy sampling kernel
- [ ] FP16 softmax kernel
- [ ] FP16 advanced sampling kernels (top-k, top-p, etc.)
- [ ] Unit tests for FP16 variants (10+ tests)
- [ ] Performance comparison (FP16 vs FP32)
- [ ] Documentation updated

---

## Technical Details

### FP16 Greedy Sampling

```cpp
/**
 * Greedy sampling kernel (FP16).
 * 
 * Same as FP32 version but with half precision.
 */
__global__ void greedy_sample_reduce_fp16(
    const half* logits,
    int vocab_size,
    float* block_max,
    int* block_idx
) {
    __shared__ half shared_max[256];
    __shared__ int shared_idx[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize with first element or -inf
    half local_max = (idx < vocab_size) ? logits[idx] : __float2half(-INFINITY);
    int local_idx = (idx < vocab_size) ? idx : 0;
    
    // Grid-stride loop
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        if (__hgt(logits[i], local_max)) {
            local_max = logits[i];
            local_idx = i;
        }
    }
    
    shared_max[tid] = local_max;
    shared_idx[tid] = local_idx;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (__hgt(shared_max[tid + s], shared_max[tid])) {
                shared_max[tid] = shared_max[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Write block result (convert to FP32 for final reduction)
    if (tid == 0) {
        block_max[blockIdx.x] = __half2float(shared_max[0]);
        block_idx[blockIdx.x] = shared_idx[0];
    }
}

int launch_greedy_sample_fp16(
    const half* logits,
    int vocab_size,
    cudaStream_t stream = 0
);
```

### FP16 Softmax

```cpp
/**
 * Softmax kernel (FP16).
 * 
 * Converts FP16 logits to FP16 probabilities.
 * Uses FP32 for accumulation (numerical stability).
 */
__global__ void softmax_fp16(
    const half* logits,
    half* probs,
    int vocab_size
) {
    __shared__ float shared_max[256];
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Find max (in FP32 for stability)
    float local_max = (idx < vocab_size) ? __half2float(logits[idx]) : -INFINITY;
    
    // Grid-stride loop
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, __half2float(logits[i]));
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
    
    // Compute exp and sum (in FP32)
    float local_exp = 0.0f;
    if (idx < vocab_size) {
        local_exp = expf(__half2float(logits[idx]) - global_max);
    }
    
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        local_exp += expf(__half2float(logits[i]) - global_max);
    }
    
    shared_sum[tid] = local_exp;
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
    
    // Normalize (convert back to FP16)
    if (idx < vocab_size) {
        float prob = expf(__half2float(logits[idx]) - global_max) / global_sum;
        probs[idx] = __float2half(prob);
    }
    
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        float prob = expf(__half2float(logits[i]) - global_max) / global_sum;
        probs[i] = __float2half(prob);
    }
}
```

---

## Testing Strategy

### Unit Tests (10 tests)

**FP16 Greedy Sampling** (3 tests):
1. FP16 simple argmax
2. FP16 large vocabulary
3. FP16 determinism

**FP16 Stochastic Sampling** (3 tests):
1. FP16 softmax normalization
2. FP16 sampling distribution
3. FP16 numerical stability

**FP16 Advanced Sampling** (4 tests):
1. FP16 top-k filtering
2. FP16 top-p filtering
3. FP16 repetition penalty
4. FP16 min-p filtering

### Performance Tests (3 tests)

1. **FP16 vs FP32 Bandwidth**: Measure memory bandwidth savings
2. **FP16 vs FP32 Latency**: Measure latency improvement
3. **FP16 Accuracy**: Verify numerical accuracy vs FP32

---

## Performance Targets

- **Bandwidth Reduction**: 50% (FP16 is half the size of FP32)
- **Latency Improvement**: 20-30% (less memory bandwidth)
- **Accuracy**: <0.1% difference in token selection vs FP32

---

## Definition of Done

- [ ] All FP16 kernels implemented
- [ ] Unit tests passing (10 tests)
- [ ] Performance tests passing (3 tests)
- [ ] Performance targets met
- [ ] Documentation updated
- [ ] Code reviewed (self-review)

---

## References

- **Spec**: `bin/.specs/01_M0_worker_orcd.md` (M0-W-1421)
- **CUDA FP16**: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html

---
Built by Foundation-Alpha 🏗️
