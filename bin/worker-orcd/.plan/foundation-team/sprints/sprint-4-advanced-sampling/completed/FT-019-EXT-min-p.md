# FT-019-EXT-4: Min-P Sampling

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Advanced Sampling  
**Size**: XS (0.5 days)  
**Days**: 7 - 7  
**Spec Ref**: M0-W-1421, GENERATION_PARAMETERS_ANALYSIS.md  
**Priority**: Low (Optional)

---

## Story Description

Implement Min-P sampling to filter tokens below a minimum probability threshold. This is a less common parameter but provides additional control over token selection.

**Formula**: Remove tokens where `prob < min_p * max_prob`

---

## Acceptance Criteria

- [ ] Kernel filters tokens below min_p threshold
- [ ] Re-normalizes remaining probabilities
- [ ] Handles edge cases (min_p=0.0, min_p=1.0)
- [ ] Unit tests validate filtering (3+ tests)
- [ ] Performance acceptable (<0.1ms per token)

---

## Technical Details

### Kernel Implementation

```cpp
/**
 * Apply min-p filtering to logits.
 * 
 * Filters tokens where prob < min_p * max_prob.
 * 
 * @param logits Device pointer to logits [vocab_size] (modified in-place)
 * @param vocab_size Vocabulary size
 * @param min_p Minimum probability threshold (0.0-1.0)
 */
__global__ void apply_min_p(
    float* logits,
    int vocab_size,
    float min_p
) {
    // If min_p disabled, skip
    if (min_p <= 0.0f) {
        return;
    }
    
    // Find max logit
    __shared__ float shared_max[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max = (idx < vocab_size) ? logits[idx] : -INFINITY;
    
    // Grid-stride loop
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, logits[i]);
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
    
    // Compute threshold: min_p * max_prob
    // Since we're in logit space: threshold_logit = log(min_p) + max_logit
    float threshold_logit = logf(min_p) + global_max;
    
    // Filter tokens below threshold
    if (idx < vocab_size) {
        if (logits[idx] < threshold_logit) {
            logits[idx] = -INFINITY;
        }
    }
    
    // Grid-stride loop
    for (int i = idx + blockDim.x * gridDim.x; i < vocab_size; 
         i += blockDim.x * gridDim.x) {
        if (logits[i] < threshold_logit) {
            logits[i] = -INFINITY;
        }
    }
}

/**
 * Launch min-p filtering kernel.
 */
void launch_min_p(
    float* logits,
    int vocab_size,
    float min_p,
    cudaStream_t stream = 0
) {
    if (min_p <= 0.0f) {
        return;  // Disabled
    }
    
    if (vocab_size <= 0 || logits == nullptr) {
        fprintf(stderr, "Invalid inputs to min-p\n");
        return;
    }
    
    int threads_per_block = 256;
    int num_blocks = (vocab_size + threads_per_block - 1) / threads_per_block;
    if (num_blocks > 256) {
        num_blocks = 256;
    }
    
    apply_min_p<<<num_blocks, threads_per_block, 0, stream>>>(
        logits, vocab_size, min_p
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Min-p kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}
```

---

## Testing Strategy

### Unit Tests (3 tests)

1. **BasicMinP**
   - Given: logits with known distribution, min_p=0.05
   - When: apply_min_p
   - Then: Tokens with prob < 5% of max filtered out

2. **MinPDisabled**
   - Given: logits, min_p=0.0
   - When: apply_min_p
   - Then: No filtering (logits unchanged)

3. **MinPOne**
   - Given: logits, min_p=1.0
   - When: apply_min_p
   - Then: Only max token kept (all others filtered)

---

## HTTP API Extension

### Request Schema Addition

```json
{
  "prompt": "Write a haiku",
  "temperature": 0.7,
  "min_p": 0.05
}
```

### Validation Rules

- `min_p`: 0.0-1.0
- Default: 0.0 (disabled)
- Optional parameter

---

## Performance Targets

- **Latency**: <0.1ms per token
- **Memory**: No additional memory
- **Overhead**: <1% of total sampling time

---

## Definition of Done

- [ ] Kernel implemented and tested (3 tests)
- [ ] HTTP API extended
- [ ] Performance within budget (<0.1ms)
- [ ] Documentation updated
- [ ] Code reviewed (self-review)

---

## Notes

**Priority**: Low (can be skipped if time-constrained)

**Rationale**: Min-P is rarely used compared to Top-P/Top-K. If Sprint 4 runs over time, defer Min-P to Sprint 5.

---

## References

- **Spec**: `bin/.specs/01_M0_worker_orcd.md` (M0-W-1421)
- **llama.cpp**: Min-P implementation reference

---
Built by Foundation-Alpha 🏗️
