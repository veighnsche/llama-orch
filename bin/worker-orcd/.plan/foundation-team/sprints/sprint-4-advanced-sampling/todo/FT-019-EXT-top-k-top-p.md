# FT-019-EXT-1: Top-K and Top-P Sampling

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Advanced Sampling  
**Size**: M (3 days)  
**Days**: 1 - 3  
**Spec Ref**: M0-W-1421, GENERATION_PARAMETERS_ANALYSIS.md

---

## Story Description

Implement Top-K and Top-P (nucleus) sampling to filter low-probability tokens and improve generation quality. These are the most commonly used advanced sampling parameters in production LLM APIs.

**Top-K**: Keep only top K tokens by probability  
**Top-P**: Keep tokens whose cumulative probability <= top_p

---

## Acceptance Criteria

### Top-K Sampling
- [ ] Kernel performs partial sort to find top K tokens
- [ ] Zeros out logits outside top K
- [ ] Handles edge cases (k=0, k=vocab_size, k>vocab_size)
- [ ] Unit tests validate filtering (5+ tests)
- [ ] Performance acceptable (<2ms for vocab=151936)

### Top-P Sampling
- [ ] Kernel sorts logits in descending order
- [ ] Computes cumulative probability
- [ ] Filters tokens where cumsum > top_p
- [ ] Handles edge cases (p=0.0, p=1.0)
- [ ] Unit tests validate filtering (5+ tests)
- [ ] Numerical stability with large logits

### Integration
- [ ] Top-K + Top-P can be used together
- [ ] Integration tests validate combined usage
- [ ] Performance profiling complete

---

## Technical Details

### Top-K Implementation

**Approach**: Use Thrust library for partial sort
```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

__global__ void apply_top_k(
    float* logits,
    int vocab_size,
    int top_k
) {
    // If top_k disabled or >= vocab_size, no filtering
    if (top_k <= 0 || top_k >= vocab_size) {
        return;
    }
    
    // Create indices array
    thrust::device_vector<int> indices(vocab_size);
    thrust::sequence(indices.begin(), indices.end());
    
    // Sort by logits (descending)
    thrust::sort_by_key(
        thrust::device,
        logits, logits + vocab_size,
        indices.begin(),
        thrust::greater<float>()
    );
    
    // Zero out logits outside top k
    for (int i = top_k; i < vocab_size; ++i) {
        logits[indices[i]] = -INFINITY;
    }
}
```

**Alternative (Custom Kernel)**:
```cpp
// If Thrust is too slow, implement custom partial sort
__global__ void partial_sort_top_k(
    const float* logits_in,
    float* logits_out,
    int vocab_size,
    int top_k
) {
    // Use parallel selection algorithm
    // Find k-th largest element (pivot)
    // Partition: keep elements >= pivot, zero others
}
```

### Top-P Implementation

**Approach**: Sort + cumulative sum + filter
```cpp
__global__ void apply_top_p(
    float* logits,
    int vocab_size,
    float top_p
) {
    // If top_p disabled, no filtering
    if (top_p >= 1.0f) {
        return;
    }
    
    // Sort logits descending
    thrust::device_vector<float> sorted_logits(logits, logits + vocab_size);
    thrust::device_vector<int> indices(vocab_size);
    thrust::sequence(indices.begin(), indices.end());
    
    thrust::sort_by_key(
        thrust::device,
        sorted_logits.begin(), sorted_logits.end(),
        indices.begin(),
        thrust::greater<float>()
    );
    
    // Compute softmax on sorted logits
    float max_logit = sorted_logits[0];
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        sum += expf(sorted_logits[i] - max_logit);
    }
    
    // Find cutoff where cumsum > top_p
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; ++i) {
        float prob = expf(sorted_logits[i] - max_logit) / sum;
        cumsum += prob;
        if (cumsum > top_p) {
            cutoff = i;
            break;
        }
    }
    
    // Zero out logits below cutoff
    for (int i = cutoff; i < vocab_size; ++i) {
        logits[indices[i]] = -INFINITY;
    }
}
```

### Integration

**Combined Usage**:
```cpp
// Apply filters in order
if (config.top_k > 0) {
    apply_top_k(logits, vocab_size, config.top_k);
}
if (config.top_p < 1.0f) {
    apply_top_p(logits, vocab_size, config.top_p);
}
// Then softmax + sample
```

---

## Testing Strategy

### Unit Tests: Top-K (5 tests)

1. **BasicTopK**: k=50, vocab=1000, verify only 50 tokens remain
2. **TopKDisabled**: k=0, verify no filtering
3. **TopKAll**: k=vocab_size, verify no filtering
4. **TopKTooLarge**: k>vocab_size, verify clamped to vocab_size
5. **TopKLargeVocab**: k=100, vocab=151936, verify correct filtering

### Unit Tests: Top-P (5 tests)

1. **BasicTopP**: p=0.9, verify cumsum cutoff correct
2. **TopPZero**: p=0.0, verify only max token kept
3. **TopPOne**: p=1.0, verify no filtering
4. **TopPNumericalStability**: large logits, verify no overflow
5. **TopPLargeVocab**: p=0.9, vocab=151936, verify correct filtering

### Integration Tests (3 tests)

1. **TopKTopPCombined**: Apply both filters, verify correct interaction
2. **TemperatureTopKTopP**: Full pipeline with temperature scaling
3. **DeterminismWithFilters**: Same seed + filters → same output

---

## Performance Profiling

### Metrics to Collect

1. **Sorting Time**: Time to sort logits (top-k and top-p)
2. **Filtering Time**: Time to zero out filtered tokens
3. **Total Overhead**: Additional latency vs core sampling
4. **Memory Usage**: Temporary buffers for sorting

### Profiling Commands

```bash
# Profile with nvprof
nvprof --metrics shared_efficiency,gld_efficiency ./build/tests/test_sampling

# Profile with Nsight Compute
ncu --set full --target-processes all ./build/tests/test_sampling
```

### Performance Targets

- **Top-K sorting**: <2ms for vocab=151936
- **Top-P filtering**: <1ms for vocab=151936
- **Total overhead**: <5ms per token
- **Memory overhead**: <1 MB temporary buffers

**If targets not met**: Optimize sorting (custom kernels, parallel primitives)

---

## Definition of Done

- [ ] Top-K kernel implemented and tested (5 tests)
- [ ] Top-P kernel implemented and tested (5 tests)
- [ ] Integration tests passing (3 tests)
- [ ] Performance profiling complete
- [ ] Performance within budget (<5ms per token)
- [ ] Documentation updated
- [ ] Code reviewed (self-review)

---

## References

- **Deferral Decision**: `../sprint-3-shared-kernels/ADVANCED_SAMPLING_DEFERRAL.md`
- **Spec**: `bin/.specs/01_M0_worker_orcd.md` (M0-W-1421)
- **Analysis**: `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`
- **Nucleus Sampling Paper**: Holtzman et al. 2019 "The Curious Case of Neural Text Degeneration"
- **Top-K Sampling Paper**: Fan et al. 2018 "Hierarchical Neural Story Generation"

---
Built by Foundation-Alpha 🏗️
