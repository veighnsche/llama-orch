# LT-019: Kernel Unit Tests

**Team**: Llama-Beta  
**Sprint**: Sprint 4 - GQA Attention + Integration  
**Size**: M (2 days)  
**Days**: 52-53  
**Spec Ref**: M0-W-1430

---

## Story Description

Create comprehensive unit test suite for all Llama CUDA kernels. Validate numerical correctness, edge cases, and performance characteristics for RoPE, RMSNorm, residual, GQA attention, and SwiGLU kernels.

---

## Acceptance Criteria

- [ ] Unit tests for RoPE kernel (LT-012)
- [ ] Unit tests for RMSNorm kernel (LT-013)
- [ ] Unit tests for Residual kernel (LT-014)
- [ ] Unit tests for GQA Attention Prefill (LT-015)
- [ ] Unit tests for GQA Attention Decode (LT-016)
- [ ] Unit tests for SwiGLU FFN (LT-017)
- [ ] Test numerical correctness (compare with reference)
- [ ] Test edge cases (zero values, boundary conditions)
- [ ] Test different tensor shapes and dimensions
- [ ] All tests pass with defined tolerance (±0.01 to ±0.05)
- [ ] Performance benchmarks recorded for each kernel
- [ ] Error handling tests (invalid inputs, shape mismatches)
- [ ] Log test results with pass/fail status

---

## Dependencies

### Upstream (Blocks This Story)
- LT-012: RoPE Kernel (needs kernel)
- LT-013: RMSNorm Kernel (needs kernel)
- LT-014: Residual Kernel (needs kernel)
- LT-015: GQA Attention Prefill (needs kernel)
- LT-016: GQA Attention Decode (needs kernel)
- LT-017: SwiGLU FFN (needs kernel)

### Downstream (This Story Blocks)
- LT-020: Gate 1 Participation (needs validated kernels)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/kernels/test_rope.cpp` - RoPE tests
- `bin/worker-orcd/tests/kernels/test_rmsnorm.cpp` - RMSNorm tests
- `bin/worker-orcd/tests/kernels/test_residual.cpp` - Residual tests
- `bin/worker-orcd/tests/kernels/test_gqa_attention.cpp` - GQA tests
- `bin/worker-orcd/tests/kernels/test_swiglu.cpp` - SwiGLU tests
- `bin/worker-orcd/tests/kernels/test_utils.h` - Test utilities

### Test Utilities
```cpp
// Numerical comparison with tolerance
bool approx_equal(const half* a, const half* b, int n, float tolerance) {
    for (int i = 0; i < n; ++i) {
        float diff = std::abs(__half2float(a[i]) - __half2float(b[i]));
        if (diff > tolerance) {
            return false;
        }
    }
    return true;
}

// Generate random tensor
void random_tensor(half* data, int n, float mean = 0.0f, float stddev = 1.0f) {
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(mean, stddev);
    for (int i = 0; i < n; ++i) {
        data[i] = __float2half(dist(gen));
    }
}

// Reference implementations (CPU)
void rope_reference(/* ... */);
void rmsnorm_reference(/* ... */);
void gqa_attention_reference(/* ... */);
// ...
```

### Test Structure
```cpp
TEST(RoPEKernel, BasicRotation) {
    // Setup
    int seq_len = 10;
    int num_heads = 14;
    int head_dim = 64;
    
    half* q_in = allocate_device(seq_len * num_heads * head_dim);
    half* q_out = allocate_device(seq_len * num_heads * head_dim);
    half* q_ref = allocate_host(seq_len * num_heads * head_dim);
    
    random_tensor(q_in, seq_len * num_heads * head_dim);
    
    // Execute kernel
    RoPEConfig config = {seq_len, num_heads, head_dim, 10000.0f, head_dim};
    rope_forward(q_out, nullptr, q_in, nullptr, config);
    
    // Compute reference
    rope_reference(q_ref, q_in, config);
    
    // Compare
    ASSERT_TRUE(approx_equal(q_out, q_ref, seq_len * num_heads * head_dim, 0.01f));
    
    // Cleanup
    free_device(q_in);
    free_device(q_out);
    free_host(q_ref);
}
```

### Test Coverage

**RoPE Tests** (5 tests):
- Basic rotation (seq_len=10, head_dim=64)
- Different frequency bases (10000, 1000000)
- Edge case: seq_len=1 (single position)
- Edge case: seq_len=32768 (max context)
- Numerical correctness (tolerance ±0.01)

**RMSNorm Tests** (5 tests):
- Basic normalization (hidden_dim=896)
- Different dimensions (896, 3072)
- Numerical stability (very small/large values)
- Weight scaling
- Numerical correctness (tolerance ±0.01)

**Residual Tests** (4 tests):
- Element-wise addition
- In-place operation
- Out-of-place operation
- Different tensor shapes

**GQA Attention Tests** (6 tests):
- Prefill attention (seq_len=128)
- Decode attention (seq_len=1, cache_len=100)
- GQA head grouping (14 Q heads, 2 KV heads)
- Causal masking (prefill)
- KV cache read/write
- Numerical correctness (tolerance ±0.05)

**SwiGLU Tests** (5 tests):
- SiLU activation
- Gate and up projections
- Element-wise multiply
- Down projection
- Numerical correctness (tolerance ±0.05)

---

## Testing Strategy

### Unit Tests
- Test each kernel individually
- Test with various input shapes
- Test numerical correctness against reference
- Test edge cases and boundary conditions
- Test error handling

### Performance Tests
- Benchmark each kernel (TFLOPS, GB/s)
- Compare with theoretical peak performance
- Record baseline performance metrics

### Integration Tests
- Test kernel combinations (RoPE + Attention)
- Test full transformer block (all kernels)

### Manual Verification
1. Run full kernel test suite
2. Verify all tests pass
3. Check performance benchmarks
4. Review logs for any warnings

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] All kernel tests passing (30+ tests total)
- [ ] Numerical validation passing (defined tolerances)
- [ ] Performance benchmarks recorded
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.5 (Inference Kernels)
- Related Stories: LT-012, LT-013, LT-014, LT-015, LT-016, LT-017
- CUDA Testing: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
