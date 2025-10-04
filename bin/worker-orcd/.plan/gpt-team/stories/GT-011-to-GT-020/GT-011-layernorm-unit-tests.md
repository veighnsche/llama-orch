# GT-011: LayerNorm Unit Tests

**Team**: GPT-Gamma  
**Sprint**: Sprint 2 (GPT Kernels)  
**Size**: S (1 day)  
**Days**: 32  
**Spec Ref**: M0-W-1432

---

## Story Description

Implement comprehensive unit tests for LayerNorm kernel to validate correctness, numerical stability, and edge case handling. Tests must verify mean, variance, normalization, and scale/bias operations match reference implementations.

---

## Acceptance Criteria

- [ ] Test validates mean computation correctness
- [ ] Test validates variance computation correctness
- [ ] Test validates normalization correctness
- [ ] Test validates scale (gamma) application
- [ ] Test validates bias (beta) application
- [ ] Test validates epsilon handling (numerical stability)
- [ ] Test validates FP16/FP32 precision handling
- [ ] Test validates edge cases (zero variance, extreme values)
- [ ] All tests passing with <0.01% error tolerance
- [ ] Documentation updated with test coverage

---

## Dependencies

### Upstream (Blocks This Story)
- GT-009: LayerNorm Mean Reduction (needs mean kernel)
- GT-010: LayerNorm Variance + Normalize (needs complete LayerNorm)

### Downstream (This Story Blocks)
- GT-014: GPT FFN Kernel (needs validated LayerNorm)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/kernels/layernorm_test.cu` - Unit test suite
- `bin/worker-orcd/tests/fixtures/layernorm_reference.json` - Reference outputs

### Key Test Cases
```cpp
TEST(LayerNorm, MeanComputation) {
    // Test mean calculation matches reference
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected_mean = 2.5f;
    // ... test implementation
}

TEST(LayerNorm, VarianceComputation) {
    // Test variance calculation matches reference
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected_var = 1.25f;
    // ... test implementation
}

TEST(LayerNorm, NormalizationCorrectness) {
    // Test full LayerNorm matches reference
    // ... test implementation
}

TEST(LayerNorm, NumericalStability) {
    // Test with extreme values
    // ... test implementation
}
```

### Implementation Notes
- Compare against CPU reference implementation
- Test with various d_model sizes (128, 512, 2048, 8192)
- Test with various batch and sequence lengths
- Validate FP16 precision loss is acceptable
- Test epsilon prevents division by zero
- Use tolerance of 0.01% for FP16 comparisons

---

## Testing Strategy

### Unit Tests
- Test mean computation
- Test variance computation
- Test normalization
- Test scale/bias application
- Test numerical stability
- Test edge cases

### Integration Tests
- Test with GPT-OSS-20B layer dimensions
- Test full forward pass
- Compare with reference implementation

### Manual Verification
1. Run full test suite
2. Verify all tests pass
3. Check error tolerances
4. Profile test execution time

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7.3 (GPT Kernels)
- Related Stories: GT-009, GT-010

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
