# GT-013: GELU Unit Tests

**Team**: GPT-Gamma  
**Sprint**: Sprint 2 (GPT Kernels)  
**Size**: S (1 day)  
**Days**: 35  
**Spec Ref**: M0-W-1433

---

## Story Description

Implement comprehensive unit tests for GELU activation kernel to validate correctness, numerical accuracy, and edge case handling. Tests must verify GELU output matches reference implementation with acceptable FP16 precision.

---

## Acceptance Criteria

- [ ] Test validates GELU output for known inputs
- [ ] Test validates numerical accuracy (error <0.1%)
- [ ] Test validates edge cases (zero, negative, large positive values)
- [ ] Test validates FP16 precision handling
- [ ] Test compares against reference implementation
- [ ] All tests passing with acceptable tolerance
- [ ] Performance benchmark included
- [ ] Documentation updated with test coverage

---

## Dependencies

### Upstream (Blocks This Story)
- GT-012: GELU Activation Kernel (needs GELU implementation)

### Downstream (This Story Blocks)
- GT-014: GPT FFN Kernel (needs validated GELU)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/kernels/gelu_test.cu` - Unit test suite
- `bin/worker-orcd/tests/fixtures/gelu_reference.json` - Reference outputs

### Key Test Cases
```cpp
TEST(GELU, KnownInputs) {
    // Test GELU(0) = 0
    // Test GELU(1) ≈ 0.8413
    // Test GELU(-1) ≈ -0.1587
}

TEST(GELU, NumericalAccuracy) {
    // Compare with reference implementation
    // Tolerance: 0.1% for FP16
}

TEST(GELU, EdgeCases) {
    // Test large positive values
    // Test large negative values
    // Test very small values near zero
}
```

---

## Testing Strategy

### Unit Tests
- Test known GELU values
- Test numerical accuracy
- Test edge cases
- Test FP16 precision

### Integration Tests
- Test with GPT-OSS-20B FFN dimensions
- Compare with reference

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7.3
- Related Stories: GT-012, GT-014

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
