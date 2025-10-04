# GT-030: MXFP4 Unit Tests

**Team**: GPT-Gamma  
**Sprint**: Sprint 5 (MXFP4 Dequant)  
**Size**: M (2 days)  
**Days**: 70-71  
**Spec Ref**: M0-W-1822

---

## Story Description

Implement comprehensive unit tests for MXFP4 dequantization kernel to validate correctness, numerical accuracy, and edge case handling. Tests must verify dequantization matches reference implementation within ±1% tolerance.

---

## Acceptance Criteria

- [ ] Test validates MXFP4 block parsing
- [ ] Test validates FP4 mantissa extraction
- [ ] Test validates FP8 scale extraction
- [ ] Test validates dequantization formula
- [ ] Test validates numerical accuracy (±1%)
- [ ] Test validates edge cases (zero, max values)
- [ ] All tests passing
- [ ] Documentation updated

---

## Dependencies

### Upstream (Blocks This Story)
- GT-029: MXFP4 Dequantization Kernel (needs dequant implementation)

### Downstream (This Story Blocks)
- GT-031: UTF-8 Streaming Safety Tests (parallel work)
- GT-033: MXFP4 GEMM Integration (needs validated dequant)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/kernels/mxfp4_test.cu` - MXFP4 test suite
- `bin/worker-orcd/tests/fixtures/mxfp4_reference.json` - Reference values

---

## Testing Strategy

### Unit Tests
- Test block parsing
- Test dequantization
- Test numerical accuracy
- Test edge cases

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.1

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
