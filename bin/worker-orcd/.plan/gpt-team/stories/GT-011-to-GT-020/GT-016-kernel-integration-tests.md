# GT-016: Kernel Integration Tests

**Team**: GPT-Gamma  
**Sprint**: Sprint 2 (GPT Kernels)  
**Size**: M (2 days)  
**Days**: 40-41  
**Spec Ref**: M0-W-1431

---

## Story Description

Implement integration tests for all GPT-specific kernels (LayerNorm, GELU, FFN, residual) to validate they work together correctly in a full transformer layer pipeline.

---

## Acceptance Criteria

- [ ] Integration test validates full transformer layer
- [ ] Test includes LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
- [ ] Test compares output with reference implementation
- [ ] Test validates numerical accuracy end-to-end
- [ ] Test validates VRAM usage stays within bounds
- [ ] All integration tests passing
- [ ] Performance benchmarks included
- [ ] Documentation updated

---

## Dependencies

### Upstream (Blocks This Story)
- GT-015: Residual Connection Kernel (needs all kernels)

### Downstream (This Story Blocks)
- GT-017: MHA Attention Prefill (needs validated kernel suite)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/gpt_kernels_test.cu` - Integration tests

---

## Testing Strategy

### Integration Tests
- Test full transformer layer
- Test kernel composition
- Compare with reference

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7.3

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
