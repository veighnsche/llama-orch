# GT-027: GPT Basic Generation Test

**Team**: GPT-Gamma  
**Sprint**: Sprint 4 (GPT Basic)  
**Size**: S (1 day)  
**Days**: 66  
**Spec Ref**: M0-W-1001

---

## Story Description

Implement basic text generation test for GPT-OSS-20B using Q4_K_M weights. Validate model can generate coherent tokens and complete simple prompts.

---

## Acceptance Criteria

- [ ] Test generates tokens from prompt
- [ ] Test validates token IDs are valid
- [ ] Test validates output is coherent
- [ ] Test validates generation completes without errors
- [ ] Test runs with temperature=0 for reproducibility
- [ ] Documentation updated with test results

---

## Dependencies

### Upstream (Blocks This Story)
- GT-026: GPT Forward Pass (needs working inference)

### Downstream (This Story Blocks)
- GT-028: Gate 2 Checkpoint (needs basic generation working)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/gpt_generation_test.rs` - Generation test

---

## Testing Strategy

### Integration Tests
- Test basic generation
- Test reproducibility
- Validate output quality

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
