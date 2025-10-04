# GT-025: GPT Weight Loading

**Team**: GPT-Gamma  
**Sprint**: Sprint 4 (GPT Basic)  
**Size**: M (2 days)  
**Days**: 61-62  
**Spec Ref**: M0-W-1220, M0-W-1221

---

## Story Description

Implement weight loading from GGUF file to VRAM for GPT architecture. Use memory-mapped I/O and chunked transfer to efficiently load GPT-OSS-20B weights into GPU memory.

---

## Acceptance Criteria

- [ ] Load all GPT weights from GGUF to VRAM
- [ ] Use memory-mapped I/O for efficient file access
- [ ] Use chunked H2D transfer for large tensors
- [ ] Validate all weights loaded correctly
- [ ] Track VRAM usage during loading
- [ ] Unit tests validate weight loading
- [ ] Integration test loads full GPT-OSS-20B model
- [ ] Error handling for insufficient VRAM

---

## Dependencies

### Upstream (Blocks This Story)
- GT-024: GPT Weight Mapping (needs weight structure)
- LT-003: Memory-Mapped I/O (needs mmap implementation)

### Downstream (This Story Blocks)
- GT-026: GPT Forward Pass (needs loaded weights)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/model/gpt_loader.cpp` - Weight loader

---

## Testing Strategy

### Unit Tests
- Test weight loading
- Test VRAM allocation
- Test error handling

### Integration Tests
- Load full GPT-OSS-20B model
- Verify VRAM usage

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation updated

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.3

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
