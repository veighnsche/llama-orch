# Foundation Team Milestones

**Team**: Foundation-Alpha  
**Purpose**: Track all critical milestones and validation procedures

---

## Milestone Overview

Foundation team has **5 critical milestones** that block downstream work:

1. **FFI Lock** (Day 11) 🔒
2. **Gate 1: Foundation Complete** (Day 52) 🎯
3. **Gate 2: Both Architectures** (Day 62) 🎯
4. **Gate 3: Adapter Complete** (Day 71) 🎯
5. **Gate 4: M0 Complete** (Day 89) 🎯

---

## Milestone 1: FFI Interface Lock (Day 11)

**Story**: FT-006 - FFI Interface Definition  
**Sprint**: Sprint 2 (FFI Layer)

### Deliverables
- [ ] `worker_ffi.h` - Complete C API header
- [ ] `coordination/FFI_INTERFACE_LOCKED.md` - Lock announcement
- [ ] Opaque handle types defined
- [ ] Error code enum complete
- [ ] All function signatures finalized

### Validation Procedure
```bash
# Verify FFI header compiles
gcc -c bin/worker-orcd/cuda/include/worker_ffi.h

# Verify lock document exists
cat coordination/FFI_INTERFACE_LOCKED.md

# Verify no TODO/FIXME in FFI header
grep -E "TODO|FIXME" bin/worker-orcd/cuda/include/worker_ffi.h
```

### Pass Criteria
- FFI header compiles without errors
- Lock document published
- No unresolved TODOs in FFI header
- Llama and GPT teams notified

### Blocks
- Llama-Beta prep work (LT-000)
- GPT-Gamma prep work (GT-000)

---

## Milestone 2: Gate 1 - Foundation Complete (Day 52)

**Story**: FT-027 - Gate 1 Checkpoint  
**Sprint**: Sprint 4 (Integration + Gate 1)

### Deliverables
- [ ] HTTP server operational
- [ ] FFI boundary working
- [ ] CUDA context management
- [ ] Basic kernels (embedding, GEMM, sampling)
- [ ] VRAM-only enforcement
- [ ] Error handling complete
- [ ] Integration tests passing

### Validation Procedure
See: `integration-gates/gate-1-foundation-complete.md`

### Pass Criteria
- All Gate 1 checklist items ✅
- HTTP-FFI-CUDA integration test passing
- VRAM residency verification working

### Blocks
- Llama Gate 1 (LT-020)
- GPT Gate 1 (GT-022)

---

## Milestone 3: Gate 2 - Both Architectures (Day 62)

**Story**: FT-032 - Gate 2 Checkpoint  
**Sprint**: Sprint 6 (Adapter + Gate 3)

### Deliverables
- [ ] Llama implementation complete
- [ ] GPT implementation complete
- [ ] Both models load successfully
- [ ] Both models generate tokens
- [ ] Integration tests pass for both

### Validation Procedure
See: `integration-gates/gate-2-both-architectures.md`

### Pass Criteria
- Llama Gate 2 complete (LT-027)
- GPT Gate 2 complete (GT-028)
- Both architectures validated

### Blocks
- Adapter pattern work (FT-033+)

---

## Milestone 4: Gate 3 - Adapter Complete (Day 71)

**Story**: FT-038 - Gate 3 Checkpoint  
**Sprint**: Sprint 6 (Adapter + Gate 3)

### Deliverables
- [ ] InferenceAdapter interface complete
- [ ] Factory pattern working
- [ ] Architecture detection automatic
- [ ] LlamaInferenceAdapter working
- [ ] GPTInferenceAdapter working
- [ ] Integration tests passing

### Validation Procedure
See: `integration-gates/gate-3-adapter-complete.md`

### Pass Criteria
- All Gate 3 checklist items ✅
- Adapter pattern tests passing
- Both adapters validated

### Blocks
- Llama Gate 3 (LT-034)
- GPT Gate 3 (GT-041)

---

## Milestone 5: Gate 4 - M0 Complete (Day 89)

**Story**: FT-047 - Gate 4 Checkpoint  
**Sprint**: Sprint 7 (Final Integration)

### Deliverables
- [ ] All Foundation stories complete (FT-001 to FT-050)
- [ ] All Llama stories complete (LT-001 to LT-039)
- [ ] All GPT stories complete (GT-001 to GT-048)
- [ ] All tests passing
- [ ] All documentation complete
- [ ] All gates passed (1, 2, 3, 4)
- [ ] Performance baselines met
- [ ] Production ready

### Validation Procedure
See: `integration-gates/gate-4-m0-complete.md`

### Pass Criteria
- All Gate 4 checklist items ✅
- M0 haiku test passing
- All models working
- Ready for deployment

### Blocks
- **M0 Release**: Production deployment

---

## Milestone Timeline

```
Day 1    Day 11   Day 52   Day 62   Day 71   Day 89
  |        |        |        |        |        |
Start   FFI Lock  Gate 1   Gate 2   Gate 3   Gate 4
         🔒        🎯       🎯       🎯       🎯
         |         |        |        |        |
      Unblocks  Unblocks  Validates Unblocks  M0
      Llama+GPT  Gate 1   Both Arch Gate 3   Complete
```

---

## Milestone Tracking

| Milestone | Day | Story | Status | Date Completed |
|-----------|-----|-------|--------|----------------|
| FFI Lock | 11 | FT-006 | ⬜ Pending | - |
| Gate 1 | 52 | FT-027 | ⬜ Pending | - |
| Gate 2 | 62 | FT-032 | ⬜ Pending | - |
| Gate 3 | 71 | FT-038 | ⬜ Pending | - |
| Gate 4 | 89 | FT-047 | ⬜ Pending | - |

---

**Last Updated**: 2025-10-04  
**Updated By**: Foundation-Alpha
