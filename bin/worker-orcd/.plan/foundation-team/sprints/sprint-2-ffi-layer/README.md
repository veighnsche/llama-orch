# Sprint 2: FFI Layer

**Team**: Foundation-Alpha  
**Days**: 10-22 (13 agent-days)  
**Goal**: Define and implement FFI interface, achieve FFI lock milestone

---

## Sprint Overview

Sprint 2 is the most critical sprint for the entire M0 project. It defines the FFI interface between Rust and C++/CUDA, implements the bindings, and achieves the **FFI Lock milestone on Day 15**. This milestone blocks both Llama-Beta and GPT-Gamma teams from starting their work.

The FFI interface must be stable, well-documented, and exception-safe. Once locked, it cannot be changed without coordinating with all teams.

---

## Stories in This Sprint

| ID | Title | Size | Days | Day Range |
|----|-------|------|------|-----------|
| FT-006 | FFI Interface Definition | M | 2 | 10-11 |
| FT-007 | Rust FFI Bindings | M | 2 | 12-13 |
| FT-008 | Error Code System (C++) | S | 1 | 14 |
| FT-009 | Error Code to Result Conversion (Rust) | S | 1 | 15 |
| FT-010 | CUDA Context Initialization | M | 2 | 16-17 |
| **FT-R001** | **Cancellation Endpoint** | **S** | **1** | **18** |

**Total**: 6 stories (5 planned + 1 retroactive), 14 agent-days (Days 10-23)

**Note**: FT-R001 is a retroactive addition identified during Sprint 1 retrospective to ensure M0-W-1330 compliance.

---

## Story Execution Order

### Days 10-11: FT-006 - FFI Interface Definition 🔒
**Goal**: Define complete FFI interface with opaque handle types  
**Key Deliverable**: FFI_INTERFACE_LOCKED.md published on Day 11  
**Blocks**: ALL Llama and GPT work

**Critical**: This story achieves the FFI Lock milestone. The interface must be:
- Stable (no changes after Day 11)
- Complete (all functions defined)
- Documented (parameter semantics, error codes)
- Exception-safe (no C++ exceptions across boundary)

### Days 12-13: FT-007 - Rust FFI Bindings
**Goal**: Implement Rust FFI bindings with RAII wrappers  
**Key Deliverable**: Safe Rust API wrapping C FFI  
**Blocks**: FT-008 (error code system)

### Day 14: FT-008 - Error Code System (C++)
**Goal**: Implement C++ error code system with stable codes  
**Key Deliverable**: ErrorCode enum with all error types  
**Blocks**: FT-009 (error code conversion)

### Day 15: FT-009 - Error Code to Result Conversion (Rust)
**Goal**: Convert C error codes to Rust Result types  
**Key Deliverable**: Error mapping with context preservation  
**Blocks**: FT-010 (CUDA context init)

### Days 16-17: FT-010 - CUDA Context Initialization
**Goal**: Initialize CUDA context via FFI  
**Key Deliverable**: Working CUDA context creation from Rust  
**Blocks**: Sprint 3 (shared kernels)

### Day 18: FT-R001 - Cancellation Endpoint (Retroactive)
**Goal**: Implement POST /cancel endpoint  
**Key Deliverable**: Idempotent job cancellation with SSE error events  
**Blocks**: M0 completion (required for M0-W-1330)  
**Type**: Retroactive addition (identified in Sprint 1 retrospective)

**Why Retroactive**: M0-W-1330 requires POST /cancel endpoint, but it was not included in original Sprint 2 plan. Added to ensure M0 compliance.

---

## Critical Milestones

### FFI Lock (Day 11)

**What**: FFI interface frozen and published  
**Why Critical**: Blocks Llama-Beta and GPT-Gamma from starting  
**Deliverable**: `coordination/FFI_INTERFACE_LOCKED.md` published  
**Blocks**: LT-000 (Llama prep), GT-000 (GPT prep)

**Checklist**:
- [ ] All FFI functions defined with signatures
- [ ] All opaque handle types defined
- [ ] All error codes enumerated
- [ ] Parameter semantics documented
- [ ] Return value semantics documented
- [ ] Memory ownership rules documented
- [ ] Thread safety guarantees documented
- [ ] FFI_INTERFACE_LOCKED.md published to coordination/

**Validation**:
```bash
# Verify FFI header exists
ls bin/worker-orcd/cuda/ffi.h

# Verify documentation published
ls coordination/FFI_INTERFACE_LOCKED.md

# Notify dependent teams
echo "FFI Lock achieved on Day 11" >> coordination/master-timeline.md
```

---

## Dependencies

### Upstream (Blocks This Sprint)
- FT-005: Request Validation Framework (provides request types)

### Downstream (This Sprint Blocks)
- **CRITICAL**: Llama-Beta Sprint 1 (LT-001 to LT-006)
- **CRITICAL**: GPT-Gamma Sprint 1 (GT-001 to GT-007)
- Sprint 3: Shared Kernels (needs FFI and CUDA context)

---

## Success Criteria

Sprint is complete when:
- [ ] All 6 stories marked complete (5 planned + 1 retroactive)
- [ ] FFI interface defined and locked (Day 11)
- [ ] FFI_INTERFACE_LOCKED.md published
- [ ] Rust FFI bindings implemented with RAII
- [ ] Error code system operational
- [ ] Error code to Result conversion working
- [ ] CUDA context initialization working
- [ ] POST /cancel endpoint implemented (M0-W-1330)
- [ ] All unit tests passing
- [ ] Llama and GPT teams notified of FFI lock
- [ ] Ready for Sprint 3 (shared kernels)

---

## Next Sprint

**Sprint 3**: Shared Kernels  
**Starts**: Day 23  
**Focus**: VRAM enforcement, memory management, shared CUDA kernels

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Coordinated by Project Management Team 📋
