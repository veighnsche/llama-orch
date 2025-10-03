# Worker-orcd Specification Integration

All specifications are now integrated into the main binary. Previously separate crate specifications have been consolidated.

**Date**: 2025-10-03
**Status**: Complete - All functionality integrated into single binary

---

## Spec Hierarchy

```
bin/worker-orcd/.specs/00_worker-orcd.md (main/binary)
├── WORKER-4000-4003: Goals
├── WORKER-4010-4032: Architecture & Lifecycle
├── WORKER-4100-4122: VRAM Residency (integrated)
├── WORKER-4200-4253: HTTP API (integrated)
├── WORKER-4300-4323: Input Validation (integrated)
├── WORKER-4400-4423: CUDA FFI & Safety (integrated)
├── WORKER-4500-4522: Scheduling (integrated)
├── WORKER-4600-4623: Capability Matching (integrated)
├── WORKER-4700-4722: Inference (integrated)
├── WORKER-4800-4822: Observability (integrated)
├── WORKER-4900-4922: Security (integrated)
├── WORKER-4950-4972: Error Handling (integrated)
├── WORKER-4980-4991: Configuration (integrated)
└── WORKER-4995-5000: Testing (integrated)
```

---

## Requirement Distribution

| Requirement Range | Module | Spec File |
|-------------------|--------|-----------|
| WORKER-4000-4003 | Goals | `bin/worker-orcd/.specs/00_worker-orcd.md` |
| WORKER-4010-4032 | Architecture & Lifecycle | `bin/worker-orcd/.specs/00_worker-orcd.md` |
| WORKER-4100-4122 | VRAM Residency | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |
| WORKER-4200-4253 | HTTP API | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |
| WORKER-4300-4323 | Input Validation | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |
| WORKER-4400-4423 | CUDA FFI & Safety | `bin/worker-orcd/.specs/01_cuda_ffi_boundary.md` (integrated) |
| WORKER-4500-4522 | Scheduling | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |
| WORKER-4600-4623 | Capability Matching | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |
| WORKER-4700-4722 | Inference | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |
| WORKER-4800-4822 | Observability | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |
| WORKER-4900-4922 | Security & Privileges | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |
| WORKER-4950-4972 | Error Handling | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |
| WORKER-4980-4991 | Configuration | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |
| WORKER-4995-5000 | Testing | `bin/worker-orcd/.specs/00_worker-orcd.md` (integrated) |

---

## Integrated Specifications

All specifications are now integrated into the main binary specification. Previously separate crate specifications have been consolidated into `bin/worker-orcd/.specs/00_worker-orcd.md`.

## Main Binary Specification

**File**: `bin/worker-orcd/.specs/00_worker-orcd.md`

**Covers all functionality**:
- Architecture & process model (WORKER-4010-4032)
- VRAM residency enforcement (WORKER-4100-4122)
- HTTP API endpoints (WORKER-4200-4253)
- Input validation (WORKER-4300-4323)
- CUDA FFI safety (WORKER-4400-4423)
- Scheduling logic (WORKER-4500-4522)
- Capability matching (WORKER-4600-4623)
- Inference execution (WORKER-4700-4722)
- Observability & telemetry (WORKER-4800-4822)
- Security & privileges (WORKER-4900-4922)
- Error handling & recovery (WORKER-4950-4972)
- Configuration & deployment (WORKER-4980-4991)
- Testing & validation (WORKER-4995-5000)

**Role**: Complete specification for integrated binary functionality

---

## Shared Specs (External)

### Input Validation

**Location**: `libs/shared-crates/input-validation/.specs/` (to be created)

**Covers**: Request validation (WORKER-4300-4305)

**Shared by**: worker-orcd, orchestratord, pool-managerd

---

## Spec Maintenance

### Adding New Requirements

1. Determine which component owns the requirement
2. Add to appropriate component spec file
3. Update main spec with reference if cross-cutting
4. Update this distribution document

### Modifying Existing Requirements

1. Update in component spec file (source of truth)
2. Update main spec reference if needed
3. Ensure requirement ID remains stable

### Cross-Cutting Requirements

Requirements that span multiple components:
- Add to main spec (`bin/worker-orcd/.specs/00_worker-orcd.md`)
- Reference from component specs as needed
- Examples: observability, security, testing

---

## Verification

Each crate spec MUST:
- ✅ Have RFC-2119 conformance language (MUST/SHOULD/MAY)
- ✅ Reference parent spec
- ✅ Define clear scope
- ✅ List dependencies
- ✅ Include traceability section
- ✅ Specify security tier (if applicable)

---

## Integration Benefits

1. **Unified specification**: All requirements in single, cohesive document
2. **Simplified maintenance**: No need to coordinate across multiple spec files
3. **Clear ownership**: Main binary specification owns all functionality
4. **Consistency**: Single source of truth for all requirements
5. **Testability**: All functionality tested as integrated binary
6. **Modularity maintained**: Code organized into modules, specs centralized

---

**Status**: Specification integration complete - All functionality covered in main binary spec
**Total specs**: 2 files (main binary + CUDA FFI boundary)
**Total requirements**: 100+ normative requirements integrated
