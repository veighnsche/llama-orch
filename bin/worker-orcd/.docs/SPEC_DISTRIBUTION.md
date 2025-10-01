# Worker-orcd Specification Distribution

Specs are distributed across crates to ensure each component carries its own requirements.

**Date**: 2025-10-01  
**Status**: Complete

---

## Spec Hierarchy

```
bin/worker-orcd/.specs/00_worker-orcd.md (main/orchestration)
├── bin/worker-orcd-crates/api/.specs/00_api.md
├── bin/worker-orcd-crates/vram-residency/.specs/00_vram-residency.md
├── bin/worker-orcd-crates/model-loader/.specs/00_model-loader.md
├── bin/worker-orcd-crates/capability-matcher/.specs/00_capability-matcher.md
├── bin/worker-orcd-crates/scheduler/.specs/00_scheduler.md
└── bin/worker-orcd/cuda/kernels/.specs/00_cuda-kernels.md
```

---

## Requirement Distribution

| Requirement Range | Component | Spec File |
|-------------------|-----------|-----------|
| WORKER-4000-4003 | Goals | `bin/worker-orcd/.specs/00_worker-orcd.md` |
| WORKER-4010-4032 | Architecture & Lifecycle | `bin/worker-orcd/.specs/00_worker-orcd.md` |
| WORKER-4100-4122 | VRAM Residency | `bin/worker-orcd-crates/vram-residency/.specs/00_vram-residency.md` |
| WORKER-4200-4253 | RPC Protocol & API | `bin/worker-orcd-crates/api/.specs/00_api.md` |
| WORKER-4300-4323 | Input Validation | `bin/worker-orcd-crates/model-loader/.specs/00_model-loader.md` |
| WORKER-4400-4423 | CUDA FFI & Safety | `bin/worker-orcd/src/cuda_ffi/mod.rs` (impl) |
| WORKER-4500-4522 | Tensor-Parallel (NCCL) | `bin/worker-orcd/.specs/00_worker-orcd.md` (deferred) |
| WORKER-4600-4623 | Capability Matching | `bin/worker-orcd-crates/capability-matcher/.specs/00_capability-matcher.md` |
| WORKER-4700-4722 | Inference Kernels | `bin/worker-orcd/cuda/kernels/.specs/00_cuda-kernels.md` |
| WORKER-4800-4822 | Observability | `bin/worker-orcd/.specs/00_worker-orcd.md` |
| WORKER-4900-4922 | Security & Privileges | `bin/worker-orcd/.specs/00_worker-orcd.md` |
| WORKER-4950-4972 | Error Handling | `bin/worker-orcd/.specs/00_worker-orcd.md` |
| WORKER-4980-4991 | Configuration | `bin/worker-orcd/.specs/00_worker-orcd.md` |
| WORKER-4995-5000 | Testing | `bin/worker-orcd/.specs/00_worker-orcd.md` |
| SCHEDULER-M0-* | Scheduler (M0) | `bin/worker-orcd-crates/scheduler/.specs/00_scheduler.md` |
| KERNEL-*-* | CUDA Kernels | `bin/worker-orcd/cuda/kernels/.specs/00_cuda-kernels.md` |

---

## Component Specs

### 1. API Crate

**File**: `bin/worker-orcd-crates/api/.specs/00_api.md`

**Covers**:
- Endpoint authentication (WORKER-4200-4203)
- Plan endpoint (WORKER-4210-4214)
- Commit endpoint (WORKER-4220-4227)
- Ready endpoint (WORKER-4230-4233)
- Execute endpoint (WORKER-4240-4248)
- SSE streaming security (WORKER-4250-4253)

**Dependencies**: vram-residency, model-loader, capability-matcher, scheduler, input-validation, auth-min

---

### 2. VRAM Residency Crate

**File**: `bin/worker-orcd-crates/vram-residency/.specs/00_vram-residency.md`

**Covers**:
- VRAM-only policy (WORKER-4100-4103)
- ModelShardHandle contract (WORKER-4110-4113)
- Seal integrity (WORKER-4120-4122)

**Dependencies**: sha2, hmac, cuda_ffi

**Security**: TIER 1 (critical)

---

### 3. Model Loader Crate

**File**: `bin/worker-orcd-crates/model-loader/.specs/00_model-loader.md`

**Covers**:
- Model validation (WORKER-4310-4314)
- GGUF format validation
- Hash verification (WORKER-4320-4323)
- Path validation (WORKER-4340-4343)

**Dependencies**: sha2, ed25519-dalek (optional)

**Security**: TIER 1 (critical)

---

### 4. Capability Matcher Crate

**File**: `bin/worker-orcd-crates/capability-matcher/.specs/00_capability-matcher.md`

**Covers**:
- Model Capability Descriptor (WORKER-4600-4603)
- Engine Capability Profile (WORKER-4610-4613)
- Capability matching (WORKER-4620-4623)

**Dependencies**: serde, serde_json, thiserror, tracing

**Security**: TIER 2 (high-importance)

---

### 5. Scheduler Crate

**File**: `bin/worker-orcd-crates/scheduler/.specs/00_scheduler.md`

**Covers**:
- Single-slot scheduling (M0)
- Job state machine
- Post-M0 enhancements (deferred)

**Dependencies**: thiserror

**Security**: TIER 3 (medium-importance)

---

### 6. CUDA Kernels

**File**: `bin/worker-orcd/cuda/kernels/.specs/00_cuda-kernels.md`

**Covers**:
- M0 kernel set (WORKER-4700-4703)
- GEMM, RoPE, Attention, RMSNorm, Sampling
- Determinism (WORKER-4710-4713)
- Memory safety (KERNEL-SAFE-*)

**Language**: CUDA C++

**Build**: Compiled via `build.rs`, linked as static library

---

## Main Spec (Orchestration)

**File**: `bin/worker-orcd/.specs/00_worker-orcd.md`

**Covers**:
- Architecture & process model (WORKER-4010-4032)
- Observability & telemetry (WORKER-4800-4822)
- Security & privileges (WORKER-4900-4922)
- Error handling & recovery (WORKER-4950-4972)
- Configuration & deployment (WORKER-4980-4991)
- Testing & validation (WORKER-4995-5000)
- References to component specs

**Role**: High-level orchestration, cross-cutting concerns, integration requirements

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

## Benefits of Distribution

1. **Locality**: Requirements live with implementation
2. **Ownership**: Clear responsibility per crate
3. **Maintainability**: Easier to update component-specific requirements
4. **Testability**: Each crate can verify its own requirements
5. **Modularity**: Crates can be understood independently
6. **Scalability**: Adding new crates doesn't bloat main spec

---

**Status**: Spec distribution complete  
**Total specs**: 7 files (1 main + 6 component)  
**Total requirements**: 100+ normative requirements distributed
