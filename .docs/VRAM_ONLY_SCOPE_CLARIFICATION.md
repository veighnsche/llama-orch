# VRAM-ONLY Policy Scope Clarification

**Date**: 2025-10-05  
**Status**: Completed  
**Issue**: VRAM-ONLY clause in system spec contradicted Apple ARM worker introduction

---

## Problem Statement

The original system spec (`bin/.specs/00_llama-orch.md`) defined VRAM-ONLY as a **system-wide requirement** in section SYS-2.2.x:

> "The system MUST enforce VRAM-only policy: model weights, KV cache, activations, and intermediate tensors MUST reside entirely in GPU VRAM."

This created a contradiction when introducing Apple ARM workers (via worker adapters in `bin/pool-managerd/.specs/10_worker_adapters.md`), which use **unified memory architecture** where CPU and GPU share memory.

---

## Solution

The VRAM-ONLY clause has been **scoped to specific worker implementations** rather than being a system-wide requirement.

### Changes Made

#### 1. System Spec (`bin/.specs/00_llama-orch.md`)

**Section 0.1 Glossary** - Updated definition:
- **Before**: "VRAM-Only Policy: Requirement that all model weights... reside entirely in GPU VRAM with no RAM/disk fallback."
- **After**: "VRAM-Only Policy: Worker-specific requirement for bespoke NVIDIA workers (worker-orcd) that all model weights... reside entirely in GPU VRAM. Other worker types (e.g., Apple ARM workers) may use unified memory architecture as appropriate for their platform."

**Section 2.2** - Renamed and restructured:
- **Before**: "2.2 VRAM-Only Policy [M0+] (SYS-2.2.x)"
- **After**: "2.2 Memory Architecture Policy [M0+] (SYS-2.2.x)"

**New SYS-2.2.1** - Worker-Specific Memory Requirements:
```
Worker implementations MUST declare their memory architecture requirements. 
The orchestrator and pool manager MUST NOT enforce a specific memory architecture—this is worker-specific.


1. Bespoke NVIDIA Workers (worker-orcd): VRAM-ONLY policy
   - Model weights, KV cache, activations, and intermediate tensors MUST reside entirely in GPU VRAM
   - Prohibited: RAM fallback, Unified memory (CUDA UMA), Zero-copy mode, CPU inference fallback, Disk swapping

2. **Apple ARM Workers (worker-aarmd)**: UNIFIED MEMORY architecture (future, see `bin/pool-managerd/.specs/10_worker_adapters.md` and `bin/worker-aarmd/.specs/00_worker-aarmd.md`)
   - MUST use Apple Metal unified memory architecture
   - Model weights and activations reside in unified memory accessible by both CPU and GPU
   - Rationale: Leverages Apple Silicon architecture for efficient memory usage
   - Binary: `bin/worker-aarmd/` (Apple ARM daemon - note: two 'a's) specs

**Section 2.5.1** - FFI Boundaries updated:
- **Before**: "Worker (CUDA only): MUST use CUDA Runtime API for VRAM allocation..."
- **After**: "Worker (architecture-specific): Bespoke NVIDIA workers (worker-orcd): MUST use CUDA Runtime API... / Apple ARM workers (worker-aarmd): MUST use Metal API for unified memory allocation..."

#### 2. Worker-orcd Spec (`bin/.specs/01_M0_worker_orcd.md`)

**Title updated**:
- **Before**: "M0: Worker-orcd Complete Specification"
- **After**: "M0: Worker-orcd Complete Specification — Bespoke NVIDIA VRAM-ONLY Worker"

**Added critical notice**:
```
**CRITICAL: This worker implements VRAM-ONLY + NVIDIA CUDA ONLY policy. 
For other architectures (Apple ARM, AMD), see separate worker specs.**
```

**Section 0.1 Purpose** - Clarified worker type:
```
**Worker Type**: Bespoke NVIDIA CUDA Worker (VRAM-ONLY)

**IMPORTANT**: This worker is NVIDIA-specific. For Apple ARM workers (unified memory), 
see `bin/worker-aarmd/` (future). For other worker types, see 
`bin/pool-managerd/.specs/10_worker_adapters.md`.
```

**Section 2** - Renamed and scoped:
- **Before**: "2. VRAM-Only Policy"
- **After**: "2. VRAM-Only Policy (NVIDIA CUDA ONLY)"

**Added critical notice**:
```
**CRITICAL**: This section applies ONLY to worker-orcd (bespoke NVIDIA worker). 
Other worker types (Apple ARM, AMD) have different memory architectures defined in their respective specs.
```

#### 3. Worker Adapter Spec (`bin/pool-managerd/.specs/10_worker_adapters.md`)

**Section 3.3 Apple Metal Adapter** - Added memory architecture clarification:
```
**IMPORTANT**: Apple ARM workers use **UNIFIED MEMORY architecture**, NOT VRAM-only. 
This is a fundamental architectural difference from bespoke NVIDIA workers.

**Requirements**:
- [POOL-1034] Adapter MUST report memory architecture as `unified` (not `vram-only`)

**Configuration**:
adapters:
  - name: "apple-metal"
    binary_path: "/usr/local/bin/worker-aarmd"
    capabilities: ["text-gen"]
    memory_architecture: "unified"  # NOT vram-only

**Memory Architecture Note**: 
- Bespoke NVIDIA workers (`worker-orcd`): VRAM-ONLY policy (see `bin/.specs/01_M0_worker_orcd.md`)
- Apple ARM workers (`worker-aarmd`): UNIFIED MEMORY policy (see `bin/worker-aarmd/.specs/00_worker-aarmd.md` when created)

**Tagline updated**:
- **Before**: "Reproducible, VRAM-only, multi-node GPU orchestration for LLM inference"
- **After**: "Reproducible, multi-architecture, multi-node GPU orchestration for LLM inference"

**Core Value Propositions updated**:
- **Before**: "3. VRAM-Only Policy: Model fully resident in GPU VRAM (no RAM fallback)"

**Worker binaries clarified**:
```
3. Workers — Dumb Executors (load one model, execute inference)
   - `worker-orcd` — Bespoke NVIDIA CUDA worker (VRAM-only)
   - `worker-aarmd` — Apple ARM worker (unified memory) [future]
   - Extensible via worker adapter pattern
```

**Architecture benefits updated**:
- **Before**: "Workers have isolated CUDA contexts: Each worker owns its VRAM allocation"
- **After**: "Workers have isolated memory contexts: Each worker owns its memory allocation (VRAM for NVIDIA, unified for Apple)"

---

## Architecture Implications

### 1. Orchestrator and Pool Manager are Memory-Agnostic

The orchestrator and pool manager **DO NOT enforce** VRAM-ONLY or any specific memory architecture. They:
- Accept worker state reports that include memory architecture metadata
- Route jobs based on worker capabilities, not memory architecture
- Delegate memory management entirely to workers

### 2. Worker-Specific Memory Policies

Each worker type defines its own memory policy:

| Worker Type | Memory Architecture | Spec Location |
|-------------|---------------------|---------------|
| `worker-orcd` (Bespoke NVIDIA) | VRAM-ONLY | `bin/.specs/01_M0_worker_orcd.md` |
| `worker-aarmd` (Apple ARM) | Unified Memory | `bin/worker-aarmd/.specs/00_worker-aarmd.md` (future) |
| `llama.cpp` adapter | Depends on llama.cpp config | `bin/pool-managerd/.specs/10_worker_adapters.md` |
| Other workers | Defined in worker spec | Worker-specific specs |

### 3. Worker Adapter Pattern Enables Multi-Architecture

The worker adapter pattern in pool-managerd allows:
- Multiple worker types with different memory architectures in the same pool
- Orchestrator routes jobs based on capabilities, not memory architecture
- Pool manager normalizes worker state regardless of memory architecture

---

## Verification

### Spec Consistency Checks

- [x] System spec (SYS-2.2.x) no longer mandates VRAM-ONLY system-wide
- [x] Worker-orcd spec (M0-SYS-2.2.1) explicitly scopes VRAM-ONLY to NVIDIA workers
- [x] Worker adapter spec clarifies Apple ARM uses unified memory
- [x] README.md reflects multi-architecture support
- [x] Glossary updated to scope VRAM-ONLY to worker-orcd

### Future Worker Specs

When creating new worker types (e.g., `worker-aarmd` for Apple ARM):
1. Create worker-specific spec in `bin/worker-aarmd/.specs/00_worker-aarmd.md`
2. Define memory architecture policy (e.g., "UNIFIED MEMORY ONLY")
3. Reference worker adapter spec for integration with pool-managerd
4. Update system spec glossary if introducing new memory architecture terms

---

## Summary

**Problem**: VRAM-ONLY was a system-wide requirement that contradicted Apple ARM worker introduction.

**Solution**: Scoped VRAM-ONLY to bespoke NVIDIA workers (`worker-orcd`). System spec now defines worker-specific memory architecture policies.

**Result**: 
- ✅ No contradiction between specs
- ✅ Apple ARM workers can use unified memory
- ✅ Orchestrator and pool manager remain memory-architecture-agnostic
- ✅ Worker adapter pattern supports heterogeneous memory architectures
- ✅ Clear separation: worker-orcd = VRAM-ONLY (NVIDIA), worker-aarmd = UNIFIED (Apple)

---

**Files Modified**:
1. `/home/vince/Projects/llama-orch/bin/.specs/00_llama-orch.md`
2. `/home/vince/Projects/llama-orch/bin/.specs/01_M0_worker_orcd.md`
3. `/home/vince/Projects/llama-orch/bin/pool-managerd/.specs/10_worker_adapters.md`
4. `/home/vince/Projects/llama-orch/README.md`
5. `/home/vince/Projects/llama-orch/.docs/VRAM_ONLY_SCOPE_CLARIFICATION.md` (this document)
