# M0 Spec Update Summary
**Date**: 2025-10-03  
**Action**: Updated M0 worker spec with hybrid approach decisions  
**Source**: M0_RESOLUTION_CONTRADICTIONS.md (final decisions)
---
## Changes Made to `01_M0_worker_orcd.md`
### 1. Document Header Updated
- **Status**: Changed to "Draft (Hybrid Scope - Performance Bundle Deferred)"
- **Timeline**: Added "4-5 weeks (optimized from 6-8 weeks)"
### 2. New Section 0.0: Scope Decision Summary
Added comprehensive scope decision summary at the beginning:
**DEFERRED to M1+ (14 items)**:
- Prometheus metrics endpoint
- Performance metrics in logs
- Graceful shutdown endpoint
- All performance targets (latency, throughput, loading time)
- Performance test suite
- Client disconnect detection
- Reproducible CUDA kernels validation
- Sensitive data handling
**KEPT in M0 (13 items)**:
- All 3 models (Qwen2.5-0.5B, Phi-3-Mini, GPT-OSS-20B)
- All 3 quantization formats (Q4_K_M, MXFP4, Q4_0)
- 2 tokenizer backends (GGUF byte-BPE, tokenizer.json)
- Narration-core logging (basic events only)
- Model load progress events ← **CRITICAL** (user feedback)
- VRAM OOM handling ← **CRITICAL** (safety)
- VRAM residency verification ← **CRITICAL** (runtime leak detection)
- Reproducible CUDA kernels (implementation, validation deferred)
- Memory-mapped I/O
- Chunked VRAM transfer
- CUDA unit tests (functional only)
- Kernel safety validation
- Temperature scaling 0.0-2.0
**REMOVED**:
-  (entire concept)
### 3. Section 0.2 Scope - Out of Scope Updated
Added specific deferred items:
- Performance metrics/observability (deferred to M1)
- Performance test suite (deferred to M1)
- Graceful shutdown endpoint (deferred to M1)
- Client disconnect detection (deferred to M1)
- Reproducible kernels validation (implementation in M0, validation in M1)
-  (removed from repo)
### 4. Section 13: Observability & Logging - Complete Rewrite
**Changed from**: Structured Logging  
**Changed to**: Narration-Core Logging (Hybrid Scope)
#### M0-W-1900: Updated to Narration-Core
- Basic event tracking only
- No performance metrics fields
- Kept model load progress events (critical UX)
#### M0-W-1901: Marked as DEFERRED
- Performance metrics in logs deferred to M1
- All performance fields removed (vram_bytes, decode_time_ms, etc.)
#### M0-W-1902: Marked as DEFERRED
- Sensitive data handling deferred to M1
- M0 may log prompts for debugging
### 5. Section 14.3: Deferred to Post-M0 - Reorganized
Added three categories:
1. **Performance Bundle (Deferred to M1)** - 14 items
2. **Advanced Features (Deferred to M2+)** - existing items
3. **Removed from Repo** - 
### 6. Section 15.1: M0 Success Criteria - Updated
**Per-Model Acceptance Criteria**:
- Changed "Deterministic SSE" to "Functional SSE"
- Added "Progress: Model load progress events (0-100%)"
- Changed "Disconnects" to "Logs: Narration-core events"
**General M0 Success Criteria**:
- Added VRAM residency verification (critical safety)
- Added VRAM OOM handling (critical safety)
- Added model load progress events
- Updated reproducibility to "implementation done, validation deferred"
- Changed "gracefully on SIGTERM" to "on SIGTERM (graceful shutdown endpoint deferred)"
- Updated CUDA unit tests to "functional only, no performance tests"
- Updated integration tests to "functional validation"
### 7. Section 15.2: Non-Goals - Reorganized
Split into two categories:
1. **Deferred to M1 (Performance Bundle)** - 7 items
2. **Deferred to M2+** - existing items
### 8. Section 15.3: Performance Exit Criteria - Marked as DEFERRED
- **Status**: DEFERRED (Performance Bundle)
- All 7 performance targets marked as deferred
- Added "M0 Behavior: Functional validation only, no performance benchmarking"
- Added "M1 Plan: Comprehensive performance test suite with validation against targets"
### 9. Section 18.4: New References Section
Added scope decision documents:
- `M0_DEFERRAL_CANDIDATES.md` — Deferral analysis (28 candidates)
- `M0_RESOLUTION_CONTRADICTIONS.md` — Contradiction resolution (hybrid approach)
- `M0_PERFORMANCE_BUNDLE_ANALYSIS.md` — Performance bundle impact analysis
### 10. End of Document - Complete Rewrite
**Updated Status**:
- Status: Draft (Hybrid Scope) — Ready for implementation
- Scope: Performance Bundle Deferred (14 items to M1)
- Timeline: 4-5 weeks (optimized from 6-8 weeks)
**Updated Next Steps**:
1. Remove  references
2. Implement narration-core logging
3. Implement CUDA modules
4. Implement Rust HTTP layer with critical features:
   - Model load progress events
   - VRAM residency verification
   - VRAM OOM handling
5. Implement FFI boundary
6. Write functional tests only (no performance tests)
7. Execute haiku test (functional validation)
**Added Deferred to M1 List**:
- Performance test suite
- Performance metrics collection
- Reproducible kernels validation
- Graceful shutdown endpoint
- Client disconnect detection
- Sensitive data handling
**Added Key Trade-offs**:
- ✅ 2-3 weeks faster delivery
- ✅ Critical safety features retained
- ✅ User experience retained
- ❌ Performance validation deferred to M1
---
## Impact Summary
### Scope Reduction
- **Original M0**: 22 items kept, 5 deferred
- **Hybrid M0**: 13 items kept, 14 deferred
- **Reduction**: 55% fewer items (9 items removed from M0)
### Timeline Improvement
- **Original**: 6-8 weeks
- **Hybrid**: 4-5 weeks
- **Savings**: 2-3 weeks (25-37% faster)
### Critical Features Retained
1. ✅ All 3 models and quantization formats
2. ✅ VRAM monitoring (periodic residency checks)
3. ✅ VRAM OOM handling (graceful error)
4. ✅ Model load progress events (user feedback)
5. ✅ Narration-core logging (basic events)
### Performance Features Deferred
1. ❌ All performance targets and benchmarks
2. ❌ Performance test suite
3. ❌ Performance metrics collection
4. ❌ Reproducible kernels validation
5. ❌ Graceful shutdown endpoint
6. ❌ Client disconnect detection
---
## Next Actions Required
### 1. Proof Bundle Removal
- [ ] Delete or archive `libs/` crate
- [ ] Remove  requirements from test specs
- [ ] Remove LLORCH_RUN_ID and LLORCH_PROOF_DIR references
- [ ] Update test documentation
### 2. Implementation Priorities
1. Narration-core logging infrastructure
2. Model load progress events (0-100%)
3. VRAM residency verification (periodic checks)
4. VRAM OOM handling (graceful error)
5. Functional CUDA unit tests (no performance tests)
### 3. Documentation Updates
- [ ] Update worker README with hybrid scope
- [ ] Update test documentation to remove 
- [ ] Update implementation guides with narration-core logging
---
**Status**: M0 spec updated with hybrid approach  
**Outcome**: Faster delivery (4-5 weeks) with critical safety features retained  
**Reference**: See M0_RESOLUTION_CONTRADICTIONS.md for full decision rationale
