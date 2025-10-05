# Main Spec Update Summary (00_llama-orch.md)
**Date**: 2025-10-03  
**Action**: Updated main system spec with M0 hybrid scope decisions  
**Source**: M0_RESOLUTION_CONTRADICTIONS.md final decisions
---
## Changes Made to `00_llama-orch.md`
### 1. Section 14: Milestone Roadmap - Added M0 Scope Decision
**New subsection added after "Cross-Cutting Foundations"**:
```markdown
**M0 Scope Decision (2025-10-03)**:
- **Approach**: Performance Bundle Deferral (Hybrid)
- **Timeline**: M0 optimized to 4-5 weeks (from 6-8 weeks)
- **Deferred to M1**: Performance metrics, performance test suite, graceful shutdown, 
  client disconnect detection, reproducible kernels validation, sensitive data handling
- **Removed**:  (entire concept removed from repo)
- **Reference**: See `bin/.specs/M0_RESOLUTION_CONTRADICTIONS.md` for full analysis
```
### 2. Section 14.0: M-1 Foundation - Removed Proof Bundle References
**Removed**:
- `libs/` crate from deliverables
-  template from exit criteria
**Added**:
- Note: " removed from repo (M0 scope decision 2025-10-03)"
### 3. Section 14.1: M0 Worker Haiku Test - Complete Rewrite
#### Title Changed
- **From**: `### 14.1 M0: Worker Haiku Test (v0.1.0)`
- **To**: `### 14.1 M0: Worker Haiku Test (v0.1.0) - HYBRID SCOPE`
#### Goal Updated
- **From**: "Prove a single worker can load a model in VRAM and pass the haiku anti-cheat test"
- **To**: "Prove a single worker can load a model in VRAM and execute inference functionally"
#### Scope Updated
- **From**: "`worker-orcd` binary only, standalone operation"
- **To**: "`worker-orcd` binary only, standalone operation (performance validation deferred to M1)"
#### Added Scope Decision Section
```markdown
**Scope Decision**: Performance Bundle Deferred (Hybrid Approach)
- **Timeline**: 4-5 weeks (optimized from 6-8 weeks)
- **Focus**: Functional correctness + critical safety features
- **Deferred to M1**: Performance validation, metrics, graceful shutdown
```
#### Deliverables - Major Updates
**Added to deliverables**:
- Model load progress events (0%, 25%, 50%, 75%, 100%) ← **CRITICAL** (user feedback)
- VRAM residency verification (periodic checks) ← **CRITICAL** (runtime safety)
- VRAM OOM handling (graceful error, not crash) ← **CRITICAL** (safety)
- Narration-core logging (basic events only, NO performance metrics)
- Temperature scaling (0.0-2.0 range)
**Changed**:
- HTTP inference API: Added `POST /cancel`
- Logging: Changed from "Basic metrics emission" to "Narration-core logging (basic events only, NO performance metrics)"
**Added new section**: "DEFERRED to M1 (Performance Bundle)"
- Performance metrics emission (latency, throughput)
- Performance test suite
- Graceful shutdown endpoint (rely on SIGTERM)
- Client disconnect detection
- Reproducible kernels validation (implementation done, validation deferred)
- Sensitive data handling in logs
#### Testing - Updated
**Changed from**:
- Unit tests with  outputs
- E2E haiku test with  emission
**Changed to**:
- CUDA unit tests (functional only, NO performance tests)
- Rust unit tests
- E2E haiku test with basic test outputs (NO  - removed from repo)
- Functional validation (reproducibility implementation done, validation deferred to M1)
**Added new section**: "DEFERRED to M1 (Performance Testing)"
- Performance test suite (latency, throughput, memory leaks)
- Reproducible kernels validation
- Client disconnect detection tests
- Graceful shutdown tests
-  emission (removed from repo)
**Removed**: All performance audit HTML comments
#### Exit Criteria - Updated
**Added critical features**:
- Model load progress events emit (0%, 25%, 50%, 75%, 100%) ← **CRITICAL**
- VRAM residency verification operational (periodic checks) ← **CRITICAL**
- VRAM OOM handling works (graceful error, not crash) ← **CRITICAL**
- Worker shuts down on SIGTERM (graceful shutdown endpoint deferred to M1)
**Removed**:
- Metrics show `tokens_generated_total` > 0 and `quant_kind` label present
-  artifacts generated and validated
**Added new section**: "DEFERRED to M1 (Performance Exit Criteria)"
- First token latency p95 <100ms
- Per-token latency p95 <50ms
- Health endpoint p99 <10ms
- Model loading time <60s
- Graceful shutdown <5s
- Zero memory leaks validation
- Client disconnect abort <100ms
-  artifacts
#### Non-Goals - Updated
**Added**:
- Performance metrics/observability (deferred to M1)
- Performance test suite (deferred to M1)
- Graceful shutdown endpoint (deferred to M1)
- Client disconnect detection (deferred to M1)
-  (removed from repo)
#### Status Updated
- **From**: "In progress"
- **To**: "In progress (Hybrid Scope - 4-5 weeks)"
### 4. Section 14.2: M1 Pool Manager Lifecycle - Major Update
#### Title Changed
- **From**: `### 14.2 M1: Pool Manager Lifecycle (v0.2.0)`
- **To**: `### 14.2 M1: Pool Manager Lifecycle + M0 Performance Bundle (v0.2.0)`
#### Goal Updated
- **From**: "Pool manager can start/stop workers, hot-load models in RAM, and report pool state"
- **To**: "Pool manager lifecycle + complete M0 performance validation (deferred items from M0)"
#### Added M0 Deferred Items Section
**New section at the beginning of M1**:
```markdown
**M0 Deferred Items** (Performance Bundle - added to M1):
1. ✅ Performance metrics emission (worker_inference_duration_ms, worker_tokens_generated_total, latency metrics)
2. ✅ Performance test suite (first token latency, per-token latency, health endpoint, model loading time)
3. ✅ Graceful shutdown endpoint (POST /shutdown with 5s deadline)
4. ✅ Client disconnect detection (abort inference on SSE close)
5. ✅ Reproducible CUDA kernels validation (prove determinism works)
6. ✅ Sensitive data handling in logs (no raw prompts, only hashes)
7. ✅ Performance exit criteria validation (all targets from M0 spec)
**M1 Core Goal**: Pool manager can start/stop workers, hot-load models in RAM, and report pool state
```
---
## Impact Summary
### M0 Changes
- **Timeline**: Reduced from 6-8 weeks to 4-5 weeks (25-37% faster)
- **Scope**: Reduced from 22 items to 13 items (55% reduction)
- **Focus**: Functional correctness + critical safety (VRAM monitoring, OOM handling, progress events)
- **Deferred**: 14 items moved to M1 (performance bundle)
### M1 Changes
- **New responsibilities**: Complete M0 performance validation (7 deferred items)
- **Core goal**: Unchanged (pool manager lifecycle)
- **Timeline impact**: M1 will be slightly longer to accommodate M0 deferred items
### Proof Bundles
- **Status**: Removed from entire repo
- **M-1**: Removed from deliverables and exit criteria
- **M0**: Removed from testing and exit criteria
- **Note added**: " removed from repo (M0 scope decision 2025-10-03)"
### Milestone Numbering
- **No renumbering needed**: M0 deferred items added to M1 without shifting other milestones
- M2 (Orchestrator Scheduling) remains unchanged
- M3 (Security & Platform) remains unchanged
- M4+ remain unchanged
---
## Cross-References Updated
### Documents Updated
1. ✅ `00_llama-orch.md` (this file) - main system spec
2. ✅ `01_M0_worker_orcd.md` - M0 worker spec
3. ✅ `M0_RESOLUTION_CONTRADICTIONS.md` - contradiction resolution
4. ✅ `M0_PERFORMANCE_BUNDLE_ANALYSIS.md` - performance bundle analysis
5. ✅ `M0_DEFERRAL_CANDIDATES.md` - deferral candidates
### Reference Added
- All M0 sections now reference: `bin/.specs/M0_RESOLUTION_CONTRADICTIONS.md` for full analysis
---
## Next Actions Required
### 1. Update Related Specs
- [ ] Update worker-orcd component spec (if exists)
- [ ] Update CUDA module specs to reflect hybrid scope
- [ ] Update test documentation to remove  references
### 2. Code Changes
- [ ] Remove `libs/` crate
- [ ] Remove  references from all test code
- [ ] Remove LLORCH_RUN_ID and LLORCH_PROOF_DIR from environment handling
### 3. Documentation Updates
- [ ] Update README.md with M0 hybrid scope
- [ ] Update TODO.md with 4-5 week M0 timeline
- [ ] Update test-case-discovery-method.md to remove 
### 4. CI/CD Updates
- [ ] Remove  validation from CI
- [ ] Update test runners to not expect  outputs
---
## Summary
The main system spec (`00_llama-orch.md`) has been successfully updated to reflect the M0 hybrid scope decision:
1. **M0 optimized**: 4-5 weeks timeline (from 6-8 weeks)
2. **Performance bundle deferred**: 14 items moved to M1
3. **Critical safety retained**: VRAM monitoring, OOM handling, progress events
4. ** removed**: Entire concept removed from repo
5. **M1 expanded**: Now includes M0 performance validation + pool manager lifecycle
**Result**: Faster M0 delivery with critical features retained, comprehensive validation deferred to M1.
---
**Status**: Main spec updated with hybrid approach  
**Milestone numbering**: Unchanged (M0 deferrals added to M1)  
**Reference**: See M0_RESOLUTION_CONTRADICTIONS.md for full decision rationale
