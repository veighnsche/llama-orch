# M0 Deferral Candidates Analysis

**Date**: 2025-10-03  
**Purpose**: Identify M0 requirements that could be deferred to later milestones to accelerate initial delivery  
**Source**: Analysis of `01_M0_worker_orcd.md` and `00_llama-orch.md`

---

## Executive Summary

This document identifies **28 deferral candidates** across 6 categories that could be moved from M0 to later milestones. Deferring these would reduce M0 scope by ~40% while maintaining core functionality: load model → execute inference → stream tokens.

**Recommended Minimal M0**: Single model (Qwen2.5-0.5B only), basic inference, no metrics, no graceful shutdown, no performance targets.

---

## Deferral Categories

### 🟢 High-Confidence Deferrals (Safe to Remove from M0)
Requirements that add complexity without blocking core M0 validation.

### 🟡 Medium-Confidence Deferrals (Consider Trade-offs)
Requirements that provide value but aren't strictly necessary for M0 success.

### 🔴 Low-Confidence Deferrals (High Risk)
Requirements that are deeply integrated; deferral may cause rework.

---

## 1. Test Models (3 → 1 Model)

### 🟢 **DEFER-M0-001**: Phi-3-Mini Model Support
- **Current**: M0 requires 3 models (Qwen2.5-0.5B, Phi-3-Mini, GPT-OSS-20B)
- **Proposal**: Defer Phi-3-Mini to M1
- **Rationale**: 
  - Qwen2.5-0.5B sufficient for smoke testing
  - Phi-3-Mini adds tokenizer variety but no new capabilities
  - Reduces test matrix complexity
- **Spec IDs**: M0-W-1230 (Model 2)
- **Impact**: Removes ~2.3GB model download and testing burden
- **Risk**: Low - same GGUF format, same Q4_K_M quantization

- **RESOLUTION**: DO NOT DEFER

### 🟢 **DEFER-M0-002**: GPT-OSS-20B Model Support
- **Current**: M0 requires GPT-OSS-20B (MXFP4, 12GB)
- **Proposal**: Defer to M2
- **Rationale**:
  - Requires tokenizer.json backend (separate implementation)
  - MXFP4 quantization adds kernel complexity
  - Large model (12GB) requires 24GB GPU
  - Trend-relevant but not essential for M0 validation
- **Spec IDs**: M0-W-1230 (Model 3)
- **Impact**: Removes MXFP4 kernel path, tokenizer.json parser, 12GB model
- **Risk**: Low - can validate quantization path with Q4_K_M first

**RESOLUTION**: DO NOT DEFER

---

## 2. Observability & Metrics

### 🟢 **DEFER-M0-003**: Prometheus Metrics Endpoint
- **Current**: M0-W-1350 - Worker SHOULD expose `/metrics` (optional)
- **Proposal**: Make explicitly deferred to M2+
- **Rationale**:
  - Already marked optional in spec
  - Basic health endpoint sufficient for M0
  - Full Prometheus exporter is M2+ scope
- **Spec IDs**: M0-W-1350, M0-W-1901
- **Impact**: Removes metrics collection, Prometheus dependency
- **Risk**: None - already optional

- **RESOLUTION**: DEFER

### 🟡 **DEFER-M0-004**: Performance Metrics in Logs
- **Current**: M0-W-1901 - Worker SHOULD include performance metrics in logs
- **Proposal**: Defer detailed performance logging to M1
- **Rationale**:
  - Basic event logging sufficient for M0 debugging
  - Performance metrics useful but not blocking
- **Spec IDs**: M0-W-1901
- **Impact**: Simpler log structure, faster implementation
- **Risk**: Low - can add metrics fields incrementally

- **RESOLUTION**: DEFER

### 🟡 **DEFER-M0-005**: Structured Logging Fields
- **Current**: M0-W-1900 - Extensive structured logging with multiple fields
- **Proposal**: Minimal logging for M0 (event type + error only)
- **Rationale**:
  - Full structured logging is observability feature
  - Simple println debugging sufficient for M0
- **Spec IDs**: M0-W-1900
- **Impact**: Faster implementation, simpler debugging
- **Risk**: Medium - may need to retrofit logging later

- **RESOLUTION**: DO NOT DEFER

### 🟢 **DEFER-M0-006**: Model Load Progress Events
- **Current**: M0-W-1621 - Worker SHOULD emit progress events during model loading
- **Proposal**: Defer to M1
- **Rationale**:
  - Qwen2.5-0.5B loads in <10s (fast enough without progress)
  - Progress events useful for large models (deferred anyway)
- **Spec IDs**: M0-W-1621
- **Impact**: Removes progress tracking logic
- **Risk**: None - nice-to-have feature

- **RESOLUTION**: DO NOT DEFER

---

## 3. Graceful Shutdown & Cleanup

### 🟡 **DEFER-M0-007**: Graceful Shutdown Endpoint
- **Current**: M0-W-1340 - Worker MAY expose `/shutdown` endpoint
- **Proposal**: Defer to M1 (rely on SIGTERM only)
- **Rationale**:
  - Already marked optional (MAY)
  - SIGTERM sufficient for M0 testing
  - Graceful shutdown is operational feature
- **Spec IDs**: M0-W-1340
- **Impact**: Removes shutdown endpoint, simpler API surface
- **Risk**: Low - SIGTERM works for testing

- **RESOLUTION**: DO NOT DEFER

### 🟡 **DEFER-M0-008**: Graceful Shutdown Performance Target
- **Current**: M0-W-1630 - Worker MUST complete shutdown within 5s
- **Proposal**: Defer performance target to M1 (allow any shutdown time for M0)
- **Rationale**:
  - Performance targets add testing complexity
  - M0 focus is functional correctness, not performance
- **Spec IDs**: M0-W-1630
- **Impact**: Removes shutdown performance testing
- **Risk**: Low - can measure later

- **RESOLUTION**: DEFER

---

## 4. Performance Requirements

### 🟢 **DEFER-M0-009**: First Token Latency Target
- **Current**: M0-W-1600 - First token <100ms (p95)
- **Proposal**: Defer all performance targets to M1
- **Rationale**:
  - M0 goal is functional correctness, not performance
  - Performance testing requires statistical analysis (p95, p99)
  - Can measure baseline in M0, enforce targets in M1
- **Spec IDs**: M0-W-1600
- **Impact**: Removes performance measurement infrastructure
- **Risk**: Low - functional tests sufficient for M0

- **RESOLUTION**: DEFER

### 🟢 **DEFER-M0-010**: Token Generation Rate Target
- **Current**: M0-W-1601 - 20-100 tok/s depending on model
- **Proposal**: Defer to M1
- **Spec IDs**: M0-W-1601
- **Impact**: No rate measurement needed
- **Risk**: None

- **RESOLUTION**: DEFER

### 🟢 **DEFER-M0-011**: Per-Token Latency Target
- **Current**: M0-W-1602 - Inter-token latency 10-50ms (p95)
- **Proposal**: Defer to M1
- **Spec IDs**: M0-W-1602
- **Impact**: No latency histograms needed
- **Risk**: None

- **RESOLUTION**: DEFER

### 🟢 **DEFER-M0-012**: Execute Endpoint Performance (<1ms parsing)
- **Current**: M0-W-1603 - Parse request in <1ms
- **Proposal**: Defer optimization to M1
- **Rationale**:
  - Premature optimization
  - Standard JSON parsing sufficient for M0
- **Spec IDs**: M0-W-1603
- **Impact**: Use standard serde_json (no zero-copy)
- **Risk**: None - can optimize later

- **RESOLUTION**: DEFER

### 🟢 **DEFER-M0-013**: Health Endpoint Performance (<10ms)
- **Current**: M0-W-1604 - Health responds in <10ms
- **Proposal**: Defer to M1
- **Spec IDs**: M0-W-1604
- **Impact**: No performance requirement on /health
- **Risk**: None

- **RESOLUTION**: DEFER

### 🟢 **DEFER-M0-014**: Cancellation Latency Target (<100ms)
- **Current**: M0-W-1610 - Stop inference within 100ms
- **Proposal**: Defer to M1 (functional cancel only for M0)
- **Spec IDs**: M0-W-1610
- **Impact**: Cancel works, but no latency guarantee
- **Risk**: Low - functional correctness sufficient

- **RESOLUTION**: DEFER

### 🟡 **DEFER-M0-015**: Client Disconnect Detection
- **Current**: M0-W-1611 - Detect disconnect and abort immediately
- **Proposal**: Defer to M1
- **Rationale**:
  - Adds complexity (connection monitoring every 10 tokens)
  - M0 can complete inference even if client disconnects
  - Optimization feature, not core functionality
- **Spec IDs**: M0-W-1611
- **Impact**: No disconnect detection, may waste GPU cycles
- **Risk**: Medium - nice-to-have for resource efficiency

- **RESOLUTION**: DEFER

### 🟢 **DEFER-M0-016**: Model Loading Time Target (<60s)
- **Current**: M0-W-1620 - Load within 60s
- **Proposal**: Defer to M1
- **Spec IDs**: M0-W-1620
- **Impact**: No startup performance requirement
- **Risk**: None

- **RESOLUTION**: DEFER

### 🟢 **DEFER-M0-017**: Performance Test Suite
- **Current**: M0-W-1830 - Comprehensive performance test suite with 7 tests
- **Proposal**: Defer entire suite to M1
- **Rationale**:
  - Large testing burden (histograms, p95/p99, proof bundles)
  - M0 focus is functional correctness
- **Spec IDs**: M0-W-1830
- **Impact**: Removes performance testing infrastructure
- **Risk**: Low - functional tests sufficient

- **RESOLUTION**: DEFER

---

## 5. Advanced Features

### 🟡 **DEFER-M0-018**: VRAM Residency Verification
- **Current**: M0-W-1012 - Periodic VRAM residency checks every 60s
- **Proposal**: Defer to M1
- **Rationale**:
  - VRAM-only enforced at startup (M0-W-1010)
  - Periodic checks are defensive monitoring
  - Adds complexity (background thread, cudaPointerGetAttributes)
- **Spec IDs**: M0-W-1012
- **Impact**: No runtime residency verification
- **Risk**: Medium - startup checks may be sufficient

- **RESOLUTION**: DEFER

### 🟡 **DEFER-M0-019**: VRAM OOM During Inference Handling
- **Current**: M0-W-1021 - Graceful OOM handling (emit error, free partial, stay alive)
- **Proposal**: Defer to M1 (crash on OOM for M0)
- **Rationale**:
  - Complex error recovery logic
  - M0 uses small model (352MB) - OOM unlikely
  - Can validate with preflight checks
- **Spec IDs**: M0-W-1021
- **Impact**: Worker crashes on OOM instead of recovering
- **Risk**: Medium - depends on KV cache sizing

- **RESOLUTION**: DO NOT DEFER

### 🟡 **DEFER-M0-020**: Reproducible CUDA Kernels
- **Current**: M0-W-1031 - Disable non-deterministic cuBLAS algorithms
- **Proposal**: Defer strict reproducibility to M1
- **Rationale**:
  - Adds cuBLAS configuration complexity
  - Determinism is testing feature, not M0 blocker
  - Can validate functional correctness without bit-exact reproducibility
- **Spec IDs**: M0-W-1031
- **Impact**: May not achieve perfect reproducibility
- **Risk**: Medium - affects testing strategy

- **RESOLUTION**: DO NOT DEFER

### 🔴 **DEFER-M0-021**: Temperature Scaling (Keep Greedy Only)
- **Current**: M0-W-1032 - Temperature 0.0-2.0 range
- **Proposal**: M0 = greedy (temp=0) only, defer temperature to M1
- **Rationale**:
  - Temperature adds sampling complexity
  - Greedy sufficient for deterministic testing
  - **CONFLICT**: Spec says temperature is product feature, not optional
- **Spec IDs**: M0-W-1032, M0-W-1310
- **Impact**: Simpler sampling logic
- **Risk**: HIGH - spec explicitly requires temperature as product feature

- **RESOLUTION**: DO NOT DEFER

### 🟡 **DEFER-M0-022**: Memory-Mapped I/O
- **Current**: M0-W-1221 - Use mmap() for efficient file reading
- **Proposal**: Defer to M1 (use standard file I/O)
- **Rationale**:
  - Optimization feature
  - Standard file read sufficient for 352MB model
- **Spec IDs**: M0-W-1221
- **Impact**: Slightly higher RAM usage during load
- **Risk**: Low - functional equivalent

- **RESOLUTION**: DO NOT DEFER

### 🟡 **DEFER-M0-023**: Chunked Transfer to VRAM
- **Current**: M0-W-1222 - Copy to VRAM in chunks (1MB)
- **Proposal**: Defer to M1 (single cudaMemcpy)
- **Rationale**:
  - Optimization for large models
  - 352MB model can be copied in one shot
- **Spec IDs**: M0-W-1222
- **Impact**: Simpler implementation
- **Risk**: Low

- **RESOLUTION**: DO NOT DEFER

---

## 6. Testing & Validation

### 🟢 **DEFER-M0-024**: CUDA Unit Tests
- **Current**: M0-W-1810 - Comprehensive CUDA unit tests with Google Test
- **Proposal**: Defer to M1 (rely on integration tests only)
- **Rationale**:
  - Unit tests add testing infrastructure overhead
  - Integration test validates end-to-end behavior
  - Can add unit tests incrementally
- **Spec IDs**: M0-W-1810
- **Impact**: No Google Test framework, no CUDA unit tests
- **Risk**: Medium - less granular testing

- **RESOLUTION**: DO NOT DEFER

### 🟡 **DEFER-M0-025**: Proof Bundle Requirements
- **Current**: M0-W-1840 - All tests emit proof bundles per standard
- **Proposal**: Defer to M1
- **Rationale**:
  - Proof bundles are reproducibility infrastructure
  - M0 can use simple test outputs
- **Spec IDs**: M0-W-1840
- **Impact**: No proof bundle emission
- **Risk**: Low - can add later

- **RESOLUTION**: PLEASE REMOVE THE CONCEPT OF PROOF BUNDLE OUT OF EXISTENCE OF THIS REPO!!

### 🟡 **DEFER-M0-026**: Sensitive Data Handling in Logs
- **Current**: M0-W-1902 - Strict rules for logging (no prompts, only hashes)
- **Proposal**: Defer to M1 (allow prompt logging in M0 for debugging)
- **Rationale**:
  - M0 is development/testing phase
  - Privacy features are production concern
- **Spec IDs**: M0-W-1902
- **Impact**: Can log prompts for debugging
- **Risk**: Low - M0 is not production

- **RESOLUTION**: DEFER

### 🟢 **DEFER-M0-027**: Kernel Safety Validation
- **Current**: M0-W-1431 - All kernels have bounds checking, dimension validation
- **Proposal**: Defer comprehensive validation to M1
- **Rationale**:
  - Safety checks add development time
  - Can validate with known-good inputs in M0
- **Spec IDs**: M0-W-1431
- **Impact**: Less defensive kernel code
- **Risk**: Medium - may encounter crashes with edge cases

- **RESOLUTION**: DO NOT DEFER

### 🟡 **DEFER-M0-028**: Multiple Quantization Format Support
- **Current**: M0-W-1201 - Support Q4_K_M, MXFP4, Q4_0
- **Proposal**: M0 = Q4_K_M only (Qwen2.5-0.5B format)
- **Rationale**:
  - MXFP4 deferred with GPT-OSS-20B
  - Q4_0 is fallback compatibility
  - Single format simplifies kernel implementation
- **Spec IDs**: M0-W-1201
- **Impact**: Only Q4_K_M kernels needed
- **Risk**: Low - can add formats incrementally

- **RESOLUTION**: DO NOT DEFER

---

## Recommended M0 Minimal Scope

### Core M0 (Must Have)
1. ✅ Single model: Qwen2.5-0.5B-Instruct (Q4_K_M)
2. ✅ CUDA context init + VRAM allocation
3. ✅ GGUF parsing + model load to VRAM
4. ✅ Basic inference (forward pass + greedy sampling)
5. ✅ HTTP server: POST /execute, GET /health
6. ✅ SSE streaming: started → token* → end
7. ✅ Basic error handling (crash on error, no recovery)
8. ✅ VRAM-only enforcement at startup
9. ✅ Haiku generation test (functional correctness)

### Deferred to M1
1. ❌ Phi-3-Mini, GPT-OSS-20B models
2. ❌ All performance targets and testing
3. ❌ Metrics endpoint
4. ❌ Graceful shutdown
5. ❌ Client disconnect detection
6. ❌ VRAM residency monitoring
7. ❌ Proof bundle emission
8. ❌ CUDA unit tests
9. ❌ Advanced error recovery
10. ❌ Structured logging

### Deferred to M2+
1. ❌ Temperature scaling (if high-risk deferral accepted)
2. ❌ MXFP4 quantization
3. ❌ Tokenizer.json backend
4. ❌ Memory-mapped I/O
5. ❌ Chunked VRAM transfer

---

## Impact Analysis

### Scope Reduction
- **Original M0**: ~100 requirements across 18 sections
- **Minimal M0**: ~60 requirements (40% reduction)
- **Time Savings**: Estimated 2-3 weeks faster delivery

### Risk Assessment
- **Low Risk Deferrals**: 17 items (safe to remove)
- **Medium Risk Deferrals**: 10 items (consider trade-offs)
- **High Risk Deferrals**: 1 item (temperature scaling - NOT recommended)

### Testing Impact
- **Original M0**: 3 models × 7 performance tests = 21 test scenarios
- **Minimal M0**: 1 model × 1 functional test = 1 test scenario
- **Test Reduction**: 95% fewer test cases

---

## Recommendations

### Phase 1: Minimal M0 (Recommended)
**Goal**: Prove CUDA pipeline works end-to-end

**Scope**:
- Single model (Qwen2.5-0.5B, Q4_K_M)
- Basic inference (greedy sampling only)
- Simple HTTP API (execute, health)
- SSE streaming (no metrics events)
- Functional haiku test
- No performance requirements
- No observability beyond basic logs

**Timeline**: 2-3 weeks

### Phase 2: M0.5 (Optional Hardening)
**Goal**: Add operational features

**Scope**:
- Add Phi-3-Mini model
- Add temperature scaling (0.0-2.0)
- Add /cancel endpoint
- Add basic metrics
- Add graceful shutdown

**Timeline**: 1-2 weeks

### Phase 3: M1 (Full Feature Set)
**Goal**: Production readiness

**Scope**:
- All deferred features
- Performance targets
- Full observability
- Pool manager integration

**Timeline**: 3-4 weeks

---

## Decision Matrix

| Requirement | Defer? | Confidence | Impact | Risk | Recommendation |
|-------------|--------|------------|--------|------|----------------|
| Phi-3-Mini model | ✅ | High | Low | Low | **Defer to M1** |
| GPT-OSS-20B model | ✅ | High | Medium | Low | **Defer to M2** |
| Prometheus metrics | ✅ | High | Low | None | **Defer to M2** |
| Performance targets | ✅ | High | Medium | Low | **Defer to M1** |
| Graceful shutdown | ✅ | Medium | Low | Low | **Defer to M1** |
| Client disconnect | ✅ | Medium | Low | Medium | **Defer to M1** |
| VRAM monitoring | ✅ | Medium | Low | Medium | **Defer to M1** |
| Temperature scaling | ❌ | Low | High | High | **KEEP in M0** |
| CUDA unit tests | ✅ | High | Medium | Medium | **Defer to M1** |
| Proof bundles | ✅ | Medium | Low | Low | **Defer to M1** |

---

## Next Steps

1. **Review with team**: Validate deferral candidates
2. **Update M0 spec**: Remove deferred requirements, mark as M1/M2
3. **Create M0.5 spec**: Optional hardening phase
4. **Update TODO.md**: Reflect new M0 scope
5. **Communicate changes**: Update stakeholders on revised timeline

---

**Status**: Draft - Awaiting Review  
**Author**: Cascade (AI Assistant)  
**Review Required**: Technical Lead, Product Owner
