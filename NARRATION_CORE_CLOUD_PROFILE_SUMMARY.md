# Narration-Core + Cloud Profile: Critical Update

**Date**: 2025-09-30 22:46  
**Status**: URGENT  
**Impact**: BLOCKS v0.2.0 Release

---

## TL;DR

The **CLOUD_PROFILE migration** (v0.2.0) makes **narration-core completion MANDATORY**, not optional. Without it, distributed deployments cannot be debugged.

**What Changed**:
- ✅ Provenance added to specs (emitted_by, trace_id, etc.)
- ✅ Implementation started (474 lines of code, 16 tests passing)
- ⚠️ **NEW**: Cloud Profile requires OpenTelemetry integration, HTTP header propagation, and cross-service correlation
- ❌ **BLOCKING**: These features are not yet implemented

---

## Why This Matters

### HOME_PROFILE (v0.1.x) - Single Machine
```
All services on one machine → shared filesystem → logs in one place
```
**Debugging**: `grep "error" logs/*.log` → find issue

**Narration-core**: Nice to have, but not critical.

### CLOUD_PROFILE (v0.2.0+) - Distributed
```
Control Plane (orchestratord) on machine 1
    ↓ HTTP
GPU Worker 1 (pool-managerd + engine-provisioner) on machine 2
    ↓ HTTP
GPU Worker 2 (pool-managerd + engine-provisioner) on machine 3
```
**Debugging**: Logs scattered across 3+ machines, no correlation

**Narration-core**: **CRITICAL**. Without it, you cannot:
- Trace requests across machines
- Correlate logs in Loki/Elasticsearch
- Link logs to OpenTelemetry traces
- Debug multi-service flows

---

## What Was Updated

### 1. Specs Updated ✅

**File**: `libs/observability/narration-core/.specs/00_narration_core.md`

**Added**:
- Provenance section (emitted_by, trace_id, span_id, source_location)
- Cloud Profile Requirements section (OTEL integration, HTTP headers, cross-service correlation)

**File**: `libs/observability/narration-core/.specs/CLOUD_PROFILE_NARRATION_REQUIREMENTS.md` (NEW)

**Content**:
- Detailed requirements for distributed deployments
- OpenTelemetry integration specs
- HTTP header propagation patterns
- Log aggregation compatibility
- Cross-service correlation examples
- Implementation priority and timeline

### 2. Implementation Started ✅

**Files Created**:
- `src/lib.rs` (151 lines) - Core API with full field taxonomy including provenance
- `src/redaction.rs` (138 lines) - Secret masking
- `src/capture.rs` (185 lines) - Test adapter
- `tests/integration.rs` (partial) - Integration tests

**Test Results**: 16/16 passing (12 unit + 4 doc)

**Provenance Fields Added**:
```rust
pub struct NarrationFields {
    // ... existing fields ...
    
    // Provenance (audit trail and debugging)
    pub emitted_by: Option<String>,      // "orchestratord@0.2.0"
    pub emitted_at_ms: Option<u64>,      // Unix timestamp
    pub trace_id: Option<String>,        // OpenTelemetry trace ID
    pub span_id: Option<String>,         // OpenTelemetry span ID
    pub source_location: Option<String>, // "data.rs:155" (dev only)
}
```

### 3. Missing for Cloud Profile ❌

**Not Yet Implemented**:
1. **OpenTelemetry Integration**
   - `narrate_with_otel_context()` - auto-extract trace/span IDs
   - W3C Trace Context propagation
   - Parent span ID tracking

2. **Auto-Injection Helpers**
   - `narrate_auto()` - auto-inject service identity + timestamp
   - Macro for ergonomics: `narrate_auto!(...)`

3. **HTTP Header Helpers**
   - `extract_correlation_from_headers()` - for axum handlers
   - `inject_correlation_into_headers()` - for reqwest clients
   - Middleware for automatic propagation

4. **Cross-Service Adoption**
   - orchestratord: Replace tracing with narration
   - pool-managerd: Replace println with narration
   - engine-provisioner: Replace println with narration

5. **BDD Cross-Service Tests**
   - Test correlation ID propagation
   - Test trace ID propagation
   - Test service identity in logs

---

## Impact on v0.2.0 Timeline

### Original Cloud Profile Timeline (from CLOUD_PROFILE_MIGRATION_PLAN.md)

**Total**: 5-6 weeks

- Week 1: Preparation & Documentation ✅ DONE
- Week 2: pool-managerd Watcher
- Week 3: orchestratord Polling
- Week 4: Testing & Validation
- Week 5-6: Production Rollout

### Narration-Core Timeline (from NARRATION_CORE_URGENT_MEMO.md)

**Total**: 3 weeks (overlaps with Cloud Profile)

- Week 1: Core Implementation (redaction, capture, field taxonomy) ✅ MOSTLY DONE
- Week 2: Cross-Crate Adoption (orchestratord, pool-managerd, provisioners)
- Week 3: Testing & Enforcement (BDD coverage, story snapshots, CI)

### **NEW**: Cloud Profile Narration Features

**Total**: +2 weeks (NEW WORK)

- Week 1: OpenTelemetry integration, HTTP header helpers
- Week 2: Cross-service adoption with correlation, BDD tests

### **REVISED TIMELINE**

**v0.2.0 Release**: 7-8 weeks (was 5-6 weeks)

**Critical Path**:
1. Week 1-2: Narration-core OTEL + HTTP features ← **BLOCKING**
2. Week 3-4: Cross-service adoption with correlation ← **BLOCKING**
3. Week 5: pool-managerd watcher + orchestratord polling
4. Week 6: Testing & validation with distributed narration
5. Week 7-8: Production rollout

---

## Risk Assessment

### HIGH RISK: Narration-Core Not Ready

**Probability**: HIGH (95% unimplemented as of 2025-09-30)

**Impact**: CRITICAL (blocks v0.2.0 or ships with poor observability)

**Consequences**:
- ❌ Cannot debug distributed deployments
- ❌ Cannot trace requests across services
- ❌ Cannot meet Cloud Profile observability requirements (`.specs/01_cloud_profile.md` line 67-75)
- ❌ Production incidents take 10x longer to resolve
- ❌ Users lose trust in platform

**Mitigation**:
1. **Assign dedicated owner** for narration-core (TODAY)
2. **Prioritize OTEL + HTTP features** (Week 1-2)
3. **Make cross-service adoption mandatory** (not optional)
4. **Add BDD tests for cross-service correlation** (Week 3-4)
5. **Block v0.2.0 release** until narration coverage ≥80%

---

## Action Items

### Immediate (This Week)

1. **Assign Owner** ✅
   - Who: TBD
   - What: Complete narration-core for Cloud Profile
   - When: Start immediately

2. **Prioritize OTEL Integration** ⏸️
   - Implement `narrate_with_otel_context()`
   - Add `parent_span_id` field
   - Unit tests

3. **Implement HTTP Header Helpers** ⏸️
   - `extract_correlation_from_headers()`
   - `inject_correlation_into_headers()`
   - axum + reqwest integration

### Short-Term (Next 2 Weeks)

4. **Cross-Service Adoption**
   - orchestratord: Replace tracing with narration
   - pool-managerd: Replace println with narration
   - engine-provisioner: Replace println with narration

5. **BDD Cross-Service Tests**
   - Test correlation ID propagation
   - Test trace ID propagation
   - Test service identity

6. **Documentation**
   - Update README with Cloud Profile examples
   - Add HTTP header propagation guide
   - Add OTEL integration guide

### Medium-Term (Weeks 3-4)

7. **E2E Distributed Tests**
   - Deploy to 2-machine test environment
   - Verify log aggregation (Loki)
   - Verify distributed tracing (Tempo)

8. **Performance Testing**
   - Measure narration overhead
   - Ensure <1ms per call
   - Load test with 1000 tasks/sec

9. **CI Enforcement**
   - Add narration coverage gate
   - Fail if coverage <80%
   - Add correlation ID validation

---

## Success Criteria

### Week 2 (OTEL + HTTP Features)
- ✅ `narrate_with_otel_context()` implemented and tested
- ✅ HTTP header helpers implemented and tested
- ✅ `narrate_auto()` implemented and tested
- ✅ Documentation updated

### Week 4 (Cross-Service Adoption)
- ✅ orchestratord uses narration (no more raw tracing)
- ✅ pool-managerd uses narration (no more println)
- ✅ engine-provisioner uses narration (no more println)
- ✅ Correlation IDs propagate across all HTTP calls
- ✅ Trace IDs link to OpenTelemetry spans

### Week 6 (Testing & Validation)
- ✅ BDD narration coverage ≥80%
- ✅ E2E tests pass in distributed environment
- ✅ Logs aggregate correctly in Loki
- ✅ Traces appear correctly in Tempo
- ✅ Performance: <1ms per narration call

### Week 8 (Production Ready)
- ✅ v0.2.0 released with Cloud Profile support
- ✅ Narration-core fully adopted across all services
- ✅ Debugging distributed deployments is fast and easy
- ✅ No more "I can't trace this request" complaints

---

## Conclusion

**Narration-core is now CRITICAL for v0.2.0**, not optional. The Cloud Profile migration introduces distributed deployments where proper observability is mandatory.

**What's Done**:
- ✅ Specs updated with Cloud Profile requirements
- ✅ Provenance fields added to implementation
- ✅ Core API implemented (474 lines, 16 tests passing)

**What's Needed**:
- ❌ OpenTelemetry integration
- ❌ HTTP header propagation
- ❌ Cross-service adoption
- ❌ BDD cross-service tests

**Timeline Impact**: +2 weeks to v0.2.0 release (now 7-8 weeks total)

**Action Required**: Assign owner and prioritize narration-core completion immediately.

---

**Files Updated**:
- `libs/observability/narration-core/.specs/00_narration_core.md` - Added Cloud Profile section
- `libs/observability/narration-core/.specs/CLOUD_PROFILE_NARRATION_REQUIREMENTS.md` - NEW detailed requirements
- `NARRATION_CORE_CLOUD_PROFILE_SUMMARY.md` - This file

**Next Steps**:
1. Review and approve updated specs
2. Assign owner for narration-core completion
3. Begin OTEL + HTTP feature implementation
4. Update v0.2.0 timeline (add 2 weeks)
