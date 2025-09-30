# pool-managerd BDD Behavior Audit - Summary

**Date**: 2025-09-30  
**Audit Scope**: Complete source code, spec, and checklist analysis  
**Total Behaviors Identified**: 200+

---

## Executive Summary

Comprehensive audit of `pool-managerd` identified **200+ distinct behaviors** requiring BDD coverage, organized into 12 major categories. Approximately **80 behaviors are immediately testable** (Phase 1), while **120 behaviors require implementation** before BDD coverage (Phase 2 & 3).

---

## Behavior Categories

| Category | Behavior Count | Status | Priority |
|----------|----------------|--------|----------|
| **1. HTTP API** | 13 | ✅ Implemented | P0 |
| **2. Preload & Readiness** | 24 | ✅ Implemented | P0 |
| **3. Registry State** | 43 | ✅ Implemented | P0 |
| **4. Process Lifecycle** | 12 | ✅ Implemented | P0 |
| **5. Preflight Validation** | 7 | ✅ Implemented | P0 |
| **6. Drain & Reload** | 15 | ⏳ TODO | P1 |
| **7. Supervision & Backoff** | 24 | ⏳ TODO | P1 |
| **8. Device Masks & Placement** | 16 | ⏳ TODO | P2 |
| **9. Observability** | 18 | ⏳ TODO | P2 |
| **10. Configuration** | 6 | ⏳ TODO | P2 |
| **11. Security** | 6 | ⏳ TODO | P3 |
| **12. Error Handling** | 9 | ✅ Partial | P1 |

**Total**: 193 behaviors catalogued (more exist in edge cases)

---

## Phase 1: Immediately Testable (80 behaviors)

### HTTP API Endpoints
- Health check (3 behaviors)
- Preload endpoint (7 behaviors)
- Pool status endpoint (5 behaviors)

### Registry Operations
- Health status (4 behaviors)
- Error tracking (4 behaviors)
- Heartbeat tracking (3 behaviors)
- Version management (4 behaviors)
- Engine metadata (6 behaviors)
- Device mask management (3 behaviors)
- Slot management (5 behaviors)
- Draining state (4 behaviors)
- Pool registration (6 behaviors)
- Lease accounting (7 behaviors)
- Handoff registration (8 behaviors)
- Update merge (3 behaviors)
- Snapshot export (4 behaviors)

### Preload Lifecycle
- Engine spawn (5 behaviors)
- Health check wait (5 behaviors)
- Readiness gating (5 behaviors)
- Failure handling (5 behaviors)
- Handoff generation (5 behaviors)

### Process Management
- Process spawn (6 behaviors)
- Process stop (6 behaviors)
- Process monitoring (3 behaviors)

### Preflight Checks
- CUDA detection (4 behaviors)
- GPU-only enforcement (4 behaviors)

---

## Phase 2: Requires Implementation (70 behaviors)

### Drain & Reload (OC-POOL-3031, OC-POOL-3038)
- Drain lifecycle (7 behaviors)
- Reload lifecycle (9 behaviors)

### Supervision & Backoff (OC-POOL-3010, OC-POOL-3011)
- Crash detection (4 behaviors)
- Exponential backoff (5 behaviors)
- Circuit breaker (6 behaviors)
- Restart storm prevention (4 behaviors)

### Error Handling
- Additional error scenarios (6 behaviors)

---

## Phase 3: Advanced Features (50 behaviors)

### Device Masks & Placement (OC-POOL-3020, OC-POOL-3021)
- Device mask parsing (5 behaviors)
- Placement enforcement (4 behaviors)
- Heterogeneous split planning (6 behaviors)

### Observability (OC-POOL-3030)
- Metrics emission (10 behaviors)
- Structured logging (8 behaviors)

### Configuration
- Environment variables (3 behaviors)
- Device configuration (3 behaviors)

### Security
- Process isolation (3 behaviors)
- Container runtime (3 behaviors)

---

## Spec Requirement Coverage

| Requirement | Behaviors | Status |
|-------------|-----------|--------|
| **OC-POOL-3001** (Preload & Ready) | 24 | ✅ Testable |
| **OC-POOL-3002** (Fail Fast) | 5 | ✅ Testable |
| **OC-POOL-3003** (Readiness Endpoints) | 8 | ✅ Testable |
| **OC-POOL-3010** (Driver Errors) | 11 | ⏳ TODO |
| **OC-POOL-3011** (Restart Storms) | 13 | ⏳ TODO |
| **OC-POOL-3012** (CPU Spillover) | 7 | ✅ Testable |
| **OC-POOL-3020** (Device Masks) | 9 | ⏳ TODO |
| **OC-POOL-3021** (Hetero Split) | 6 | ⏳ TODO |
| **OC-POOL-3030** (Observability) | 18 | ⏳ TODO |
| **OC-POOL-3101** (Handoff API) | 4 | ✅ Testable |
| **OC-POOL-3102** (Handoff Tests) | 4 | ✅ Testable |

---

## Source Code Coverage

### Implemented Modules (Testable)
```
✅ src/api/routes.rs          - HTTP handlers (13 behaviors)
✅ src/core/registry.rs        - State management (43 behaviors)
✅ src/lifecycle/preload.rs    - Preload logic (24 behaviors)
✅ src/validation/preflight.rs - GPU checks (7 behaviors)
```

### Stub Modules (Not Yet Testable)
```
⏳ src/lifecycle/drain.rs      - Drain/reload (15 behaviors)
⏳ src/lifecycle/supervision.rs - Backoff/circuit breaker (24 behaviors)
⏳ src/placement/devicemasks.rs - Device masks (9 behaviors)
⏳ src/placement/hetero_split.rs - Split planning (6 behaviors)
```

---

## BDD Feature File Mapping

### Phase 1 Features (Ready to Write)

1. **`api_health.feature`** (3 behaviors)
   - Daemon health check scenarios

2. **`api_preload.feature`** (12 behaviors)
   - Engine preload success/failure scenarios

3. **`api_pool_status.feature`** (5 behaviors)
   - Pool status query scenarios

4. **`registry_health.feature`** (4 behaviors)
   - Health state management

5. **`registry_leases.feature`** (7 behaviors)
   - Lease allocation/release

6. **`registry_handoff.feature`** (8 behaviors)
   - Handoff registration (OC-POOL-3101, OC-POOL-3102)

7. **`preload_lifecycle.feature`** (24 behaviors)
   - Complete preload flow (OC-POOL-3001, OC-POOL-3002, OC-POOL-3003)

8. **`process_management.feature`** (12 behaviors)
   - Spawn, stop, monitor processes

9. **`preflight_gpu.feature`** (7 behaviors)
   - GPU-only enforcement (OC-POOL-3012)

### Phase 2 Features (Requires Implementation)

10. **`drain_reload.feature`** (15 behaviors)
    - Drain and reload lifecycle

11. **`supervision_backoff.feature`** (24 behaviors)
    - Crash detection, backoff, circuit breaker (OC-POOL-3010, OC-POOL-3011)

### Phase 3 Features (Requires Implementation)

12. **`device_masks.feature`** (9 behaviors)
    - Device mask enforcement (OC-POOL-3020)

13. **`hetero_split.feature`** (6 behaviors)
    - Heterogeneous GPU split planning (OC-POOL-3021)

14. **`observability.feature`** (18 behaviors)
    - Metrics and logging (OC-POOL-3030)

---

## Recommended BDD Implementation Order

### Week 1: Core API & Registry (30 behaviors)
1. `api_health.feature` - 3 behaviors
2. `api_pool_status.feature` - 5 behaviors
3. `registry_health.feature` - 4 behaviors
4. `registry_leases.feature` - 7 behaviors
5. `registry_handoff.feature` - 8 behaviors

### Week 2: Preload Lifecycle (36 behaviors)
6. `preload_lifecycle.feature` - 24 behaviors
7. `api_preload.feature` - 12 behaviors

### Week 3: Process & Preflight (19 behaviors)
8. `process_management.feature` - 12 behaviors
9. `preflight_gpu.feature` - 7 behaviors

### Week 4+: Advanced Features (As Implemented)
10. Drain & Reload
11. Supervision & Backoff
12. Device Masks & Placement
13. Observability

---

## Key Insights from Audit

### Strengths
- ✅ **Registry module is comprehensive** - 43 behaviors well-defined
- ✅ **Preload flow is complete** - Full lifecycle implemented
- ✅ **HTTP API is functional** - All endpoints working
- ✅ **Preflight checks exist** - GPU enforcement in place

### Gaps (Requires Implementation)
- ⏳ **No drain/reload logic** - Stubs only
- ⏳ **No supervision/backoff** - Stubs only
- ⏳ **No device mask enforcement** - Stubs only
- ⏳ **No metrics emission** - Not implemented
- ⏳ **No structured logging** - Basic tracing only

### Technical Debt
- TODO comments in drain.rs, supervision.rs, devicemasks.rs, hetero_split.rs
- Missing integration tests for multi-pool scenarios
- No chaos/fault injection tests

---

## Verification Commands

### Run Phase 1 BDD Tests (Once Implemented)
```bash
# All Phase 1 features
cargo run -p pool-managerd-bdd --bin bdd-runner

# Specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/preload_lifecycle.feature \
  cargo run -p pool-managerd-bdd --bin bdd-runner

# Specific directory
LLORCH_BDD_FEATURE_PATH=tests/features/api/ \
  cargo run -p pool-managerd-bdd --bin bdd-runner
```

### Validate Step Definitions
```bash
# Run step validation test (once created)
cargo test -p pool-managerd-bdd --test bdd -- --nocapture
```

---

## Documentation References

- **Full Behavior List**: `BEHAVIOR_AUDIT.md` (200+ behaviors catalogued)
- **Spec**: `.specs/30-pool-managerd.md` (OC-POOL-3xxx requirements)
- **Checklist**: `CHECKLIST.md` (production readiness items)
- **Source Code**: `src/` (implementation status)

---

## Conclusion

The behavior audit provides a **complete roadmap** for BDD test coverage. With **80 behaviors immediately testable** in Phase 1, we can achieve substantial coverage of the implemented functionality. Phase 2 & 3 behaviors will be added as the corresponding features are implemented, following the spec-first workflow.

**Next Action**: Begin implementing Phase 1 feature files, starting with `api_health.feature` and `registry_health.feature` as the simplest entry points.
