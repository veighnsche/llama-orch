# Dead Code Analysis: Cloud Profile Migration

**Date**: 2025-10-01  
**Status**: COMPLETE  
**Context**: Phase 9 Documentation - Cloud Profile v0.2.0

---

## Executive Summary

Analysis of pre-cloud-profile code identified **1 file** that is deprecated for CLOUD_PROFILE but must be retained for HOME_PROFILE backward compatibility.

**Finding**: The migration developers DID properly clean up dead code. Only the intentionally-preserved HOME_PROFILE handoff watcher remains, which is properly feature-gated.

---

## Analysis Methodology

1. Read all cloud profile migration documentation (Phases 1-8)
2. Identify components that existed pre-migration
3. Trace which components were replaced by new CLOUD_PROFILE features
4. Search codebase for filesystem dependencies and HOME_PROFILE-only patterns
5. Verify feature gating and deprecation markers

---

## Findings

### 1. orchestratord Handoff Watcher (REMOVED) ✅

**File**: ~~`bin/orchestratord/src/services/handoff.rs`~~ **DELETED**

**Lines**: 243 lines
- Module documentation: Lines 1-20
- Implementation: Lines 21-147
- Unit tests: Lines 149-242 (3 tests)

**Purpose** (HOME_PROFILE):
- Watches local filesystem for engine handoff files (`.runtime/engines/*.json`)
- Auto-binds adapters when engines become ready
- Updates pool registry from handoff metadata
- Provides idempotent processing

**Why Deprecated for CLOUD_PROFILE**:
1. **Filesystem Coupling**: Requires orchestratord and engine-provisioner share filesystem
2. **Cannot Scale**: Only works on single machine
3. **Replaced By**: HTTP polling architecture where:
   - pool-managerd owns handoff watcher (same node as engine-provisioner)
   - orchestratord polls pool-managerd via HTTP (`GET /v2/pools/{id}/status`)
   - Nodes report readiness via heartbeats (`POST /v2/nodes/{id}/heartbeat`)

**Current Status**:

✅ **Properly Documented**:
```rust
//! ⚠️ HOME_PROFILE ONLY - CLOUD_PROFILE LIMITATION ⚠️
//!
//! This implementation assumes orchestratord and engine-provisioner share a filesystem.
//! This ONLY works for HOME_PROFILE (single machine deployment).
//!
//! For CLOUD_PROFILE (distributed deployment), the handoff watcher MUST be owned by
//! pool-managerd (which runs on the same machine as engine-provisioner). orchestratord
//! will poll pool-managerd via HTTP instead of watching the filesystem directly.
```

✅ **Properly Feature-Gated** (`bin/orchestratord/src/app/bootstrap.rs:29-31`):
```rust
if !state.cloud_profile_enabled() {
    crate::services::handoff::spawn_handoff_autobind_watcher(state.clone());
}
```

✅ **Has Unit Tests**: 3 tests verifying HOME_PROFILE behavior

**Action Taken**: 
- ✅ **REMOVED** — File deleted along with integration tests
- ✅ Module reference removed from `bin/orchestratord/src/services/mod.rs`
- ✅ Spawn call removed from `bin/orchestratord/src/app/bootstrap.rs`
- ✅ BDD step updated to no-op in `bin/orchestratord/bdd/src/steps/background.rs`
- ✅ Documentation updated to reflect cloud-only architecture

**Files Deleted**:
- `bin/orchestratord/src/services/handoff.rs` (243 lines)
- `bin/orchestratord/tests/handoff_autobind_integration.rs` (200+ lines)

**Migration Complete**: Project is now cloud-profile only (v0.2.0+)

---

## Code That Is NOT Dead

### New CLOUD_PROFILE Features (Keep All)

All code added in Phases 1-8 is active and required:

#### Phase 1-4: Core Infrastructure
- `libs/shared/pool-registry-types/` - Type definitions for node communication
- `libs/control-plane/service-registry/` - Node tracking and health management
- `libs/gpu-node/handoff-watcher/` - Handoff watcher for pool-managerd (NEW location)
- `libs/gpu-node/node-registration/` - Node registration with control plane

#### Phase 5: Authentication (Active)
- `bin/orchestratord/src/app/auth_min.rs` - Bearer token middleware
- `bin/pool-managerd/src/api/auth.rs` - Bearer token validation
- `libs/auth-min/` - Timing-safe comparison utilities

**Status**: Required for all CLOUD_PROFILE deployments ✅

#### Phase 6: Observability (Active)
- `bin/orchestratord/src/metrics.rs` - Cloud-specific metrics
- `ci/dashboards/cloud_profile_overview.json` - Grafana dashboard
- `ci/alerts/cloud_profile.yml` - Prometheus alerting rules
- `docs/runbooks/CLOUD_PROFILE_INCIDENTS.md` - Incident runbook

**Status**: Required for production observability ✅

#### Phase 7: Catalog Distribution (Active)
- `bin/orchestratord/src/api/catalog_availability.rs` - Multi-node catalog endpoint
- `bin/orchestratord/src/services/placement_v2.rs` - Model-aware placement
- `docs/MANUAL_MODEL_STAGING.md` - Operator guide

**Status**: Required for multi-node deployments ✅

#### Phase 8: Testing (Active)
- `bin/orchestratord/tests/cloud_profile_integration.rs` - Integration tests
- `bin/orchestratord/tests/placement_v2_tests.rs` - Placement unit tests

**Status**: Required for CI/CD pipeline ✅

### Profile-Agnostic Code (Keep All)

Code that works for both HOME_PROFILE and CLOUD_PROFILE:
- `bin/orchestratord/src/api/data.rs` - Task admission (works for both)
- `bin/orchestratord/src/api/control.rs` - Pool control endpoints
- `bin/orchestratord/src/api/catalog.rs` - Catalog CRUD
- `libs/orchestrator-core/` - Queue logic (profile-agnostic)
- `libs/adapter-host/` - Adapter management
- `libs/worker-adapters/` - All adapter implementations
- `libs/catalog-core/` - Catalog storage

**Status**: Core functionality, required for both profiles ✅

---

## Filesystem Dependency Audit

Searched for filesystem access patterns that would break in CLOUD_PROFILE:

### ✅ Safe (Local-Only Filesystem)
- `pool-managerd` handoff watcher - local filesystem on same node as engine-provisioner
- `engine-provisioner` handoff writes - local filesystem
- `catalog-core` model storage - local filesystem per node (by design)
- Config file loading - local filesystem (startup only)

### ❌ Unsafe (Cross-Service Filesystem)
- orchestratord handoff watcher - **PROPERLY GATED** for HOME_PROFILE only

**Conclusion**: All filesystem access is either:
1. Local-only (safe for CLOUD_PROFILE)
2. Properly gated for HOME_PROFILE

---

## Migration Quality Assessment

### Code Cleanup: ✅ EXCELLENT

The migration developers:
1. ✅ Moved handoff watcher to pool-managerd (correct location)
2. ✅ Properly feature-gated old orchestratord watcher
3. ✅ Added clear deprecation warnings in comments
4. ✅ Documented architecture limitations
5. ✅ Maintained backward compatibility with HOME_PROFILE
6. ✅ Added comprehensive tests for new code
7. ✅ No dangling dead code found

### Documentation Quality: ✅ EXCELLENT

1. ✅ Module docs clearly state HOME_PROFILE limitation
2. ✅ TODOs reference migration tasks
3. ✅ Spec documents explain architecture change
4. ✅ Migration plan documents rationale

### Feature Gating: ✅ CORRECT

```rust
// Correct pattern used throughout:
if state.cloud_profile_enabled() {
    // New CLOUD_PROFILE code
} else {
    // Old HOME_PROFILE code
}
```

---

## Recommendations

### Immediate (v0.2.0)

1. ✅ **Keep handoff.rs** - Required for HOME_PROFILE backward compatibility
2. ⏸️ **Add deprecation attributes** - Mark functions as deprecated (optional)
3. ✅ **Document in README** - Already documented in Phase 9
4. ✅ **No other cleanup needed** - Migration was clean

### Future (v1.0.0)

If HOME_PROFILE is dropped in v1.0.0:
1. Delete `bin/orchestratord/src/services/handoff.rs`
2. Remove feature gate from `bootstrap.rs`
3. Remove HOME_PROFILE config options
4. Update documentation

---

## Conclusion

**Cloud Profile migration is COMPLETE.** 

The cloud profile migration (Phases 1-9) cleanup:
- ✅ Old orchestratord handoff watcher **REMOVED**
- ✅ All CLOUD_PROFILE features are active and tested
- ✅ No dead code remaining
- ✅ Project is now cloud-first (v0.2.0+)

**HOME_PROFILE deprecated**: For single-machine deployments, run orchestratord and pool-managerd on the same node with CLOUD_PROFILE mode.

---

## Appendix: Search Commands Used

```bash
# Search for filesystem access
rg "std::fs::|PathBuf|File::" --type rust bin/orchestratord/src/

# Search for HOME_PROFILE/CLOUD_PROFILE markers
rg "CLOUD_PROFILE|HOME_PROFILE|ORCHD_PROFILE" --type rust bin/orchestratord/src/

# Search for handoff-related code
rg "handoff|filesystem" --type rust bin/orchestratord/src/

# Search for deprecated markers
rg "#\[deprecated\]|TODO\[CLOUD" --type rust bin/orchestratord/src/
