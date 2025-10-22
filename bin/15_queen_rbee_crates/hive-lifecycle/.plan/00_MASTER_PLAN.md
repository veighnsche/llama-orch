# TEAM-208: Hive Lifecycle Migration Master Plan

**Created by:** TEAM-208  
**Date:** 2025-10-22  
**Status:** PLANNING

---

## Mission

Migrate all hive lifecycle logic from `job_router.rs` (1,115 LOC) to the dedicated `hive-lifecycle` crate to reduce complexity and improve maintainability.

---

## Current State Analysis

### Problem
`job_router.rs` has grown to **1,114 lines** (TEAM-209: verified actual count) and contains:
- Job routing logic (core responsibility)
- **Hive lifecycle operations** (should be in dedicated crate)
- SSH testing
- Health checking
- Capabilities fetching/caching
- Binary path resolution
- Process spawning/stopping

### Inspiration
Following the pattern from:
- `rbee-keeper/src/queen_lifecycle.rs` (306 LOC) - Clean lifecycle management
- `daemon-lifecycle` crate - Shared daemon spawning utilities

### Target Architecture
```
job_router.rs (THIN)
    ↓
hive-lifecycle crate (THICK)
    ↓
Shared utilities:
    - daemon-lifecycle
    - narration-core
    - timeout-enforcer
```

---

## Operations to Migrate

From `job_router.rs` lines 252-1011:

1. **SshTest** (lines 253-279) - ✅ Already in crate
2. **HiveInstall** (lines 280-401) - 121 LOC
3. **HiveUninstall** (lines 402-484) - 82 LOC
4. **HiveStart** (lines 485-717) - 232 LOC
5. **HiveStop** (lines 718-820) - 102 LOC
6. **HiveList** (lines 821-863) - 42 LOC
7. **HiveGet** (lines 864-877) - 13 LOC
8. **HiveStatus** (lines 878-921) - 43 LOC
9. **HiveRefreshCapabilities** (lines 922-1011) - 89 LOC

**Total:** ~724 LOC to migrate

### Supporting Functions
- `validate_hive_exists()` (lines 98-160) - 62 LOC
- Uses `hive_client::check_hive_health()` and `fetch_hive_capabilities()`

---

## Migration Strategy

### Phase 1: Foundation (TEAM-210)
**Goal:** Set up crate structure and request/response types

**Deliverables:**
- Request/Response types for all 9 operations
- Module structure (install.rs, start.rs, stop.rs, etc.)
- Update Cargo.toml dependencies
- Validation helpers

**LOC:** ~150 lines

---

### Phase 2: Simple Operations (TEAM-211)
**Goal:** Migrate read-only and simple operations

**Operations:**
- HiveList (42 LOC)
- HiveGet (13 LOC)
- HiveStatus (43 LOC)

**Deliverables:**
- `list.rs` - List all hives from config
- `get.rs` - Get single hive details
- `status.rs` - Check hive health

**LOC:** ~100 lines

---

### Phase 3: Lifecycle Core (TEAM-212)
**Goal:** Migrate start/stop operations (most complex)

**Operations:**
- HiveStart (232 LOC) - Binary resolution, spawning, health polling, capabilities
- HiveStop (102 LOC) - Graceful shutdown with SIGTERM/SIGKILL

**Deliverables:**
- `start.rs` - Complete hive startup with capabilities caching
- `stop.rs` - Graceful shutdown with fallback to force-kill

**LOC:** ~350 lines

---

### Phase 4: Install/Uninstall (TEAM-213)
**Goal:** Migrate installation operations

**Operations:**
- HiveInstall (121 LOC) - Binary path resolution, localhost vs remote
- HiveUninstall (82 LOC) - Cleanup with cache removal

**Deliverables:**
- `install.rs` - Hive installation logic
- `uninstall.rs` - Hive uninstallation with cleanup

**LOC:** ~220 lines

---

### Phase 5: Capabilities (TEAM-214)
**Goal:** Migrate capabilities refresh operation

**Operations:**
- HiveRefreshCapabilities (89 LOC)

**Deliverables:**
- `capabilities.rs` - Fetch and cache device capabilities
- Integration with existing `hive_client` module

**LOC:** ~100 lines

---

### Phase 6: Integration (TEAM-215)
**Goal:** Wire up job_router.rs to use new crate

**Deliverables:**
- Update `job_router.rs` to call hive-lifecycle functions
- Remove old implementation (~724 LOC deleted)
- Update imports and error handling
- Ensure job_id propagation for SSE routing

**LOC:** ~50 lines added, ~724 lines removed

---

### Phase 7: Verification (TEAM-209 - PEER REVIEW)
**Goal:** Critical peer review of entire migration

**Deliverables:**
- Code review checklist
- Test coverage verification
- SSE routing verification (job_id propagation)
- Error handling audit
- Performance comparison
- Documentation review

**Acceptance Criteria:**
- All operations work identically
- SSE narration flows correctly
- No regressions in error messages
- job_router.rs reduced to <400 LOC

---

## Success Metrics

### Before (TEAM-209: ACTUAL STATE)
- `job_router.rs`: **1,114 LOC** (verified: wc -l)
- Hive logic mixed with routing logic
- Hard to test hive operations in isolation
- **Current hive-lifecycle implementation:** Only SSH test (155 LOC)
- **Status:** NO migration done yet (Phase 1-6 not started)

### After (TARGET STATE)
- `job_router.rs`: ~350 LOC (routing only)
- `hive-lifecycle`: ~900 LOC (all hive operations)
- Clean separation of concerns
- Testable hive operations

**LOC Reduction in job_router.rs:** ~65% (1,114 → 350)

---

## Critical Requirements (Engineering Rules)

1. ✅ **Add TEAM-XXX signatures** to all new/modified code
2. ✅ **No TODO markers** - Implement fully or ask for help
3. ✅ **Complete previous team's TODO** before moving on
4. ✅ **Max 2 pages per handoff** with code examples
5. ✅ **Show actual progress** (function count, LOC migrated)
6. ✅ **No background testing** - All tests run in foreground
7. ✅ **Update existing docs** - Don't create multiple .md files

---

## Dependencies

### Existing Crates (Already Used)
- `daemon-lifecycle` - Process spawning
- `narration-core` - Observability
- `timeout-enforcer` - Timeout with countdown
- `rbee-config` - Configuration management
- `anyhow` - Error handling
- `tokio` - Async runtime
- `reqwest` - HTTP client

### Internal Modules (Need to Extract)
- `hive_client::check_hive_health()` - May need to move to hive-lifecycle
- `hive_client::fetch_hive_capabilities()` - May need to move to hive-lifecycle

### CRITICAL DEPENDENCY (TEAM-209 FINDING)
**⚠️  MISSING FROM ORIGINAL PLAN:**
- `rbee-hive-device-detection` (in `bin/25_rbee_hive_crates/device-detection/`)
- **Used by rbee-hive** to detect GPUs/CPUs
- **Required for capabilities flow**: queen → hive `/capabilities` → device-detection → JSON response
- **Impact on Phase 3 & 5**: Must understand this architectural flow
- See: `bin/20_rbee_hive/src/main.rs:156` - `rbee_hive_device_detection::detect_gpus()`

---

## Risk Mitigation

### Risk 1: SSE Routing Breaks
**Mitigation:** Ensure all narration includes `.job_id(&job_id)` (see MEMORY about SSE routing)

### Risk 2: Error Messages Change
**Mitigation:** Copy exact error messages, preserve user-facing text

### Risk 3: Capabilities Caching Breaks
**Mitigation:** Test cache hit/miss paths thoroughly

### Risk 4: Process Management Issues
**Mitigation:** Follow daemon-lifecycle patterns exactly

---

## Team Assignments

- **TEAM-209:** Peer Review (runs after all other teams)
- **TEAM-210:** Foundation (Phase 1)
- **TEAM-211:** Simple Operations (Phase 2)
- **TEAM-212:** Lifecycle Core (Phase 3)
- **TEAM-213:** Install/Uninstall (Phase 4)
- **TEAM-214:** Capabilities (Phase 5)
- **TEAM-215:** Integration (Phase 6)

---

## Next Steps

1. TEAM-210 reads `01_PHASE_1_FOUNDATION.md` and starts implementation
2. Each team completes their phase fully before handoff
3. TEAM-209 performs final peer review after TEAM-215 completes
4. All teams read `START_HERE.md` for workflow instructions

---

## References

- Engineering Rules: `/home/vince/Projects/llama-orch/.windsurf/rules/engineering-rules.md`
- Inspiration: `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/src/queen_lifecycle.rs`
- Shared Crate: `/home/vince/Projects/llama-orch/bin/99_shared_crates/daemon-lifecycle`
- Source File: `/home/vince/Projects/llama-orch/bin/10_queen_rbee/src/job_router.rs`
