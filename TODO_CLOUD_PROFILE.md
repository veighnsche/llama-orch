# Cloud Profile Migration TODO

**Status**: 🔴 **PAUSED AT PHASE 5 - SECURITY GATE**  
**Date**: 2025-09-30  
**Pause Reason**: Critical security vulnerabilities in authentication  
**Resume ETA**: ~1 week after P0 fixes  
**Target**: v1.0.0

---

## ⚠️ MIGRATION PAUSED - SECURITY ALERT

**DO NOT PROCEED TO PHASE 6** until Phase 5 security vulnerabilities are fixed.

**Critical Issues**:
- 🔴 Timing attack in orchestratord node registration (CWE-208)
- 🔴 Zero authentication on pool-managerd endpoints
- ❌ 65% of Phase 5 authentication work missing

**See**: `.docs/CLOUD_MIGRATION_PAUSED.md` for full details

**Fix Checklist**: `.docs/PHASE5_FIX_CHECKLIST.md`

---

## ✅ Phase 1: Foundation Libraries (COMPLETE)

### Created Libraries

- [x] `libs/shared/pool-registry-types/` - Common types for service communication
- [x] `libs/control-plane/service-registry/` - Track GPU nodes
- [x] `libs/gpu-node/handoff-watcher/` - Watch for engine readiness (moved from orchestratord)
- [x] `libs/gpu-node/node-registration/` - Register with control plane
- [x] Updated `Cargo.toml` workspace members
- [x] Created architecture documentation

---

## ⏳ Phase 2: Integrate into orchestratord (NEXT)

### Tasks

- [ ] Add `/v2/nodes/register` endpoint in `bin/orchestratord/src/api/nodes.rs`
- [ ] Add `/v2/nodes/{id}/heartbeat` endpoint
- [ ] Add `/v2/nodes/{id}` DELETE endpoint (deregister)
- [ ] Add `/v2/nodes` GET endpoint (list nodes)
- [ ] Embed `ServiceRegistry` in `AppState`
- [ ] Spawn stale checker task on startup
- [ ] Update placement logic to query service registry
- [ ] Remove old handoff watcher (or feature-gate for HOME_PROFILE)
- [ ] Add config: `ORCHESTRATORD_CLOUD_PROFILE=false` (default)
- [ ] Add config: `ORCHESTRATORD_NODE_TIMEOUT_SECS=30`
- [ ] Add config: `ORCHESTRATORD_HEARTBEAT_INTERVAL_SECS=10`
- [ ] Update OpenAPI contracts for new endpoints
- [ ] Write integration tests

### Files to Modify

```
bin/orchestratord/
├── src/
│   ├── api/
│   │   └── nodes.rs                    # NEW: Node management endpoints
│   ├── app/
│   │   ├── bootstrap.rs                # MODIFY: Spawn service registry + stale checker
│   │   └── state.rs                    # MODIFY: Add service_registry field
│   ├── services/
│   │   ├── handoff.rs                  # MODIFY: Feature-gate or remove
│   │   └── placement.rs                # MODIFY: Query service registry
│   └── config.rs                       # MODIFY: Add cloud profile flags
└── Cargo.toml                          # MODIFY: Add service-registry dependency
```

---

## ⏳ Phase 3: Integrate into pool-managerd

### Tasks

- [ ] Add `HandoffWatcher` in `bin/pool-managerd/src/main.rs`
- [ ] Add `NodeRegistration` lifecycle (register on startup, deregister on shutdown)
- [ ] Implement callback: handoff detected → update registry
- [ ] Spawn heartbeat task with pool status
- [ ] Add HTTP server (if not exists) for health queries
- [ ] Add config: `POOL_MANAGERD_NODE_ID=gpu-node-1`
- [ ] Add config: `POOL_MANAGERD_MACHINE_ID=machine-alpha`
- [ ] Add config: `ORCHESTRATORD_URL=http://control-plane:8080`
- [ ] Add config: `ORCHESTRATORD_REGISTER_ON_STARTUP=true`
- [ ] Add config: `POOL_MANAGERD_RUNTIME_DIR=.runtime/engines`
- [ ] Add config: `POOL_MANAGERD_HEARTBEAT_INTERVAL_SECS=10`
- [ ] Write integration tests

### Files to Modify

```
bin/pool-managerd/
├── src/
│   ├── main.rs                         # MODIFY: Add handoff watcher + registration
│   ├── config.rs                       # MODIFY: Add cloud profile config
│   └── registry.rs                     # MODIFY: Callback for handoff events
└── Cargo.toml                          # MODIFY: Add handoff-watcher, node-registration deps
```

---

## ⏳ Phase 4: Multi-Node Placement

### Tasks

- [ ] Create `libs/control-plane/multi-node-placement/`
- [ ] Extend placement to consider all online nodes
- [ ] Filter by model availability (track per-node)
- [ ] Filter by VRAM capacity
- [ ] Implement least-loaded selection across nodes
- [ ] Handle node failures (re-route tasks)
- [ ] Add affinity/anti-affinity rules
- [ ] Update metrics (add `node_id` label)
- [ ] Write chaos tests (node failures)

---

## 🔴 Phase 5: Authentication (BLOCKED - SECURITY VULNERABILITIES)

**Status**: ⚠️ **INCOMPLETE - 35% with critical flaws**  
**Blocking**: Phases 6-9 cannot proceed  
**Priority**: **P0 - FIX IMMEDIATELY**

### Critical Security Issues

- 🔴 **CRITICAL**: Timing attack in `orchestratord/api/nodes.rs` (CWE-208, CVSS 7.5)
- 🔴 **CRITICAL**: No authentication on pool-managerd endpoints (CVSS 9.1)
- ❌ Ignored existing `auth-min` security library
- ❌ No security tests, no token fingerprinting, no audit logging

### P0 Tasks (Must Fix Before Continuing Migration)

- [ ] **FIX**: Replace manual validation with `auth_min::timing_safe_eq()` in orchestratord/api/nodes.rs (2h)
- [ ] **ADD**: Timing attack regression test (2h)
- [ ] **IMPLEMENT**: pool-managerd authentication middleware (3h)
- [ ] **TEST**: pool-managerd auth integration tests (1h)
- [ ] **ADD**: Bearer tokens to orchestratord → pool-managerd client (1h)
- [ ] **TEST**: E2E with authentication enabled (3h)
- [ ] **REVIEW**: Security team code review (4h)
- [ ] **DEPLOY**: Monitor auth in production (2h)

**Total P0 Effort**: 18 hours (2-3 days)

### P1 Tasks (Complete Coverage)

- [ ] Add auth to data plane endpoints (`/v2/tasks/*`)
- [ ] Add auth to control plane endpoints (`/control/*`)
- [ ] Add auth to catalog/artifacts endpoints
- [ ] Comprehensive auth-min test suite
- [ ] BDD auth scenarios per `.specs/11_min_auth_hooks.md`
- [ ] Security documentation (token generation, deployment checklist)

**Total P1 Effort**: 22 hours (3 days)

### Security Gate Criteria

Phase 5 complete ONLY when:
- [ ] All token comparisons use `auth_min::timing_safe_eq()`
- [ ] All Bearer parsing uses `auth_min::parse_bearer()`
- [ ] All auth logs use `auth_min::token_fp6()` for identity
- [ ] pool-managerd has Bearer token validation on all endpoints
- [ ] Timing attack tests pass (< 10% variance)
- [ ] Token leakage tests pass (no raw tokens in logs)
- [ ] E2E tests pass with auth enabled
- [ ] Security team sign-off received

### References

- `.docs/CLOUD_MIGRATION_PAUSED.md` - Why we paused
- `.docs/PHASE5_FIX_CHECKLIST.md` - Step-by-step fixes
- `.specs/12_auth-min-hardening.md` - Security spec
- `.docs/SECURITY_AUDIT_AUTH_MIN.md` - Full audit
- `.docs/PHASE5_SECURITY_FINDINGS.md` - Detailed findings

---

## ⏸️ Phase 6: Catalog Distribution (BLOCKED - Waiting for Phase 5)

**Status**: 🔴 **BLOCKED** - Cannot start until Phase 5 security gate passed  
**Depends On**: Phase 5 authentication complete

### Tasks (DO NOT START)

- [ ] Implement per-node model tracking
- [ ] Add `GET /v2/catalog/availability` endpoint
- [ ] Update placement to check model availability
- [ ] Fail with `MODEL_NOT_AVAILABLE` if model missing on node
- [ ] Document manual model staging procedure
- [ ] (Future v2.0) Add catalog sync protocol

**Blocked Reason**: Catalog distribution requires secure pool-managerd communication

---

## ⏸️ Phase 7: Observability (BLOCKED - Waiting for Phase 5)

**Status**: 🔴 **BLOCKED** - Cannot start until Phase 5 security gate passed

### Tasks (DO NOT START)

- [ ] Add `node_id` label to all metrics
- [ ] Update Prometheus scrape config for multi-node
- [ ] Create Grafana dashboards for cluster view
- [ ] Add centralized logging (Loki/ELK)
- [ ] Add distributed tracing with correlation IDs
- [ ] Document monitoring setup

**Blocked Reason**: Need secure foundations before observability

---

## ⏸️ Phase 8: Testing & Validation (BLOCKED - Waiting for Phase 5)

**Status**: 🔴 **BLOCKED** - Cannot start until Phase 5 security gate passed

### Tasks (DO NOT START)

- [ ] Create `test-harness/e2e-cloud/` for multi-node tests
- [ ] Test: 2-node cluster (1 control + 1 GPU)
- [ ] Test: 3-node cluster (1 control + 2 GPU)
- [ ] Test: Node registration/deregistration
- [ ] Test: Heartbeat timeout (mark offline)
- [ ] Test: Placement across nodes
- [ ] Test: Node failure handling
- [ ] Test: Network partition scenarios
- [ ] Test: Load balancing across nodes
- [ ] Update BDD features for cloud profile

**Blocked Reason**: Would test insecure implementation

---

## ⏸️ Phase 9: Documentation (BLOCKED - Waiting for Phase 5)

**Status**: 🔴 **BLOCKED** - Cannot start until Phase 5 security gate passed

### Tasks (DO NOT START)

- [ ] Update README.md with cloud profile instructions
- [ ] Create deployment guide (Kubernetes)
- [ ] Create deployment guide (Docker Compose)
- [ ] Create deployment guide (Bare Metal)
- [ ] Document configuration options
- [ ] Create troubleshooting guide
- [ ] Update architecture diagrams
- [ ] Create migration guide from HOME_PROFILE

**Blocked Reason**: Would document vulnerable setup

---

## Feature Flags & Backward Compatibility

### HOME_PROFILE (Default)
```bash
ORCHESTRATORD_CLOUD_PROFILE=false
ORCHESTRATORD_ADDR=127.0.0.1:8080
# Single machine, embedded registry, localhost
```

### CLOUD_PROFILE (Opt-in)
```bash
ORCHESTRATORD_CLOUD_PROFILE=true
ORCHESTRATORD_ADDR=0.0.0.0:8080
ORCHESTRATORD_NODE_TIMEOUT_SECS=30
LLORCH_API_TOKEN=<secret>
# Multi-machine, service discovery, network
```

---

## Verification Commands

### Check New Crates Compile
```bash
cargo check -p pool-registry-types
cargo check -p service-registry
cargo check -p handoff-watcher
cargo check -p node-registration
```

### Run Unit Tests
```bash
cargo test -p pool-registry-types
cargo test -p service-registry
cargo test -p handoff-watcher
cargo test -p node-registration
```

### Full Dev Loop
```bash
cargo xtask dev:loop
```

---

## Success Criteria

- [ ] orchestratord and pool-managerd run on separate machines
- [ ] Multiple pool-managerd instances register automatically
- [ ] Placement considers all available nodes
- [ ] Node failures detected within 30 seconds
- [ ] Authentication prevents unauthorized access
- [ ] All metrics visible in centralized dashboard
- [ ] E2E tests pass in multi-node setup
- [ ] HOME_PROFILE continues to work (backward compatible)

---

## Timeline Estimate (UPDATED)

- **Phase 1**: 2 weeks ✅ COMPLETE
- **Phase 2**: 2 weeks ✅ COMPLETE (orchestratord integration)
- **Phase 3**: 1 week ✅ COMPLETE (pool-managerd integration)
- **Phase 4**: 2 weeks ✅ COMPLETE (multi-node placement)
- **Phase 5**: 2 weeks 🔴 IN PROGRESS (was incorrectly marked complete)
  - **P0 Security Fixes**: 1 week (18 hours)
  - **P1 Complete Coverage**: 1 week (22 hours)
- **Phase 6**: 1 week ⏸️ BLOCKED (catalog)
- **Phase 7**: 1 week ⏸️ BLOCKED (observability)
- **Phase 8**: 2 weeks ⏸️ BLOCKED (testing)
- **Phase 9**: 1 week ⏸️ BLOCKED (documentation)

**Original Estimate**: ~13 weeks to v1.0.0  
**Revised Estimate**: ~14 weeks (added 1 week for proper Phase 5)  
**Current Progress**: ~4.5 weeks complete (Phase 1-4 done, Phase 5 35% with flaws)

---

## References

- `.specs/01_cloud_profile_.md` - Cloud profile specification
- `.docs/CLOUD_PROFILE_KNOWLEDGE.md` - Architecture deep dive
- `.docs/CLOUD_PROFILE_MIGRATION.md` - Migration plan
- `.docs/ARCHITECTURE_LIBRARY_ORGANIZATION.md` - Library organization
- `.docs/CLOUD_PROFILE_IMPLEMENTATION_SUMMARY.md` - What was implemented
