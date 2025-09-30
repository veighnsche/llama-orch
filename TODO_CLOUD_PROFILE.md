# Cloud Profile Migration TODO

**Status**: Phase 1 Complete - Foundation Libraries  
**Date**: 2025-09-30  
**Target**: v1.0.0

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

## ⏳ Phase 5: Authentication

### Tasks

- [ ] Add Bearer token validation in orchestratord
- [ ] Add Bearer token validation in pool-managerd
- [ ] Configure `LLORCH_API_TOKEN` environment variable
- [ ] Update all inter-service HTTP calls to include token
- [ ] Document token generation and rotation
- [ ] Add security tests

---

## ⏳ Phase 6: Catalog Distribution

### Tasks

- [ ] Implement per-node model tracking
- [ ] Add `GET /v2/catalog/availability` endpoint
- [ ] Update placement to check model availability
- [ ] Fail with `MODEL_NOT_AVAILABLE` if model missing on node
- [ ] Document manual model staging procedure
- [ ] (Future v2.0) Add catalog sync protocol

---

## ⏳ Phase 7: Observability

### Tasks

- [ ] Add `node_id` label to all metrics
- [ ] Update Prometheus scrape config for multi-node
- [ ] Create Grafana dashboards for cluster view
- [ ] Add centralized logging (Loki/ELK)
- [ ] Add distributed tracing with correlation IDs
- [ ] Document monitoring setup

---

## ⏳ Phase 8: Testing & Validation

### Tasks

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

---

## ⏳ Phase 9: Documentation

### Tasks

- [ ] Update README.md with cloud profile instructions
- [ ] Create deployment guide (Kubernetes)
- [ ] Create deployment guide (Docker Compose)
- [ ] Create deployment guide (Bare Metal)
- [ ] Document configuration options
- [ ] Create troubleshooting guide
- [ ] Update architecture diagrams
- [ ] Create migration guide from HOME_PROFILE

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

## Timeline Estimate

- **Phase 1**: 2 weeks ✅ COMPLETE
- **Phase 2**: 2 weeks (orchestratord integration)
- **Phase 3**: 1 week (pool-managerd integration)
- **Phase 4**: 2 weeks (multi-node placement)
- **Phase 5**: 1 week (authentication)
- **Phase 6**: 1 week (catalog)
- **Phase 7**: 1 week (observability)
- **Phase 8**: 2 weeks (testing)
- **Phase 9**: 1 week (documentation)

**Total**: ~13 weeks to v1.0.0

---

## References

- `.specs/01_cloud_profile_.md` - Cloud profile specification
- `.docs/CLOUD_PROFILE_KNOWLEDGE.md` - Architecture deep dive
- `.docs/CLOUD_PROFILE_MIGRATION.md` - Migration plan
- `.docs/ARCHITECTURE_LIBRARY_ORGANIZATION.md` - Library organization
- `.docs/CLOUD_PROFILE_IMPLEMENTATION_SUMMARY.md` - What was implemented
