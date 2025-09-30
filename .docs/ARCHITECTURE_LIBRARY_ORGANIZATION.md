# Library Organization for Cloud Profile

**Date**: 2025-09-30  
**Status**: IMPLEMENTATION  
**Version**: 1.0

---

## Overview

This document defines the library organization for llama-orch to support both HOME_PROFILE (single machine) and CLOUD_PROFILE (distributed) deployments.

---

## Component Naming

### Control Plane Node
- **Name**: `control-plane` or `orchestrator-node`
- **Runs**: `orchestratord`
- **Hardware**: CPU-only (no GPU required)
- **Responsibilities**:
  - Accept client requests
  - Admission & queueing
  - Placement decisions across multiple GPU nodes
  - Service registry (which GPU nodes are online)
  - Unified API surface
  - Metrics aggregation

### GPU Nodes
- **Name**: `gpu-node` or `worker-node`
- **Runs**: `pool-managerd`
- **Hardware**: NVIDIA GPUs required
- **Responsibilities**:
  - Manage local GPU pools
  - Spawn engine processes
  - Watch handoff files (local filesystem)
  - Report health to control plane
  - Execute inference tasks
  - Manage local catalog

---

## Library Categories

### Control Plane Only (`libs/control-plane/`)

Libraries used **only** by orchestratord:

- **service-registry**: Track online GPU nodes, heartbeat monitoring
- **multi-node-placement**: Placement logic across distributed nodes
- **node-health-poller**: Poll GPU nodes for health status
- **admission-coordinator**: Coordinate admission across multiple nodes

**Dependencies**: HTTP client, metrics aggregation

### GPU Node Only (`libs/gpu-node/`)

Libraries used **only** by pool-managerd:

- **handoff-watcher**: Watch `.runtime/engines/*.json` for readiness
- **node-registration**: Register with control plane on startup
- **heartbeat-reporter**: Send periodic heartbeat to control plane
- **local-pool-manager**: Manage pools on this machine
- **gpu-discovery**: NVIDIA GPU detection (may move from engine-provisioner)

**Dependencies**: filesystem watching, GPU APIs, HTTP client

### Shared Libraries (`libs/shared/`)

Libraries used by **both** control plane and GPU nodes:

- **orchestrator-core**: Queue implementation (used by orchestratord)
- **pool-registry-types**: Common types for pool registry (used by both)
- **adapter-api**: Adapter trait definitions
- **api-types**: OpenAPI contracts
- **auth-min**: Authentication/authorization
- **observability/narration-core**: Logging
- **proof-bundle**: Test artifact generation

**Dependencies**: serde, HTTP types, tracing

### Provisioners (`libs/provisioners/`)

**Current Location**: `libs/provisioners/`  
**Used By**: GPU nodes (pool-managerd coordinates them)  
**Keep Separate**: Yes (they're already well-organized)

- **engine-provisioner**: Build/fetch engines, spawn processes
- **model-provisioner**: Fetch GGUF models

### Adapters (`libs/worker-adapters/`)

**Current Location**: `libs/worker-adapters/`  
**Used By**: Control plane (orchestratord routes via adapters)  
**Keep Separate**: Yes

- **adapter-host**: Adapter binding/routing
- **llamacpp-http**, **vllm-http**, etc.: Engine-specific adapters

---

## Proposed Directory Structure

```
libs/
├── control-plane/              # NEW: Control plane only
│   ├── service-registry/
│   ├── multi-node-placement/
│   ├── node-health-poller/
│   └── admission-coordinator/
│
├── gpu-node/                   # NEW: GPU node only
│   ├── handoff-watcher/
│   ├── node-registration/
│   ├── heartbeat-reporter/
│   ├── local-pool-manager/     # Extracted from pool-managerd
│   └── gpu-discovery/          # May extract from engine-provisioner
│
├── shared/                     # NEW: Shared between both
│   ├── pool-registry-types/    # Extracted from pool-managerd
│   ├── node-types/             # Node metadata, health status
│   └── service-discovery-types/
│
├── orchestrator-core/          # EXISTING: Queue (used by orchestratord)
├── catalog-core/               # EXISTING: Catalog storage (used by orchestratord)
├── adapter-host/               # EXISTING: Adapter routing (used by orchestratord)
├── auth-min/                   # EXISTING: Auth (used by both)
├── observability/              # EXISTING: Logging (used by both)
├── proof-bundle/               # EXISTING: Testing (used by both)
├── provisioners/               # EXISTING: Engine/model provisioning (used by pool-managerd)
└── worker-adapters/            # EXISTING: Adapters (used by orchestratord)
```

---

## Migration Strategy

### Phase 1: Extract Shared Types (Week 1)
1. Create `libs/shared/pool-registry-types/`
2. Move common types from `pool-managerd` to shared crate
3. Update imports in orchestratord and pool-managerd

### Phase 2: Move Handoff Watcher (Week 2)
1. Create `libs/gpu-node/handoff-watcher/`
2. Move handoff watching logic from orchestratord
3. pool-managerd uses it to watch local filesystem
4. orchestratord polls pool-managerd HTTP instead

### Phase 3: Service Discovery (Week 3-4)
1. Create `libs/control-plane/service-registry/`
2. Create `libs/gpu-node/node-registration/`
3. Implement registration/heartbeat protocol
4. Update orchestratord to track multiple nodes

### Phase 4: Multi-Node Placement (Week 5-6)
1. Create `libs/control-plane/multi-node-placement/`
2. Extend placement logic to consider all nodes
3. Filter by model availability, GPU capacity
4. Implement node failure handling

---

## Dependency Rules

### orchestratord Dependencies
```toml
[dependencies]
# Control plane libs
service-registry = { path = "../../libs/control-plane/service-registry" }
multi-node-placement = { path = "../../libs/control-plane/multi-node-placement" }
node-health-poller = { path = "../../libs/control-plane/node-health-poller" }

# Shared libs
orchestrator-core = { path = "../../libs/orchestrator-core" }
catalog-core = { path = "../../libs/catalog-core" }
adapter-host = { path = "../../libs/adapter-host" }
auth-min = { path = "../../libs/auth-min" }
pool-registry-types = { path = "../../libs/shared/pool-registry-types" }

# Contracts
api-types = { path = "../../contracts/api-types" }

# Observability
narration-core = { path = "../../libs/observability/narration-core" }
```

### pool-managerd Dependencies
```toml
[dependencies]
# GPU node libs
handoff-watcher = { path = "../../libs/gpu-node/handoff-watcher" }
node-registration = { path = "../../libs/gpu-node/node-registration" }
heartbeat-reporter = { path = "../../libs/gpu-node/heartbeat-reporter" }
local-pool-manager = { path = "../../libs/gpu-node/local-pool-manager" }

# Provisioners
engine-provisioner = { path = "../../libs/provisioners/engine-provisioner" }
model-provisioner = { path = "../../libs/provisioners/model-provisioner" }

# Shared libs
auth-min = { path = "../../libs/auth-min" }
pool-registry-types = { path = "../../libs/shared/pool-registry-types" }

# Contracts
api-types = { path = "../../contracts/api-types" }

# Observability
narration-core = { path = "../../libs/observability/narration-core" }
```

---

## HOME_PROFILE vs CLOUD_PROFILE

### HOME_PROFILE (Current)
- Both orchestratord and pool-managerd on same machine
- Shared filesystem
- Localhost communication
- orchestratord embeds pool-managerd registry as library
- Handoff watcher in orchestratord works (same filesystem)

### CLOUD_PROFILE (Future)
- orchestratord on control plane node (no GPU)
- pool-managerd on each GPU node
- Network communication (HTTP)
- Authentication required
- Handoff watcher in pool-managerd (local filesystem)
- Service discovery via registration/heartbeat

---

## Backward Compatibility

**Requirement**: HOME_PROFILE must continue to work

**Strategy**:
1. Feature flags: `ORCHESTRATORD_CLOUD_PROFILE=false` (default)
2. When cloud profile disabled:
   - orchestratord uses localhost URLs
   - Service registry disabled
   - Handoff watcher can stay in orchestratord (or poll local pool-managerd)
3. When cloud profile enabled:
   - orchestratord binds 0.0.0.0
   - Service registry active
   - pool-managerd registers on startup

---

## Testing Strategy

### Unit Tests
- Each new crate has unit tests
- Shared types have property tests

### Integration Tests
- HOME_PROFILE: Single process tests (current behavior)
- CLOUD_PROFILE: Multi-process tests (orchestratord + 2+ pool-managerd)

### E2E Tests
- test-harness/e2e-cloud: Multi-node cluster tests
- Chaos testing: Node failures, network partitions

---

## Next Steps

1. ✅ Document library organization (this file)
2. ⏳ Create `libs/shared/pool-registry-types/`
3. ⏳ Create `libs/gpu-node/handoff-watcher/`
4. ⏳ Create `libs/control-plane/service-registry/`
5. ⏳ Update orchestratord to use service registry
6. ⏳ Update pool-managerd to register and send heartbeats
7. ⏳ Update placement logic for multi-node
8. ⏳ Add authentication layer
9. ⏳ Write multi-node E2E tests

---

## References

- `.specs/01_cloud_profile_.md` - Cloud profile spec
- `.docs/CLOUD_PROFILE_KNOWLEDGE.md` - Architecture deep dive
- `.docs/CLOUD_PROFILE_MIGRATION.md` - Migration plan
- `bin/pool-managerd/RESPONSIBILITY_AUDIT.md` - Current state analysis
