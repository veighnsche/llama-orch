# Cloud Profile Implementation Summary

**Date**: 2025-09-30  
**Status**: PHASE 1 COMPLETE - Foundation Libraries Created

---

## What Was Implemented

### 1. Library Organization Structure

Created three-tier library organization for cloud profile:

- **`libs/shared/`** - Types used by both control plane and GPU nodes
- **`libs/control-plane/`** - Libraries for orchestratord (no GPU)
- **`libs/gpu-node/`** - Libraries for pool-managerd (GPU required)

### 2. Shared Libraries Created

**`libs/shared/pool-registry-types/`** - Common types for service communication
- `HealthStatus`, `HealthState` - Pool health tracking
- `NodeInfo`, `NodeStatus`, `NodeCapabilities` - Node metadata
- `PoolSnapshot`, `PoolMetadata` - Pool state for placement
- **Purpose**: Enable HTTP communication between orchestratord and pool-managerd

### 3. Control Plane Libraries Created

**`libs/control-plane/service-registry/`** - Track GPU nodes
- `ServiceRegistry` - Manage node registration/heartbeat
- `register()`, `heartbeat()`, `deregister()` - Node lifecycle
- `check_stale_nodes()` - Detect offline nodes (30s timeout)
- `spawn_stale_checker()` - Background task for monitoring
- **Purpose**: orchestratord tracks which GPU nodes are online

### 4. GPU Node Libraries Created

**`libs/gpu-node/handoff-watcher/`** - Watch for engine readiness
- `HandoffWatcher` - Poll `.runtime/engines/*.json`
- `HandoffPayload` - Engine metadata from provisioner
- Callback-based architecture for pool registry updates
- **Purpose**: Detect when engines are ready (local filesystem)

**`libs/gpu-node/node-registration/`** - Register with control plane
- `NodeRegistration` - HTTP client for registration
- `register()`, `deregister()` - Startup/shutdown
- `spawn_heartbeat()` - Periodic health reporting
- **Purpose**: pool-managerd announces itself to orchestratord

---

## Architectural Changes

### Before (HOME_PROFILE)
```
orchestratord (watches .runtime/engines/*.json)
      ↓ localhost
pool-managerd (embedded registry)
      ↓ spawns
engine processes
```

### After (CLOUD_PROFILE)
```
Control Plane Node:
  orchestratord + service-registry
        ↓ HTTP + Auth
        ↓
GPU Node 1:
  pool-managerd + handoff-watcher + node-registration
        ↓ spawns
  engine processes
```

---

## Key Design Decisions

### 1. **Component Naming**
- **Control plane node**: Runs orchestratord (CPU-only)
- **GPU nodes**: Run pool-managerd (one per machine)

### 2. **Handoff Watcher Relocation**
- **From**: orchestratord (breaks in cloud - can't access remote filesystem)
- **To**: pool-managerd (local filesystem access)
- **Communication**: pool-managerd includes readiness in heartbeat

### 3. **Shared Types**
- Extract common types to `libs/shared/pool-registry-types/`
- Both orchestratord and pool-managerd depend on shared crate
- Enables JSON serialization for HTTP transport

### 4. **Service Discovery Pattern**
- GPU nodes register on startup (`POST /v2/nodes/register`)
- Periodic heartbeat every 10s (`POST /v2/nodes/{id}/heartbeat`)
- Graceful deregister on shutdown (`DELETE /v2/nodes/{id}`)
- Timeout detection: 30s missed heartbeat → offline

---

## Next Steps

### Phase 2: Integrate into orchestratord (Week 2-3)
1. Add `/v2/nodes/*` HTTP endpoints
2. Embed `ServiceRegistry` in app state
3. Update placement logic to query registry
4. Remove old handoff watcher

### Phase 3: Integrate into pool-managerd (Week 3-4)
1. Add `HandoffWatcher` on startup
2. Add `NodeRegistration` lifecycle
3. Report pool status in heartbeat
4. Update registry on handoff detection

### Phase 4: Multi-Node Placement (Week 5-6)
1. Extend placement to consider all online nodes
2. Filter by model availability
3. Handle node failures gracefully
4. Implement affinity/anti-affinity

### Phase 5: Authentication (Week 7-8)
1. Add Bearer token validation
2. Configure `LLORCH_API_TOKEN`
3. Secure all inter-service HTTP calls

### Phase 6: Testing (Week 9-10)
1. Multi-node E2E tests
2. Chaos scenarios (network partitions, crashes)
3. Load testing across nodes

---

## Backward Compatibility

**HOME_PROFILE continues to work**:
- Feature flag: `ORCHESTRATORD_CLOUD_PROFILE=false` (default)
- When disabled: uses localhost, no service discovery
- Gradual migration path

---

## References

- `.specs/01_cloud_profile_.md` - Cloud profile specification
- `.docs/CLOUD_PROFILE_KNOWLEDGE.md` - Architecture deep dive
- `.docs/CLOUD_PROFILE_MIGRATION.md` - Full migration plan
- `.docs/ARCHITECTURE_LIBRARY_ORGANIZATION.md` - Library organization
- `bin/pool-managerd/RESPONSIBILITY_AUDIT.md` - Responsibility analysis
- `bin/orchestratord/RESPONSIBILITY_AUDIT.md` - Boundary verification
