# Cloud Profile Migration Plan

**Date**: 2025-09-30  
**Status**: PLANNING  
**Target**: v1.0.0  
**Current**: v0.1.0 (HOME_PROFILE only)

---

## Executive Summary

This document outlines the complete migration from **HOME_PROFILE** (single-machine) to **CLOUD_PROFILE** (distributed multi-machine) architecture. The migration involves breaking filesystem coupling, implementing service discovery, adding authentication, and ensuring all components work across network boundaries.

---

## Table of Contents

1. [Current State (HOME_PROFILE)](#current-state-home_profile)
2. [Target State (CLOUD_PROFILE)](#target-state-cloud_profile)
3. [Breaking Changes Required](#breaking-changes-required)
4. [Migration Phases](#migration-phases)
5. [Component-by-Component Changes](#component-by-component-changes)
6. [Testing Strategy](#testing-strategy)
7. [Deployment Architecture](#deployment-architecture)
8. [Risk Assessment](#risk-assessment)
9. [Timeline & Milestones](#timeline--milestones)

---

## Current State (HOME_PROFILE)

### Architecture

```
Single Workstation (localhost):
  ┌─────────────────────────────────────┐
  │ orchestratord (port 8080)           │
  │   ├── Handoff watcher (filesystem)  │ ← BREAKS IN CLOUD
  │   ├── Catalog (local filesystem)    │ ← BREAKS IN CLOUD
  │   └── Adapter-host                  │
  └─────────────────────────────────────┘
              ↓ HTTP (localhost)
  ┌─────────────────────────────────────┐
  │ pool-managerd (port 9200)           │
  │   ├── Registry (in-memory)          │
  │   ├── Engine processes              │
  │   └── Catalog (local filesystem)    │ ← BREAKS IN CLOUD
  └─────────────────────────────────────┘
              ↓ spawns
  ┌─────────────────────────────────────┐
  │ llamacpp (GPU:0)  llamacpp (GPU:1)  │
  └─────────────────────────────────────┘
```

### Assumptions (HOME_PROFILE)

1. **Single Machine**: All components on same host
2. **Shared Filesystem**: orchestratord can read pool-managerd files
3. **Localhost Networking**: No authentication required
4. **Single Catalog**: One filesystem-based catalog
5. **Direct Access**: orchestratord watches `.runtime/` directory
6. **No Service Discovery**: Hardcoded `localhost:9200`

### What Works

- ✅ Single workstation with multiple GPUs
- ✅ Fast development iteration
- ✅ Simple deployment (single binary per service)
- ✅ No network overhead
- ✅ No authentication complexity

### What Breaks in Cloud

- ❌ **Handoff watcher**: orchestratord can't read pool-managerd filesystem
- ❌ **Catalog**: Each machine has separate catalog (no sync)
- ❌ **Service discovery**: Can't find pool-managerd on other machines
- ❌ **Authentication**: Open localhost is insecure across network
- ❌ **Model distribution**: Models must be staged per-machine
- ❌ **Metrics aggregation**: Each machine reports separately

---

## Target State (CLOUD_PROFILE)

### Architecture

```
Control Plane (Machine A - No GPU):
  ┌─────────────────────────────────────┐
  │ orchestratord (0.0.0.0:8080)        │
  │   ├── Service registry              │ ← NEW
  │   ├── Pool health poller            │ ← NEW
  │   ├── Catalog API (coordinator)     │ ← CHANGED
  │   └── Adapter-host                  │
  └─────────────────────────────────────┘
              ↓ HTTP (TLS + Auth)
  ┌─────────────────────────────────────┐
  │ Load Balancer / Service Mesh        │ ← NEW
  └─────────────────────────────────────┘
         ↓                    ↓
  ┌──────────────┐    ┌──────────────┐
  │ GPU Node B   │    │ GPU Node C   │
  └──────────────┘    └──────────────┘

GPU Node B (Machine B):
  ┌─────────────────────────────────────┐
  │ pool-managerd (0.0.0.0:9200)        │
  │   ├── Handoff watcher               │ ← MOVED HERE
  │   ├── Registry (local)              │
  │   ├── Catalog (local filesystem)    │
  │   ├── Health reporter               │ ← NEW
  │   └── Callback client               │ ← NEW
  └─────────────────────────────────────┘
              ↓ spawns
  ┌─────────────────────────────────────┐
  │ llamacpp (GPU:0)  llamacpp (GPU:1)  │
  └─────────────────────────────────────┘

GPU Node C (Machine C):
  ┌─────────────────────────────────────┐
  │ pool-managerd (0.0.0.0:9200)        │
  │   ├── Handoff watcher               │
  │   ├── Registry (local)              │
  │   ├── Catalog (local filesystem)    │
  │   ├── Health reporter               │
  │   └── Callback client               │
  └─────────────────────────────────────┐
              ↓ spawns
  ┌─────────────────────────────────────┐
  │ llamacpp (GPU:2)  llamacpp (GPU:3)  │
  └─────────────────────────────────────┘
```

### Requirements (CLOUD_PROFILE)

1. **Multi-Machine**: Services distributed across hosts
2. **Network Isolation**: No shared filesystem
3. **Authentication**: mTLS or Bearer tokens
4. **Service Discovery**: Dynamic pool registration
5. **HTTP-Only Communication**: All inter-service via HTTP
6. **Catalog Sync**: Strategy for model distribution
7. **Observability**: Centralized metrics/logs

---

## Breaking Changes Required

### 1. Handoff Watcher Ownership

**Current**: orchestratord watches filesystem  
**Required**: pool-managerd watches filesystem

**Impact**: HIGH  
**Effort**: 2 weeks  
**Blocker**: Yes (for distributed deployment)

**Changes**:
- Move watcher from `bin/orchestratord/src/services/handoff.rs`
- Create `bin/pool-managerd/src/watcher/handoff.rs`
- orchestratord polls pool-managerd HTTP instead
- Update all tests

### 2. Service Discovery

**Current**: Hardcoded `localhost:9200`  
**Required**: Dynamic pool registry

**Impact**: HIGH  
**Effort**: 3 weeks  
**Blocker**: Yes

**Changes**:
- Add `ServiceRegistry` in orchestratord
- pool-managerd registers on startup: `POST /v2/nodes/register`
- Heartbeat mechanism: `POST /v2/nodes/{id}/heartbeat`
- Deregistration on shutdown
- Health checks detect offline nodes

### 3. Authentication

**Current**: Open localhost (no auth)  
**Required**: mTLS or Bearer tokens

**Impact**: MEDIUM  
**Effort**: 2 weeks  
**Blocker**: No (can use network isolation initially)

**Changes**:
- Implement `.specs/11_min_auth_hooks.md`
- Add `LLORCH_API_TOKEN` environment variable
- Bearer token validation in pool-managerd
- Optional mTLS for machine-to-machine
- Token rotation mechanism

### 4. Catalog Synchronization

**Current**: Per-machine filesystem catalog  
**Required**: Distributed catalog strategy

**Impact**: HIGH  
**Effort**: 4 weeks  
**Blocker**: Partial (can stage manually initially)

**Options**:

**Option A: Shared Filesystem** (NFS/S3)
- Mount shared volume on all nodes
- Single source of truth
- Simple but requires infrastructure

**Option B: Catalog Replication**
- orchestratord as catalog coordinator
- Replicate entries to all nodes
- Complex but flexible

**Option C: Manual Staging** (v1.0 interim)
- Operator stages models per-node
- orchestratord tracks which nodes have which models
- Simple but manual

**Recommendation**: Option C for v1.0, Option B for v2.0

### 5. Network Binding

**Current**: `127.0.0.1` (localhost only)  
**Required**: `0.0.0.0` (all interfaces)

**Impact**: LOW  
**Effort**: 1 day  
**Blocker**: No

**Changes**:
```bash
# orchestratord
ORCHESTRATORD_ADDR=0.0.0.0:8080  # was 127.0.0.1:8080

# pool-managerd
POOL_MANAGERD_ADDR=0.0.0.0:9200  # was 127.0.0.1:9200
```

### 6. Metrics Aggregation

**Current**: Single Prometheus scrape target  
**Required**: Multi-target scraping

**Impact**: LOW  
**Effort**: 1 week  
**Blocker**: No

**Changes**:
- Prometheus scrapes all pool-managerd instances
- Service discovery integration (Consul/K8s)
- Centralized Grafana dashboards
- Alert aggregation

### 7. Logging Correlation

**Current**: Single log stream  
**Required**: Distributed tracing

**Impact**: MEDIUM  
**Effort**: 2 weeks  
**Blocker**: No

**Changes**:
- Add `machine_id` to all logs
- Correlation IDs across services
- Centralized log aggregation (Loki/ELK)
- Trace context propagation

---

## Migration Phases

### Phase 0: Preparation (Week 1-2)

**Goal**: Document and plan

- ✅ Document current architecture
- ✅ Identify breaking changes
- ✅ Create migration plan
- ✅ Design cloud profile spec
- ✅ Update roadmap

**Deliverables**:
- This document
- `.specs/01_cloud_profile.md`
- Updated TODO.md

### Phase 1: Foundation (Week 3-4)

**Goal**: Break filesystem coupling

**Tasks**:
1. Move handoff watcher to pool-managerd
2. Implement HTTP polling in orchestratord
3. Add network binding configuration
4. Update tests for HTTP-based communication

**Deliverables**:
- `bin/pool-managerd/src/watcher/handoff.rs`
- `bin/orchestratord/src/services/pool_health.rs`
- E2E tests with separate processes
- Documentation updates

**Success Criteria**:
- orchestratord and pool-managerd can run on different machines
- Handoff detection works across network
- All tests pass

### Phase 2: Service Discovery (Week 5-7)

**Goal**: Dynamic node registration

**Tasks**:
1. Design service registry API
2. Implement node registration in pool-managerd
3. Implement registry in orchestratord
4. Add heartbeat mechanism
5. Handle node failures gracefully

**Deliverables**:
- `bin/orchestratord/src/services/registry.rs`
- `POST /v2/nodes/register` endpoint
- `POST /v2/nodes/{id}/heartbeat` endpoint
- Health check monitoring
- Deregistration logic

**Success Criteria**:
- Multiple pool-managerd instances register automatically
- orchestratord discovers all nodes
- Failed nodes are detected and removed
- Placement considers all available nodes

### Phase 3: Authentication (Week 8-9)

**Goal**: Secure inter-service communication

**Tasks**:
1. Implement minimal auth hooks (`.specs/11_min_auth_hooks.md`)
2. Add Bearer token validation
3. Generate and distribute tokens
4. Optional: mTLS setup
5. Token rotation mechanism

**Deliverables**:
- `libs/auth-min/` enhancements
- Token validation middleware
- Configuration guide
- Security documentation

**Success Criteria**:
- Unauthorized requests are rejected
- Token validation works
- No secrets in logs
- Audit trail for access

### Phase 4: Catalog Strategy (Week 10-12)

**Goal**: Model distribution across nodes

**Tasks**:
1. Design catalog sync strategy
2. Implement model tracking per-node
3. Add model staging API
4. Handle model availability in placement
5. Garbage collection across nodes

**Deliverables**:
- Catalog sync design doc
- `GET /v2/catalog/availability` endpoint
- Model staging workflow
- GC policy implementation

**Success Criteria**:
- orchestratord knows which nodes have which models
- Placement only considers nodes with required model
- Models can be staged to specific nodes
- Disk space managed across cluster

### Phase 5: Observability (Week 13-14)

**Goal**: Unified monitoring and logging

**Tasks**:
1. Multi-target Prometheus scraping
2. Centralized logging setup
3. Distributed tracing
4. Unified dashboards
5. Alert aggregation

**Deliverables**:
- Prometheus service discovery config
- Grafana dashboards for multi-node
- Logging aggregation setup
- Trace context propagation
- Alert rules

**Success Criteria**:
- All nodes visible in Grafana
- Logs searchable across cluster
- Traces span multiple services
- Alerts fire correctly

### Phase 6: Testing & Validation (Week 15-16)

**Goal**: Comprehensive testing

**Tasks**:
1. Multi-node E2E tests
2. Failure scenario testing
3. Load testing across nodes
4. Security testing
5. Documentation updates

**Deliverables**:
- E2E test suite for cloud profile
- Chaos testing scenarios
- Load test results
- Security audit report
- Deployment guide

**Success Criteria**:
- All tests pass in multi-node setup
- Handles node failures gracefully
- Performance meets SLOs
- Security audit passes
- Documentation complete

---

## Component-by-Component Changes

### orchestratord Changes

#### New Modules

1. **`src/services/registry.rs`** - Service registry
   ```rust
   pub struct ServiceRegistry {
       nodes: HashMap<NodeId, NodeInfo>,
   }
   
   pub struct NodeInfo {
       node_id: String,
       address: String,
       pools: Vec<PoolId>,
       last_heartbeat: Instant,
       status: NodeStatus,
   }
   ```

2. **`src/services/pool_health.rs`** - Health poller
   ```rust
   pub fn spawn_pool_health_poller(state: AppState) {
       // Poll all registered nodes
       // Update placement data
       // Bind adapters when ready
   }
   ```

3. **`src/api/nodes.rs`** - Node management API
   ```rust
   // POST /v2/nodes/register
   // POST /v2/nodes/{id}/heartbeat
   // DELETE /v2/nodes/{id}
   // GET /v2/nodes
   ```

#### Modified Modules

1. **`src/services/handoff.rs`** - Remove or deprecate
2. **`src/services/placement.rs`** - Multi-node aware
3. **`src/app/bootstrap.rs`** - Start registry and poller
4. **`src/app/state.rs`** - Add service registry

#### Configuration

```bash
# Network
ORCHESTRATORD_ADDR=0.0.0.0:8080

# Service Discovery
ORCHESTRATORD_NODE_TIMEOUT_SECS=30
ORCHESTRATORD_HEARTBEAT_INTERVAL_SECS=10

# Polling
ORCHESTRATORD_POOL_POLL_INTERVAL_MS=5000

# Authentication
LLORCH_API_TOKEN=<secret>
ORCHESTRATORD_REQUIRE_AUTH=true
```

### pool-managerd Changes

#### New Modules

1. **`src/watcher/handoff.rs`** - Handoff watcher
   ```rust
   pub fn spawn_handoff_watcher(
       registry: Arc<Mutex<Registry>>,
       config: HandoffWatcherConfig,
   ) -> JoinHandle<()>
   ```

2. **`src/services/registration.rs`** - Node registration
   ```rust
   pub async fn register_with_orchestratord(
       node_id: &str,
       orchestratord_url: &str,
   ) -> Result<()>
   
   pub fn spawn_heartbeat_task(
       node_id: String,
       orchestratord_url: String,
   ) -> JoinHandle<()>
   ```

3. **`src/services/callback.rs`** - Callback client
   ```rust
   pub async fn notify_pool_ready(
       orchestratord_url: &str,
       pool_id: &str,
   ) -> Result<()>
   ```

#### Modified Modules

1. **`src/main.rs`** - Start watcher and registration
2. **`src/api/routes.rs`** - Add auth middleware
3. **`src/core/registry.rs`** - No changes needed (already good!)

#### Configuration

```bash
# Network
POOL_MANAGERD_ADDR=0.0.0.0:9200

# Node Identity
POOL_MANAGERD_NODE_ID=gpu-node-1
POOL_MANAGERD_MACHINE_ID=machine-b

# orchestratord Connection
ORCHESTRATORD_URL=https://orchestratord.example.com:8080
ORCHESTRATORD_REGISTER_ON_STARTUP=true

# Handoff Watcher
POOL_MANAGERD_RUNTIME_DIR=.runtime/engines
POOL_MANAGERD_WATCH_INTERVAL_MS=1000

# Authentication
LLORCH_API_TOKEN=<secret>
POOL_MANAGERD_REQUIRE_AUTH=true

# Callbacks
POOL_MANAGERD_CALLBACK_ENABLED=true
```

### catalog-core Changes

#### Strategy Decision

**v1.0**: Manual staging per-node (no changes needed)  
**v2.0**: Add sync protocol

#### Future Enhancements

1. **`src/sync.rs`** - Catalog synchronization
2. **`src/availability.rs`** - Track which nodes have which models
3. **`src/replication.rs`** - Replicate catalog entries

### Provisioners Changes

#### Minimal Changes

- No changes for v1.0 (manual staging)
- Models staged per-node by operator
- Provisioners work as-is on each node

#### Future Enhancements

- Add model distribution protocol
- Peer-to-peer model sharing
- Centralized model repository

---

## Testing Strategy

### Unit Tests

**orchestratord**:
- Service registry operations
- Node registration/deregistration
- Heartbeat timeout detection
- Multi-node placement logic

**pool-managerd**:
- Handoff watcher file detection
- Registration client
- Heartbeat sender
- Callback client

### Integration Tests

**Cross-Service**:
- pool-managerd registers with orchestratord
- Handoff detection triggers adapter binding
- Node failure detection and recovery
- Multi-node placement decisions

### E2E Tests

**Multi-Node Scenarios**:
```rust
#[tokio::test]
async fn test_three_node_cluster() {
    // Start orchestratord
    // Start 3 pool-managerd instances
    // Verify all register
    // Submit task
    // Verify placement across nodes
    // Kill one node
    // Verify failover
}
```

### Chaos Tests

**Failure Scenarios**:
- Network partition
- Node crash
- Slow network
- Packet loss
- Clock skew
- Disk full

### Load Tests

**Performance**:
- 100 concurrent tasks
- 10 nodes
- Model switching
- Node scaling (add/remove)

---

## Deployment Architecture

### Kubernetes (Recommended)

```yaml
# orchestratord Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestratord
spec:
  replicas: 1  # Single control plane
  template:
    spec:
      containers:
      - name: orchestratord
        image: llama-orch/orchestratord:v1.0.0
        env:
        - name: ORCHESTRATORD_ADDR
          value: "0.0.0.0:8080"
        - name: LLORCH_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: llorch-secrets
              key: api-token

---
# pool-managerd DaemonSet (one per GPU node)
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: pool-managerd
spec:
  selector:
    matchLabels:
      app: pool-managerd
  template:
    spec:
      nodeSelector:
        gpu: nvidia
      containers:
      - name: pool-managerd
        image: llama-orch/pool-managerd:v1.0.0
        env:
        - name: POOL_MANAGERD_ADDR
          value: "0.0.0.0:9200"
        - name: ORCHESTRATORD_URL
          value: "http://orchestratord:8080"
        - name: POOL_MANAGERD_NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        resources:
          limits:
            nvidia.com/gpu: 2
```

### Docker Compose (Development)

```yaml
version: '3.8'

services:
  orchestratord:
    image: llama-orch/orchestratord:v1.0.0
    ports:
      - "8080:8080"
    environment:
      ORCHESTRATORD_ADDR: "0.0.0.0:8080"
      LLORCH_API_TOKEN: "dev-token"
    networks:
      - llama-orch

  pool-managerd-node1:
    image: llama-orch/pool-managerd:v1.0.0
    ports:
      - "9201:9200"
    environment:
      POOL_MANAGERD_ADDR: "0.0.0.0:9200"
      ORCHESTRATORD_URL: "http://orchestratord:8080"
      POOL_MANAGERD_NODE_ID: "node1"
      LLORCH_API_TOKEN: "dev-token"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']
              capabilities: [gpu]
    networks:
      - llama-orch

  pool-managerd-node2:
    image: llama-orch/pool-managerd:v1.0.0
    ports:
      - "9202:9200"
    environment:
      POOL_MANAGERD_ADDR: "0.0.0.0:9200"
      ORCHESTRATORD_URL: "http://orchestratord:8080"
      POOL_MANAGERD_NODE_ID: "node2"
      LLORCH_API_TOKEN: "dev-token"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2', '3']
              capabilities: [gpu]
    networks:
      - llama-orch

networks:
  llama-orch:
    driver: bridge
```

### Bare Metal (Manual)

```bash
# Machine A (Control Plane)
./orchestratord \
  --addr 0.0.0.0:8080 \
  --api-token $LLORCH_API_TOKEN

# Machine B (GPU Node 1)
./pool-managerd \
  --addr 0.0.0.0:9200 \
  --node-id gpu-node-1 \
  --orchestratord-url https://machine-a:8080 \
  --api-token $LLORCH_API_TOKEN

# Machine C (GPU Node 2)
./pool-managerd \
  --addr 0.0.0.0:9200 \
  --node-id gpu-node-2 \
  --orchestratord-url https://machine-a:8080 \
  --api-token $LLORCH_API_TOKEN
```

---

## Risk Assessment

### High Risk

1. **Catalog Synchronization**
   - **Risk**: Models out of sync across nodes
   - **Mitigation**: Manual staging for v1.0, checksums for verification
   - **Impact**: HIGH (placement failures)

2. **Network Partitions**
   - **Risk**: Nodes can't reach orchestratord
   - **Mitigation**: Heartbeat timeout, graceful degradation
   - **Impact**: HIGH (service unavailable)

3. **Authentication Bypass**
   - **Risk**: Unauthorized access to pools
   - **Mitigation**: Mandatory auth in production, audit logging
   - **Impact**: HIGH (security breach)

### Medium Risk

4. **Service Discovery Failures**
   - **Risk**: Nodes don't register correctly
   - **Mitigation**: Retry logic, manual registration fallback
   - **Impact**: MEDIUM (reduced capacity)

5. **Handoff Detection Latency**
   - **Risk**: Slow pool readiness detection
   - **Mitigation**: Configurable poll interval, callbacks
   - **Impact**: MEDIUM (slower startup)

6. **Metrics Aggregation**
   - **Risk**: Incomplete observability
   - **Mitigation**: Multi-target scraping, centralized dashboards
   - **Impact**: MEDIUM (reduced visibility)

### Low Risk

7. **Configuration Complexity**
   - **Risk**: Operators misconfigure services
   - **Mitigation**: Validation, good defaults, documentation
   - **Impact**: LOW (operator error)

8. **Log Volume**
   - **Risk**: Too many logs in distributed setup
   - **Mitigation**: Log levels, sampling, aggregation
   - **Impact**: LOW (storage cost)

---

## Timeline & Milestones

### v0.1.0 (Current) - HOME_PROFILE ✅
- **Status**: COMPLETE
- **Features**: Single machine, multiple GPUs
- **Limitations**: Documented

### v0.2.0 (Week 4) - Foundation
- **Milestone**: Break filesystem coupling
- **Features**:
  - Handoff watcher in pool-managerd
  - HTTP polling in orchestratord
  - Network binding configuration
- **Success**: Works across 2 machines

### v0.5.0 (Week 7) - Service Discovery
- **Milestone**: Dynamic node registration
- **Features**:
  - Service registry
  - Node registration/heartbeat
  - Multi-node placement
- **Success**: 3+ nodes auto-discover

### v0.8.0 (Week 12) - Authentication & Catalog
- **Milestone**: Secure and distributed
- **Features**:
  - Bearer token auth
  - Catalog availability tracking
  - Model staging workflow
- **Success**: Secure multi-node cluster

### v1.0.0 (Week 16) - Production Ready
- **Milestone**: CLOUD_PROFILE complete
- **Features**:
  - Full observability
  - Comprehensive testing
  - Production documentation
  - Deployment guides
- **Success**: Production deployment

### v2.0.0 (Future) - Optimizations
- **Milestone**: Advanced features
- **Features**:
  - Catalog replication
  - Callbacks instead of polling
  - Auto-scaling
  - Advanced placement policies

---

## Success Criteria

### Technical

- ✅ orchestratord and pool-managerd run on separate machines
- ✅ Multiple pool-managerd instances register automatically
- ✅ Placement considers all available nodes
- ✅ Node failures detected within 30 seconds
- ✅ Authentication prevents unauthorized access
- ✅ Models staged per-node with verification
- ✅ All metrics visible in centralized dashboard
- ✅ Logs searchable across cluster
- ✅ E2E tests pass in multi-node setup
- ✅ Load tests meet performance SLOs

### Operational

- ✅ Deployment guide complete
- ✅ Configuration examples provided
- ✅ Troubleshooting documentation
- ✅ Monitoring dashboards created
- ✅ Alert rules defined
- ✅ Backup/restore procedures
- ✅ Upgrade path documented
- ✅ Security audit passed

### Business

- ✅ Supports 10+ GPU nodes
- ✅ Scales to 100+ concurrent tasks
- ✅ 99.9% uptime SLA achievable
- ✅ Cost-effective resource utilization
- ✅ Easy to operate and maintain

---

## Next Steps

### Immediate (This Week)

1. ✅ Review and approve this migration plan
2. ✅ Create `.specs/01_cloud_profile.md`
3. ✅ Update TODO.md with cloud profile tasks
4. ✅ Communicate plan to all teams

### Short Term (Next 2 Weeks)

1. Implement handoff watcher in pool-managerd
2. Implement HTTP polling in orchestratord
3. Update network binding configuration
4. Create E2E test framework for multi-node

### Medium Term (Next 2 Months)

1. Implement service discovery
2. Add authentication
3. Design catalog sync strategy
4. Build observability stack

### Long Term (Next 4 Months)

1. Complete v1.0.0 release
2. Production deployment
3. Monitor and optimize
4. Plan v2.0.0 features

---

## References

- **Specs**: `.specs/00_llama-orch.md`, `.specs/30-pool-managerd.md`
- **Proposals**: `proposals/batch_2/2025-09-19-centralized-placement-and-priority-policy.md`
- **Architecture**: `bin/pool-managerd/bdd/ARCHITECTURE_CLARIFICATION.md`
- **Handoff Issue**: `bin/pool-managerd/bdd/HANDOFF_WATCHER_RESPONSE.md`
- **Auth Spec**: `.specs/11_min_auth_hooks.md`

---

## Approval

- [ ] Architecture Team
- [ ] pool-managerd Team
- [ ] orchestratord Team
- [ ] DevOps Team
- [ ] Security Team
- [ ] Management

**Approved By**: _______________  
**Date**: _______________
