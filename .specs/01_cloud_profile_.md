# Cloud Profile Specification

**Status**: Draft  
**Version**: 1.0  
**Date**: 2025-09-30  
**Applies To**: All llama-orch components

---

## 0. Overview

The **Cloud Profile** defines the architecture, requirements, and contracts for deploying llama-orch in a distributed, multi-machine environment. This profile enables:

- Multiple GPU nodes managed by a single control plane
- Network-based communication (no shared filesystem)
- Secure inter-service authentication
- Dynamic service discovery
- Horizontal scalability

This specification complements `.specs/00_llama-orch.md` (HOME_PROFILE) and defines additional requirements for cloud deployments.

---

## 1. Architecture Principles

### 1.1 Service Boundaries

**[CLOUD-1001]** Services MUST communicate exclusively via HTTP/HTTPS. Filesystem coupling is FORBIDDEN.

**[CLOUD-1002]** Each service MUST only access its local filesystem. No shared filesystem assumptions.

**[CLOUD-1003]** Services MUST be location-agnostic. They MUST NOT assume co-location with other services.

**[CLOUD-1004]** Services MUST handle network failures gracefully with retries and timeouts.

### 1.2 Network Topology

**[CLOUD-1010]** orchestratord MUST run on a control plane node (GPU optional, not required).

**[CLOUD-1011]** pool-managerd MUST run on GPU nodes (one instance per machine).

**[CLOUD-1012]** All services MUST bind to `0.0.0.0` (all interfaces), not `127.0.0.1`.

**[CLOUD-1013]** Services MUST support TLS termination (direct or via reverse proxy).

### 1.3 State Management

**[CLOUD-1020]** orchestratord MUST maintain authoritative state for:
- Service registry (which nodes are online)
- Placement decisions
- Task routing

**[CLOUD-1021]** pool-managerd MUST maintain authoritative state for:
- Local pool registry
- GPU inventory
- Engine processes

**[CLOUD-1022]** State MUST NOT be shared via filesystem. All state synchronization via HTTP.

---

## 2. Service Discovery

### 2.1 Node Registration

**[CLOUD-2001]** pool-managerd MUST register with orchestratord on startup via `POST /v2/nodes/register`.

**[CLOUD-2002]** Registration payload MUST include:
```json
{
  "node_id": "string",        // Unique node identifier
  "address": "string",        // HTTP(S) address (host:port)
  "machine_id": "string",     // Physical machine identifier
  "pools": ["string"],        // Pool IDs on this node
  "capabilities": {
    "gpus": [
      {
        "device_id": 0,
        "name": "RTX 3090",
        "vram_total_bytes": 24000000000,
        "compute_capability": "8.6"
      }
    ]
  }
}
```

**[CLOUD-2003]** orchestratord MUST respond with registration confirmation or rejection.

**[CLOUD-2004]** Registration MUST be idempotent. Re-registration updates node info.

### 2.2 Heartbeat Mechanism

**[CLOUD-2010]** pool-managerd MUST send heartbeat to orchestratord via `POST /v2/nodes/{id}/heartbeat`.

**[CLOUD-2011]** Heartbeat interval MUST be configurable (default: 10 seconds).

**[CLOUD-2012]** Heartbeat payload MUST include:
```json
{
  "timestamp": "ISO8601",
  "pools": [
    {
      "pool_id": "string",
      "ready": true,
      "slots_free": 2,
      "vram_free_bytes": 18000000000
    }
  ]
}
```

**[CLOUD-2013]** orchestratord MUST mark nodes as offline if heartbeat not received within timeout (default: 30 seconds).

**[CLOUD-2014]** Offline nodes MUST be excluded from placement decisions.

### 2.3 Deregistration

**[CLOUD-2020]** pool-managerd MUST deregister on graceful shutdown via `DELETE /v2/nodes/{id}`.

**[CLOUD-2021]** orchestratord MUST remove node from registry on deregistration.

**[CLOUD-2022]** Ungraceful shutdown (crash) MUST be detected via heartbeat timeout.

---

## 3. Handoff Detection (Cloud-Aware)

### 3.1 Ownership

**[CLOUD-3001]** pool-managerd MUST own the handoff file watcher.

**[CLOUD-3002]** orchestratord MUST NOT access pool-managerd filesystem.

**[CLOUD-3003]** Handoff detection MUST occur on the same machine as engine-provisioner.

### 3.2 Detection Flow

**[CLOUD-3010]** pool-managerd MUST watch `.runtime/engines/*.json` for handoff files.

**[CLOUD-3011]** When handoff detected, pool-managerd MUST:
1. Update local registry (mark pool ready)
2. Include updated status in next heartbeat
3. Optionally: POST callback to orchestratord

**[CLOUD-3012]** orchestratord MUST detect pool readiness via:
- **Primary**: Heartbeat payload
- **Optional**: Callback from pool-managerd

**[CLOUD-3013]** Handoff files MUST be processed within 2 seconds of creation.

### 3.3 Adapter Binding

**[CLOUD-3020]** orchestratord MUST bind adapter when pool becomes ready.

**[CLOUD-3021]** Adapter binding MUST use pool's HTTP endpoint from registry.

**[CLOUD-3022]** Binding MUST be idempotent (re-binding same pool is safe).

---

## 4. Authentication & Security

### 4.1 Inter-Service Authentication

**[CLOUD-4001]** All inter-service HTTP requests MUST include authentication.

**[CLOUD-4002]** Authentication MUST use Bearer tokens or mTLS.

**[CLOUD-4003]** Tokens MUST be configured via `LLORCH_API_TOKEN` environment variable.

**[CLOUD-4004]** Services MUST validate tokens on all non-loopback requests.

**[CLOUD-4005]** Invalid tokens MUST result in HTTP 401 Unauthorized.

### 4.2 Token Management

**[CLOUD-4010]** Tokens MUST be at least 32 bytes of cryptographic randomness.

**[CLOUD-4011]** Tokens MUST be rotatable without service restart.

**[CLOUD-4012]** Token rotation MUST be coordinated (grace period for old tokens).

**[CLOUD-4013]** Tokens MUST NOT be logged or exposed in error messages.

### 4.3 TLS Requirements

**[CLOUD-4020]** Production deployments MUST use TLS for all HTTP traffic.

**[CLOUD-4021]** TLS certificates MUST be valid and not self-signed in production.

**[CLOUD-4022]** Certificate validation MUST be enforced (no `--insecure` flags).

**[CLOUD-4023]** mTLS MAY be used for machine-to-machine communication.

### 4.4 Network Isolation

**[CLOUD-4030]** Services SHOULD run in isolated network segments.

**[CLOUD-4031]** Firewall rules MUST restrict access to service ports.

**[CLOUD-4032]** Only orchestratord SHOULD be exposed to external clients.

**[CLOUD-4033]** pool-managerd SHOULD only accept connections from orchestratord.

---

## 5. Catalog Management (Cloud)

### 5.1 Per-Node Catalogs

**[CLOUD-5001]** Each node MUST maintain its own local catalog.

**[CLOUD-5002]** Catalogs MUST NOT be shared via filesystem.

**[CLOUD-5003]** orchestratord MUST track which nodes have which models.

### 5.2 Model Availability

**[CLOUD-5010]** orchestratord MUST expose `GET /v2/catalog/availability`.

**[CLOUD-5011]** Response MUST include per-node model availability:
```json
{
  "models": [
    {
      "model_id": "llama-3-8b-instruct",
      "nodes": ["node1", "node2"],
      "total_nodes": 2,
      "total_capacity": 3
    }
  ]
}
```

**[CLOUD-5012]** Placement MUST only consider nodes that have the required model.

**[CLOUD-5013]** Placement to node without model MUST fail with `MODEL_NOT_AVAILABLE`.

### 5.3 Model Staging

**[CLOUD-5020]** Models MUST be staged per-node by operator or automation.

**[CLOUD-5021]** pool-managerd MUST report available models in registration/heartbeat.

**[CLOUD-5022]** orchestratord MUST update availability tracking on heartbeat.

### 5.4 Synchronization Strategy (v2.0)

**[CLOUD-5030]** Future: Catalog replication protocol for automatic sync.

**[CLOUD-5031]** Future: Peer-to-peer model distribution.

**[CLOUD-5032]** Future: Centralized model repository with pull mechanism.

---

## 6. Placement (Multi-Node)

### 6.1 Node Selection

**[CLOUD-6001]** Placement MUST consider all registered, online nodes.

**[CLOUD-6002]** Placement MUST filter nodes by:
- Model availability
- GPU capacity (VRAM)
- Pool readiness
- Device mask compatibility

**[CLOUD-6003]** Placement MUST use least-loaded selection across all nodes:
1. Filter: ready, not draining, has model, sufficient VRAM
2. Score: most free VRAM, then fewest active slots
3. Tie-break: deterministic (node_id, pool_id)

**[CLOUD-6004]** Placement MUST NOT assume node co-location.

### 6.2 Affinity & Anti-Affinity

**[CLOUD-6010]** Session affinity SHOULD prefer same node for session continuity.

**[CLOUD-6011]** Anti-affinity rules MAY spread load across nodes.

**[CLOUD-6012]** Node pinning MAY be supported via `TaskRequest.placement.node_id`.

**[CLOUD-6013]** Invalid node pin MUST fail with `NODE_UNAVAILABLE`.

### 6.3 Failure Handling

**[CLOUD-6020]** Node failure MUST trigger immediate re-placement.

**[CLOUD-6021]** In-flight tasks on failed node MUST be marked as failed.

**[CLOUD-6022]** Clients MUST receive `WORKER_RESET` error for failed tasks.

**[CLOUD-6023]** Retry MUST route to different node.

---

## 7. Observability (Multi-Node)

### 7.1 Metrics Collection

**[CLOUD-7001]** Each pool-managerd MUST expose Prometheus metrics on `/metrics`.

**[CLOUD-7002]** orchestratord MUST expose aggregated metrics on `/metrics`.

**[CLOUD-7003]** All metrics MUST include `node_id` label.

**[CLOUD-7004]** Prometheus MUST scrape all pool-managerd instances.

### 7.2 Service Discovery Integration

**[CLOUD-7010]** Prometheus SHOULD use service discovery (Consul, K8s, etc.).

**[CLOUD-7011]** Scrape targets MUST be dynamically updated as nodes join/leave.

**[CLOUD-7012]** Scrape interval MUST be configurable (default: 15 seconds).

### 7.3 Logging

**[CLOUD-7020]** All logs MUST include `node_id` and `machine_id` fields.

**[CLOUD-7021]** Logs MUST include correlation IDs for cross-service tracing.

**[CLOUD-7022]** Logs SHOULD be aggregated in centralized system (Loki, ELK).

**[CLOUD-7023]** Log retention MUST be configurable per environment.

### 7.4 Distributed Tracing

**[CLOUD-7030]** Trace context MUST propagate across service boundaries.

**[CLOUD-7031]** Trace IDs MUST be included in HTTP headers (`X-Trace-Id`).

**[CLOUD-7032]** Spans MUST be emitted for:
- Placement decisions
- Inter-service HTTP calls
- Pool operations (drain, reload)
- Adapter binding

---

## 8. Configuration

### 8.1 orchestratord Configuration

```bash
# Network
ORCHESTRATORD_ADDR=0.0.0.0:8080
ORCHESTRATORD_TLS_CERT=/path/to/cert.pem
ORCHESTRATORD_TLS_KEY=/path/to/key.pem

# Service Discovery
ORCHESTRATORD_NODE_TIMEOUT_SECS=30
ORCHESTRATORD_HEARTBEAT_INTERVAL_SECS=10

# Polling (if not using callbacks)
ORCHESTRATORD_POOL_POLL_INTERVAL_MS=5000

# Authentication
LLORCH_API_TOKEN=<secret>
ORCHESTRATORD_REQUIRE_AUTH=true

# Observability
ORCHESTRATORD_METRICS_ADDR=0.0.0.0:9090
ORCHESTRATORD_LOG_LEVEL=info
```

### 8.2 pool-managerd Configuration

```bash
# Network
POOL_MANAGERD_ADDR=0.0.0.0:9200
POOL_MANAGERD_TLS_CERT=/path/to/cert.pem
POOL_MANAGERD_TLS_KEY=/path/to/key.pem

# Node Identity
POOL_MANAGERD_NODE_ID=gpu-node-1
POOL_MANAGERD_MACHINE_ID=machine-b

# orchestratord Connection
ORCHESTRATORD_URL=https://orchestratord.example.com:8080
ORCHESTRATORD_REGISTER_ON_STARTUP=true
ORCHESTRATORD_HEARTBEAT_INTERVAL_SECS=10

# Handoff Watcher
POOL_MANAGERD_RUNTIME_DIR=.runtime/engines
POOL_MANAGERD_WATCH_INTERVAL_MS=1000

# Authentication
LLORCH_API_TOKEN=<secret>
POOL_MANAGERD_REQUIRE_AUTH=true

# Callbacks (optional)
POOL_MANAGERD_CALLBACK_ENABLED=true

# Observability
POOL_MANAGERD_METRICS_ADDR=0.0.0.0:9091
POOL_MANAGERD_LOG_LEVEL=info
```

---

## 9. Deployment Patterns

### 9.1 Kubernetes

**[CLOUD-9001]** orchestratord SHOULD run as Deployment (single replica).

**[CLOUD-9002]** pool-managerd SHOULD run as DaemonSet (one per GPU node).

**[CLOUD-9003]** Services MUST use Kubernetes service discovery.

**[CLOUD-9004]** Secrets MUST be managed via Kubernetes Secrets.

**[CLOUD-9005]** GPU allocation MUST use device plugins.

### 9.2 Docker Compose

**[CLOUD-9010]** Suitable for development and small deployments.

**[CLOUD-9011]** Each service MUST run in separate container.

**[CLOUD-9012]** Network MUST be bridge or overlay mode.

**[CLOUD-9013]** GPU passthrough MUST use nvidia-docker runtime.

### 9.3 Bare Metal

**[CLOUD-9020]** Services MUST run as systemd units.

**[CLOUD-9021]** Service discovery MAY use Consul or static configuration.

**[CLOUD-9022]** TLS certificates MUST be managed by operator.

**[CLOUD-9023]** Firewall rules MUST be configured manually.

---

## 10. Failure Modes & Recovery

### 10.1 Node Failure

**[CLOUD-10001]** orchestratord MUST detect node failure within heartbeat timeout.

**[CLOUD-10002]** Failed node MUST be removed from placement candidates.

**[CLOUD-10003]** In-flight tasks on failed node MUST be marked as failed.

**[CLOUD-10004]** Node recovery MUST trigger automatic re-registration.

### 10.2 Network Partition

**[CLOUD-10010]** Partitioned nodes MUST continue serving local tasks.

**[CLOUD-10011]** orchestratord MUST not route new tasks to partitioned nodes.

**[CLOUD-10012]** Partition recovery MUST trigger heartbeat resume.

**[CLOUD-10013]** Split-brain scenarios MUST be prevented via single orchestratord.

### 10.3 orchestratord Failure

**[CLOUD-10020]** pool-managerd MUST continue serving existing tasks.

**[CLOUD-10021]** pool-managerd MUST queue registration attempts with backoff.

**[CLOUD-10022]** orchestratord recovery MUST trigger node re-registration.

**[CLOUD-10023]** High availability MAY use orchestratord replicas (future).

---

## 11. Performance Requirements

### 11.1 Latency

**[CLOUD-11001]** Node registration MUST complete within 5 seconds.

**[CLOUD-11002]** Heartbeat processing MUST complete within 1 second.

**[CLOUD-11003]** Placement decision MUST complete within 100ms.

**[CLOUD-11004]** Handoff detection MUST occur within 2 seconds.

### 11.2 Throughput

**[CLOUD-11010]** System MUST support 10+ GPU nodes.

**[CLOUD-11011]** System MUST support 100+ concurrent tasks.

**[CLOUD-11012]** System MUST handle 1000+ requests/second (admission).

### 11.3 Scalability

**[CLOUD-11020]** Horizontal scaling MUST be linear up to 50 nodes.

**[CLOUD-11021]** Adding nodes MUST NOT require orchestratord restart.

**[CLOUD-11022]** Removing nodes MUST NOT affect other nodes.

---

## 12. Migration from HOME_PROFILE

### 12.1 Compatibility

**[CLOUD-12001]** HOME_PROFILE deployments MUST continue to work.

**[CLOUD-12002]** Feature flags MUST control CLOUD_PROFILE features.

**[CLOUD-12003]** Configuration MUST support both profiles.

### 12.2 Migration Path

**[CLOUD-12010]** Step 1: Update network binding (`0.0.0.0`)
**[CLOUD-12011]** Step 2: Move handoff watcher to pool-managerd
**[CLOUD-12012]** Step 3: Implement service discovery
**[CLOUD-12013]** Step 4: Add authentication
**[CLOUD-12014]** Step 5: Enable multi-node placement

---

## 13. Testing Requirements

### 13.1 Multi-Node Tests

**[CLOUD-13001]** E2E tests MUST run with 3+ nodes.

**[CLOUD-13002]** Tests MUST verify node registration/deregistration.

**[CLOUD-13003]** Tests MUST verify placement across nodes.

**[CLOUD-13004]** Tests MUST verify node failure handling.

### 13.2 Chaos Testing

**[CLOUD-13010]** Tests MUST simulate network partitions.

**[CLOUD-13011]** Tests MUST simulate node crashes.

**[CLOUD-13012]** Tests MUST simulate slow networks.

**[CLOUD-13013]** Tests MUST verify recovery scenarios.

---

## 14. References

- **HOME_PROFILE**: `.specs/00_llama-orch.md`
- **pool-managerd**: `.specs/30-pool-managerd.md`
- **orchestratord**: `.specs/20-orchestratord.md`
- **Auth**: `.specs/11_min_auth_hooks.md`
- **Migration Plan**: `.docs/CLOUD_PROFILE_MIGRATION.md`

---

## 15. Approval & Versioning

**Version**: 1.0 (Draft)  
**Status**: Under Review  
**Target Release**: v1.0.0  

**Approved By**:
- [ ] Architecture Team
- [ ] Security Team
- [ ] DevOps Team
- [ ] Management
