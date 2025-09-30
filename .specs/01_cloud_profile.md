# Cloud Profile Specification

**Version**: 0.1.0  
**Status**: DRAFT  
**Date**: 2025-09-30  
**Owner**: Architecture Team

---

## Overview

The **Cloud Profile** defines the architecture, deployment model, and operational characteristics for llama-orch in **distributed, multi-machine deployments** where services run on separate nodes without shared filesystem access.

This contrasts with **HOME_PROFILE** (single machine, shared filesystem) and enables:
- Horizontal scaling across multiple GPU nodes
- Cloud-native deployments (Kubernetes, Docker Swarm, etc.)
- Geographic distribution of compute resources
- Separation of control plane from GPU workers

---

## Architecture Principles

### 1. No Shared Filesystem

**Rule**: Services MUST communicate via HTTP/gRPC only. No filesystem coupling.

**Rationale**:
- Different machines cannot share local filesystems
- Network filesystems (NFS, etc.) add complexity and failure modes
- HTTP is universal, well-understood, and cloud-native

**Implications**:
- Handoff files cannot be read across machines
- Configuration must be via environment variables or HTTP APIs
- Artifacts must be stored in distributed storage (S3, object store)

### 2. Service Boundaries

**Rule**: Each service runs independently on its designated node type.

**Node Types**:
- **Control Plane**: No GPU, runs orchestratord
- **GPU Worker**: Has GPU(s), runs pool-managerd + engine-provisioner + engines

**Communication**:
```
Control Plane (orchestratord)
    ↓ HTTP
GPU Worker 1 (pool-managerd + engine-provisioner)
    ↓ HTTP
GPU Worker 2 (pool-managerd + engine-provisioner)
    ↓ HTTP
GPU Worker N (pool-managerd + engine-provisioner)
```

### 3. Stateless Where Possible

**Rule**: Services SHOULD be stateless or use external state stores.

**State Management**:
- **orchestratord**: Stateless (admission queue in-memory, acceptable to lose)
- **pool-managerd**: Registry state (can rebuild from engine discovery)
- **engine-provisioner**: Stateless (provisions on demand)
- **Persistent state**: External (PostgreSQL, Redis, S3)

### 4. Observability First

**Rule**: All inter-service communication MUST be observable.

**Requirements**:
- Distributed tracing (OpenTelemetry)
- Structured logging with correlation IDs
- Prometheus metrics for all HTTP calls
- Health checks on all services

---

## Service Topology

### Control Plane Node

**Services**:
- `orchestratord` (port 8080)

**Responsibilities**:
- Accept client requests (HTTP API)
- Admission control and queueing
- Task routing and placement decisions
- Adapter binding (owns adapter-host)
- SSE streaming to clients

**Does NOT**:
- Access GPU hardware
- Read handoff files
- Provision engines
- Manage pool state

**Network Requirements**:
- Inbound: Client traffic (port 8080)
- Outbound: pool-managerd on GPU workers (port 9200)

### GPU Worker Node

**Services**:
- `pool-managerd` (port 9200)
- `engine-provisioner` (background service)
- `llama.cpp` or other engines (dynamic ports)

**Responsibilities**:
- GPU discovery and management
- Engine provisioning (download, compile, start)
- Pool lifecycle management
- Handoff file watching (local filesystem)
- Health monitoring of engines

**Does NOT**:
- Accept client requests directly
- Make placement decisions
- Own adapter-host

**Network Requirements**:
- Inbound: orchestratord polling (port 9200)
- Outbound: Engine HTTP endpoints (dynamic ports)
- Local: Filesystem access for handoff files

---

## Communication Patterns

### 1. Handoff Flow (Engine Readiness)

**Problem**: engine-provisioner writes handoff files, orchestratord needs to know when engines are ready.

**HOME_PROFILE Solution** (BROKEN in cloud):
```
engine-provisioner → writes .runtime/engines/pool-0.json
orchestratord → watches filesystem → binds adapter
```

**CLOUD_PROFILE Solution**:
```
engine-provisioner → writes .runtime/engines/pool-0.json (local)
pool-managerd → watches local filesystem → updates registry
orchestratord → polls pool-managerd HTTP → binds adapter
```

**Implementation**:

**pool-managerd**:
```rust
// Watches local filesystem (same machine as engine-provisioner)
pub fn spawn_handoff_watcher(registry: Arc<Mutex<Registry>>) {
    tokio::spawn(async move {
        loop {
            // Scan .runtime/engines/*.json
            // Update registry when new handoff detected
            // Mark pool as ready
        }
    });
}
```

**orchestratord**:
```rust
// Polls pool-managerd via HTTP
pub fn spawn_pool_health_poller(state: AppState) {
    tokio::spawn(async move {
        loop {
            // GET http://gpu-worker-1:9200/v2/pools/{id}/status
            // If ready=true and not bound, bind adapter
        }
    });
}
```

### 2. Task Dispatch Flow

**Sequence**:
```
1. Client → POST /v2/tasks → orchestratord
2. orchestratord → admission queue → placement decision
3. orchestratord → GET /v2/pools/{id}/status → pool-managerd
4. orchestratord → bind adapter (if not already bound)
5. orchestratord → dispatch to adapter → engine HTTP endpoint
6. orchestratord → SSE stream → Client
```

**Key Points**:
- orchestratord never talks to engines directly
- Adapters abstract engine communication
- pool-managerd provides pool status via HTTP

### 3. Health Monitoring

**Polling Pattern**:
```
orchestratord → GET /v2/pools/{id}/status (every 5s)
pool-managerd → responds with:
  - live: bool (is pool process running)
  - ready: bool (has capacity)
  - slots_free: i32
  - vram_free_bytes: i64
```

**Callback Pattern** (future optimization):
```
orchestratord → POST /v2/callbacks/register
pool-managerd → stores callback URL
pool-managerd → POST http://orchestratord:8080/callbacks/pool-ready
orchestratord → binds adapter immediately
```

---

## Configuration

### Environment Variables

#### orchestratord (Control Plane)

```bash
# Service binding
ORCHD_BIND_ADDR=0.0.0.0:8080

# pool-managerd endpoints (comma-separated)
ORCHD_POOL_MANAGERS=http://gpu-worker-1:9200,http://gpu-worker-2:9200

# Polling configuration
ORCHD_POOL_POLL_INTERVAL_MS=5000

# Admission queue
ORCHD_ADMISSION_CAPACITY=100
ORCHD_ADMISSION_POLICY=drop-lru

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
PROMETHEUS_METRICS_PORT=9090

# Profile mode
ORCHD_PROFILE=cloud  # or "home"
```

#### pool-managerd (GPU Worker)

```bash
# Service binding
POOL_MANAGERD_BIND_ADDR=0.0.0.0:9200

# Handoff watcher
POOL_MANAGERD_RUNTIME_DIR=/var/lib/llama-orch/engines
POOL_MANAGERD_WATCH_INTERVAL_MS=1000
POOL_MANAGERD_AUTO_DELETE_HANDOFF=true

# GPU discovery
POOL_MANAGERD_GPU_DISCOVERY_INTERVAL_MS=10000

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
PROMETHEUS_METRICS_PORT=9091

# Callback (optional)
POOL_MANAGERD_ORCHESTRATORD_CALLBACK_URL=http://orchestratord:8080/callbacks/pool-ready
```

#### engine-provisioner (GPU Worker)

```bash
# Handoff output directory
ENGINE_PROVISIONER_HANDOFF_DIR=/var/lib/llama-orch/engines

# Engine download cache
ENGINE_PROVISIONER_CACHE_DIR=/var/cache/llama-orch

# GPU selection
ENGINE_PROVISIONER_GPU_MASK=0,1  # Use GPUs 0 and 1

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
```

---

## Deployment Models

### Kubernetes

**Namespace**: `llama-orch`

**Deployments**:

```yaml
# Control Plane
apiVersion: apps/v1
kind: Deployment
metadata:
  name: orchestratord
spec:
  replicas: 3  # HA
  selector:
    matchLabels:
      app: orchestratord
  template:
    spec:
      containers:
      - name: orchestratord
        image: llama-orch/orchestratord:v0.2.0
        env:
        - name: ORCHD_PROFILE
          value: "cloud"
        - name: ORCHD_POOL_MANAGERS
          value: "http://pool-managerd-0:9200,http://pool-managerd-1:9200"
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 2
            memory: 4Gi
---
# GPU Worker
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: pool-managerd
spec:
  replicas: 2  # One per GPU node
  selector:
    matchLabels:
      app: pool-managerd
  template:
    spec:
      nodeSelector:
        gpu: "true"
      containers:
      - name: pool-managerd
        image: llama-orch/pool-managerd:v0.2.0
        env:
        - name: POOL_MANAGERD_RUNTIME_DIR
          value: "/var/lib/llama-orch/engines"
        ports:
        - containerPort: 9200
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: runtime
          mountPath: /var/lib/llama-orch
      - name: engine-provisioner
        image: llama-orch/engine-provisioner:v0.2.0
        env:
        - name: ENGINE_PROVISIONER_HANDOFF_DIR
          value: "/var/lib/llama-orch/engines"
        volumeMounts:
        - name: runtime
          mountPath: /var/lib/llama-orch
  volumeClaimTemplates:
  - metadata:
      name: runtime
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

### Docker Compose

**File**: `docker-compose.cloud.yml`

```yaml
version: '3.8'

services:
  orchestratord:
    image: llama-orch/orchestratord:v0.2.0
    ports:
      - "8080:8080"
    environment:
      ORCHD_PROFILE: cloud
      ORCHD_POOL_MANAGERS: http://pool-managerd-1:9200,http://pool-managerd-2:9200
      ORCHD_POOL_POLL_INTERVAL_MS: 5000
    networks:
      - llama-orch
    deploy:
      replicas: 2

  pool-managerd-1:
    image: llama-orch/pool-managerd:v0.2.0
    ports:
      - "9201:9200"
    environment:
      POOL_MANAGERD_RUNTIME_DIR: /var/lib/llama-orch/engines
    volumes:
      - gpu1-runtime:/var/lib/llama-orch
    networks:
      - llama-orch
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  engine-provisioner-1:
    image: llama-orch/engine-provisioner:v0.2.0
    environment:
      ENGINE_PROVISIONER_HANDOFF_DIR: /var/lib/llama-orch/engines
      ENGINE_PROVISIONER_GPU_MASK: "0"
    volumes:
      - gpu1-runtime:/var/lib/llama-orch
    networks:
      - llama-orch
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  pool-managerd-2:
    image: llama-orch/pool-managerd:v0.2.0
    ports:
      - "9202:9200"
    environment:
      POOL_MANAGERD_RUNTIME_DIR: /var/lib/llama-orch/engines
    volumes:
      - gpu2-runtime:/var/lib/llama-orch
    networks:
      - llama-orch
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

  engine-provisioner-2:
    image: llama-orch/engine-provisioner:v0.2.0
    environment:
      ENGINE_PROVISIONER_HANDOFF_DIR: /var/lib/llama-orch/engines
      ENGINE_PROVISIONER_GPU_MASK: "1"
    volumes:
      - gpu2-runtime:/var/lib/llama-orch
    networks:
      - llama-orch
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

volumes:
  gpu1-runtime:
  gpu2-runtime:

networks:
  llama-orch:
```

---

## Service Discovery

### Static Configuration (v0.2.0)

**Method**: Environment variables with comma-separated URLs

```bash
ORCHD_POOL_MANAGERS=http://gpu-1:9200,http://gpu-2:9200,http://gpu-3:9200
```

**Pros**:
- Simple
- No dependencies
- Works everywhere

**Cons**:
- Manual updates when scaling
- No automatic failover

### DNS-Based (v0.3.0)

**Method**: DNS SRV records or headless services

```bash
ORCHD_POOL_MANAGERS_DNS=_pool-managerd._tcp.llama-orch.svc.cluster.local
```

**Pros**:
- Automatic discovery
- Kubernetes-native
- Scales automatically

**Cons**:
- Requires DNS infrastructure
- More complex

### Consul/etcd (v1.0.0)

**Method**: Service registry with health checks

**Pros**:
- Dynamic registration
- Health-aware routing
- Multi-datacenter support

**Cons**:
- Additional infrastructure
- Operational complexity

---

## Failure Modes & Recovery

### 1. orchestratord Failure

**Impact**: Client requests fail, no new tasks admitted

**Recovery**:
- Load balancer fails over to healthy replica
- In-flight tasks continue (adapters still bound)
- Admission queue lost (acceptable - clients retry)

**Mitigation**:
- Run 3+ replicas
- Use Kubernetes liveness/readiness probes
- Stateless design enables fast recovery

### 2. pool-managerd Failure

**Impact**: One GPU node unavailable, tasks cannot be routed there

**Recovery**:
- orchestratord detects via health check timeout
- Routes new tasks to healthy nodes
- In-flight tasks on that node fail (clients see error)

**Mitigation**:
- Health checks with timeout
- Automatic retry to different pool
- Client-side retry logic

### 3. engine-provisioner Failure

**Impact**: No new engines provisioned on that node

**Recovery**:
- Restart engine-provisioner
- Re-scans GPUs and provisions engines
- Writes new handoff files
- pool-managerd detects and updates registry

**Mitigation**:
- Systemd auto-restart
- Kubernetes restart policy
- Idempotent provisioning

### 4. Network Partition

**Impact**: orchestratord cannot reach pool-managerd

**Recovery**:
- Health checks timeout
- orchestratord marks pool as unavailable
- Routes tasks to reachable pools
- When network recovers, resumes routing

**Mitigation**:
- Multi-region deployment
- Network redundancy
- Circuit breaker pattern

---

## Security

### 1. mTLS Between Services

**Requirement**: All inter-service communication MUST use mTLS

```bash
# orchestratord
ORCHD_TLS_CERT=/etc/llama-orch/certs/orchestratord.crt
ORCHD_TLS_KEY=/etc/llama-orch/certs/orchestratord.key
ORCHD_TLS_CA=/etc/llama-orch/certs/ca.crt

# pool-managerd
POOL_MANAGERD_TLS_CERT=/etc/llama-orch/certs/pool-managerd.crt
POOL_MANAGERD_TLS_KEY=/etc/llama-orch/certs/pool-managerd.key
POOL_MANAGERD_TLS_CA=/etc/llama-orch/certs/ca.crt
```

### 2. API Authentication

**Requirement**: Client requests MUST be authenticated

```bash
# orchestratord validates X-API-Key header
ORCHD_API_KEYS=key1,key2,key3

# Or use OAuth2/OIDC
ORCHD_OIDC_ISSUER=https://auth.example.com
ORCHD_OIDC_AUDIENCE=llama-orch
```

### 3. Network Policies

**Kubernetes NetworkPolicy**:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: orchestratord
spec:
  podSelector:
    matchLabels:
      app: orchestratord
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: pool-managerd
    ports:
    - protocol: TCP
      port: 9200
```

---

## Observability

### 1. Distributed Tracing

**Tool**: OpenTelemetry + Tempo/Jaeger

**Trace Spans**:
```
Client Request
├─ orchestratord: admission
├─ orchestratord: placement
├─ orchestratord: pool_health_check
│  └─ pool-managerd: get_status
├─ orchestratord: adapter_bind
├─ orchestratord: dispatch
│  └─ adapter: forward_to_engine
│     └─ llama.cpp: completion
└─ orchestratord: sse_stream
```

**Implementation**:
```rust
use opentelemetry::trace::{Tracer, SpanKind};

async fn dispatch_task(task_id: &str) {
    let tracer = global::tracer("orchestratord");
    let span = tracer
        .span_builder("dispatch_task")
        .with_kind(SpanKind::Client)
        .with_attributes(vec![
            KeyValue::new("task_id", task_id.to_string()),
        ])
        .start(&tracer);
    
    let _guard = span.enter();
    // ... dispatch logic ...
}
```

### 2. Metrics

**orchestratord Metrics**:
```
# Admission
orchd_admission_queue_depth{pool_id}
orchd_admission_enqueued_total{pool_id, outcome}
orchd_admission_rejected_total{pool_id, reason}

# Pool Health
orchd_pool_health_checks_total{pool_id, outcome}
orchd_pool_health_check_duration_ms{pool_id}
orchd_pools_available{pool_id}

# Dispatch
orchd_tasks_dispatched_total{pool_id, outcome}
orchd_dispatch_latency_ms{pool_id}
```

**pool-managerd Metrics**:
```
# Handoff Watcher
pool_handoff_files_processed_total{pool_id, outcome}
pool_handoff_processing_duration_ms{pool_id}

# Pool State
pool_slots_total{pool_id}
pool_slots_free{pool_id}
pool_vram_total_bytes{pool_id}
pool_vram_free_bytes{pool_id}

# Health
pool_health_checks_total{pool_id, outcome}
pool_ready{pool_id}  # 0 or 1
```

### 3. Logging

**Structured Logging**:
```rust
tracing::info!(
    target: "orchestratord::dispatch",
    task_id = %task_id,
    pool_id = %pool_id,
    correlation_id = %corr_id,
    latency_ms = dispatch_latency,
    "task dispatched successfully"
);
```

**Log Aggregation**: Loki, Elasticsearch, CloudWatch

---

## Performance Characteristics

### Latency Budget

**Target**: P99 < 100ms for admission + placement

**Breakdown**:
- Admission queue: 5ms
- Placement decision: 10ms
- Pool health check: 20ms (cached)
- Adapter binding: 15ms (if needed)
- Dispatch to engine: 30ms
- Engine processing: Variable (model-dependent)
- SSE streaming: 20ms overhead

**Optimization**:
- Cache pool health status (5s TTL)
- Pre-bind adapters on pool ready
- Connection pooling to pool-managerd
- HTTP/2 multiplexing

### Throughput

**Target**: 1000 tasks/sec per orchestratord instance

**Scaling**:
- Horizontal: Add more orchestratord replicas
- Vertical: Increase admission queue capacity
- GPU: Add more GPU worker nodes

---

## Migration from HOME_PROFILE

### Phase 1: Preparation (Week 1)

- [ ] Document all filesystem dependencies
- [ ] Add profile detection (`ORCHD_PROFILE=home|cloud`)
- [ ] Create cloud_profile deployment manifests
- [ ] Set up test environment (2 machines minimum)

### Phase 2: pool-managerd Watcher (Week 2)

- [ ] Implement handoff watcher in pool-managerd
- [ ] Add HTTP endpoints for pool status
- [ ] Unit tests for watcher
- [ ] Integration tests with engine-provisioner

### Phase 3: orchestratord Polling (Week 3)

- [ ] Implement HTTP polling in orchestratord
- [ ] Remove filesystem watcher (or make HOME_PROFILE only)
- [ ] Update adapter binding logic
- [ ] E2E tests with real pool-managerd

### Phase 4: Testing & Validation (Week 4)

- [ ] BDD tests with distributed setup
- [ ] Load testing (1000 tasks/sec)
- [ ] Failure injection (chaos engineering)
- [ ] Performance benchmarking

### Phase 5: Production Rollout (Week 5-6)

- [ ] Deploy to staging environment
- [ ] Monitor metrics and logs
- [ ] Gradual rollout (canary deployment)
- [ ] Documentation and runbooks

---

## Refinement Opportunities

1. **Service Mesh Integration**: Istio/Linkerd for mTLS, retries, circuit breaking
2. **Callback Optimization**: Reduce polling overhead with webhooks
3. **Multi-Region**: Deploy across geographic regions for low latency
4. **Auto-Scaling**: HPA based on queue depth and GPU utilization
5. **Cost Optimization**: Spot instances for GPU workers, reserved for control plane
6. **Advanced Placement**: ML-based placement considering GPU memory, model size, load
7. **Persistent Queue**: Redis/PostgreSQL for admission queue durability
8. **GraphQL API**: More flexible client queries
9. **gRPC**: Lower latency than HTTP/JSON for inter-service communication
10. **Edge Deployment**: Run orchestratord closer to clients (CDN edge)

---

## References

- **HOME_PROFILE**: `.specs/00_home_profile.md`
- **orchestratord**: `.specs/20_orchestratord.md`
- **pool-managerd**: `.specs/30_pool-managerd.md`
- **Handoff Issue**: `bin/orchestratord/HANDOFF_WATCHER_ARCHITECTURE_ISSUE.md`
- **Resolution**: `bin/orchestratord/HANDOFF_WATCHER_RESOLUTION.md`

---

**Status**: DRAFT - Awaiting implementation  
**Target Version**: v0.2.0  
**Owner**: Architecture Team  
**Last Updated**: 2025-09-30
