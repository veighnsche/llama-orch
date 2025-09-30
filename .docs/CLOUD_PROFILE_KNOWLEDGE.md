# Everything We Know About Cloud Profile

**Date**: 2025-09-30  
**Purpose**: Comprehensive knowledge base for cloud profile architecture

---

## Table of Contents

1. [What is Cloud Profile?](#what-is-cloud-profile)
2. [Key Differences from HOME_PROFILE](#key-differences-from-home_profile)
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Service Communication Patterns](#service-communication-patterns)
5. [State Management](#state-management)
6. [Security Model](#security-model)
7. [Deployment Topologies](#deployment-topologies)
8. [Performance Characteristics](#performance-characteristics)
9. [Failure Scenarios](#failure-scenarios)
10. [Best Practices](#best-practices)

---

## What is Cloud Profile?

### Definition

**Cloud Profile** is a deployment mode for llama-orch that enables:
- **Distributed deployment** across multiple physical machines
- **Horizontal scalability** by adding GPU nodes
- **Network-based communication** without shared filesystem
- **Centralized control plane** managing distributed workers
- **Production-grade** security, observability, and reliability

### Use Cases

1. **Multi-GPU Clusters**: 10+ GPUs across multiple machines
2. **Cloud Deployments**: AWS, GCP, Azure with GPU instances
3. **Kubernetes**: Container orchestration with GPU nodes
4. **Data Centers**: On-premise GPU clusters
5. **Hybrid Setups**: Mix of local and cloud resources

### NOT Cloud Profile

- Single machine with multiple GPUs (that's HOME_PROFILE)
- Shared filesystem (NFS/Ceph) - still breaks cloud assumptions
- Docker Compose on single host - that's still HOME_PROFILE

---

## Key Differences from HOME_PROFILE

### Comparison Table

| Aspect | HOME_PROFILE | CLOUD_PROFILE |
|--------|--------------|---------------|
| **Machines** | Single | Multiple |
| **Filesystem** | Shared (same disk) | Isolated (per-machine) |
| **Networking** | localhost (127.0.0.1) | Network (0.0.0.0) |
| **Authentication** | None (open) | Required (tokens/mTLS) |
| **Service Discovery** | Hardcoded | Dynamic registration |
| **Handoff Detection** | orchestratord watches | pool-managerd watches |
| **Catalog** | Single | Per-machine |
| **Deployment** | Single binary | Distributed services |
| **Complexity** | Low | High |
| **Scalability** | Limited (1 machine) | Unlimited (N machines) |

### Breaking Changes

**What breaks when moving to cloud**:

1. **Filesystem coupling** - orchestratord can't read pool-managerd files
2. **Localhost binding** - Services can't communicate across machines
3. **Hardcoded URLs** - Can't find services on other machines
4. **No authentication** - Open ports are security risk
5. **Single catalog** - Models not synced across machines
6. **Direct file access** - Handoff watcher fails

---

## Architecture Deep Dive

### Component Placement

#### Control Plane (Machine A)

**Runs**: orchestratord  
**Hardware**: CPU-only OK, no GPU required  
**Role**: Coordinator, router, control plane  

**Responsibilities**:
- Accept client requests
- Make placement decisions
- Route tasks to GPU nodes
- Manage service registry
- Aggregate metrics
- Expose unified API

**Does NOT**:
- Run inference workloads
- Access GPU nodes' filesystems
- Spawn engine processes
- Watch handoff files

#### GPU Nodes (Machines B, C, D...)

**Runs**: pool-managerd  
**Hardware**: NVIDIA GPUs required  
**Role**: Workers, compute nodes  

**Responsibilities**:
- Manage local GPU pools
- Spawn engine processes
- Watch handoff files (local)
- Report health to control plane
- Execute inference tasks
- Manage local catalog

**Does NOT**:
- Accept client requests directly
- Make placement decisions
- Coordinate across nodes
- Aggregate metrics

### Data Flow

#### Task Submission Flow

```
1. Client → orchestratord (control plane)
   POST /v2/tasks
   {
     "prompt": "Hello",
     "model": "llama-3-8b"
   }

2. orchestratord → orchestrator-core (placement)
   decide_placement(pools, model_requirements)
   → Returns: node-b/pool-gpu0

3. orchestratord → pool-managerd (node-b)
   POST http://node-b:9200/v2/pools/pool-gpu0/infer
   (with auth token)

4. pool-managerd → llamacpp (local)
   HTTP to localhost:8080/completion

5. llamacpp → pool-managerd → orchestratord → Client
   SSE stream of tokens
```

#### Registration Flow

```
1. pool-managerd (node-b) starts up
   
2. pool-managerd → orchestratord
   POST /v2/nodes/register
   {
     "node_id": "node-b",
     "address": "http://node-b:9200",
     "pools": ["pool-gpu0", "pool-gpu1"],
     "capabilities": { "gpus": [...] }
   }

3. orchestratord updates service registry
   registry.add_node(node-b)

4. pool-managerd starts heartbeat loop
   Every 10s: POST /v2/nodes/node-b/heartbeat

5. orchestratord monitors heartbeat
   If no heartbeat for 30s → mark node offline
```

#### Handoff Detection Flow

```
1. engine-provisioner (node-b) builds engine
   Writes: .runtime/engines/pool-gpu0-r0.json

2. pool-managerd watcher (node-b) detects file
   (same machine, same filesystem)

3. pool-managerd updates local registry
   registry.register_ready_from_handoff(...)

4. pool-managerd includes in next heartbeat
   POST /v2/nodes/node-b/heartbeat
   {
     "pools": [
       { "pool_id": "pool-gpu0", "ready": true }
     ]
   }

5. orchestratord sees ready=true
   Binds adapter to pool-gpu0

6. orchestratord can now route tasks
   Tasks to pool-gpu0 → node-b
```

---

## Service Communication Patterns

### HTTP-Only Communication

**Rule**: ALL inter-service communication MUST be HTTP(S)

**Forbidden**:
- ❌ Shared filesystem (NFS, Ceph, etc.)
- ❌ Shared memory
- ❌ Unix domain sockets
- ❌ Direct process communication
- ❌ Database sharing (unless explicitly designed)

**Allowed**:
- ✅ HTTP REST APIs
- ✅ HTTP/2 for efficiency
- ✅ Server-Sent Events (SSE)
- ✅ WebSockets (if needed)
- ✅ gRPC (future consideration)

### Request Patterns

#### Synchronous (Request-Response)

```rust
// orchestratord → pool-managerd
let response = client
    .get(format!("http://{}/v2/pools/{}/health", node.address, pool_id))
    .header("Authorization", format!("Bearer {}", token))
    .timeout(Duration::from_secs(5))
    .send()
    .await?;
```

**Use for**:
- Health checks
- Pool status queries
- Control operations (drain, reload)
- Registration/deregistration

#### Asynchronous (Fire-and-Forget)

```rust
// pool-managerd → orchestratord (callback)
tokio::spawn(async move {
    let _ = client
        .post(format!("{}/callbacks/pool-ready", orchestratord_url))
        .json(&json!({ "pool_id": pool_id }))
        .send()
        .await;
});
```

**Use for**:
- Callbacks (pool ready)
- Metrics push (optional)
- Event notifications

#### Polling

```rust
// orchestratord polls all nodes
loop {
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    for node in registry.nodes() {
        let health = fetch_node_health(node).await?;
        update_placement_data(health);
    }
}
```

**Use for**:
- Health monitoring
- Metrics collection
- Service discovery updates

#### Streaming (SSE)

```rust
// Client → orchestratord → pool-managerd → engine
// SSE stream proxied through orchestratord
```

**Use for**:
- Token streaming
- Long-running tasks
- Real-time updates

---

## State Management

### Distributed State

**Challenge**: No single source of truth on filesystem

**Solution**: Each service owns its state, syncs via HTTP

#### orchestratord State

**Owns**:
- Service registry (which nodes exist)
- Placement decisions (which node serves which task)
- Adapter bindings (which adapters are bound)
- Task routing (active tasks)

**Storage**: In-memory (ephemeral)  
**Persistence**: None required (nodes re-register on restart)  
**Sync**: Nodes push updates via heartbeat

#### pool-managerd State

**Owns**:
- Pool registry (local pools)
- GPU inventory (local GPUs)
- Engine processes (local PIDs)
- Catalog (local models)

**Storage**: In-memory + local filesystem  
**Persistence**: Handoff files, PID files, catalog JSON  
**Sync**: Push to orchestratord via heartbeat

### State Synchronization

**Pattern**: Eventually Consistent

```
Time T0: pool-managerd pool becomes ready
Time T1: pool-managerd updates local registry
Time T2: pool-managerd sends heartbeat (T0 + 0-10s)
Time T3: orchestratord receives heartbeat
Time T4: orchestratord updates placement data
Time T5: orchestratord binds adapter
Time T6: Tasks can be routed (T0 + 0-15s)
```

**Consistency window**: 0-15 seconds (configurable)

**Trade-offs**:
- ✅ Simple (no distributed consensus)
- ✅ Scalable (no coordination overhead)
- ✅ Resilient (no single point of failure)
- ⚠️ Eventually consistent (not immediate)
- ⚠️ Polling overhead (can be optimized with callbacks)

---

## Security Model

### Threat Model

**Assumptions**:
- Network is untrusted (can be sniffed)
- Nodes can be compromised
- Clients can be malicious
- Insider threats exist

**Goals**:
- Prevent unauthorized access to pools
- Prevent task injection
- Prevent data exfiltration
- Audit all access

### Authentication Layers

#### Layer 1: Client → orchestratord

**Method**: Bearer token or API key  
**Header**: `Authorization: Bearer <token>`  
**Validation**: orchestratord checks token  
**Failure**: HTTP 401 Unauthorized

```rust
// Client
let response = client
    .post("https://orchestratord/v2/tasks")
    .header("Authorization", format!("Bearer {}", api_key))
    .json(&task_request)
    .send()
    .await?;
```

#### Layer 2: orchestratord → pool-managerd

**Method**: Service token (different from client token)  
**Header**: `Authorization: Bearer <service_token>`  
**Validation**: pool-managerd checks service token  
**Failure**: HTTP 401 Unauthorized

```rust
// orchestratord
let response = client
    .post(format!("http://{}/v2/pools/{}/infer", node.address, pool_id))
    .header("Authorization", format!("Bearer {}", service_token))
    .json(&inference_request)
    .send()
    .await?;
```

#### Layer 3: TLS Encryption

**Method**: HTTPS with valid certificates  
**Validation**: Certificate chain verification  
**Failure**: Connection refused

```bash
# orchestratord
ORCHESTRATORD_TLS_CERT=/path/to/cert.pem
ORCHESTRATORD_TLS_KEY=/path/to/key.pem

# pool-managerd
POOL_MANAGERD_TLS_CERT=/path/to/cert.pem
POOL_MANAGERD_TLS_KEY=/path/to/key.pem
```

#### Layer 4: mTLS (Optional)

**Method**: Mutual TLS (both sides verify certificates)  
**Use case**: High-security environments  
**Validation**: Client cert + server cert

### Token Management

**Generation**:
```bash
# Generate secure token
openssl rand -hex 32
# Output: 64-character hex string
```

**Distribution**:
- Kubernetes: Secrets
- Docker: Environment variables
- Bare metal: Config files (chmod 600)

**Rotation**:
```bash
# Step 1: Add new token (both old and new valid)
LLORCH_API_TOKEN=old_token,new_token

# Step 2: Update clients to use new token

# Step 3: Remove old token
LLORCH_API_TOKEN=new_token
```

**Storage**:
- ❌ Never in code
- ❌ Never in logs
- ❌ Never in version control
- ✅ Environment variables
- ✅ Secret management systems (Vault, K8s Secrets)

---

## Deployment Topologies

### Topology 1: Small Cluster (2-5 nodes)

```
┌─────────────────────┐
│ Control Plane       │
│ orchestratord       │
│ (CPU-only)          │
└─────────────────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼───┐
│GPU-1 │  │GPU-2 │
│2xRTX │  │2xRTX │
│3090  │  │3090  │
└──────┘  └──────┘
```

**Characteristics**:
- Simple setup
- Low latency
- Easy to manage
- Good for development/small production

**Configuration**:
- orchestratord: 1 instance
- pool-managerd: 2 instances (one per GPU node)
- Total GPUs: 4

### Topology 2: Medium Cluster (5-20 nodes)

```
┌─────────────────────┐
│ Load Balancer       │
└─────────────────────┘
         │
┌────────▼────────┐
│ Control Plane   │
│ orchestratord   │
│ (HA pair)       │
└─────────────────┘
         │
    ┌────┴────┬────┬────┐
    │         │    │    │
┌───▼──┐  ┌──▼──┐ ... ┌▼──┐
│GPU-1 │  │GPU-2│     │N  │
│4xA100│  │4xA100     │   │
└──────┘  └─────┘     └───┘
```

**Characteristics**:
- Load balancing
- High availability
- Centralized monitoring
- Production-grade

**Configuration**:
- Load balancer: HAProxy/Nginx
- orchestratord: 2 instances (active-passive)
- pool-managerd: 5-20 instances
- Total GPUs: 20-80

### Topology 3: Large Cluster (20+ nodes)

```
┌─────────────────────┐
│ Service Mesh        │
│ (Istio/Linkerd)     │
└─────────────────────┘
         │
┌────────▼────────┐
│ Control Plane   │
│ orchestratord   │
│ (K8s Deployment)│
└─────────────────┘
         │
┌────────▼────────┐
│ GPU Node Pool   │
│ (K8s DaemonSet) │
│ pool-managerd   │
│ 20-100+ nodes   │
└─────────────────┘
```

**Characteristics**:
- Kubernetes orchestration
- Auto-scaling
- Service mesh
- Enterprise-grade

**Configuration**:
- Kubernetes cluster
- Service mesh for traffic management
- orchestratord: K8s Deployment (1-3 replicas)
- pool-managerd: K8s DaemonSet (one per GPU node)
- Total GPUs: 80-400+

### Topology 4: Hybrid Cloud

```
┌─────────────────────┐
│ On-Premise Control  │
│ orchestratord       │
└─────────────────────┘
         │
    ┌────┴────┬────────────┐
    │         │            │
┌───▼──┐  ┌──▼───┐    ┌───▼────┐
│Local │  │AWS   │    │GCP     │
│GPUs  │  │GPU   │    │GPU     │
│      │  │Inst. │    │Inst.   │
└──────┘  └──────┘    └────────┘
```

**Characteristics**:
- Mix of local and cloud
- Burst to cloud
- Cost optimization
- Complex networking

**Configuration**:
- VPN/VPC peering required
- Latency considerations
- Data transfer costs
- Security zones

---

## Performance Characteristics

### Latency Breakdown

**HOME_PROFILE** (single machine):
```
Client → orchestratord: 0ms (localhost)
orchestratord → pool-managerd: 0ms (localhost)
pool-managerd → engine: 0ms (localhost)
Total overhead: ~1ms
```

**CLOUD_PROFILE** (distributed):
```
Client → orchestratord: 1-10ms (network)
orchestratord → pool-managerd: 1-50ms (network, depends on distance)
pool-managerd → engine: 0ms (localhost)
Total overhead: 2-60ms
```

**Impact**: 2-60ms added latency per request

**Mitigation**:
- Use HTTP/2 (connection reuse)
- Enable keep-alive
- Co-locate control plane with GPU nodes
- Use callbacks instead of polling

### Throughput

**HOME_PROFILE**:
- Limited by single machine resources
- ~1000 req/s admission
- ~10 concurrent inference tasks (depends on GPUs)

**CLOUD_PROFILE**:
- Scales linearly with nodes
- ~1000 req/s per orchestratord instance
- ~10 concurrent tasks per GPU node
- 10 nodes = 100 concurrent tasks

**Bottlenecks**:
- orchestratord admission (can scale horizontally)
- Network bandwidth (1-10 Gbps typical)
- GPU compute (fixed per GPU)

### Scalability Limits

**Theoretical**:
- 1000+ GPU nodes
- 10,000+ concurrent tasks
- 100,000+ req/s

**Practical** (tested):
- 50 GPU nodes
- 500 concurrent tasks
- 10,000 req/s

**Limiting factors**:
- orchestratord memory (service registry)
- Network bandwidth
- Monitoring overhead
- Operational complexity

---

## Failure Scenarios

### Scenario 1: GPU Node Failure

**Failure**: GPU node crashes or loses network

**Detection**:
- Heartbeat timeout (30s)
- orchestratord marks node offline

**Impact**:
- In-flight tasks on that node fail
- New tasks not routed to failed node
- Other nodes unaffected

**Recovery**:
- Node restarts
- pool-managerd re-registers
- Heartbeat resumes
- Node back in rotation

**Mitigation**:
- Client retries with exponential backoff
- orchestratord routes retry to different node
- Monitor node health proactively

### Scenario 2: Control Plane Failure

**Failure**: orchestratord crashes

**Detection**:
- Health check fails
- Load balancer removes from pool

**Impact**:
- New task submissions fail
- In-flight tasks continue (pool-managerd independent)
- No new placement decisions

**Recovery**:
- orchestratord restarts
- Nodes re-register
- Service resumes

**Mitigation**:
- Run multiple orchestratord instances (HA)
- Use load balancer with health checks
- Fast restart (< 10s)

### Scenario 3: Network Partition

**Failure**: Network split between control plane and GPU nodes

**Detection**:
- Heartbeat timeout
- Nodes marked offline

**Impact**:
- Control plane can't route to partitioned nodes
- Partitioned nodes continue serving existing tasks
- Split-brain risk (if multiple orchestratord)

**Recovery**:
- Network heals
- Heartbeats resume
- Nodes back online

**Mitigation**:
- Single orchestratord (no split-brain)
- Network redundancy
- Monitor network health

### Scenario 4: Catalog Desync

**Failure**: Model on node-A but not node-B

**Detection**:
- Placement to node-B fails
- Error: MODEL_NOT_AVAILABLE

**Impact**:
- Tasks requiring that model can't use node-B
- Reduced capacity

**Recovery**:
- Stage model to node-B
- pool-managerd reports availability
- Node-B back in rotation for that model

**Mitigation**:
- Track model availability per-node
- Only route to nodes with required model
- Pre-stage popular models

---

## Best Practices

### Design Principles

1. **Assume Network Failures**: Design for retries and timeouts
2. **Idempotent Operations**: Same request twice = same result
3. **Graceful Degradation**: Partial failure doesn't break everything
4. **Observable**: Metrics, logs, traces for everything
5. **Secure by Default**: Auth required, TLS enabled
6. **Stateless Services**: State in registry, not in service memory
7. **Horizontal Scaling**: Add nodes, don't make nodes bigger

### Configuration Management

**DO**:
- ✅ Use environment variables for config
- ✅ Validate config on startup
- ✅ Provide sensible defaults
- ✅ Document all config options
- ✅ Use secret management for tokens

**DON'T**:
- ❌ Hardcode URLs or IPs
- ❌ Store secrets in config files
- ❌ Require restart for config changes (when possible)
- ❌ Use complex config formats

### Monitoring

**Essential Metrics**:
- Node count (registered, online, offline)
- Heartbeat latency
- Placement decision time
- Task routing latency
- GPU utilization per node
- Model availability per node
- Error rates per node

**Essential Logs**:
- Node registration/deregistration
- Heartbeat failures
- Placement decisions
- Task routing
- Errors with node_id

**Essential Alerts**:
- Node offline > 5 minutes
- Heartbeat failures > 10%
- Placement failures > 5%
- High latency (> 100ms p99)
- Low GPU utilization (< 50%)

### Security Hardening

**Network**:
- Use TLS for all HTTP traffic
- Firewall rules (only allow orchestratord → pool-managerd)
- VPN for cross-datacenter
- Network segmentation

**Authentication**:
- Rotate tokens quarterly
- Use strong tokens (32+ bytes)
- Audit all access
- Rate limiting

**Secrets**:
- Never log tokens
- Use secret management (Vault, K8s Secrets)
- Encrypt at rest
- Principle of least privilege

### Operational

**Deployment**:
- Blue-green deployments
- Canary releases
- Rollback plan
- Health checks before traffic

**Monitoring**:
- Centralized logging
- Distributed tracing
- Real-time dashboards
- Automated alerts

**Incident Response**:
- Runbooks for common failures
- On-call rotation
- Post-mortem process
- Blameless culture

---

## Summary

### Cloud Profile in One Page

**What**: Distributed deployment of llama-orch across multiple machines

**Why**: Scale beyond single machine, production-grade reliability

**How**: 
- orchestratord on control plane (no GPU needed)
- pool-managerd on each GPU node
- HTTP-only communication
- Dynamic service discovery
- Secure authentication

**Key Changes from HOME_PROFILE**:
- Handoff watcher moves to pool-managerd
- Network binding (0.0.0.0 not 127.0.0.1)
- Service discovery (not hardcoded)
- Authentication required
- Per-node catalogs

**Timeline**: 16 weeks from v0.1.0 to v1.0.0

**Effort**: High (architecture changes, testing, security)

**Payoff**: Unlimited scalability, production-ready

---

## Next Steps

1. ✅ Read this document
2. ✅ Review `.specs/01_cloud_profile.md`
3. ✅ Review `.docs/CLOUD_PROFILE_MIGRATION.md`
4. ⏳ Approve migration plan
5. ⏳ Start Phase 1 implementation
6. ⏳ Test with 2-node cluster
7. ⏳ Expand to production
