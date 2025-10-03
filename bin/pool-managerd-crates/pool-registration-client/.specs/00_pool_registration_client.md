# Pool Registration Client SPEC — Orchestrator Registration (POOLREG-18xxx)

**Status**: Draft  
**Applies to**: `bin/pool-managerd-crates/pool-registration-client/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `pool-registration-client` crate handles registration of pool manager with orchestratord or platform. It maintains heartbeat and reports pool availability.

**Why it exists:**
- Orchestratord needs to discover available pool managers
- Platform needs to track registered providers
- Need continuous heartbeat to detect pool failures

**What it does:**
- Register pool manager with orchestratord/platform at startup
- Send periodic heartbeats with pool state (GPUs, workers, VRAM)
- Handle re-registration on connection loss
- Report pool shutdown gracefully

**What it does NOT do:**
- ❌ Make placement decisions (orchestratord does this)
- ❌ Spawn workers (worker-lifecycle does this)
- ❌ Query GPU state (gpu-inventory does this)

---

## 1. Core Responsibilities

### [POOLREG-18001] Registration
The crate MUST register pool manager with orchestrator at startup.

### [POOLREG-18002] Heartbeat
The crate MUST send periodic heartbeats with pool state.

### [POOLREG-18003] Re-registration
The crate MUST handle re-registration on connection loss.

### [POOLREG-18004] Graceful Shutdown
The crate MUST deregister pool on shutdown.

---

## 2. Registration Flow

### [POOLREG-18010] Initial Registration
At startup, register with orchestrator:

`POST /v2/pools/register`

Request:
```json
{
  "pool_id": "pool-1",
  "endpoint": "http://192.168.1.100:9200",
  "node_id": "gpu-node-1",
  "gpus": [
    {
      "id": 0,
      "model": "RTX 4090",
      "total_vram_gb": 24,
      "compute_capability": "8.9"
    }
  ],
  "region": "EU",
  "version": "0.1.0"
}
```

Response:
```json
{
  "pool_id": "pool-1",
  "status": "registered",
  "heartbeat_interval_ms": 10000
}
```

### [POOLREG-18011] Registration Retry
If registration fails:
- Retry with exponential backoff (1s, 2s, 4s, 8s, max 30s)
- Log registration attempts
- Pool manager can still start (offline mode), but won't receive jobs

---

## 3. Heartbeat

### [POOLREG-18020] Heartbeat Loop
Send heartbeat every N seconds (default 10s):

`POST /v2/pools/{id}/heartbeat`

Request:
```json
{
  "pool_id": "pool-1",
  "gpus": [
    {
      "id": 0,
      "total_vram_gb": 24,
      "available_vram_gb": 8,
      "workers": ["worker-abc"],
      "temperature_celsius": 65
    }
  ],
  "workers_total": 1,
  "uptime_seconds": 3600
}
```

Response:
```json
{
  "status": "healthy",
  "next_heartbeat_ms": 10000
}
```

### [POOLREG-18021] Heartbeat Failure
If heartbeat fails:
- Log warning
- Continue trying (don't deregister)
- After 3 consecutive failures, orchestrator may mark pool as unavailable

### [POOLREG-18022] Dynamic Heartbeat Interval
Orchestrator can adjust heartbeat interval in response:
- Normal: 10s
- High load: 5s (more frequent updates)
- Low activity: 30s (reduce overhead)

---

## 4. State Updates

### [POOLREG-18030] State Snapshot
Heartbeat includes current state:
```rust
pub struct PoolState {
    pub pool_id: String,
    pub gpus: Vec<GpuState>,
    pub workers_total: usize,
    pub workers_ready: usize,
    pub workers_busy: usize,
    pub uptime_seconds: u64,
}
```

### [POOLREG-18031] State Collection
Collect state from other components:
- `gpu-inventory`: GPU VRAM state
- `worker-lifecycle`: Worker registry
- System: Uptime, health

---

## 5. Re-registration

### [POOLREG-18040] Connection Loss
If connection to orchestrator lost:
1. Log error
2. Continue operating (offline mode)
3. Retry registration with backoff
4. Once reconnected, send full state update

### [POOLREG-18041] Orchestrator Restart
If orchestrator restarts:
1. Pool manager detects via heartbeat failure
2. Re-register with full state
3. Resume normal operation

---

## 6. Graceful Shutdown

### [POOLREG-18050] Deregistration
On pool manager shutdown:

`POST /v2/pools/{id}/deregister`

Request:
```json
{
  "pool_id": "pool-1",
  "reason": "graceful_shutdown"
}
```

This tells orchestrator to stop sending jobs to this pool.

### [POOLREG-18051] Shutdown Timeout
If deregistration doesn't complete within 5s, proceed with shutdown anyway.

---

## 7. Platform Mode

### [POOLREG-18060] Platform Registration
For marketplace mode, register with platform instead:

`POST /v2/platform/providers/register`

Request includes pricing, SLA, geo information (see platform-api spec).

### [POOLREG-18061] Dual Registration
Pool manager MAY register with both orchestrator AND platform (for hybrid deployments).

---

## 8. Configuration

### [POOLREG-18070] Config
```yaml
registration:
  orchestrator_url: "http://orchestrator:8080"
  pool_id: "pool-1"
  heartbeat_interval_ms: 10000
  retry_backoff_base_ms: 1000
  retry_backoff_max_ms: 30000
```

---

## 9. Metrics

### [POOLREG-18080] Metrics
The crate MUST emit:
- `registration_attempts_total{outcome}`
- `heartbeats_sent_total{outcome}`
- `heartbeat_latency_ms`
- `connection_status` (gauge: 1=connected, 0=disconnected)

---

## 10. Dependencies

### [POOLREG-18090] Required Crates
```toml
[dependencies]
gpu-inventory = { path = "../gpu-inventory" }
tokio = { workspace = true }
reqwest = { workspace = true }
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
tracing = { workspace = true }
thiserror = { workspace = true }
```

---

## 11. Traceability

**Code**: `bin/pool-managerd-crates/pool-registration-client/src/`  
**Tests**: `bin/pool-managerd-crates/pool-registration-client/tests/`  
**Parent**: `bin/pool-managerd/.specs/00_pool-managerd.md`  
**Used by**: `pool-managerd`  
**Depends on**: `gpu-inventory`  
**Spec IDs**: POOLREG-18001 to POOLREG-18090

---

**End of Specification**
