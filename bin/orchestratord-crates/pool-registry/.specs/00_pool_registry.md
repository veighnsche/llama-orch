# Pool Registry SPEC — Pool Manager Tracking (POOLREG-19xxx)

**Status**: Draft  
**Applies to**: `bin/orchestratord-crates/pool-registry/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `pool-registry` crate tracks available pool managers for orchestratord. It maintains the registry of pools, their state, and worker availability.

**Why it exists:**
- Orchestratord needs to know which pool managers are available
- Need current state for scheduling decisions (GPU VRAM, workers)
- Track pool health and detect failures

**What it does:**
- Maintain registry of registered pool managers
- Accept pool registrations and heartbeats
- Track pool state (GPUs, workers, VRAM)
- Detect pool failures (missed heartbeats)
- Query pools for scheduling

**What it does NOT do:**
- ❌ Make scheduling decisions (scheduling crate does this)
- ❌ Command pools (orchestratord does this via HTTP)
- ❌ Spawn workers (pool managers do this)

---

## 1. Core Responsibilities

### [POOLREG-19001] Pool Registration
The crate MUST accept pool registrations.

### [POOLREG-19002] Pool State Tracking
The crate MUST track current state of all pools (GPUs, workers, VRAM).

### [POOLREG-19003] Heartbeat Processing
The crate MUST process pool heartbeats and update state.

### [POOLREG-19004] Failure Detection
The crate MUST detect pool failures (missed heartbeats).

---

## 2. Pool Registration

### [POOLREG-19010] Registration Handler
Accept pool registration:
```rust
pub fn register_pool(&mut self, registration: PoolRegistration) -> Result<()> {
    let pool = Pool {
        pool_id: registration.pool_id,
        endpoint: registration.endpoint,
        gpus: registration.gpus,
        status: PoolStatus::Healthy,
        last_heartbeat: Utc::now(),
        registered_at: Utc::now(),
    };
    
    self.pools.insert(pool.pool_id.clone(), pool);
    Ok(())
}
```

### [POOLREG-19011] Pool Metadata
Store pool metadata:
```rust
pub struct Pool {
    pub pool_id: String,
    pub endpoint: String,         // e.g., "http://192.168.1.100:9200"
    pub node_id: String,
    pub gpus: Vec<GpuInfo>,
    pub status: PoolStatus,
    pub last_heartbeat: DateTime<Utc>,
    pub registered_at: DateTime<Utc>,
    pub version: String,
}
```

---

## 3. State Tracking

### [POOLREG-19020] Pool State
Track dynamic state per pool:
```rust
pub struct PoolState {
    pub pool_id: String,
    pub gpus: Vec<GpuState>,      // Current VRAM, workers per GPU
    pub workers: Vec<WorkerInfo>,  // All workers in this pool
    pub status: PoolStatus,
}

pub struct GpuState {
    pub id: u32,
    pub total_vram_gb: u64,
    pub available_vram_gb: u64,
    pub workers: Vec<String>,      // Worker IDs on this GPU
    pub temperature_celsius: u32,
}
```

### [POOLREG-19021] State Updates
Update state on every heartbeat:
```rust
pub fn update_pool_state(&mut self, pool_id: &str, state: PoolState) {
    if let Some(pool) = self.pools.get_mut(pool_id) {
        pool.gpus = state.gpus;
        pool.workers = state.workers;
        pool.last_heartbeat = Utc::now();
        pool.status = PoolStatus::Healthy;
    }
}
```

---

## 4. Heartbeat Processing

### [POOLREG-19030] Heartbeat Handler
Process heartbeat:
```rust
pub fn process_heartbeat(&mut self, pool_id: &str, heartbeat: Heartbeat) -> Result<()> {
    let pool = self.pools.get_mut(pool_id)
        .ok_or(PoolRegistryError::PoolNotFound)?;
    
    pool.last_heartbeat = Utc::now();
    pool.status = PoolStatus::Healthy;
    
    // Update GPU state
    for gpu_state in heartbeat.gpus {
        pool.update_gpu_state(gpu_state);
    }
    
    Ok(())
}
```

### [POOLREG-19031] Heartbeat Timeout
Detect missed heartbeats:
- Expected interval: 10s
- Timeout threshold: 30s (3 missed heartbeats)
- Mark pool as `Unhealthy` after timeout

### [POOLREG-19032] Health Check Loop
Background task to check pool health:
```rust
async fn health_check_loop(&mut self) {
    loop {
        tokio::time::sleep(Duration::from_secs(10)).await;
        
        for pool in self.pools.values_mut() {
            let since_last_heartbeat = Utc::now() - pool.last_heartbeat;
            if since_last_heartbeat > Duration::from_secs(30) {
                pool.status = PoolStatus::Unhealthy;
                tracing::warn!(pool_id = %pool.pool_id, "Pool missed heartbeats");
            }
        }
    }
}
```

---

## 5. Query Interface

### [POOLREG-19040] Query All Pools
```rust
pub fn get_all_pools(&self) -> Vec<&Pool> {
    self.pools.values().collect()
}
```

### [POOLREG-19041] Query Healthy Pools
```rust
pub fn get_healthy_pools(&self) -> Vec<&Pool> {
    self.pools.values()
        .filter(|p| p.status == PoolStatus::Healthy)
        .collect()
}
```

### [POOLREG-19042] Query by Model
```rust
pub fn get_pools_with_model(&self, model_ref: &str) -> Vec<&Pool> {
    self.pools.values()
        .filter(|p| p.has_model_loaded(model_ref))
        .collect()
}
```

### [POOLREG-19043] Query by VRAM
```rust
pub fn get_pools_with_vram(&self, min_vram_gb: u64) -> Vec<&Pool> {
    self.pools.values()
        .filter(|p| p.max_available_vram_gb() >= min_vram_gb)
        .collect()
}
```

---

## 6. Pool Status

### [POOLREG-19050] Status Enum
```rust
pub enum PoolStatus {
    Healthy,      // Receiving heartbeats
    Unhealthy,    // Missed heartbeats
    Draining,     // Finishing jobs, won't accept new ones
    Offline,      // Explicitly deregistered
}
```

### [POOLREG-19051] Status Transitions
- Healthy → Unhealthy (missed heartbeats)
- Unhealthy → Healthy (heartbeat resumed)
- Healthy → Draining (explicit drain command)
- Draining → Offline (shutdown complete)

---

## 7. Pool Deregistration

### [POOLREG-19060] Graceful Deregistration
```rust
pub fn deregister_pool(&mut self, pool_id: &str) {
    if let Some(pool) = self.pools.get_mut(pool_id) {
        pool.status = PoolStatus::Offline;
        tracing::info!(pool_id = %pool_id, "Pool deregistered");
    }
}
```

### [POOLREG-19061] Cleanup
Remove offline pools after grace period (e.g., 5 minutes).

---

## 8. Metrics

### [POOLREG-19070] Metrics
```rust
pub struct PoolRegistryMetrics {
    pub pools_registered_total: Counter,
    pub pools_healthy: Gauge,
    pub pools_unhealthy: Gauge,
    pub heartbeats_received_total: Counter,
    pub heartbeat_latency_ms: Histogram,
}
```

---

## 9. Dependencies

### [POOLREG-19080] Required Crates
```toml
[dependencies]
tokio = { workspace = true }
serde = { workspace = true, features = ["derive"] }
chrono = { workspace = true }
tracing = { workspace = true }
thiserror = { workspace = true }
```

---

## 10. Traceability

**Code**: `bin/orchestratord-crates/pool-registry/src/`  
**Tests**: `bin/orchestratord-crates/pool-registry/tests/`  
**Parent**: `bin/orchestratord/.specs/00_orchestratord.md`  
**Used by**: `orchestratord`, `scheduling`  
**Spec IDs**: POOLREG-19001 to POOLREG-19080

---

**End of Specification**
