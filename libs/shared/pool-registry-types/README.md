# pool-registry-types

**Shared types for pool registry communication in multi-node deployments**

`libs/shared/pool-registry-types` — Common data types used by orchestratord (control plane) and pool-managerd (GPU nodes).

---

## What This Library Does

pool-registry-types provides **shared type definitions** for llama-orch:

- **Node types** — NodeInfo, NodeCapabilities, NodeStatus
- **Pool types** — PoolSnapshot, PoolMetadata
- **Health types** — HealthStatus, HealthState
- **GPU types** — GpuInfo, GpuCapabilities
- **Serializable** — All types support serde for HTTP/JSON transport
- **No business logic** — Pure data types only

**Used by**: `orchestratord`, `pool-managerd`, `service-registry`, `node-registration`, `handoff-watcher`

---

## Key Types

### NodeInfo

```rust
use pool_registry_types::NodeInfo;

pub struct NodeInfo {
    pub node_id: String,
    pub hostname: String,
    pub endpoint: String,
    pub pools: Vec<String>,
    pub capabilities: NodeCapabilities,
    pub registered_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
}

impl NodeInfo {
    pub fn is_available(&self) -> bool;
    pub fn is_online(&self) -> bool;
}
```

### NodeCapabilities

```rust
pub struct NodeCapabilities {
    pub gpus: Vec<GpuInfo>,
    pub cpu_cores: Option<usize>,
    pub ram_total_bytes: Option<u64>,
}
```

### GpuInfo

```rust
pub struct GpuInfo {
    pub device_id: usize,
    pub name: String,
    pub vram_total_mb: u64,
    pub cuda_compute_capability: String,
}
```

### HealthStatus

```rust
pub struct HealthStatus {
    pub state: HealthState,
    pub slots_total: usize,
    pub slots_free: usize,
    pub last_check: DateTime<Utc>,
}
```

### HealthState

```rust
pub enum HealthState {
    Initializing,  // Pool starting up
    Ready,         // Pool accepting requests
    Draining,      // Pool finishing existing requests
    Failed,        // Pool unhealthy
    Offline,       // Pool stopped
}
```

### PoolSnapshot

```rust
pub struct PoolSnapshot {
    pub pool_id: String,
    pub node_id: String,
    pub health: HealthStatus,
    pub model: Option<String>,
}
```

---

## Usage Examples

### Create NodeInfo

```rust
use pool_registry_types::{NodeInfo, NodeCapabilities, GpuInfo};
use chrono::Utc;

let node = NodeInfo {
    node_id: "gpu-node-1".to_string(),
    hostname: "machine-alpha".to_string(),
    endpoint: "http://192.168.1.100:9200".to_string(),
    pools: vec!["pool-0".to_string(), "pool-1".to_string()],
    capabilities: NodeCapabilities {
        gpus: vec![
            GpuInfo {
                device_id: 0,
                name: "NVIDIA RTX 4090".to_string(),
                vram_total_mb: 24_000,
                cuda_compute_capability: "8.9".to_string(),
            },
            GpuInfo {
                device_id: 1,
                name: "NVIDIA RTX 4090".to_string(),
                vram_total_mb: 24_000,
                cuda_compute_capability: "8.9".to_string(),
            },
        ],
        cpu_cores: Some(16),
        ram_total_bytes: Some(64_000_000_000),
    },
    registered_at: Utc::now(),
    last_heartbeat: Utc::now(),
};

// Check availability
if node.is_available() {
    println!("Node {} is online and ready", node.node_id);
}
```

### Create HealthStatus

```rust
use pool_registry_types::{HealthStatus, HealthState};
use chrono::Utc;

let health = HealthStatus {
    state: HealthState::Ready,
    slots_total: 4,
    slots_free: 2,
    last_check: Utc::now(),
};

if health.state == HealthState::Ready && health.slots_free > 0 {
    println!("Pool has {} free slots", health.slots_free);
}
```

### Create PoolSnapshot

```rust
use pool_registry_types::{PoolSnapshot, HealthStatus, HealthState};

let snapshot = PoolSnapshot {
    pool_id: "pool-0".to_string(),
    node_id: "gpu-node-1".to_string(),
    health: HealthStatus {
        state: HealthState::Ready,
        slots_total: 4,
        slots_free: 3,
        last_check: Utc::now(),
    },
    model: Some("llama-3.1-8b-instruct".to_string()),
};
```

---

## Serialization

All types support JSON serialization via serde:

```rust
use pool_registry_types::NodeInfo;

// Serialize to JSON
let json = serde_json::to_string(&node)?;

// Deserialize from JSON
let node: NodeInfo = serde_json::from_str(&json)?;
```

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p pool-registry-types -- --nocapture

# Run specific test
cargo test -p pool-registry-types -- test_node_availability --nocapture
```

---

## Dependencies

### Internal

- None (this is a foundational shared library)

### External

- `serde` — Serialization
- `chrono` — Timestamps

---

## Design Goals

1. **Shared**: Used by both control plane and GPU nodes
2. **Serializable**: All types support serde for HTTP transport
3. **Simple**: No business logic, just data types
4. **Tested**: Unit tests for availability checks and serialization
5. **Versioned**: Types can evolve with backward compatibility

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha
- **Maintainers**: @llama-orch-maintainers
