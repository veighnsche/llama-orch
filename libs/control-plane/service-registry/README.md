# service-registry

Service registry for tracking GPU nodes in CLOUD_PROFILE deployments.

## Purpose

Maintains authoritative state of which GPU nodes are online, their capabilities, and pool availability. Used by orchestratord for multi-node placement decisions.

## Features

- **Node Registration**: GPU nodes register on startup
- **Heartbeat Monitoring**: Detect offline nodes via heartbeat timeout
- **Stale Detection**: Automatic offline marking for missed heartbeats
- **Pool Mapping**: Track which pools exist on which nodes
- **Query API**: Find nodes by ID, pool, or online status

## Usage

```rust
use service_registry::{ServiceRegistry, RegisterRequest};
use pool_registry_types::{NodeInfo, NodeCapabilities};

// Create registry with 30s heartbeat timeout
let registry = ServiceRegistry::new(30_000);

// Register a node
let node = NodeInfo::new(
    "gpu-node-1".to_string(),
    "machine-alpha".to_string(),
    "http://192.168.1.100:9200".to_string(),
    vec!["pool-0".to_string(), "pool-1".to_string()],
    NodeCapabilities { /* ... */ },
);
registry.register(node)?;

// Process heartbeat
registry.heartbeat("gpu-node-1")?;

// Get online nodes for placement
let online_nodes = registry.get_online_nodes();

// Spawn stale checker task
use service_registry::heartbeat::spawn_stale_checker;
let checker = spawn_stale_checker(registry.clone(), 10);
```

## Architecture

```
GPU Node 1 ──register──> ServiceRegistry
     │                        │
     └────heartbeat(10s)──────┤
                              │
GPU Node 2 ──register────────>│
     │                        │
     └────heartbeat(10s)──────┤
                              │
                     Stale Checker (background task)
                              │
                     orchestratord queries
```

## Specifications

- **CLOUD-2001**: Node registration endpoint
- **CLOUD-2010**: Heartbeat mechanism
- **CLOUD-2013**: Offline detection (30s timeout)
- **CLOUD-2020**: Graceful deregistration

## References

- `.specs/01_cloud_profile_.md` - Service discovery requirements
- `.docs/CLOUD_PROFILE_MIGRATION.md` - Phase 2 implementation
