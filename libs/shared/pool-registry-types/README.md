# pool-registry-types

Shared types for pool registry communication between orchestratord (control plane) and pool-managerd (GPU nodes).

## Purpose

This crate provides common data types used by both:
- **orchestratord**: Track nodes, make placement decisions
- **pool-managerd**: Report pool health, register with control plane

## Types

### Health Types
- `HealthStatus`: Complete health info for a pool
- `HealthState`: Pool state (Initializing, Ready, Draining, Failed, Offline)

### Node Types
- `NodeInfo`: Complete node information (id, address, capabilities)
- `NodeStatus`: Node status in cluster
- `NodeCapabilities`: Hardware info (GPUs, RAM, CPU)
- `GpuInfo`: Individual GPU details

### Pool Types
- `PoolSnapshot`: Pool state for placement decisions
- `PoolMetadata`: Pool configuration

## Usage

```rust
use pool_registry_types::{NodeInfo, NodeCapabilities, HealthStatus};

// Create node info
let node = NodeInfo::new(
    "node-1".to_string(),
    "machine-1".to_string(),
    "http://192.168.1.100:9200".to_string(),
    vec!["pool-0".to_string(), "pool-1".to_string()],
    NodeCapabilities {
        gpus: vec![/* ... */],
        cpu_cores: Some(16),
        ram_total_bytes: Some(64_000_000_000),
    },
);

// Check availability
if node.is_available() {
    println!("Node is online and ready");
}
```

## Design Goals

1. **Shared**: Used by both control plane and GPU nodes
2. **Serializable**: All types support serde for HTTP transport
3. **Simple**: No business logic, just data types
4. **Tested**: Unit tests for availability checks

## References

- `.specs/01_cloud_profile_.md` - Cloud profile specification
- `.docs/ARCHITECTURE_LIBRARY_ORGANIZATION.md` - Library organization
