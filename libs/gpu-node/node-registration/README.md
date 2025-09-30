# node-registration

Node registration and heartbeat for GPU nodes in CLOUD_PROFILE.

## Purpose

GPU nodes (running pool-managerd) register with the control plane (orchestratord) on startup and send periodic heartbeats to report health.

## Usage

```rust
use node_registration::{NodeRegistration, NodeRegistrationConfig};
use pool_registry_types::NodeCapabilities;

// Configure registration
let config = NodeRegistrationConfig {
    node_id: "gpu-node-1".to_string(),
    machine_id: "machine-alpha".to_string(),
    address: "http://192.168.1.100:9200".to_string(),
    orchestratord_url: "http://192.168.1.1:8080".to_string(),
    pools: vec!["pool-0".to_string(), "pool-1".to_string()],
    capabilities: NodeCapabilities { /* ... */ },
    heartbeat_interval_secs: 10,
    api_token: Some("secret".to_string()),
};

// Register on startup
let registration = NodeRegistration::new(config);
registration.register().await?;

// Spawn heartbeat task
let get_pool_status = || {
    // Return current pool status
    vec![/* ... */]
};
let heartbeat_handle = registration.spawn_heartbeat(get_pool_status);

// On shutdown
registration.deregister().await?;
```

## Flow

```
pool-managerd startup
    │
    └─> NodeRegistration::register()
            │
            └─> POST /v2/nodes/register
                    │
                    └─> orchestratord adds to registry
                            │
                            └─> response: success

pool-managerd running
    │
    └─> spawn_heartbeat() every 10s
            │
            └─> POST /v2/nodes/{id}/heartbeat
                    │
                    └─> orchestratord updates last_heartbeat_ms

pool-managerd shutdown
    │
    └─> NodeRegistration::deregister()
            │
            └─> DELETE /v2/nodes/{id}
```

## Specifications

- **CLOUD-2001**: Register on startup
- **CLOUD-2002**: Registration payload (node_id, address, pools, capabilities)
- **CLOUD-2010**: Heartbeat mechanism
- **CLOUD-2011**: Heartbeat interval (configurable, default 10s)
- **CLOUD-2012**: Heartbeat payload (timestamp, pool status)
- **CLOUD-2020**: Deregister on shutdown

## References

- `.specs/01_cloud_profile_.md` - Service discovery
- `libs/control-plane/service-registry` - Control plane counterpart
