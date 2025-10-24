# hive-contract

**TEAM-284**: Contract definition for hive implementations

## Purpose

Defines the types and protocols that ALL hives must implement. Mirrors `worker-contract` but for hives.

## Architecture

```text
shared-contract (common types)
    ↓
hive-contract (hive-specific types)
    ↓
rbee-hive (implementation)
```

## Types

### HiveInfo (`types.rs`)
Complete hive state:
- `id` - Hive identifier
- `hostname` - IP/hostname
- `port` - HTTP port
- `operational_status` - Ready/Busy/Stopped
- `health_status` - Healthy/Degraded/Unhealthy
- `version` - Hive version

### HiveHeartbeat (`heartbeat.rs`)
Periodic status update:
- `hive` - Full HiveInfo
- `timestamp` - When heartbeat was sent

### API Specification (`api.rs`)
Required HTTP endpoints:
- `GET /health` - Health check
- `GET /capabilities` - Device capabilities
- `POST /workers` - Spawn worker
- `GET /workers` - List workers

## Usage

```rust
use hive_contract::{HiveInfo, HiveHeartbeat};
use shared_contract::{OperationalStatus, HealthStatus};

// Create hive info
let hive = HiveInfo {
    id: "localhost".to_string(),
    hostname: "127.0.0.1".to_string(),
    port: 9200,
    operational_status: OperationalStatus::Ready,
    health_status: HealthStatus::Healthy,
    version: "0.1.0".to_string(),
};

// Create heartbeat
let heartbeat = HiveHeartbeat::new(hive);

// Send to queen
// POST http://queen:8500/v1/hive-heartbeat
```

## Relationship to worker-contract

| Aspect | worker-contract | hive-contract |
|--------|----------------|---------------|
| **Component** | Worker | Hive |
| **Info Type** | WorkerInfo | HiveInfo |
| **Heartbeat** | WorkerHeartbeat | HiveHeartbeat |
| **Endpoint** | `/v1/worker-heartbeat` | `/v1/hive-heartbeat` |
| **Registry** | WorkerRegistry | HiveRegistry |

Both use `shared-contract` for common types.
