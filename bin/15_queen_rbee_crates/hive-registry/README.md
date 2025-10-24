# queen-rbee-hive-registry

**TEAM-284**: Hive registry for queen-rbee

## Purpose

Tracks hive state in RAM based on heartbeats. Mirrors `worker-registry` but for hives.

## Architecture

```text
Hive → POST /v1/hive-heartbeat → Queen → HiveRegistry
```

## Usage

```rust
use queen_rbee_hive_registry::HiveRegistry;
use hive_contract::{HiveInfo, HiveHeartbeat};
use shared_contract::{OperationalStatus, HealthStatus};

let registry = HiveRegistry::new();

// Hive sends heartbeat
let hive = HiveInfo {
    id: "localhost".to_string(),
    hostname: "127.0.0.1".to_string(),
    port: 9200,
    operational_status: OperationalStatus::Ready,
    health_status: HealthStatus::Healthy,
    version: "0.1.0".to_string(),
};

registry.update_hive(HiveHeartbeat::new(hive));

// Query hives
let online = registry.list_online_hives();
let available = registry.list_available_hives();
let count = registry.count_online();

// Cleanup stale entries
let removed = registry.cleanup_stale();
```

## API

### Core Operations
- `new()` - Create empty registry
- `update_hive(heartbeat)` - Update hive from heartbeat
- `get_hive(id)` - Get hive by ID
- `remove_hive(id)` - Remove hive

### Queries
- `list_all_hives()` - All hives (including stale)
- `list_online_hives()` - Hives with recent heartbeats
- `list_available_hives()` - Online + ready hives
- `count_online()` - Count of online hives
- `count_available()` - Count of available hives

### Maintenance
- `cleanup_stale()` - Remove hives with old heartbeats

## Thread Safety

Uses `RwLock` for concurrent access:
- Multiple readers can access simultaneously
- Writers get exclusive access
- No deadlocks or race conditions

## Relationship to worker-registry

| Aspect | worker-registry | hive-registry |
|--------|----------------|---------------|
| **Component** | Worker | Hive |
| **Contract** | worker-contract | hive-contract |
| **Heartbeat** | WorkerHeartbeat | HiveHeartbeat |
| **Info Type** | WorkerInfo | HiveInfo |
| **Endpoint** | `/v1/worker-heartbeat` | `/v1/hive-heartbeat` |

Both use identical patterns and `shared-contract` types.
