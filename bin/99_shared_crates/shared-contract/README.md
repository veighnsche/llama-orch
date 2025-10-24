# shared-contract

**TEAM-284**: Common contract types shared between workers and hives

## Purpose

This crate provides the foundation types that both `worker-contract` and `hive-contract` build upon.

## Architecture

```text
shared-contract (common types)
    ↓
    ├─→ worker-contract (worker-specific types)
    └─→ hive-contract (hive-specific types)
```

## What Lives Here

### Status Types (`status.rs`)
- `HealthStatus` - Health state (Healthy, Degraded, Unhealthy)
- `OperationalStatus` - Operational state (Starting, Ready, Busy, Stopping, Stopped)

### Heartbeat Protocol (`heartbeat.rs`)
- `HeartbeatTimestamp` - Consistent timestamp handling
- `HeartbeatPayload` - Trait for all heartbeat types

### Constants (`constants.rs`)
- `HEARTBEAT_INTERVAL_SECS` - 30 seconds
- `HEARTBEAT_TIMEOUT_SECS` - 90 seconds
- `CLEANUP_INTERVAL_SECS` - 60 seconds

### Errors (`error.rs`)
- `ContractError` - Common error types

## Usage

```rust
use shared_contract::{HealthStatus, OperationalStatus, HeartbeatTimestamp};

// Health status
let health = HealthStatus::Healthy;
assert!(health.is_operational());

// Operational status
let status = OperationalStatus::Ready;
assert!(status.is_available());

// Timestamps
let ts = HeartbeatTimestamp::now();
assert!(ts.is_recent(90));
```

## Design Principles

1. **DRY** - No duplication between worker and hive contracts
2. **Type Safety** - Strong types with helper methods
3. **Consistency** - Same types used everywhere
4. **Testability** - Comprehensive unit tests
5. **Documentation** - Every type is documented

## Dependencies

- `serde` - Serialization
- `chrono` - Timestamp handling
- `thiserror` - Error definitions
