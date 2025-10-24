# Migration Guide: Using Generic HeartbeatRegistry

**TEAM-285:** Guide for migrating worker-registry and hive-registry to use the generic implementation.

## Overview

The generic `HeartbeatRegistry<T>` eliminates ~250 LOC of duplication between worker-registry and hive-registry by providing a single, well-tested implementation.

## Step 1: Implement HeartbeatItem Trait

### For Worker Contract

```rust
// In bin/97_contracts/worker-contract/src/lib.rs

use heartbeat_registry::HeartbeatItem;

impl HeartbeatItem for WorkerHeartbeat {
    type Info = WorkerInfo;

    fn id(&self) -> &str {
        &self.worker.id
    }

    fn info(&self) -> Self::Info {
        self.worker.clone()
    }

    fn is_recent(&self) -> bool {
        self.timestamp.is_recent()
    }

    fn is_available(&self) -> bool {
        self.worker.is_available()
    }
}
```

### For Hive Contract

```rust
// In bin/97_contracts/hive-contract/src/lib.rs

use heartbeat_registry::HeartbeatItem;

impl HeartbeatItem for HiveHeartbeat {
    type Info = HiveInfo;

    fn id(&self) -> &str {
        &self.hive.id
    }

    fn info(&self) -> Self::Info {
        self.hive.clone()
    }

    fn is_recent(&self) -> bool {
        self.timestamp.is_recent()
    }

    fn is_available(&self) -> bool {
        self.hive.is_available()
    }
}
```

## Step 2: Update Registry Implementations

### Worker Registry

```rust
// In bin/15_queen_rbee_crates/worker-registry/src/registry.rs

use heartbeat_registry::HeartbeatRegistry;
use worker_contract::WorkerHeartbeat;

pub struct WorkerRegistry {
    inner: HeartbeatRegistry<WorkerHeartbeat>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        Self {
            inner: HeartbeatRegistry::new(),
        }
    }

    // Delegate to generic registry
    pub fn update_worker(&self, heartbeat: WorkerHeartbeat) {
        self.inner.update(heartbeat);
    }

    pub fn get_worker(&self, worker_id: &str) -> Option<WorkerInfo> {
        self.inner.get(worker_id)
    }

    pub fn list_online_workers(&self) -> Vec<WorkerInfo> {
        self.inner.list_online()
    }

    pub fn list_available_workers(&self) -> Vec<WorkerInfo> {
        self.inner.list_available()
    }

    pub fn count_online(&self) -> usize {
        self.inner.count_online()
    }

    pub fn count_available(&self) -> usize {
        self.inner.count_available()
    }

    // ... other methods delegate to self.inner
}
```

### Hive Registry

```rust
// In bin/15_queen_rbee_crates/hive-registry/src/registry.rs

use heartbeat_registry::HeartbeatRegistry;
use hive_contract::HiveHeartbeat;

pub struct HiveRegistry {
    inner: HeartbeatRegistry<HiveHeartbeat>,
}

impl HiveRegistry {
    pub fn new() -> Self {
        Self {
            inner: HeartbeatRegistry::new(),
        }
    }

    // Delegate to generic registry
    pub fn update_hive(&self, heartbeat: HiveHeartbeat) {
        self.inner.update(heartbeat);
    }

    pub fn get_hive(&self, hive_id: &str) -> Option<HiveInfo> {
        self.inner.get(hive_id)
    }

    pub fn list_online_hives(&self) -> Vec<HiveInfo> {
        self.inner.list_online()
    }

    pub fn list_available_hives(&self) -> Vec<HiveInfo> {
        self.inner.list_available()
    }

    pub fn count_online(&self) -> usize {
        self.inner.count_online()
    }

    pub fn count_available(&self) -> usize {
        self.inner.count_available()
    }

    // ... other methods delegate to self.inner
}
```

## Step 3: Update Dependencies

### Worker Registry Cargo.toml

```toml
[dependencies]
heartbeat-registry = { path = "../../99_shared_crates/heartbeat-registry" }
worker-contract = { path = "../../97_contracts/worker-contract" }
```

### Hive Registry Cargo.toml

```toml
[dependencies]
heartbeat-registry = { path = "../../99_shared_crates/heartbeat-registry" }
hive-contract = { path = "../../97_contracts/hive-contract" }
```

## Step 4: Remove Duplicate Code

After migration, you can delete:
- ~150 LOC from worker-registry/src/registry.rs
- ~150 LOC from hive-registry/src/registry.rs
- All duplicate tests (generic registry has comprehensive tests)

## Benefits

### Code Reduction
- **Before:** ~300 LOC across 2 registries
- **After:** ~100 LOC (thin wrappers) + generic registry
- **Savings:** ~200 LOC

### Single Source of Truth
- All heartbeat logic in one place
- Bugs fixed once, benefit both registries
- Easier to add new features

### Type Safety
- Generic implementation with trait bounds
- Compile-time guarantees
- No runtime overhead

### Easier to Extend
- Want a model-registry? Just implement HeartbeatItem
- Want a cluster-registry? Just implement HeartbeatItem
- No code duplication needed

## Testing

The generic registry has comprehensive tests:
- ✅ 12 unit tests
- ✅ 1 doctest
- ✅ All edge cases covered

After migration, run:
```bash
cargo test -p worker-registry
cargo test -p hive-registry
cargo test -p heartbeat-registry
```

## Migration Checklist

- [ ] Implement `HeartbeatItem` for `WorkerHeartbeat`
- [ ] Implement `HeartbeatItem` for `HiveHeartbeat`
- [ ] Update `worker-registry` to use generic registry
- [ ] Update `hive-registry` to use generic registry
- [ ] Add dependencies to Cargo.toml files
- [ ] Run tests to verify behavior unchanged
- [ ] Remove duplicate code
- [ ] Update documentation

## Rollback Plan

If issues arise, the old implementations are preserved in git history. Simply revert the changes to worker-registry and hive-registry.

## Timeline

**Estimated Effort:** 2-3 hours
- Implement traits: 30 minutes
- Update registries: 1 hour
- Testing: 1 hour
- Documentation: 30 minutes

## Questions?

See the generic registry source code for implementation details:
- `bin/99_shared_crates/heartbeat-registry/src/lib.rs`
