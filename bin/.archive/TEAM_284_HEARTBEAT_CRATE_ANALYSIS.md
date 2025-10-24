# TEAM-284: rbee-heartbeat Crate Analysis

**Date:** Oct 24, 2025  
**Status:** 🚨 **MOSTLY DEAD CODE - CAN BE DELETED**

## TL;DR

**The `rbee-heartbeat` crate is now OBSOLETE and can be deleted.**

- ✅ We have `shared-contract` with proper heartbeat types
- ✅ We have `worker-contract` with `WorkerHeartbeat`
- ✅ We have `hive-contract` with `HiveHeartbeat`
- ❌ `rbee-heartbeat` is NOT used by any product code
- ❌ Only referenced by ONE re-export in worker binary (unused)
- ❌ Contains tons of old, obsolete code from TEAM-115, TEAM-151, TEAM-159

## What's in rbee-heartbeat?

### Files (6 total)
1. `lib.rs` - Module declarations and re-exports
2. `types.rs` - `WorkerHeartbeatPayload`, `HiveHeartbeatPayload`, `HealthStatus` (DUPLICATE of shared-contract!)
3. `worker.rs` - `WorkerHeartbeatConfig`, `start_worker_heartbeat_task()` (NOT USED!)
4. `hive.rs` - `HiveHeartbeatConfig`, `start_hive_heartbeat_task()` (NOT USED!)
5. `queen.rs` - `HeartbeatAcknowledgement` (NOT USED!)
6. `traits.rs` - HUGE file with old trait abstractions (NOT USED!)

### What's Actually Used?

**NOTHING in product code!**

The only reference is:
```rust
// bin/30_llm_worker_rbee/src/lib.rs
pub use rbee_heartbeat as heartbeat;
```

But this re-export is **NEVER USED** anywhere. It's dead code.

## Why is it Dead?

### Before TEAM-284
- Workers used `rbee_heartbeat::WorkerHeartbeatPayload` (lightweight, just ID + status)
- This was the OLD system

### After TEAM-284
- Workers use `worker_contract::WorkerHeartbeat` (full WorkerInfo)
- Hives use `hive_contract::HiveHeartbeat` (full HiveInfo)
- Both use `shared_contract` for common types
- **rbee-heartbeat is completely bypassed**

## Detailed Analysis

### types.rs (DUPLICATE!)
```rust
// rbee-heartbeat/src/types.rs
pub struct WorkerHeartbeatPayload {
    pub worker_id: String,
    pub timestamp: String,
    pub health_status: HealthStatus,
}

pub enum HealthStatus {
    Healthy,
    Degraded,
}
```

**This is a DUPLICATE of shared-contract types!**
- `shared-contract` has `HealthStatus` (with Unhealthy too)
- `worker-contract` has `WorkerHeartbeat` (with full WorkerInfo)
- `hive-contract` has `HiveHeartbeat` (with full HiveInfo)

### worker.rs (NOT USED!)
```rust
pub fn start_worker_heartbeat_task(config: WorkerHeartbeatConfig) -> JoinHandle<()>
```

**This is NOT used anywhere!**
- Worker binary has its own `heartbeat.rs` that uses `worker-contract` types
- This old implementation sends lightweight payloads
- We need full `WorkerInfo`, not just worker_id

### hive.rs (NOT USED!)
```rust
pub fn start_hive_heartbeat_task(config: HiveHeartbeatConfig) -> JoinHandle<()>
```

**This is NOT used anywhere!**
- Hive binary has its own `heartbeat.rs` that uses `hive-contract` types
- This old implementation sends lightweight payloads
- We need full `HiveInfo`, not just hive_id

### queen.rs (NOT USED!)
```rust
pub struct HeartbeatAcknowledgement {
    pub acknowledged: bool,
}
```

**This is NOT used anywhere!**
- Queen returns `HttpHeartbeatAcknowledgement` instead
- Different structure (has `status` and `message` fields)

### traits.rs (ANCIENT CODE!)
**216 lines of OLD trait abstractions from TEAM-159!**

Contains:
- `WorkerRegistry` trait (not used)
- `HiveCatalog` trait (not used - we have HiveRegistry now)
- `DeviceDetector` trait (not used)
- `Narrator` trait (not used)
- Tons of structs: `DeviceCapabilities`, `CpuDevice`, `GpuDevice`, etc.

**ALL OF THIS IS OBSOLETE!**

## What Product Code Actually Uses

### Worker Binary
```rust
// bin/30_llm_worker_rbee/src/heartbeat.rs
use worker_contract::{WorkerHeartbeat, WorkerInfo};

pub async fn send_heartbeat_to_queen(
    worker_info: &WorkerInfo,
    queen_url: &str,
) -> Result<()> {
    let heartbeat = WorkerHeartbeat::new(worker_info.clone());
    // Send to queen
}
```

**Uses:** `worker-contract`, NOT `rbee-heartbeat`

### Hive Binary
```rust
// bin/20_rbee_hive/src/heartbeat.rs
use hive_contract::{HiveHeartbeat, HiveInfo};

pub async fn send_heartbeat_to_queen(
    hive_info: &HiveInfo,
    queen_url: &str,
) -> Result<()> {
    let heartbeat = HiveHeartbeat::new(hive_info.clone());
    // Send to queen
}
```

**Uses:** `hive-contract`, NOT `rbee-heartbeat`

### Queen Binary
```rust
// bin/10_queen_rbee/src/http/heartbeat.rs
use worker_contract::WorkerHeartbeat;
use hive_contract::HiveHeartbeat;

pub async fn handle_worker_heartbeat(
    Json(heartbeat): Json<WorkerHeartbeat>,
) -> Result<...> {
    state.worker_registry.update_worker(heartbeat);
}

pub async fn handle_hive_heartbeat(
    Json(heartbeat): Json<HiveHeartbeat>,
) -> Result<...> {
    state.hive_registry.update_hive(heartbeat);
}
```

**Uses:** `worker-contract` and `hive-contract`, NOT `rbee-heartbeat`

## Recommendation

### DELETE THE ENTIRE CRATE

```bash
rm -rf bin/99_shared_crates/heartbeat/
```

### Remove from Cargo.toml
```toml
# Remove this line from workspace members:
"bin/99_shared_crates/heartbeat",
```

### Remove from Dependencies
```toml
# bin/30_llm_worker_rbee/Cargo.toml
# DELETE: rbee-heartbeat = { path = "../99_shared_crates/heartbeat" }

# bin/20_rbee_hive/Cargo.toml
# DELETE: rbee-heartbeat = { path = "../99_shared_crates/heartbeat" }

# bin/10_queen_rbee/Cargo.toml
# DELETE: rbee-heartbeat = { path = "../99_shared_crates/heartbeat" }
```

### Remove Re-export
```rust
// bin/30_llm_worker_rbee/src/lib.rs
// DELETE: pub use rbee_heartbeat as heartbeat;
```

## Why We Don't Need It

### We Have Better Alternatives

| Old (rbee-heartbeat) | New (contracts) |
|---------------------|-----------------|
| `WorkerHeartbeatPayload` | `worker_contract::WorkerHeartbeat` |
| `HiveHeartbeatPayload` | `hive_contract::HiveHeartbeat` |
| `HealthStatus` | `shared_contract::HealthStatus` |
| `start_worker_heartbeat_task()` | Custom impl in worker binary |
| `start_hive_heartbeat_task()` | Custom impl in hive binary |
| `HeartbeatAcknowledgement` | `HttpHeartbeatAcknowledgement` |

### The New System is Better

**Old System (rbee-heartbeat):**
- Lightweight payloads (just ID + status)
- Registry can't track full component info
- Separate types for each component
- Generic trait abstractions (over-engineered)

**New System (contracts):**
- Full component info (WorkerInfo, HiveInfo)
- Registry tracks everything
- Shared foundation (shared-contract)
- Simple, direct types
- Type-safe with helper methods

## History

- **TEAM-115**: Created original heartbeat system
- **TEAM-151**: Extended for hive aggregation
- **TEAM-159**: Added trait abstractions
- **TEAM-261**: Simplified (workers → queen directly)
- **TEAM-262**: Cleaned up (removed hive aggregation)
- **TEAM-284**: Created contract system (made rbee-heartbeat obsolete)

## Conclusion

**The `rbee-heartbeat` crate is 100% dead code.**

- ❌ Not used by any product code
- ❌ Only has one unused re-export
- ❌ Contains tons of obsolete code
- ✅ We have better alternatives (contracts)
- ✅ Can be safely deleted

**Action:** Delete the entire crate and remove all references.
