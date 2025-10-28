# TEAM-314: Generic DaemonHandle Proposal

**Status:** 🔍 ANALYSIS  
**Date:** 2025-10-27  
**Issue:** Handle pattern is duplicated, should be generic

---

## Problem

Currently, `QueenHandle` is specific to queen-lifecycle, but the pattern should be generic for all daemons (queen, hive, workers).

**Current State:**
```
queen-lifecycle/src/types.rs:
  - QueenHandle (specific to queen)
  - Tracks: started_by_us, base_url, pid

hive-lifecycle:
  - No handle (should have one!)

worker-lifecycle (future):
  - No handle (will need one!)
```

**Issues:**
1. ❌ Code duplication (each daemon reimplements)
2. ❌ Inconsistent API across daemons
3. ❌ No handle for hive (missing cleanup tracking)
4. ❌ No handle for workers (missing cleanup tracking)

---

## Proposed Solution

### Option 1: Generic `DaemonHandle` in `daemon-lifecycle`

Create a generic handle in the shared `daemon-lifecycle` crate:

```rust
// bin/99_shared_crates/daemon-lifecycle/src/handle.rs

/// Generic daemon handle for lifecycle management
///
/// Tracks whether we started the daemon and provides cleanup.
/// IMPORTANT: Only shuts down daemon if we started it!
#[derive(Debug, Clone)]
pub struct DaemonHandle {
    /// Daemon name (e.g., "queen-rbee", "rbee-hive")
    daemon_name: String,
    
    /// True if we started the daemon (must cleanup)
    /// False if daemon was already running (don't touch it)
    started_by_us: bool,

    /// Base URL of the daemon
    base_url: String,

    /// Process ID if we started it
    pid: Option<u32>,
}

impl DaemonHandle {
    /// Create handle for daemon that was already running
    pub fn already_running(daemon_name: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            started_by_us: false,
            base_url: base_url.into(),
            pid: None,
        }
    }

    /// Create handle for daemon that we just started
    pub fn started_by_us(
        daemon_name: impl Into<String>,
        base_url: impl Into<String>,
        pid: Option<u32>,
    ) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            started_by_us: true,
            base_url: base_url.into(),
            pid,
        }
    }

    /// Check if we started the daemon (and should clean it up)
    pub const fn should_cleanup(&self) -> bool {
        self.started_by_us
    }

    /// Get the daemon's base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
    
    /// Get the daemon name
    pub fn daemon_name(&self) -> &str {
        &self.daemon_name
    }
    
    /// Get the process ID (if we started it)
    pub const fn pid(&self) -> Option<u32> {
        self.pid
    }
    
    /// Update the handle with discovered URL
    ///
    /// Service discovery - update URL after fetching from /v1/info
    pub fn with_discovered_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Keep the daemon alive (no shutdown after task)
    ///
    /// Daemon stays running for future tasks.
    pub fn shutdown(self) -> Result<()> {
        n!("daemon_stop", "Task complete, keeping {} alive for future tasks", self.daemon_name);
        Ok(())
    }
}
```

**Usage:**

```rust
// queen-lifecycle
pub type QueenHandle = DaemonHandle;

pub async fn ensure_queen_running(queen_url: &str) -> Result<QueenHandle> {
    ensure_daemon_with_handle(
        "queen-rbee",
        queen_url,
        None,
        || spawn_queen(),
        || DaemonHandle::already_running("queen-rbee", queen_url),
        || DaemonHandle::started_by_us("queen-rbee", queen_url, Some(pid)),
    ).await
}

// hive-lifecycle (NEW!)
pub type HiveHandle = DaemonHandle;

pub async fn ensure_hive_running(hive_url: &str) -> Result<HiveHandle> {
    ensure_daemon_with_handle(
        "rbee-hive",
        hive_url,
        None,
        || spawn_hive(),
        || DaemonHandle::already_running("rbee-hive", hive_url),
        || DaemonHandle::started_by_us("rbee-hive", hive_url, Some(pid)),
    ).await
}
```

---

### Option 2: Move to Contracts

If the handle is truly a cross-cutting concern (used by CLI, UI, API), move it to contracts:

```
contracts/
  └── daemon-handle/
      ├── Cargo.toml
      └── src/
          └── lib.rs (DaemonHandle)
```

**Pros:**
- ✅ Available to all components (CLI, UI, API, lifecycle crates)
- ✅ Clear contract for daemon management
- ✅ Can be serialized for API responses

**Cons:**
- ❌ Might be overkill if only lifecycle crates use it
- ❌ Adds another contract crate

---

## Recommendation

**Use Option 1: Generic `DaemonHandle` in `daemon-lifecycle`**

**Reasons:**
1. ✅ Handles are primarily used by lifecycle crates
2. ✅ Keeps related code together
3. ✅ Simpler dependency graph
4. ✅ Can still be re-exported by lifecycle crates for compatibility

**Migration Path:**

1. Create `daemon-lifecycle/src/handle.rs` with generic `DaemonHandle`
2. Update `queen-lifecycle` to use `pub type QueenHandle = DaemonHandle`
3. Add `HiveHandle` to `hive-lifecycle`
4. Add `WorkerHandle` to worker lifecycle (future)
5. Update all consumers to use generic handle

---

## Current Usage Analysis

### QueenHandle (queen-lifecycle/src/types.rs)

**Used by:**
- `queen-lifecycle/src/ensure.rs` - Returns `QueenHandle`
- `rbee-keeper/src/job_client.rs` - Uses `QueenHandle` for cleanup

**API:**
```rust
impl QueenHandle {
    pub const fn already_running(base_url: String) -> Self
    pub const fn started_by_us(base_url: String, pid: Option<u32>) -> Self
    pub const fn should_cleanup(&self) -> bool
    pub fn base_url(&self) -> &str
    pub fn with_discovered_url(mut self, url: String) -> Self
    pub fn shutdown(self) -> Result<()>
}
```

**Observations:**
- ✅ All methods are generic (not queen-specific)
- ✅ No queen-specific logic
- ✅ Perfect candidate for generalization

### Missing Handles

**HiveHandle (should exist):**
- Hive operations don't track cleanup
- No way to know if keeper started the hive
- Missing lifecycle management

**WorkerHandle (future):**
- Worker operations will need cleanup tracking
- Should follow same pattern

---

## Benefits of Generic Handle

### 1. Consistency

All daemons use the same API:

```rust
// Queen
let queen = ensure_queen_running(url).await?;
println!("Queen at: {}", queen.base_url());
queen.shutdown()?;

// Hive
let hive = ensure_hive_running(url).await?;
println!("Hive at: {}", hive.base_url());
hive.shutdown()?;

// Worker (future)
let worker = ensure_worker_running(url).await?;
println!("Worker at: {}", worker.base_url());
worker.shutdown()?;
```

### 2. Code Reuse

No duplication:

```rust
// Before (duplicated)
struct QueenHandle { started_by_us: bool, base_url: String, pid: Option<u32> }
struct HiveHandle { started_by_us: bool, base_url: String, pid: Option<u32> }
struct WorkerHandle { started_by_us: bool, base_url: String, pid: Option<u32> }

// After (shared)
type QueenHandle = DaemonHandle;
type HiveHandle = DaemonHandle;
type WorkerHandle = DaemonHandle;
```

### 3. Easier Testing

Generic handle can be tested once:

```rust
#[test]
fn test_daemon_handle_cleanup() {
    let handle = DaemonHandle::started_by_us("test-daemon", "http://localhost:8080", None);
    assert!(handle.should_cleanup());
    
    let handle = DaemonHandle::already_running("test-daemon", "http://localhost:8080");
    assert!(!handle.should_cleanup());
}
```

### 4. Future-Proof

Easy to add new daemons:

```rust
// New daemon? Just use DaemonHandle!
type CacheHandle = DaemonHandle;
type ProxyHandle = DaemonHandle;
type GatewayHandle = DaemonHandle;
```

---

## Implementation Plan

### Phase 1: Create Generic Handle

1. Create `daemon-lifecycle/src/handle.rs`
2. Implement `DaemonHandle` with all methods from `QueenHandle`
3. Export from `daemon-lifecycle/src/lib.rs`
4. Add tests

### Phase 2: Migrate Queen

1. Update `queen-lifecycle/src/types.rs`:
   ```rust
   pub use daemon_lifecycle::DaemonHandle as QueenHandle;
   ```
2. Remove old `QueenHandle` implementation
3. Update imports in `ensure.rs`
4. Verify tests pass

### Phase 3: Add Hive Handle

1. Add to `hive-lifecycle/src/lib.rs`:
   ```rust
   pub use daemon_lifecycle::DaemonHandle as HiveHandle;
   ```
2. Update hive operations to return `HiveHandle`
3. Track cleanup in keeper

### Phase 4: Documentation

1. Update `daemon-lifecycle/README.md`
2. Add examples for all daemon types
3. Document handle pattern

---

## Breaking Changes

### For queen-lifecycle Consumers

**Before:**
```rust
use queen_lifecycle::QueenHandle;
```

**After (compatible):**
```rust
use queen_lifecycle::QueenHandle; // Still works! (type alias)
```

**No breaking changes** - type alias maintains compatibility.

---

## Alternative: Trait-Based Approach

If we need daemon-specific behavior:

```rust
pub trait DaemonHandleTrait {
    fn daemon_name(&self) -> &str;
    fn base_url(&self) -> &str;
    fn should_cleanup(&self) -> bool;
    fn shutdown(self) -> Result<()>;
}

pub struct DaemonHandle<T: DaemonHandleTrait> {
    inner: T,
}
```

**Verdict:** ❌ Overkill - current daemons have identical behavior

---

## Questions for Discussion

1. **Should handles be serializable?** (for API responses)
   - If yes, add `#[derive(Serialize, Deserialize)]`
   
2. **Should handles track more metadata?**
   - Start time?
   - Health check interval?
   - Capabilities?

3. **Should shutdown be async?**
   - Currently sync, but might need async for graceful shutdown

4. **Should we add a `DaemonInfo` struct?**
   - Separate data from behavior
   - `DaemonHandle` contains `DaemonInfo`

---

## Conclusion

**Recommendation:** Implement generic `DaemonHandle` in `daemon-lifecycle`

**Benefits:**
- ✅ Eliminates duplication
- ✅ Consistent API across all daemons
- ✅ Easy to add new daemons
- ✅ No breaking changes (type aliases)
- ✅ Better testing
- ✅ Cleaner architecture

**Next Steps:**
1. Get approval for approach
2. Implement `daemon-lifecycle/src/handle.rs`
3. Migrate `QueenHandle` to use generic
4. Add `HiveHandle` and `WorkerHandle`

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27  
**Status:** PROPOSAL 🔍
