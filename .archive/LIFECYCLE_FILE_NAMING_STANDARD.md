# Lifecycle Crates File Naming Standard

**Date:** Oct 23, 2025  
**Status:** PROPOSED  
**Purpose:** Improve DX by standardizing file names across all lifecycle crates

## Problem

Developers switching between lifecycle crates face inconsistent naming:

| Operation | daemon-lifecycle | queen-lifecycle | hive-lifecycle | worker-lifecycle |
|-----------|-----------------|-----------------|----------------|------------------|
| **Start** | lifecycle.rs | start.rs | start.rs | spawn.rs |
| **Stop** | shutdown.rs | stop.rs | stop.rs | delete.rs |
| **List** | list.rs | ❌ | list.rs | process_list.rs |
| **Get** | get.rs | ❌ | get.rs | process_get.rs |
| **Status** | status.rs | status.rs | status.rs | ❌ |

**Result:** Confusion, slower navigation, cognitive overhead

## Proposed Standard

### Core Operations (All Lifecycle Crates)

```
src/
├── lib.rs           # Exports and documentation
├── types.rs         # Request/Response types
├── start.rs         # Start/spawn operation
├── stop.rs          # Stop/shutdown/delete operation
├── status.rs        # Status check operation
├── list.rs          # List all instances
├── get.rs           # Get single instance details
├── install.rs       # Install binary
├── uninstall.rs     # Uninstall binary
└── health.rs        # Health checking utilities
```

### Optional Operations (Daemon-Specific)

```
# Queen-specific
├── rebuild.rs       # Rebuild with features
├── info.rs          # Build info

# Hive-specific
├── capabilities.rs  # Refresh capabilities
├── ssh_helper.rs    # SSH utilities
├── ssh_test.rs      # SSH testing
├── validation.rs    # Validation helpers

# Worker-specific
├── heartbeat.rs     # Heartbeat to queen

# Daemon-lifecycle specific
├── lifecycle.rs     # High-level start/stop
├── manager.rs       # Low-level spawn
├── ensure.rs        # Ensure running pattern
├── shutdown.rs      # Low-level shutdown
├── timeout.rs       # Timeout utilities
```

## Standardization Plan

### Phase 1: Rename Files (No Code Changes)

#### daemon-lifecycle
- ✅ Keep as-is (already well-organized)
- Note: `lifecycle.rs` contains high-level start/stop
- Note: `shutdown.rs` contains low-level shutdown utilities

#### queen-lifecycle
- ✅ Keep as-is (already follows standard!)
- Files: start.rs, stop.rs, status.rs, install.rs, uninstall.rs
- Bonus: info.rs, rebuild.rs (queen-specific)

#### hive-lifecycle
- ✅ Keep as-is (already follows standard!)
- Files: start.rs, stop.rs, status.rs, list.rs, get.rs, install.rs, uninstall.rs
- Bonus: capabilities.rs, ssh_helper.rs, validation.rs (hive-specific)

#### worker-lifecycle ⚠️ NEEDS CHANGES
- ❌ `spawn.rs` → ✅ `start.rs` (consistency!)
- ❌ `delete.rs` → ✅ `stop.rs` (consistency!)
- ❌ `process_list.rs` → ✅ `list.rs` (consistency!)
- ❌ `process_get.rs` → ✅ `get.rs` (consistency!)

### Phase 2: Update Documentation

Add to each crate's lib.rs:

```rust
//! # File Organization
//!
//! This crate follows the standard lifecycle file naming convention:
//!
//! - `start.rs` - Start/spawn operation
//! - `stop.rs` - Stop/shutdown operation
//! - `status.rs` - Status check
//! - `list.rs` - List all instances
//! - `get.rs` - Get single instance
//! - `install.rs` - Install binary
//! - `uninstall.rs` - Uninstall binary
//!
//! See `/LIFECYCLE_FILE_NAMING_STANDARD.md` for complete standard.
```

## Rationale

### Why "start.rs" not "spawn.rs"?
- ✅ "start" is user-facing terminology
- ✅ Consistent with "stop"
- ✅ 3/4 crates already use "start"
- ❌ "spawn" is implementation detail

### Why "stop.rs" not "delete.rs" or "shutdown.rs"?
- ✅ "stop" is user-facing terminology
- ✅ Consistent with "start"
- ✅ 3/4 crates already use "stop"
- ❌ "delete" implies removal (not just stopping)
- ❌ "shutdown" is too formal

### Why "list.rs" not "process_list.rs"?
- ✅ Shorter, clearer
- ✅ Consistent across crates
- ✅ "process" is implementation detail
- Note: Can still export as `list_worker_processes()` if needed

### Why "get.rs" not "process_get.rs"?
- ✅ Shorter, clearer
- ✅ Consistent across crates
- ✅ "process" is implementation detail
- Note: Can still export as `get_worker_process()` if needed

## Implementation: worker-lifecycle Refactoring

### 1. Rename spawn.rs → start.rs

```bash
git mv src/spawn.rs src/start.rs
```

**Update lib.rs:**
```rust
// Before
pub mod spawn;
pub use spawn::spawn_worker;

// After
pub mod start;
pub use start::start_worker;  // or keep spawn_worker for compat
```

**Update start.rs:**
```rust
// Before
pub async fn spawn_worker(config: WorkerSpawnConfig) -> Result<SpawnResult>

// After
pub async fn start_worker(config: WorkerStartConfig) -> Result<StartResult>
// OR keep function name: pub async fn spawn_worker(...) for compatibility
```

### 2. Rename delete.rs → stop.rs

```bash
git mv src/delete.rs src/stop.rs
```

**Update lib.rs:**
```rust
// Before
pub mod delete;
pub use delete::delete_worker;

// After
pub mod stop;
pub use stop::stop_worker;  // or keep delete_worker for compat
```

**Update stop.rs:**
```rust
// Before
pub async fn delete_worker(pid: u32, job_id: &str) -> Result<()>

// After
pub async fn stop_worker(pid: u32, job_id: &str) -> Result<()>
// OR keep function name for compatibility
```

### 3. Rename process_list.rs → list.rs

```bash
git mv src/process_list.rs src/list.rs
```

**Update lib.rs:**
```rust
// Before
pub mod process_list;
pub use process_list::{list_worker_processes, WorkerProcessInfo};

// After
pub mod list;
pub use list::{list_workers, WorkerInfo};  // Simplified names
// OR keep old names: pub use list::{list_worker_processes, WorkerProcessInfo};
```

### 4. Rename process_get.rs → get.rs

```bash
git mv src/process_get.rs src/get.rs
```

**Update lib.rs:**
```rust
// Before
pub mod process_get;
pub use process_get::get_worker_process;

// After
pub mod get;
pub use get::get_worker;  // Simplified name
// OR keep old name: pub use get::get_worker_process;
```

## Migration Strategy

### Option A: Breaking Changes (Clean Slate)
- Rename files AND functions
- Update all callers
- Clean, consistent API
- **Downside**: Breaking changes

### Option B: Backward Compatible (Recommended)
- Rename files ONLY
- Keep function names unchanged
- Add deprecation warnings
- Migrate callers gradually
- **Upside**: No breaking changes

### Option C: Hybrid (Best of Both)
- Rename files
- Add new function names (e.g., `start_worker`)
- Keep old function names (e.g., `spawn_worker`) as aliases
- Deprecate old names
- **Upside**: Smooth migration path

## Recommended Approach: Option C (Hybrid)

```rust
// worker-lifecycle/src/start.rs (renamed from spawn.rs)

/// Start a worker (new name)
pub async fn start_worker(config: WorkerStartConfig) -> Result<StartResult> {
    // Implementation
}

/// Spawn a worker (deprecated, use start_worker)
#[deprecated(since = "0.2.0", note = "Use start_worker instead")]
pub async fn spawn_worker(config: WorkerSpawnConfig) -> Result<SpawnResult> {
    // Alias to start_worker or keep old implementation
    start_worker(config).await
}
```

## Benefits

### 1. **Faster Navigation** ⭐⭐⭐
- Know exactly where to find operations
- No guessing "is it spawn.rs or start.rs?"
- Muscle memory works across crates

### 2. **Easier Onboarding** ⭐⭐⭐
- New developers learn one pattern
- Documentation is consistent
- Less cognitive load

### 3. **Better Refactoring** ⭐⭐
- Easy to copy patterns between crates
- Clear which files to update
- Consistent structure aids tooling

### 4. **Reduced Errors** ⭐⭐
- Less confusion = fewer mistakes
- Clear expectations
- Easier code review

## File Organization Matrix (After Standardization)

| File | daemon-lifecycle | queen-lifecycle | hive-lifecycle | worker-lifecycle |
|------|-----------------|-----------------|----------------|------------------|
| **lib.rs** | ✅ | ✅ | ✅ | ✅ |
| **types.rs** | ✅ | ✅ | ✅ | ✅ |
| **start.rs** | ✅ (lifecycle.rs) | ✅ | ✅ | ✅ (was spawn.rs) |
| **stop.rs** | ✅ (shutdown.rs) | ✅ | ✅ | ✅ (was delete.rs) |
| **status.rs** | ✅ | ✅ | ✅ | ❌ (stateless) |
| **list.rs** | ✅ | ❌ (singleton) | ✅ | ✅ (was process_list.rs) |
| **get.rs** | ✅ | ❌ (singleton) | ✅ | ✅ (was process_get.rs) |
| **install.rs** | ✅ | ✅ | ✅ | ❌ (catalog) |
| **uninstall.rs** | ✅ | ✅ | ✅ | ❌ (catalog) |
| **health.rs** | ✅ | ✅ | ❌ (uses daemon) | ❌ (no HTTP) |

**Legend:**
- ✅ = Present
- ❌ = Not applicable (architectural reason)
- (was X) = Needs renaming

## Action Items

### Immediate
- [ ] Review and approve this standard
- [ ] Decide on migration strategy (A, B, or C)

### Phase 1 (worker-lifecycle)
- [ ] Rename spawn.rs → start.rs
- [ ] Rename delete.rs → stop.rs
- [ ] Rename process_list.rs → list.rs
- [ ] Rename process_get.rs → get.rs
- [ ] Update lib.rs exports
- [ ] Add deprecation warnings (if Option C)
- [ ] Update documentation

### Phase 2 (Documentation)
- [ ] Add file organization section to each lib.rs
- [ ] Update README files
- [ ] Create migration guide (if breaking changes)

### Phase 3 (Verification)
- [ ] Verify all imports still work
- [ ] Run tests
- [ ] Update any external callers

## Conclusion

Standardizing file names across lifecycle crates will significantly improve developer experience:

- ✅ **Faster navigation** - Know where to find operations
- ✅ **Easier onboarding** - One pattern to learn
- ✅ **Better consistency** - Same structure everywhere
- ✅ **Reduced errors** - Clear expectations

**Recommendation**: Implement Option C (Hybrid) for worker-lifecycle to achieve consistency without breaking changes.
