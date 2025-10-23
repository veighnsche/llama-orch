# TEAM-276: Ensure Pattern Added to hive-lifecycle

**Status:** ✅ COMPLETE  
**Date:** Oct 23, 2025  
**Files:** `src/ensure.rs`, `src/types.rs` (HiveHandle added)

## Mission

Add ensure pattern to hive-lifecycle for consistency with queen-lifecycle, enabling auto-start of hives when needed.

## Pattern Overview

### Ensure Pattern (Same as queen-lifecycle)

1. **Check health** via HTTP GET /health
2. **If healthy** → Return HiveHandle (already_running)
3. **If not running** → Start hive + poll health → Return HiveHandle (started_by_us)

### Why This Matters

**rbee-keeper:**
- Calls `ensure_queen_running()` → Auto-starts queen if needed
- Queen stays alive for future tasks

**queen-rbee:**
- Should call `ensure_hive_running()` → Auto-starts hive if needed
- Hive stays alive for future tasks
- **Same pattern, different level of orchestration**

## Implementation

### 1. src/ensure.rs (140 LOC)

**Function:**
```rust
pub async fn ensure_hive_running(
    hive_alias: &str,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveHandle>
```

**Steps:**
1. Get hive configuration from RbeeConfig
2. Check if hive is already running (HTTP health check)
3. If running → Return `HiveHandle::already_running()`
4. If not running → Call `execute_hive_start()`
5. Return `HiveHandle::started_by_us()`

**Features:**
- ✅ 60-second timeout with progress bar
- ✅ Full narration integration
- ✅ Delegates to `execute_hive_start()` (handles SSH/local)
- ✅ Returns HiveHandle for cleanup tracking

### 2. src/types.rs - HiveHandle (80 LOC)

**Type:**
```rust
pub struct HiveHandle {
    started_by_us: bool,
    alias: String,
    endpoint: String,
}
```

**Methods:**
- `already_running(alias, endpoint)` - Hive was already running
- `started_by_us(alias, endpoint)` - We just started the hive
- `should_cleanup()` - Check if we should shut down on cleanup
- `alias()` - Get hive alias
- `endpoint()` - Get hive endpoint
- `shutdown()` - Keep hive alive (emits narration)

**Pattern matches QueenHandle:**
- Same structure
- Same cleanup semantics
- Same "keep alive" behavior

## Comparison with queen-lifecycle

### queen-lifecycle/src/ensure.rs

```rust
pub async fn ensure_queen_running(base_url: &str) -> Result<QueenHandle> {
    // 1. Check health
    if is_queen_healthy(base_url).await? {
        return Ok(QueenHandle::already_running(base_url.to_string()));
    }
    
    // 2. Start queen
    start_queen_inner(base_url).await?;
    
    // 3. Return handle
    Ok(QueenHandle::started_by_us(base_url.to_string(), pid))
}
```

### hive-lifecycle/src/ensure.rs (NEW)

```rust
pub async fn ensure_hive_running(
    hive_alias: &str,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveHandle> {
    // 1. Check health
    if is_daemon_healthy(&health_url, ...).await {
        return Ok(HiveHandle::already_running(alias, endpoint));
    }
    
    // 2. Start hive (via SSH or local)
    execute_hive_start(start_request, config).await?;
    
    // 3. Return handle
    Ok(HiveHandle::started_by_us(alias, endpoint))
}
```

**Key Difference:**
- Queen: Starts locally, simple spawn
- Hive: Starts via SSH (or locally if attached), more complex

**Same Pattern:**
- Check health → Start if needed → Return handle
- Handle tracks if we started it
- Cleanup only if we started it

## Usage Example

### In queen-rbee (future)

```rust
use queen_rbee_hive_lifecycle::ensure_hive_running;
use rbee_config::RbeeConfig;
use std::sync::Arc;

// Ensure hive is running before forwarding work
let config = Arc::new(RbeeConfig::load()?);
let hive_handle = ensure_hive_running("hive-1", config, "job-123").await?;

// Forward work to hive
forward_to_hive(operation, hive_handle.endpoint()).await?;

// Keep hive alive for future tasks
hive_handle.shutdown()?;
```

### Narration Output

**If hive already running:**
```
🔍 Checking if hive 'hive-1' is running at http://192.168.1.100:8600
✅ Hive 'hive-1' is already running and healthy
```

**If hive needs to start:**
```
🔍 Checking if hive 'hive-1' is running at http://192.168.1.100:8600
⚠️  Hive 'hive-1' is not running, starting...
[execute_hive_start narration...]
✅ Hive 'hive-1' started and healthy
```

## Architecture

### Orchestration Hierarchy

```
rbee-keeper
    ↓ ensure_queen_running()
queen-rbee
    ↓ ensure_hive_running()
rbee-hive
    ↓ start_worker()
worker
```

**Each level:**
- Checks if child is running
- Auto-starts if needed
- Returns handle for cleanup
- Keeps child alive for future tasks

### SSH vs Local

**execute_hive_start() handles both:**
- **Remote hive:** SSH to remote machine, spawn rbee-hive
- **Attached hive:** Spawn rbee-hive locally (rare case)

**ensure_hive_running() doesn't care:**
- Delegates to `execute_hive_start()`
- Just checks health and returns handle

## Benefits

### 1. **Consistency** ⭐⭐⭐
- Same pattern as queen-lifecycle
- Developers know what to expect
- Easy to understand orchestration flow

### 2. **Auto-Start** ⭐⭐⭐
- Queen can auto-start hives when needed
- No manual intervention required
- Seamless user experience

### 3. **Cleanup Tracking** ⭐⭐
- HiveHandle tracks if we started it
- Only cleanup if we started it
- Prevents shutting down shared hives

### 4. **Future-Proof** ⭐⭐
- Ready for queen to use
- Handles SSH and local cases
- Extensible for new scenarios

## Files Modified

1. **src/ensure.rs** (140 LOC) - NEW
   - `ensure_hive_running()` function
   - `ensure_hive_running_inner()` helper
   - Full narration integration
   - Timeout with progress bar

2. **src/types.rs** (+90 LOC) - UPDATED
   - Added `HiveHandle` struct
   - Added `already_running()` constructor
   - Added `started_by_us()` constructor
   - Added accessor methods
   - Added `shutdown()` method

3. **src/lib.rs** - UPDATED
   - Added ensure module
   - Exported `ensure_hive_running`
   - Updated module documentation

## Verification

```bash
# Compilation
cargo check -p queen-rbee-hive-lifecycle
# ✅ SUCCESS

# Exports
pub use ensure::ensure_hive_running;
pub use types::HiveHandle;  # Via pub use types::*

# Module structure
pub mod ensure;
```

## Complete Lifecycle API

hive-lifecycle now has all standard operations + ensure:

| Operation | File | Function | Status |
|-----------|------|----------|--------|
| **Install** | install.rs | `execute_hive_install()` | ✅ Implemented |
| **Uninstall** | uninstall.rs | `execute_hive_uninstall()` | ✅ Implemented |
| **Start** | start.rs | `execute_hive_start()` | ✅ Implemented |
| **Stop** | stop.rs | `execute_hive_stop()` | ✅ Implemented |
| **Ensure** | ensure.rs | `ensure_hive_running()` | ✅ NEW |
| **Status** | status.rs | `execute_hive_status()` | ✅ Implemented |
| **List** | list.rs | `execute_hive_list()` | ✅ Implemented |
| **Get** | get.rs | `execute_hive_get()` | ✅ Implemented |
| **Capabilities** | capabilities.rs | `execute_hive_refresh_capabilities()` | ✅ Implemented |

## Comparison with Other Lifecycle Crates

### queen-lifecycle
- ✅ ensure.rs - `ensure_queen_running()` (rbee-keeper uses this)
- ✅ QueenHandle for cleanup tracking

### hive-lifecycle (NOW)
- ✅ ensure.rs - `ensure_hive_running()` (queen-rbee can use this)
- ✅ HiveHandle for cleanup tracking

### worker-lifecycle
- ❌ No ensure pattern (workers are ephemeral, not long-lived)
- Workers are started on-demand, not kept alive

## Future Usage in queen-rbee

When queen needs to forward work to a hive:

```rust
// Instead of:
execute_hive_start(request, config).await?;
forward_to_hive(operation).await?;

// Can do:
let hive = ensure_hive_running(hive_alias, config, job_id).await?;
forward_to_hive(operation, hive.endpoint()).await?;
hive.shutdown()?;  // Keep alive
```

**Benefits:**
- Simpler code (one call instead of two)
- Automatic health check
- Proper cleanup tracking
- Consistent with rbee-keeper pattern

## Conclusion

Successfully added ensure pattern to hive-lifecycle:

- ✅ **140 LOC** ensure.rs implementation
- ✅ **90 LOC** HiveHandle type
- ✅ **Same pattern** as queen-lifecycle
- ✅ **Auto-start** hives when needed
- ✅ **Cleanup tracking** via HiveHandle
- ✅ **SSH and local** support (delegates to execute_hive_start)
- ✅ **Clean compilation** with no errors

hive-lifecycle now has the complete ensure pattern matching queen-lifecycle! 🎉
