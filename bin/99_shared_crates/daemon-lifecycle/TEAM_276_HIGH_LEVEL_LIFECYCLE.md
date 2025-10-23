# TEAM-276: High-Level Lifecycle Operations

**Status:** ✅ COMPLETE  
**Date:** Oct 23, 2025  
**File:** `src/lifecycle.rs`

## Mission

Add high-level `start_http_daemon()` and `stop_http_daemon()` functions that combine lower-level utilities for complete daemon lifecycle management.

## Problem

Each lifecycle crate was reimplementing the same pattern:

**queen-lifecycle:**
```rust
// start.rs: spawn + poll
pub async fn start_queen(url: &str) -> Result<()> {
    let handle = ensure_queen_running(url).await?;
    // Keep alive
    std::mem::forget(handle);
    Ok(())
}
```

**hive-lifecycle:**
```rust
// start.rs: resolve + spawn + poll + capabilities
pub async fn execute_hive_start(...) -> Result<()> {
    // 1. Resolve binary
    // 2. Spawn daemon
    // 3. Poll health
    // 4. Fetch capabilities
}
```

**worker-lifecycle:**
```rust
// spawn.rs: spawn with specific args
pub async fn spawn_worker(config: WorkerSpawnConfig) -> Result<()> {
    // Spawn worker process
}
```

**Common pattern**: spawn → poll health → return

## Solution

Created `lifecycle.rs` with high-level functions:

### 1. HttpDaemonConfig

Unified configuration for HTTP-based daemons:

```rust
#[derive(Clone)]
pub struct HttpDaemonConfig {
    pub daemon_name: String,
    pub binary_path: PathBuf,
    pub args: Vec<String>,
    pub health_url: String,
    pub shutdown_endpoint: String,
    pub job_id: Option<String>,
    pub max_health_attempts: Option<usize>,
    pub health_initial_delay_ms: Option<u64>,
}
```

**Builder pattern:**
```rust
let config = HttpDaemonConfig::new(
    "queen-rbee",
    PathBuf::from("target/release/queen-rbee"),
    "http://localhost:8500",
)
.with_args(vec!["--config".to_string(), "config.toml".to_string()])
.with_job_id("job-123")
.with_max_health_attempts(10);
```

### 2. start_http_daemon()

Combines spawn + health polling:

```rust
pub async fn start_http_daemon(config: HttpDaemonConfig) -> Result<Child>
```

**Steps:**
1. Spawn daemon using `DaemonManager`
2. Poll health endpoint using `poll_until_healthy`
3. Return Child process

**Usage:**
```rust
let config = HttpDaemonConfig::new(
    "queen-rbee",
    PathBuf::from("target/release/queen-rbee"),
    "http://localhost:8500",
).with_job_id("job-123");

let child = start_http_daemon(config).await?;
std::mem::forget(child); // Keep daemon alive
```

### 3. stop_http_daemon()

Graceful shutdown via HTTP:

```rust
pub async fn stop_http_daemon(config: HttpDaemonConfig) -> Result<()>
```

**Steps:**
1. Check if daemon is running
2. Send shutdown request to endpoint
3. Handle expected connection errors

**Usage:**
```rust
let config = HttpDaemonConfig::new(
    "queen-rbee",
    PathBuf::from("target/release/queen-rbee"),
    "http://localhost:8500",
).with_job_id("job-123");

stop_http_daemon(config).await?;
```

## Benefits

### 1. **Single Source of Truth** ⭐⭐⭐
- One implementation for HTTP daemon lifecycle
- Bugs fixed in one place
- Consistent behavior across all daemons

### 2. **Reduced Duplication** ⭐⭐⭐
- No need to reimplement spawn + poll in each crate
- Common pattern extracted
- Less code to maintain

### 3. **Easier to Use** ⭐⭐
- Simple, clear API
- Builder pattern for configuration
- One function call instead of multiple steps

### 4. **Flexible** ⭐⭐
- Low-level utilities still available for special cases
- High-level functions for common cases
- Best of both worlds

### 5. **Consistent Narration** ⭐⭐⭐
- All daemons emit same narration events
- Job ID propagation built-in
- Proper error handling

## Usage Examples

### Queen Startup (Simplified)

**Before (queen-lifecycle):**
```rust
pub async fn start_queen(queen_url: &str) -> Result<()> {
    let queen_handle = ensure_queen_running(queen_url).await?;
    
    NARRATE
        .action("queen_start")
        .context(queen_handle.base_url())
        .human("✅ Queen started on {}")
        .emit();
    
    std::mem::forget(queen_handle);
    Ok(())
}
```

**After (using daemon-lifecycle):**
```rust
pub async fn start_queen(queen_url: &str) -> Result<()> {
    let config = HttpDaemonConfig::new(
        "queen-rbee",
        find_queen_binary()?,
        queen_url,
    );
    
    let child = start_http_daemon(config).await?;
    std::mem::forget(child);
    Ok(())
}
```

### Hive Startup (Simplified)

**Before (hive-lifecycle):**
```rust
pub async fn execute_hive_start(request: HiveStartRequest, config: Arc<RbeeConfig>) -> Result<()> {
    // 1. Validate hive exists
    // 2. Check if already running
    // 3. Resolve binary path
    // 4. Spawn daemon
    // 5. Poll health with exponential backoff
    // 6. Fetch capabilities
    // ~385 LOC
}
```

**After (using daemon-lifecycle for spawn + poll):**
```rust
pub async fn execute_hive_start(request: HiveStartRequest, config: Arc<RbeeConfig>) -> Result<()> {
    // 1. Validate hive exists
    // 2. Check if already running
    // 3. Resolve binary path
    
    // 4-5. Spawn + poll (now one call!)
    let daemon_config = HttpDaemonConfig::new(
        "rbee-hive",
        binary_path,
        &hive_endpoint,
    ).with_job_id(&request.job_id);
    
    let child = start_http_daemon(daemon_config).await?;
    std::mem::forget(child);
    
    // 6. Fetch capabilities
    // ~200 LOC (185 LOC saved!)
}
```

### Worker Startup (Different Pattern)

Workers don't have HTTP endpoints, so they use lower-level utilities:

```rust
pub async fn spawn_worker(config: WorkerSpawnConfig) -> Result<()> {
    // Workers use DaemonManager directly (no HTTP health check)
    let manager = DaemonManager::new(binary_path, args);
    let child = manager.spawn().await?;
    
    // Register with queen via heartbeat (not HTTP health)
    Ok(())
}
```

## Architecture

### Layered Approach

```
┌─────────────────────────────────────────┐
│  High-Level (lifecycle.rs)             │
│  - start_http_daemon()                  │
│  - stop_http_daemon()                   │
│  - HttpDaemonConfig                     │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Mid-Level (health.rs, shutdown.rs)    │
│  - poll_until_healthy()                 │
│  - graceful_shutdown()                  │
│  - force_shutdown()                     │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│  Low-Level (manager.rs)                 │
│  - DaemonManager::spawn()               │
│  - is_daemon_healthy()                  │
└─────────────────────────────────────────┘
```

### When to Use Each Layer

**High-Level (lifecycle.rs):**
- ✅ Standard HTTP daemons (queen, hive)
- ✅ Simple start/stop needs
- ✅ Want automatic health polling

**Mid-Level (health.rs, shutdown.rs):**
- ✅ Custom spawn logic needed
- ✅ Need fine-grained control
- ✅ Special health check requirements

**Low-Level (manager.rs):**
- ✅ Non-HTTP daemons (workers)
- ✅ Custom process management
- ✅ No health endpoint available

## Code Metrics

### lifecycle.rs
- **240 LOC** total
- `HttpDaemonConfig` struct with builder (90 LOC)
- `start_http_daemon()` function (40 LOC)
- `stop_http_daemon()` function (30 LOC)
- Documentation and tests (80 LOC)

### Expected Impact

**queen-lifecycle:**
- Can simplify `start_queen()` by ~20 LOC
- Can simplify `stop_queen()` by ~15 LOC

**hive-lifecycle:**
- Can simplify `execute_hive_start()` by ~185 LOC (spawn + poll)
- Can simplify `execute_hive_stop()` by ~50 LOC

**Total potential savings: ~270 LOC across lifecycle crates**

## Files Modified

1. **src/lifecycle.rs** (240 LOC) - NEW
   - HttpDaemonConfig struct
   - start_http_daemon() function
   - stop_http_daemon() function
   - Tests

2. **src/lib.rs** - UPDATED
   - Added lifecycle module
   - Exported new types and functions
   - Added usage examples

## Verification

```bash
# Compilation
cargo check -p daemon-lifecycle
# ✅ SUCCESS

# Exports
# - start_http_daemon, stop_http_daemon, HttpDaemonConfig
```

## Design Decisions

### 1. **Why HttpDaemonConfig?**
- Chosen: Unified config struct
- Reason: Reduces parameter count, easier to extend
- Alternative: Separate parameters (too many, hard to extend)

### 2. **Why Return Child?**
- Chosen: Return Child from start_http_daemon
- Reason: Caller can decide to keep alive or manage
- Alternative: Auto-forget (less flexible)

### 3. **Why Clone on HttpDaemonConfig?**
- Chosen: Derive Clone
- Reason: Can reuse config for start + stop
- Alternative: Separate configs (more boilerplate)

### 4. **Why Not Include Workers?**
- Chosen: HTTP-only for now
- Reason: Workers don't have HTTP endpoints
- Future: Could add `start_process_daemon()` for workers

## Future Enhancements (Optional)

### Phase 2
1. Add `restart_http_daemon()` (stop + start)
2. Add `start_process_daemon()` for non-HTTP daemons
3. Add `batch_start_daemons()` for multiple daemons

### Phase 3
4. Add health check customization (custom endpoints)
5. Add startup hooks (callbacks before/after spawn)
6. Add metrics collection (startup time, health check attempts)

## Conclusion

Successfully added high-level lifecycle operations to `daemon-lifecycle`:

- ✅ **240 LOC implementation**
- ✅ **HttpDaemonConfig** with builder pattern
- ✅ **start_http_daemon()** combines spawn + poll
- ✅ **stop_http_daemon()** graceful shutdown
- ✅ **~270 LOC potential savings** across lifecycle crates
- ✅ **Layered architecture** (high/mid/low level)
- ✅ **Clean compilation** with tests

This provides a complete, reusable pattern for HTTP daemon lifecycle management that can be used by queen-lifecycle, hive-lifecycle, and any future HTTP-based daemons! 🎉
