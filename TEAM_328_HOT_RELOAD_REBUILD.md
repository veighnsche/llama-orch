# TEAM-328: Conditional Hot Reload for Daemon Rebuild

**Status:** ✅ COMPLETE

**Mission:** Add conditional hot reload behavior to daemon rebuild operations

## Problem

The rebuild operation always required manual stop/start workflow:
```bash
# Old workflow - manual
rbee-keeper queen stop
rbee-keeper queen rebuild  
rbee-keeper queen start
```

This was tedious and error-prone. Users had to remember the daemon state.

## Solution

Implemented conditional hot reload in `daemon-lifecycle/src/rebuild.rs`:

### Behavior

**Hot Reload (daemon was running):**
```
running → stop → rebuild → start → running
```

**Cold Rebuild (daemon was stopped):**
```
stopped → rebuild → stopped
```

### New API

```rust
pub async fn rebuild_with_hot_reload(
    rebuild_config: RebuildConfig,
    daemon_config: HttpDaemonConfig,
) -> Result<bool>
```

**Returns:** `true` if daemon was restarted (hot reload), `false` if left stopped

### Usage Example

```rust
use daemon_lifecycle::rebuild::{RebuildConfig, rebuild_with_hot_reload};
use daemon_contract::HttpDaemonConfig;

let rebuild_config = RebuildConfig::new("queen-rbee")
    .with_features(vec!["local-hive".to_string()])
    .with_job_id("job-123");

let daemon_config = HttpDaemonConfig {
    daemon_name: "queen-rbee".to_string(),
    binary_path: None, // Auto-resolve
    health_url: "http://localhost:7833".to_string(),
    args: vec![],
    env: vec![],
    job_id: Some("job-123".to_string()),
};

let was_restarted = rebuild_with_hot_reload(rebuild_config, daemon_config).await?;
if was_restarted {
    println!("Hot reload complete - daemon restarted");
} else {
    println!("Cold rebuild complete - daemon left stopped");
}
```

## Implementation Details

### Step-by-Step Flow

1. **Check daemon state** - Use health check to detect if running
2. **Conditional stop** - If running, stop gracefully using signals
3. **Build** - Run `cargo build --release --bin <name>`
4. **Conditional start** - If was running, restart with new binary
5. **Return state** - Indicate whether hot reload occurred

### Narration Events

**Hot Reload:**
- `hot_reload_start` - Detected running daemon
- `hot_reload_stop` - Stopping daemon
- `hot_reload_stopped` - Stop complete
- `build_start` / `build_success` - Build progress
- `hot_reload_restart` - Restarting daemon
- `hot_reload_complete` - Hot reload complete

**Cold Rebuild:**
- `cold_rebuild_start` - Detected stopped daemon
- `build_start` / `build_success` - Build progress
- `cold_rebuild_complete` - Rebuild complete

### Helper Functions

**Uses existing:**
```rust
crate::health::is_daemon_healthy() // Check if daemon is running
```

**Removed:**
```rust
// TEAM-328: No longer needed - rebuild_with_hot_reload handles state checking
pub async fn check_not_running_before_rebuild(...) -> Result<()>
```

## Files Changed

**Modified:**
- `bin/99_shared_crates/daemon-lifecycle/src/rebuild.rs` (+91 LOC, -52 LOC)
  - Added `rebuild_with_hot_reload()` function
  - Uses existing `is_daemon_healthy()` for state detection
  - Removed `check_not_running_before_rebuild()` (no longer needed)
  - Updated module documentation
- `bin/00_rbee_keeper/src/handlers/queen.rs` (+6 LOC, -3 LOC)
  - Updated to use `rebuild_with_hot_reload()`
  - Passes daemon config for hot reload support
- `bin/00_rbee_keeper/src/handlers/hive.rs` (+6 LOC, -3 LOC)
  - Updated to use `rebuild_with_hot_reload()`
  - Passes daemon config for hot reload support

## Benefits

✅ **User Experience**
- Single command for rebuild (auto-detects state)
- No manual stop/start required
- Preserves daemon state (running stays running, stopped stays stopped)

✅ **Safety**
- Graceful shutdown before rebuild (no file conflicts)
- Health check verification
- Proper error handling at each step

✅ **Observability**
- Clear narration for hot reload vs cold rebuild
- Progress indicators for each step
- SSE routing support via job_id

## Integration Complete

✅ **rbee-keeper queen rebuild** - Now uses `rebuild_with_hot_reload()`
✅ **rbee-keeper hive rebuild** - Now uses `rebuild_with_hot_reload()`

**Migration pattern used:**
```rust
// Old (manual state management)
let config = RebuildConfig::new("queen-rbee");
build_daemon_local(config).await?;

// New (automatic state management)
let rebuild_config = RebuildConfig::new("queen-rbee");
let daemon_config = HttpDaemonConfig::new("queen-rbee", queen_url.to_string())
    .with_args(vec!["--port".to_string(), port.to_string()]);

rebuild_with_hot_reload(rebuild_config, daemon_config).await?;
```

## Compilation

✅ `cargo check -p daemon-lifecycle` - PASS  
✅ `cargo build --bin rbee-keeper` - PASS

## Testing

Try it now:
```bash
# Start queen
./rbee queen start

# Rebuild while running (hot reload)
./rbee queen rebuild
# Expected: stop → rebuild → start → running

# Rebuild while stopped (cold rebuild)
./rbee queen stop
./rbee queen rebuild
# Expected: rebuild → stopped
```

## Code Signatures

All changes marked with `// TEAM-328:`

---

**Status:** ✅ COMPLETE - Hot reload fully integrated and ready to test
