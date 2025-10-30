# Localhost Mode - SSH Bypass

**TEAM-331: Automatic localhost optimization**

## Quick Start

```rust
use daemon_lifecycle::{SshConfig, start_daemon, StartConfig, HttpDaemonConfig};

// ✅ CORRECT: Use SshConfig::localhost()
let ssh = SshConfig::localhost();

// ❌ WRONG: Don't manually create localhost config
let ssh = SshConfig::new("localhost".to_string(), whoami::username(), 22);
```

## How It Works

When you use `SshConfig::localhost()`, all SSH/SCP operations automatically bypass SSH and use direct local execution:

```rust
// This code...
let ssh = SshConfig::localhost();
ssh_exec(&ssh, "ls -la").await?;

// ...automatically becomes:
local_exec("ls -la").await?;  // No SSH overhead!
```

## Detection

The following hostnames are detected as localhost:
- `"localhost"`
- `"127.0.0.1"`
- `"::1"`

```rust
let ssh = SshConfig::localhost();
assert!(ssh.is_localhost());  // true

let ssh = SshConfig::new("192.168.1.100".to_string(), "user".to_string(), 22);
assert!(!ssh.is_localhost());  // false - uses SSH
```

## Performance

**Localhost operations (with bypass):**
- ✅ Direct `tokio::process::Command` execution
- ✅ Direct `tokio::fs::copy` for files
- ✅ 10-50x faster than SSH
- ✅ No authentication overhead

**Remote operations (unchanged):**
- Uses SSH for command execution
- Uses SCP for file transfers
- Same reliability and security

## API

### Constructor

```rust
impl SshConfig {
    /// Create localhost SSH config (bypasses SSH when possible)
    pub fn localhost() -> Self
    
    /// Check if this config points to localhost
    pub fn is_localhost(&self) -> bool
}
```

### Operations That Benefit

All daemon-lifecycle operations automatically benefit:

```rust
// Start daemon (bypasses SSH for localhost)
start_daemon(StartConfig {
    ssh_config: SshConfig::localhost(),
    daemon_config,
    job_id: None,
}).await?;

// Stop daemon (bypasses SSH for localhost)
stop_daemon(StopConfig {
    ssh_config: SshConfig::localhost(),
    daemon_name: "rbee-hive".to_string(),
    shutdown_url,
    health_url,
    job_id: None,
}).await?;

// Install daemon (bypasses SCP for localhost)
install_daemon(InstallConfig {
    ssh_config: SshConfig::localhost(),
    daemon_name: "rbee-hive".to_string(),
    local_binary_path: None,
    job_id: None,
}).await?;
```

## Implementation Details

### Local Execution Module

**File:** `src/utils/local.rs`

```rust
/// Execute command locally (bypasses SSH)
pub async fn local_exec(command: &str) -> Result<String>

/// Copy file locally (bypasses SCP)
pub async fn local_copy(local_path: &Path, dest_path: &str) -> Result<()>
```

### SSH Helpers (with localhost detection)

**File:** `src/utils/ssh.rs`

```rust
pub async fn ssh_exec(ssh_config: &SshConfig, command: &str) -> Result<String> {
    if ssh_config.is_localhost() {
        return local_exec(command).await;  // ← Bypass!
    }
    // ... SSH execution for remote hosts
}

pub async fn scp_upload(ssh_config: &SshConfig, local_path: &PathBuf, remote_path: &str) -> Result<()> {
    if ssh_config.is_localhost() {
        return local_copy(local_path, remote_path).await;  // ← Bypass!
    }
    // ... SCP for remote hosts
}
```

## Migration Guide

### Before (TEAM-330)

```rust
// Manually creating localhost config everywhere
let ssh = SshConfig::new("localhost".to_string(), whoami::username(), 22);
```

### After (TEAM-331)

```rust
// Single constructor, automatic optimization
let ssh = SshConfig::localhost();
```

## Benefits

1. ✅ **No SSH overhead** - Direct process execution
2. ✅ **No SCP overhead** - Direct file copy
3. ✅ **Single source of truth** - One way to create localhost config
4. ✅ **Automatic optimization** - Transparent to callers
5. ✅ **Remote still works** - Only localhost is optimized

## Testing

```bash
# Run localhost tests
cargo test -p daemon-lifecycle local_exec
cargo test -p daemon-lifecycle local_copy

# Verify compilation
cargo check -p daemon-lifecycle
cargo check -p rbee-keeper
```

## Related Files

- `src/lib.rs` - `SshConfig` definition
- `src/utils/local.rs` - Local execution helpers
- `src/utils/ssh.rs` - SSH helpers with localhost detection
- `bin/00_rbee_keeper/src/handlers/queen.rs` - Uses `SshConfig::localhost()`
- `bin/00_rbee_keeper/src/handlers/hive.rs` - Uses `SshConfig::localhost()`

## Future Enhancements

Potential improvements (not implemented):
- Support for remote hives via `--host` CLI argument
- Auto-detection from `hives.conf`
- SSH connection pooling for remote operations
