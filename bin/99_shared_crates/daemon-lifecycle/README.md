# remote-daemon-lifecycle

Remote daemon lifecycle management via SSH.

## Purpose

Provides SSH-based remote execution for daemon lifecycle operations. Wraps `daemon-lifecycle` functions to work over SSH connections.

## Architecture

```
rbee-keeper (local)
    ↓
remote-daemon-lifecycle (this crate)
    ↓ SSH/SCP
remote machine → daemon-lifecycle (local execution)
```

## Design Principles

1. **Minimal SSH calls** - Bundle operations into scripts when possible
2. **HTTP for monitoring** - Use HTTP health checks, not SSH
3. **Reuse daemon-lifecycle types** - Don't duplicate config structs
4. **Shell scripts** - Use portable shell scripts for remote execution

## Status

🚧 **STUB IMPLEMENTATION** - All functions return "NOT YET IMPLEMENTED"

## Migration Plan

Migrate functions one by one from `daemon-lifecycle` to support remote execution:

1. ✅ Stubs created with requirements documented
2. ⏳ Implement `start.rs` (most complex)
3. ⏳ Implement `stop.rs`
4. ⏳ Implement `status.rs` (simplest - HTTP only)
5. ⏳ Implement `install.rs`
6. ⏳ Implement `uninstall.rs`
7. ⏳ Implement `build.rs` (local only, no SSH)
8. ⏳ Implement `rebuild.rs` (combines others)
9. ⏳ Implement `shutdown.rs` (graceful with fallback)

## SSH Call Budget

Each operation documents its SSH call count:

- **start**: 2 SSH calls (find binary, start daemon) + HTTP health polling
- **stop**: 0-1 SSH calls (try HTTP first, fallback to SSH)
- **status**: 0 SSH calls (HTTP only)
- **install**: 1 SCP + 1 SSH (copy, chmod)
- **uninstall**: 1 SSH call (rm)
- **build**: 0 SSH calls (local only)
- **rebuild**: 3-4 calls (stop + install + start)
- **shutdown**: 0-2 SSH calls (try HTTP, fallback to SIGTERM/SIGKILL)

## Usage Example (Future)

```rust
use remote_daemon_lifecycle::{start_daemon_remote, SshConfig};
use daemon_lifecycle::HttpDaemonConfig;

let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
let config = HttpDaemonConfig::new("rbee-hive", "http://192.168.1.100:7835")
    .with_args(vec!["--port".to_string(), "7835".to_string()]);

let pid = start_daemon_remote(ssh, config).await?;
println!("Started daemon with PID: {}", pid);
```

## Testing Strategy

1. **Unit tests**: Mock SSH calls
2. **Integration tests**: Real SSH to localhost
3. **Manual tests**: Real SSH to remote machine

## Next Steps

1. Implement `start_daemon_remote()` first (most complex)
2. Add SSH helper utilities (ssh_exec, scp_copy)
3. Test with real remote machine
4. Migrate remaining functions
5. Update handlers to use remote-daemon-lifecycle
