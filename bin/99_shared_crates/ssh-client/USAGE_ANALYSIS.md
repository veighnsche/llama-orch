# SSH Client Usage Analysis

**Date:** October 24, 2025  
**Status:** ✅ COMPLETE

## Summary

The `ssh-client` crate has been moved from `bin/15_queen_rbee_crates/` to `bin/99_shared_crates/` because it is used by multiple components across the system.

## Consumers

### 1. ✅ daemon-sync (Shared Crate)

**Location:** `bin/99_shared_crates/daemon-sync`

**Purpose:** Package installation and validation across remote hives

**SSH Usage:**
- **Query Operations** (`query.rs`):
  - Check if rbee-hive is installed: `~/.local/bin/rbee-hive --version`
  - List installed workers: `ls -1 ~/.local/bin/rbee-worker-*`

- **Validation Operations** (`validate.rs`):
  - Test SSH connectivity: `echo 'SSH OK'`
  - Verify worker binaries exist: `test -f ~/.local/share/rbee/workers/{binary}`

- **Installation Operations** (`install.rs`):
  - Install hive from Git (clone, build, install)
  - Install workers from Git (clone, build, install)
  - Install from GitHub releases (download, extract, install)

**Key Functions:**
```rust
// Direct SSH client usage
RbeeSSHClient::connect(&hive.hostname, hive.ssh_port, &hive.ssh_user).await?
client.exec(command).await?
client.copy_file(local_path, remote_path).await?
```

### 2. ✅ hive-lifecycle (Queen-Rbee Crate)

**Location:** `bin/15_queen_rbee_crates/hive-lifecycle`

**Purpose:** Remote hive lifecycle management (start/stop)

**SSH Usage:**
- **Start Operations** (`start.rs`):
  - Start remote rbee-hive daemon
  - Command: `nohup ~/.local/bin/rbee-hive --port {port} > /dev/null 2>&1 & echo $!`

- **Stop Operations** (`stop.rs`):
  - Graceful shutdown: `pkill -TERM rbee-hive`
  - Force kill: `pkill -KILL rbee-hive`

**Helper Module** (`ssh_helper.rs`):
```rust
// Wrapper functions for common SSH operations
pub async fn ssh_exec(hive_config, command, job_id, action, description) -> Result<String>
pub async fn scp_copy(hive_config, local_path, remote_path, job_id) -> Result<()>
```

### 3. ❌ xtask (REMOVED - Dead Dependency)

**Location:** `xtask/Cargo.toml`

**Status:** Dependency listed but **NOT USED** in code

**Action:** Removed from `Cargo.toml` (TEAM-260)

## Architecture Decision

### Why Shared Crate?

The ssh-client is used by:
1. **daemon-sync** - Shared utility for package management
2. **hive-lifecycle** - Queen-specific lifecycle operations

Since daemon-sync is a shared crate (used by multiple binaries), the ssh-client must also be shared.

### Dependency Graph

```
daemon-sync (shared) ──┐
                       ├──> ssh-client (shared)
hive-lifecycle (queen) ┘
```

## Implementation Comparison

### daemon-sync: Direct SSH Client Usage

```rust
// daemon-sync uses RbeeSSHClient directly
let client = RbeeSSHClient::connect(&hive.hostname, hive.ssh_port, &hive.ssh_user).await?;
let (stdout, stderr, exit_code) = client.exec(command).await?;
client.close().await?;
```

**Characteristics:**
- Direct client usage
- Multiple operations per connection
- Handles stdout/stderr/exit_code directly
- Used for: installation, validation, querying

### hive-lifecycle: Wrapper Functions

```rust
// hive-lifecycle uses helper wrappers
use crate::ssh_helper::{ssh_exec, scp_copy};

let output = ssh_exec(hive_config, command, job_id, action, description).await?;
scp_copy(hive_config, local_path, remote_path, job_id).await?;
```

**Characteristics:**
- Wrapper functions with narration
- One operation per connection
- Simplified error handling
- Used for: start/stop operations

## SSH Operations Summary

### daemon-sync Operations

| Operation | Command | Purpose |
|-----------|---------|---------|
| Check hive installed | `~/.local/bin/rbee-hive --version` | Verify hive binary exists |
| List workers | `ls -1 ~/.local/bin/rbee-worker-*` | Find installed workers |
| Test connectivity | `echo 'SSH OK'` | Verify SSH works |
| Verify worker binary | `test -f ~/.local/share/rbee/workers/{binary}` | Check worker exists |
| Git clone | `git clone {repo} {dir}` | Clone repository |
| Build from source | `cd {dir} && cargo build --release` | Compile binary |
| Install binary | `install -D {src} {dst}` | Copy to install location |
| Download release | `curl -L {url} -o {file}` | Download from GitHub |
| Extract archive | `tar xzf {file}` | Extract tarball |

### hive-lifecycle Operations

| Operation | Command | Purpose |
|-----------|---------|---------|
| Start hive | `nohup ~/.local/bin/rbee-hive --port {port} > /dev/null 2>&1 & echo $!` | Start daemon |
| Stop hive (graceful) | `pkill -TERM rbee-hive` | Send SIGTERM |
| Stop hive (force) | `pkill -KILL rbee-hive` | Send SIGKILL |

## Security Considerations

Both implementations use the same SSH security model:

1. **StrictHostKeyChecking=no** - Auto-accept host keys (matches russh behavior)
2. **BatchMode=yes** - No interactive prompts
3. **ConnectTimeout=30** - 30-second connection timeout

### Authentication

- SSH agent (standard)
- Unencrypted keys in `~/.ssh/` (fallback)
- No password authentication

## Future Considerations

### Potential Improvements

1. **Connection Pooling**: Reuse SSH connections for multiple operations
   - Most beneficial for daemon-sync (multiple operations)
   - Less beneficial for hive-lifecycle (single operations)

2. **Unified Wrapper**: Consider moving ssh_helper.rs to shared crate
   - Would benefit daemon-sync with narration
   - Would reduce code duplication

3. **Error Handling**: Structured error types instead of string parsing
   - Better error messages
   - Easier debugging

### Migration Path

If connection pooling is needed:
1. Add connection cache to RbeeSSHClient
2. Implement connection reuse
3. Add connection timeout/cleanup
4. Update both daemon-sync and hive-lifecycle

## Verification

✅ All crates compile successfully:
```bash
cargo check -p queen-rbee-ssh-client       # ✅ PASS
cargo check -p daemon-sync                 # ✅ PASS
cargo check -p queen-rbee-hive-lifecycle   # ✅ PASS
```

✅ Workspace updated:
- ssh-client moved to `bin/99_shared_crates/`
- daemon-sync added to workspace
- xtask dependency removed

## Team Signatures

- **TEAM-260**: Moved ssh-client to shared crates, removed xtask dependency
- **Previous Teams**: See `TEAM_260_RUSSH_TO_COMMAND_MIGRATION.md`
