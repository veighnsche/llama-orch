# TEAM-260: SSH Client Relocation to Shared Crates

**Date:** October 24, 2025  
**Status:** ✅ COMPLETE

## Mission

Move `ssh-client` from `bin/15_queen_rbee_crates/` to `bin/99_shared_crates/` because it is used by multiple components, not just queen-rbee.

## Changes Made

### 1. File System Changes

```bash
# Moved directory
mv bin/15_queen_rbee_crates/ssh-client bin/99_shared_crates/ssh-client
```

**New Location:** `bin/99_shared_crates/ssh-client/`

### 2. Workspace Updates (`Cargo.toml`)

**Added to shared crates section:**
```toml
"bin/99_shared_crates/daemon-sync",         # TEAM-280: Daemon synchronization and validation
"bin/99_shared_crates/ssh-client",          # TEAM-260: SSH client for remote operations
"bin/99_shared_crates/ssh-client/bdd",
```

**Removed from queen-rbee crates section:**
```toml
# REMOVED:
# "bin/15_queen_rbee_crates/ssh-client",
# "bin/15_queen_rbee_crates/ssh-client/bdd",
```

### 3. Dependency Path Updates

**daemon-sync** (`bin/99_shared_crates/daemon-sync/Cargo.toml`):
```toml
# Before:
queen-rbee-ssh-client = { path = "../../15_queen_rbee_crates/ssh-client" }

# After:
queen-rbee-ssh-client = { path = "../ssh-client" }
```

**hive-lifecycle** (`bin/15_queen_rbee_crates/hive-lifecycle/Cargo.toml`):
```toml
# Before:
queen-rbee-ssh-client = { path = "../ssh-client" }

# After:
queen-rbee-ssh-client = { path = "../../99_shared_crates/ssh-client" }
```

**xtask** (`xtask/Cargo.toml`):
```toml
# REMOVED (dead dependency - not actually used):
# queen-rbee-ssh-client = { path = "../bin/15_queen_rbee_crates/ssh-client" }
```

## Rationale

### Why Move to Shared Crates?

The ssh-client is used by **2 active consumers**:

1. **daemon-sync** (shared crate) - Package installation and validation
2. **hive-lifecycle** (queen-rbee crate) - Remote hive start/stop

Since `daemon-sync` is a shared utility crate (not specific to queen-rbee), the ssh-client must also be shared.

### Consumer Analysis

| Consumer | Location | Purpose | SSH Operations |
|----------|----------|---------|----------------|
| **daemon-sync** | `bin/99_shared_crates/` | Package management | Install, validate, query |
| **hive-lifecycle** | `bin/15_queen_rbee_crates/` | Hive lifecycle | Start, stop |
| ~~xtask~~ | ~~`xtask/`~~ | ~~Testing~~ | **REMOVED** (not used) |

## Implementation Differences

### daemon-sync: Direct Client Usage

```rust
// Direct SSH client usage for multiple operations
let client = RbeeSSHClient::connect(&hive.hostname, hive.ssh_port, &hive.ssh_user).await?;

// Operation 1: Check installation
let (stdout, _, exit_code) = client.exec("~/.local/bin/rbee-hive --version").await?;

// Operation 2: List workers
let (stdout, _, _) = client.exec("ls -1 ~/.local/bin/rbee-worker-*").await?;

client.close().await?;
```

**Characteristics:**
- Multiple operations per connection
- Direct stdout/stderr/exit_code handling
- Used for: installation, validation, querying

### hive-lifecycle: Wrapper Functions

```rust
// Wrapper functions with integrated narration
use crate::ssh_helper::{ssh_exec, scp_copy};

// Single operation per connection
let output = ssh_exec(hive_config, command, job_id, action, description).await?;
```

**Characteristics:**
- One operation per connection
- Integrated narration/logging
- Simplified error handling
- Used for: start/stop operations

## Verification

✅ **All crates compile successfully:**

```bash
# SSH client and consumers
cargo check -p queen-rbee-ssh-client       # ✅ PASS
cargo check -p daemon-sync                 # ✅ PASS
cargo check -p queen-rbee-hive-lifecycle   # ✅ PASS

# Core binaries
cargo check --bin rbee-keeper              # ✅ PASS
cargo check --bin queen-rbee               # ✅ PASS
cargo check --bin rbee-hive                # ✅ PASS
```

✅ **Workspace structure:**
- ssh-client in shared crates section
- daemon-sync added to workspace
- All paths updated correctly
- Dead dependency removed from xtask

## Architecture

### Dependency Graph

```
┌─────────────────────────────────────────────────────┐
│ SHARED CRATES (bin/99_shared_crates/)               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  daemon-sync ──────┐                                │
│                    │                                │
│                    ├──> ssh-client                  │
│                    │                                │
│  (other shared)    │                                │
│                    │                                │
└────────────────────┼────────────────────────────────┘
                     │
┌────────────────────┼────────────────────────────────┐
│ QUEEN-RBEE CRATES  │                                │
├────────────────────┼────────────────────────────────┤
│                    │                                │
│  hive-lifecycle ───┘                                │
│                                                     │
│  worker-registry                                    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Why This Structure?

1. **daemon-sync is shared** → Used by multiple binaries for package management
2. **ssh-client is shared** → Used by daemon-sync (shared) and hive-lifecycle (queen-specific)
3. **hive-lifecycle stays in queen-rbee** → Only used by queen-rbee for remote hive management

## Documentation

Created comprehensive documentation:

1. **TEAM_260_RUSSH_TO_COMMAND_MIGRATION.md** - Migration from russh to command-based SSH
2. **USAGE_ANALYSIS.md** - Detailed analysis of both consumers
3. **TEAM_260_RELOCATION_SUMMARY.md** - This document

## Team Signatures

- **TEAM-260**: Relocated ssh-client to shared crates, removed dead xtask dependency
- **Previous Teams**:
  - TEAM-135: Created scaffolding
  - TEAM-188: Implemented SSH test connection
  - TEAM-222: Behavior inventory
  - TEAM-256: Migrated to russh
  - TEAM-260: Reverted to command-based SSH

## Next Steps

### For Future Teams

1. **Connection Pooling**: Consider implementing connection reuse for daemon-sync (multiple operations)
2. **Unified Wrapper**: Consider moving `ssh_helper.rs` to shared crate for consistency
3. **Error Types**: Consider structured error types instead of string parsing

### Testing Recommendations

Manual testing required:
1. Test daemon-sync package installation
2. Test hive-lifecycle start/stop operations
3. Test SSH connectivity validation
4. Verify error handling for failed connections

## References

- SSH client implementation: `bin/99_shared_crates/ssh-client/src/lib.rs`
- daemon-sync usage: `bin/99_shared_crates/daemon-sync/src/{query,validate,install}.rs`
- hive-lifecycle usage: `bin/15_queen_rbee_crates/hive-lifecycle/src/{start,stop,ssh_helper}.rs`
