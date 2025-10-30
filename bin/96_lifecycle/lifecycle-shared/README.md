# lifecycle-shared

Shared types and utilities for lifecycle-local and lifecycle-ssh.

## Purpose

TEAM-367: Eliminate code duplication between `lifecycle-local` and `lifecycle-ssh` by extracting common types and utilities.

## What's Shared

### Types

- **`BuildConfig`** - Build configuration (daemon_name, target, job_id, features)
- **`HttpDaemonConfig`** - Daemon configuration (name, health_url, args, etc.)
- **`DaemonStatus`** - Daemon status information (is_running, is_installed)

### Utils

- **`utils::serde`** - Serde helpers for SystemTime serialization

## What's NOT Shared

- **Execution logic** - `lifecycle-local` uses local commands, `lifecycle-ssh` uses SSH
- **Binary resolution** - Different paths for local vs remote
- **Process management** - Different mechanisms for local vs SSH

## Usage

### lifecycle-local

```rust
use lifecycle_shared::{BuildConfig, HttpDaemonConfig, DaemonStatus};

// Use shared types, implement local execution
```

### lifecycle-ssh

```rust
use lifecycle_shared::{BuildConfig, HttpDaemonConfig, DaemonStatus};

// Use shared types, implement SSH execution
```

## Architecture

```text
lifecycle-local ──┐
                  ├──> lifecycle-shared (this crate)
lifecycle-ssh  ───┘
```

## Code Reduction

**Before:**
- `build.rs`: 189 LOC (local) + 178 LOC (ssh) = 367 LOC
- `status.rs`: 164 LOC (local) + 167 LOC (ssh) = 331 LOC
- `start.rs`: HttpDaemonConfig duplicated in both
- `utils/serde.rs`: 52 LOC duplicated in both

**After:**
- `lifecycle-shared`: ~200 LOC (shared types)
- `lifecycle-local`: Imports from shared
- `lifecycle-ssh`: Imports from shared

**Savings:** ~300-400 LOC eliminated
