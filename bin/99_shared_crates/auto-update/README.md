# auto-update

**Category:** Utility  
**Pattern:** Builder Pattern  
**Created by:** TEAM-193

## Purpose

Dependency-aware auto-update logic for rbee binaries. Ensures binaries are rebuilt when ANY dependency changes (including transitive dependencies).

## Problem

Simple mtime checks on a binary's source directory miss changes in shared crates:

```bash
# Edit shared crate
vim bin/99_shared_crates/daemon-lifecycle/src/lib.rs

# Run keeper
./rbee queen start

# ❌ BUG: xtask only checks bin/00_rbee_keeper/
# ❌ Skips rebuild
# ❌ Runs stale binary with old daemon-lifecycle code
```

## Solution

Parse `Cargo.toml` to find ALL local path dependencies, check them recursively:

```rust
use auto_update::AutoUpdater;

let updater = AutoUpdater::new("queen-rbee", "bin/10_queen_rbee")?;

// Checks:
// 1. bin/10_queen_rbee/ (source)
// 2. bin/99_shared_crates/daemon-lifecycle/ (dependency)
// 3. bin/99_shared_crates/narration-core/ (dependency)
// 4. bin/15_queen_rbee_crates/rbee-config/ (dependency)
// ... and ALL transitive dependencies

if updater.needs_rebuild()? {
    updater.rebuild()?;
}

let binary_path = updater.find_binary()?;
```

## API

### Create Updater

```rust
let updater = AutoUpdater::new("queen-rbee", "bin/10_queen_rbee")?;
```

### Check if Rebuild Needed

```rust
if updater.needs_rebuild()? {
    println!("Rebuild needed!");
}
```

### Rebuild Binary

```rust
updater.rebuild()?;  // Runs: cargo build --bin queen-rbee
```

### Find Binary Path

```rust
let path = updater.find_binary()?;  // target/debug/queen-rbee or target/release/queen-rbee
```

### One-Shot: Ensure Built

```rust
let binary_path = updater.ensure_built().await?;
// Checks, rebuilds if needed, returns path
```

## Lifecycle Integration

This crate is designed to be used by lifecycle crates:

### daemon-lifecycle (keeper → queen)

```rust
// bin/99_shared_crates/daemon-lifecycle/src/lib.rs
use auto_update::AutoUpdater;

pub async fn spawn_queen(config: &Config) -> Result<Child> {
    // Auto-update queen if enabled
    let queen_binary = if config.auto_update_queen {
        AutoUpdater::new("queen-rbee", "bin/10_queen_rbee")?
            .ensure_built()
            .await?
    } else {
        PathBuf::from("target/debug/queen-rbee")
    };
    
    // Spawn daemon
    Command::new(&queen_binary)
        .arg("--port").arg("8500")
        .spawn()?
}
```

### hive-lifecycle (queen → hive)

```rust
// bin/15_queen_rbee_crates/hive-lifecycle/src/lib.rs
use auto_update::AutoUpdater;

pub async fn spawn_hive(config: &Config) -> Result<Child> {
    // Auto-update hive if enabled
    let hive_binary = if config.auto_update_hive {
        AutoUpdater::new("rbee-hive", "bin/20_rbee_hive")?
            .ensure_built()
            .await?
    } else {
        PathBuf::from("target/debug/rbee-hive")
    };
    
    // Spawn daemon
    Command::new(&hive_binary)
        .arg("--port").arg(port.to_string())
        .spawn()?
}
```

### worker-lifecycle (hive → worker)

```rust
// bin/25_rbee_hive_crates/worker-lifecycle/src/lib.rs
use auto_update::AutoUpdater;

pub async fn spawn_worker(config: &Config) -> Result<Child> {
    // Auto-update worker if enabled
    let worker_binary = if config.auto_update_worker {
        AutoUpdater::new("llm-worker-rbee", "bin/30_llm_worker_rbee")?
            .ensure_built()
            .await?
    } else {
        PathBuf::from("target/debug/llm-worker-rbee")
    };
    
    // Spawn daemon
    Command::new(&worker_binary)
        .arg("--model").arg(model)
        .spawn()?
}
```

## Dependency Tracking

### Example: rbee-keeper Dependencies

```
rbee-keeper (bin/00_rbee_keeper/)
├── daemon-lifecycle (bin/99_shared_crates/daemon-lifecycle/)
│   └── observability-narration-core (bin/99_shared_crates/narration-core/)
├── observability-narration-core (bin/99_shared_crates/narration-core/)
├── timeout-enforcer (bin/99_shared_crates/timeout-enforcer/)
├── rbee-operations (bin/99_shared_crates/rbee-operations/)
└── rbee-config (bin/15_queen_rbee_crates/rbee-config/)
```

**If ANY of these change → rebuild triggered**

### Cross-Binary Shared Crates

| Crate | Used By |
|-------|---------|
| `observability-narration-core` | keeper, queen, hive (ALL 3!) |
| `daemon-lifecycle` | keeper, queen |
| `rbee-operations` | keeper, queen |
| `rbee-config` | keeper, queen |
| `rbee-heartbeat` | queen, hive |

**Editing `narration-core` → All 3 binaries rebuild when spawned**

## Configuration

### Enable/Disable Auto-Update

**Keeper config** (`~/.config/rbee/config.toml`):
```toml
queen_port = 8500
auto_update_queen = true  # Enable auto-update for queen
```

**Queen config** (TBD):
```toml
auto_update_hive = true  # Enable auto-update for hive
```

**Hive config** (TBD):
```toml
auto_update_worker = true  # Enable auto-update for worker
```

## Manual Update Commands

### Force Rebuild Queen

```bash
./rbee queen update
```

### Force Rebuild Hive

```bash
./rbee hive update --id localhost
```

### Force Rebuild Worker

```bash
./rbee worker update --hive-id localhost --id worker-1
```

## Testing

### Unit Tests

```bash
cargo test -p auto-update
```

Tests:
- `test_find_workspace_root()` - Workspace detection
- `test_parse_dependencies()` - Dependency parsing
- `test_new_rbee_keeper()` - Updater creation
- `test_find_binary()` - Binary discovery

### Integration Test

```bash
# 1. Build keeper
cargo build --bin rbee-keeper

# 2. Edit shared crate
echo "// test" >> bin/99_shared_crates/daemon-lifecycle/src/lib.rs

# 3. Run keeper (should auto-rebuild)
./rbee queen start

# 4. Verify rebuild happened
# Should see: "🔨 Rebuilding rbee-keeper..."
```

## Performance

**Overhead:** ~10-50ms per check (parsing Cargo.toml + checking file mtimes)

**Negligible** compared to build time (seconds to minutes)

## Observability

All operations emit narration events:

```
🔨 Initializing auto-updater for queen-rbee
📦 Found 8 dependencies
🔍 Checking if queen-rbee needs rebuild
🔨 Dependency bin/99_shared_crates/daemon-lifecycle changed, rebuild needed
🔨 Rebuilding queen-rbee...
✅ Rebuilt queen-rbee successfully (duration: 12.3s)
```

## Related Crates

- **daemon-lifecycle** - Spawns daemons (keeper → queen)
- **hive-lifecycle** - Spawns hives (queen → hive)
- **worker-lifecycle** - Spawns workers (hive → worker)

## Future Enhancements

1. **Release builds** - Support `--release` flag
2. **Parallel builds** - Build multiple binaries concurrently
3. **Build caching** - Skip rebuild if Cargo cache is fresh
4. **Cross-compilation** - Support remote hives with different architectures
5. **File locking** - Prevent concurrent rebuilds of same binary
