# TEAM-328: DaemonManager Analysis - RULE ZERO Violation?

## Question
Does `daemon-lifecycle/src/manager.rs` violate RULE ZERO (backwards compatibility vs breaking changes)?

## Analysis

### Current Usage

**Internal to daemon-lifecycle:**
1. `start.rs` - Uses `DaemonManager::find_binary()` and `DaemonManager::new()` + `spawn()`
2. `install.rs` - Uses `DaemonManager::find_binary()`
3. Tests - Uses `DaemonManager` directly

**External usage:**
1. `worker-lifecycle/src/start.rs` - Uses `DaemonManager::new()` + `enable_auto_update()` + `spawn()`

**Public API exports (lib.rs):**
```rust
pub use manager::{spawn_daemon, DaemonManager};
```

### What DaemonManager Provides

**Three distinct responsibilities:**

1. **Binary Resolution** - `find_binary()`, `find_in_target()`
2. **Process Spawning** - `spawn()` with Stdio::null() configuration
3. **Auto-Update Integration** - `enable_auto_update()` (only used by worker-lifecycle)

### RULE ZERO Analysis

**Does this violate RULE ZERO?**

**YES - Multiple violations:**

#### Violation 1: Binary Resolution Duplication

`DaemonManager` provides TWO ways to find binaries:
- `find_binary()` - Checks installed → debug → release
- `find_in_target()` - Only checks debug → release

**Why this is entropy:**
- Two functions doing similar things
- Callers must know which one to use
- Both are public API (exported)

**Should be:**
- ONE function: `find_binary()` (already does everything)
- DELETE `find_in_target()` (redundant)

#### Violation 2: Spawning Wrapper

`spawn_daemon()` is a thin wrapper around `DaemonManager::new()` + `spawn()`:

```rust
pub async fn spawn_daemon<P: AsRef<Path>>(binary_path: P, args: Vec<String>) -> Result<Child> {
    let manager = DaemonManager::new(binary_path.as_ref().to_path_buf(), args);
    manager.spawn().await
}
```

**Why this is entropy:**
- Two ways to spawn: `spawn_daemon()` vs `DaemonManager::new().spawn()`
- No clear reason to have both
- Adds confusion: which should I use?

**Current callers:**
- NONE! Nobody uses `spawn_daemon()` - they all use `DaemonManager` directly

**Should be:**
- DELETE `spawn_daemon()` (unused wrapper)
- Use `DaemonManager::new().spawn()` directly

#### Violation 3: Auto-Update Complexity

`enable_auto_update()` is only used by `worker-lifecycle`:

```rust
let manager = DaemonManager::new(binary_path, args)
    .enable_auto_update(worker_type.binary_name(), source_dir);
```

**Why this might be entropy:**
- Only ONE caller in entire codebase
- Adds complexity to DaemonManager
- Could be handled differently

**However:**
- This is a legitimate feature (not duplication)
- Worker spawning needs auto-rebuild
- Not a RULE ZERO violation (it's a feature, not backwards compatibility)

### What Should Be Done

**RULE ZERO fixes:**

1. **DELETE `find_in_target()`** - Use `find_binary()` everywhere
   - Update internal callers to use `find_binary()`
   - Remove from public API

2. **DELETE `spawn_daemon()`** - Unused wrapper function
   - Remove from public API
   - No callers to update (nobody uses it!)

3. **KEEP `DaemonManager`** - Core functionality is sound
   - `new()` + `spawn()` - Essential
   - `find_binary()` - Essential
   - `enable_auto_update()` - Legitimate feature

### Impact Assessment

**Breaking changes:**
- `find_in_target()` removal - Only used internally, easy to fix
- `spawn_daemon()` removal - NO IMPACT (unused)

**Benefits:**
- Simpler API (2 fewer public functions)
- Less confusion (one way to find binaries, one way to spawn)
- Follows RULE ZERO (delete redundant code)

### Recommended Action

**Phase 1: Delete Redundant Functions**
```rust
// DELETE from manager.rs
pub fn find_in_target(name: &str) -> Result<PathBuf> { ... }

// DELETE from manager.rs
pub async fn spawn_daemon<P: AsRef<Path>>(...) -> Result<Child> { ... }

// UPDATE lib.rs exports
- pub use manager::{spawn_daemon, DaemonManager};
+ pub use manager::DaemonManager;
```

**Phase 2: Update Internal Callers**
- No changes needed! `find_binary()` already does everything `find_in_target()` does

**Phase 3: Update Tests**
- Change `DaemonManager::find_in_target()` → `DaemonManager::find_binary()`

## Conclusion

**YES, manager.rs violates RULE ZERO** by providing multiple ways to do the same thing:
- Two binary resolution functions (keep one)
- Unused wrapper function (delete)

**Recommended fixes:**
- DELETE `find_in_target()` (redundant)
- DELETE `spawn_daemon()` (unused)
- KEEP core `DaemonManager` functionality

**Estimated effort:** 30 minutes
**Risk:** Very low (minimal usage, easy to fix)
**Benefit:** Cleaner API, less confusion, follows RULE ZERO

---

**TEAM-328 Assessment:** manager.rs needs cleanup to eliminate backwards compatibility entropy
