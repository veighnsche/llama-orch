# TEAM-231: Daemon Lifecycle Behavior Inventory

**Date:** Oct 22, 2025  
**Crate:** `daemon-lifecycle`  
**Complexity:** High  
**Status:** ✅ COMPLETE

// TEAM-231: Investigated

---

## Executive Summary

Shared daemon lifecycle management for spawning and managing daemon processes across rbee-keeper, queen-rbee, and rbee-hive. Solves the "daemon holds parent's pipes open" bug with `Stdio::null()`.

**Key Behaviors:**
- DaemonManager pattern for process spawning
- Binary resolution (config → debug → release)
- SSH agent propagation
- Stdio::null() to prevent pipe hangs
- Narration-based observability

---

## 1. Core Architecture

### 1.1 DaemonManager

**Purpose:** Spawn and manage daemon processes with proper stdio handling

**Pattern:** Builder-like configuration + spawn

```rust
let manager = DaemonManager::new(
    PathBuf::from("target/debug/queen-rbee"),
    vec!["--config".to_string(), "config.toml".to_string()]
);

let child = manager.spawn().await?;
```

**Fields:**
- `binary_path: PathBuf` - Path to daemon binary
- `args: Vec<String>` - Command-line arguments

### 1.2 Key Methods

**`new(binary_path, args)`**
- Creates daemon manager instance
- No validation at construction time
- Validation happens at spawn time

**`spawn() -> Result<Child>`**
- Spawns daemon process
- Sets `Stdio::null()` for stdout/stderr (CRITICAL)
- Propagates SSH_AUTH_SOCK environment variable
- Returns tokio::process::Child handle

**`find_in_target(name) -> Result<PathBuf>`**
- Static method for binary resolution
- Search order: `target/debug/{name}` → `target/release/{name}`
- Returns first found binary
- Errors if not found in either location

---

## 2. Critical Bug Fix (TEAM-164)

### 2.1 The Problem

**Symptom:** E2E tests hang when using `Command::output()` to run rbee-keeper

**Root Cause:**
1. Daemon spawned with `Stdio::inherit()`
2. Parent runs via `Command::output()` → stdout/stderr are PIPES
3. Daemon inherits parent's pipe file descriptors
4. Parent exits, but daemon still holds pipes open
5. `Command::output()` waits for ALL pipe readers to close
6. Result: infinite hang

### 2.2 The Fix

**Solution:** Use `Stdio::null()` for daemon stdout/stderr

```rust
let mut cmd = Command::new(&self.binary_path);
cmd.args(&self.args)
    .stdout(Stdio::null())  // Don't inherit parent's stdout pipe
    .stderr(Stdio::null()); // Don't inherit parent's stderr pipe
```

**Why This Works:**
- Daemon no longer holds parent's pipes
- Parent can exit immediately
- `Command::output()` completes as expected
- Daemon continues running independently

**Testing:**
- ✅ `cargo xtask e2e:queen` (was hanging, now passes)
- ✅ Direct: `target/debug/rbee-keeper queen start` (still works)
- ✅ With output capture: works (no more hang)

---

## 3. SSH Agent Propagation (TEAM-189)

### 3.1 Purpose

Allow daemon to use parent's SSH agent for authentication

### 3.2 Implementation

```rust
// Propagate SSH agent socket if available
if let Ok(ssh_auth_sock) = std::env::var("SSH_AUTH_SOCK") {
    cmd.env("SSH_AUTH_SOCK", ssh_auth_sock);
}
```

**Behavior:**
- Checks for `SSH_AUTH_SOCK` environment variable
- If present, passes to daemon
- If absent, daemon runs without SSH agent
- No error if SSH agent not available

---

## 4. Binary Resolution

### 4.1 Search Order

1. **Config-provided path** (if absolute path given)
2. **Debug build:** `target/debug/{name}`
3. **Release build:** `target/release/{name}`

### 4.2 Usage Pattern

```rust
// Find binary in target directory
let binary_path = DaemonManager::find_in_target("queen-rbee")?;

// Create manager with found binary
let manager = DaemonManager::new(binary_path, args);
```

### 4.3 Error Handling

**Not Found:**
```
Binary 'queen-rbee' not found in target/debug or target/release
```

**Narration:**
- Success: `"Found binary '{name}' at: {path}"`
- Failure: `"Binary '{name}' not found in target/debug or target/release"`

---

## 5. Narration Integration

### 5.1 Actor

**Name:** `"dmn-life"` (8 chars, ≤10 limit)

**Pattern:** NarrationFactory

```rust
const NARRATE: NarrationFactory = NarrationFactory::new("dmn-life");
```

### 5.2 Actions

**`spawn`** - Starting daemon spawn
```rust
NARRATE.action("spawn")
    .context(binary_path)
    .context(args)
    .human("Spawning daemon: {0} with args: {1}")
    .emit();
```

**`spawned`** - Daemon successfully spawned
```rust
NARRATE.action("spawned")
    .context(pid)
    .human("Daemon spawned with PID: {}")
    .emit();
```

**`find_binary`** - Binary resolution
```rust
// Success
NARRATE.action("find_binary")
    .context(name)
    .context(path)
    .human("Found binary '{0}' at: {1}")
    .emit();

// Failure
NARRATE.action("find_binary")
    .context(name)
    .human("Binary '{}' not found in target/debug or target/release")
    .error_kind("binary_not_found")
    .emit_error();
```

---

## 6. Integration Points

### 6.1 Used By

**rbee-keeper:**
- Spawns queen-rbee daemon
- Usage: 1 import in product code

**Pattern:**
```rust
use daemon_lifecycle::DaemonManager;

// Find binary
let binary_path = DaemonManager::find_in_target("queen-rbee")?;

// Spawn daemon
let manager = DaemonManager::new(binary_path, vec![]);
let child = manager.spawn().await?;
```

### 6.2 NOT Used By (Yet)

**queen-rbee:**
- Should use for rbee-hive lifecycle
- Currently has custom implementation in hive-lifecycle crate

**rbee-hive:**
- Should use for worker lifecycle
- Currently has custom implementation

**Consolidation Opportunity:** All 3 binaries manage daemon lifecycles but only rbee-keeper uses this crate

---

## 7. Helper Functions

### 7.1 spawn_daemon()

**Purpose:** Convenience function for simple use cases

```rust
pub async fn spawn_daemon<P: AsRef<Path>>(
    binary_path: P,
    args: Vec<String>
) -> Result<Child>
```

**Usage:**
```rust
let child = spawn_daemon("target/debug/queen-rbee", vec![]).await?;
```

**Equivalent to:**
```rust
let manager = DaemonManager::new(binary_path.as_ref().to_path_buf(), args);
manager.spawn().await
```

---

## 8. Error Handling

### 8.1 Error Types

**Binary Not Found:**
- Error: `anyhow::Error`
- Message: `"Binary '{name}' not found in target/debug or target/release"`
- Narration: `error_kind("binary_not_found")`

**Spawn Failed:**
- Error: `anyhow::Error`
- Context: `"Failed to spawn daemon: {binary_path}"`
- Includes underlying OS error

### 8.2 Error Propagation

All errors use `anyhow::Result` for easy propagation with context

---

## 9. Test Coverage

### 9.1 Existing Tests

**BDD Tests:**
- Located in `bdd/` subdirectory
- Feature files in `bdd/features/`
- Step definitions in `bdd/src/steps/`

### 9.2 Test Gaps

**Missing Tests:**
- ❌ Daemon spawn success (with mock binary)
- ❌ Binary not found error handling
- ❌ SSH agent propagation verification
- ❌ Stdio::null() behavior (no pipe inheritance)
- ❌ PID capture and validation
- ❌ Child process lifecycle (spawn → run → exit)
- ❌ Concurrent daemon spawning
- ❌ Binary resolution priority (debug vs release)

**Why No Tests:**
- Requires actual binary compilation
- Needs process isolation
- Hard to test stdio behavior in unit tests
- BDD framework exists but no scenarios implemented

---

## 10. Performance Characteristics

**Spawn Time:**
- Dominated by OS process creation
- Narration overhead: <1ms
- Binary resolution: 2 filesystem checks (fast)

**Memory:**
- DaemonManager: ~100 bytes (PathBuf + Vec<String>)
- Child handle: OS-managed

**No Blocking:**
- All operations are async
- No busy-waiting
- No polling

---

## 11. Dependencies

**Core:**
- `tokio` - Async runtime, process management
- `anyhow` - Error handling
- `observability-narration-core` - Narration

**Standard Library:**
- `std::process::Stdio` - Stdio configuration
- `std::path::{Path, PathBuf}` - Path handling
- `std::env` - Environment variable access

---

## 12. Critical Behaviors Summary

1. **Stdio::null() is MANDATORY** - Prevents pipe hangs in E2E tests
2. **SSH agent propagation** - Enables daemon to use parent's SSH agent
3. **Binary resolution** - Debug first, then release
4. **Narration integration** - All operations emit narration
5. **Async-first** - All methods are async
6. **Error context** - All errors include helpful context
7. **No graceful shutdown** - Just spawns, doesn't manage lifecycle

---

## 13. Design Patterns

**Pattern:** Factory + Spawn

**Not Implemented:**
- ❌ Graceful shutdown (SIGTERM → SIGKILL)
- ❌ Health polling
- ❌ Process state tracking
- ❌ PID file management
- ❌ Auto-restart on crash

**Why:** This crate is minimal - just spawning. Lifecycle management is in individual binaries.

---

## 14. Future Consolidation

**Opportunity:** All 3 binaries manage daemon lifecycles

**Current State:**
- rbee-keeper → queen-rbee (uses daemon-lifecycle ✅)
- queen-rbee → rbee-hive (custom implementation ❌)
- rbee-hive → worker (custom implementation ❌)

**Recommendation:**
- Extend daemon-lifecycle with graceful shutdown
- Migrate queen-rbee hive lifecycle to use this crate
- Migrate rbee-hive worker lifecycle to use this crate
- Potential savings: ~500-800 LOC

---

**Handoff:** Ready for Phase 5 integration analysis  
**Next:** TEAM-232 (rbee-config + rbee-operations)
