# daemon-lifecycle Usage Guide

**Purpose:** Shared lifecycle management for all daemon-managing binaries

---

## Overview

The `daemon-lifecycle` crate provides **common lifecycle operations** for managing daemon processes. It is used by all binary-specific lifecycle crates to eliminate code duplication.

---

## Architecture

### Lifecycle Chain

```
┌─────────────────┐
│  rbee-keeper    │  ← Uses rbee-keeper-crates/queen-lifecycle
│     (CLI)       │     └─> Uses daemon-lifecycle (shared)
└────────┬────────┘
         │ Manages
         ↓
┌─────────────────┐
│   queen-rbee    │  ← Uses queen-rbee-crates/hive-lifecycle
│ (orchestratord) │     └─> Uses daemon-lifecycle (shared)
└────────┬────────┘
         │ Manages
         ↓
┌─────────────────┐
│   rbee-hive     │  ← Uses rbee-hive-crates/worker-lifecycle
│ (pool-managerd) │     └─> Uses daemon-lifecycle (shared)
└────────┬────────┘
         │ Manages
         ↓
┌─────────────────┐
│ llm-worker-rbee │  ← Worker (no lifecycle management)
│  (worker-orcd)  │
└─────────────────┘
```

---

## Used By

### 1. rbee-keeper-crates/queen-lifecycle

**Purpose:** Manage queen-rbee daemon from keeper CLI

**Location:** `bin/rbee-keeper-crates/queen-lifecycle/`

**Cargo.toml:**
```toml
[dependencies]
daemon-lifecycle = { path = "../../shared-crates/daemon-lifecycle" }
```

**Usage:**
```rust
use daemon_lifecycle::{DaemonManager, DaemonConfig};

// Start queen-rbee
let config = DaemonConfig {
    binary_path: "/usr/local/bin/queen-rbee",
    args: vec!["--config", "/etc/rbee/queen.toml"],
    pid_file: "/var/run/queen-rbee.pid",
    // ...
};

let manager = DaemonManager::new(config);
manager.start()?;
```

---

### 2. queen-rbee-crates/hive-lifecycle

**Purpose:** Manage rbee-hive daemons from queen orchestrator

**Location:** `bin/queen-rbee-crates/hive-lifecycle/`

**Cargo.toml:**
```toml
[dependencies]
daemon-lifecycle = { path = "../../shared-crates/daemon-lifecycle" }
```

**Usage:**
```rust
use daemon_lifecycle::{DaemonManager, DaemonConfig};

// Start rbee-hive on remote node
let config = DaemonConfig {
    binary_path: "/usr/local/bin/rbee-hive",
    args: vec!["--config", "/etc/rbee/hive.toml"],
    pid_file: "/var/run/rbee-hive.pid",
    // ...
};

let manager = DaemonManager::new(config);
manager.start()?;
```

---

### 3. rbee-hive-crates/worker-lifecycle

**Purpose:** Manage llm-worker-rbee processes from hive

**Location:** `bin/rbee-hive-crates/worker-lifecycle/`

**Cargo.toml:**
```toml
[dependencies]
daemon-lifecycle = { path = "../../shared-crates/daemon-lifecycle" }
```

**Usage:**
```rust
use daemon_lifecycle::{DaemonManager, DaemonConfig};

// Start worker
let config = DaemonConfig {
    binary_path: "/usr/local/bin/llm-worker-rbee",
    args: vec!["--model", "llama-7b", "--gpu", "0"],
    pid_file: "/var/run/worker-123.pid",
    // ...
};

let manager = DaemonManager::new(config);
manager.start()?;
```

---

## Common Operations

### Start Daemon

```rust
use daemon_lifecycle::{DaemonManager, DaemonConfig};

let config = DaemonConfig::new("/usr/local/bin/daemon");
let manager = DaemonManager::new(config);

manager.start()?;
```

### Stop Daemon

```rust
manager.stop()?;
```

### Check Status

```rust
use daemon_lifecycle::DaemonStatus;

let status = manager.status()?;
match status {
    DaemonStatus::Running(pid) => println!("Running with PID {}", pid),
    DaemonStatus::Stopped => println!("Stopped"),
    DaemonStatus::Failed(error) => println!("Failed: {}", error),
}
```

### Restart Daemon

```rust
manager.restart()?;
```

---

## Code Savings

**Before (duplicated across 3 binaries):**
- rbee-keeper → queen-rbee: ~132 LOC
- queen-rbee → rbee-hive: ~800 LOC
- rbee-hive → llm-worker: ~386 LOC
- **Total:** ~1,318 LOC

**After (shared crate):**
- daemon-lifecycle: ~500 LOC (shared)
- Binary-specific wrappers: ~50 LOC each × 3 = ~150 LOC
- **Total:** ~650 LOC

**Savings:** ~668 LOC (~50% reduction)

---

## Related Documentation

- **Architecture:** `bin/.plan/TEAM_130G_FINAL_ARCHITECTURE.md`
- **Consolidation:** `bin/.plan/TEAM_130E_CONSOLIDATION_SUMMARY.md`
- **Lifecycle crates:**
  - `bin/rbee-keeper-crates/queen-lifecycle/README.md`
  - `bin/queen-rbee-crates/hive-lifecycle/README.md`
  - `bin/rbee-hive-crates/worker-lifecycle/README.md`

---

**Last Updated:** 2025-10-19
