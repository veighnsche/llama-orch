# rbee-keeper-queen-lifecycle

**Status:** 🚧 STUB (Created by TEAM-135)  
**Purpose:** Queen-rbee lifecycle management for rbee-keeper  
**Location:** `bin/rbee-keeper-crates/queen-lifecycle/` (keeper-specific)

---

## Overview

The `rbee-keeper-queen-lifecycle` crate manages the **lifecycle of queen-rbee daemon** from the rbee-keeper CLI. It provides commands to start, stop, restart, and check the status of the queen-rbee orchestrator.

### System Context

In the llama-orch architecture:

```
┌─────────────────┐
│  rbee-keeper    │  ← CLI tool (THIS CRATE)
│     (CLI)       │  ← Manages queen-rbee lifecycle
└────────┬────────┘
         │
         │ Lifecycle commands (start/stop/status)
         │ Uses daemon-lifecycle crate
         ↓
┌─────────────────┐
│   queen-rbee    │  ← Orchestrator daemon
│ (orchestratord) │  ← Managed by keeper
└─────────────────┘
```

**Key Responsibilities:**
- Start queen-rbee daemon
- Stop queen-rbee daemon
- Check queen-rbee status
- Restart queen-rbee daemon
- Handle daemon failures

---

## Uses Shared daemon-lifecycle

This crate uses the **shared `daemon-lifecycle` crate** for common lifecycle operations:

```rust
use daemon_lifecycle::{
    DaemonManager,
    DaemonConfig,
    DaemonStatus,
};
```

The `daemon-lifecycle` crate provides:
- Process spawning and monitoring
- PID file management
- Status checking
- Graceful shutdown
- Restart logic

---

## API Design

### Core Functions

```rust
/// Start queen-rbee daemon
pub fn start_queen(config: QueenConfig) -> Result<()>;

/// Stop queen-rbee daemon
pub fn stop_queen() -> Result<()>;

/// Check queen-rbee status
pub fn status_queen() -> Result<DaemonStatus>;

/// Restart queen-rbee daemon
pub fn restart_queen(config: QueenConfig) -> Result<()>;
```

---

## Dependencies

### Required

- **`daemon-lifecycle`**: Shared lifecycle management (from `shared-crates/`)
- **`tokio`**: Async runtime
- **`tracing`**: Structured logging

---

## Implementation Status

### Phase 1: Core Lifecycle (M1)
- [ ] Start queen-rbee daemon
- [ ] Stop queen-rbee daemon
- [ ] Check status
- [ ] PID file management

### Phase 2: Advanced Features (M2)
- [ ] Restart with config reload
- [ ] Health checks
- [ ] Log management
- [ ] Crash recovery

---

## Related Crates

### Used By
- **`rbee-keeper`**: Main CLI binary

### Uses
- **`shared-crates/daemon-lifecycle`**: Shared lifecycle management

### Similar Crates
- **`queen-rbee-crates/hive-lifecycle`**: Queen manages hive lifecycle
- **`rbee-hive-crates/worker-lifecycle`**: Hive manages worker lifecycle

---

## Team History

- **TEAM-135**: Scaffolding for new crate-based architecture

---

**Next Steps:**
1. Implement start/stop/status commands
2. Integrate with `daemon-lifecycle` crate
3. Add tests
4. Integrate with rbee-keeper CLI
