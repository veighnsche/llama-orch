# daemon-contract

Generic daemon lifecycle contracts for the rbee ecosystem.

## Purpose

This crate provides the foundation types for daemon lifecycle management across all rbee daemons:
- queen-rbee
- rbee-hive
- vllm-worker
- Any future daemons

## Components

### DaemonHandle

Generic handle for tracking daemon lifecycle:

```rust
use daemon_contract::DaemonHandle;

// Daemon already running
let handle = DaemonHandle::already_running("queen-rbee", "http://localhost:7833");

// We started the daemon
let handle = DaemonHandle::started_by_us("rbee-hive", "http://localhost:7835", Some(12345));

// Check if we should cleanup
if handle.should_cleanup() {
    // Shutdown daemon
}
```

### Status Types

Protocol for checking daemon status:

```rust
use daemon_contract::{StatusRequest, StatusResponse};

let request = StatusRequest {
    id: "workstation".to_string(),
    job_id: None,
};
```

### Install Types

Protocol for daemon installation:

```rust
use daemon_contract::InstallConfig;

let config = InstallConfig {
    binary_name: "rbee-hive".to_string(),
    binary_path: None,
    target_path: Some("/usr/local/bin".to_string()),
    job_id: None,
};
```

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
daemon-contract = { path = "../97_contracts/daemon-contract" }
```

## Type Aliases

Lifecycle crates can create type aliases for clarity:

```rust
// queen-lifecycle
pub type QueenHandle = daemon_contract::DaemonHandle;

// hive-lifecycle
pub type HiveHandle = daemon_contract::DaemonHandle;

// worker-lifecycle
pub type WorkerHandle = daemon_contract::DaemonHandle;
```

## License

GPL-3.0-or-later
