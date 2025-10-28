# TEAM-314: daemon-contract Implementation

**Priority:** 🔴 CRITICAL  
**Estimated Time:** 2-3 days  
**Status:** 📋 PLAN

---

## Goal

Create a generic daemon lifecycle contract that works for queen, hive, and workers.

---

## Step 1: Create Crate Structure

```bash
cd /home/vince/Projects/llama-orch/bin/97_contracts
cargo new --lib daemon-contract
cd daemon-contract
```

### Cargo.toml

```toml
[package]
name = "daemon-contract"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"
description = "Generic daemon lifecycle contracts for rbee ecosystem"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
anyhow = "1.0"

[dev-dependencies]
serde_json = "1.0"

[lints]
workspace = true
```

---

## Step 2: Implement DaemonHandle (handle.rs)

### Source to Copy From

**Location:** `bin/05_rbee_keeper_crates/queen-lifecycle/src/types.rs`

**Search Command:**
```bash
# Find the current QueenHandle implementation
rg "pub struct QueenHandle" -A 30 bin/05_rbee_keeper_crates/queen-lifecycle/src/types.rs
```

### Implementation

**File:** `src/handle.rs`

```rust
//! Generic daemon handle for lifecycle management
//!
//! TEAM-314: Extracted from queen-lifecycle, made generic

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Generic daemon handle for lifecycle management
///
/// Tracks whether we started the daemon and provides cleanup.
/// IMPORTANT: Only shuts down daemon if we started it!
///
/// # Example
///
/// ```rust
/// use daemon_contract::DaemonHandle;
///
/// // Daemon already running
/// let handle = DaemonHandle::already_running("queen-rbee", "http://localhost:7833");
/// assert!(!handle.should_cleanup());
///
/// // We started the daemon
/// let handle = DaemonHandle::started_by_us("rbee-hive", "http://localhost:7835", Some(12345));
/// assert!(handle.should_cleanup());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonHandle {
    /// Daemon name (e.g., "queen-rbee", "rbee-hive", "vllm-worker")
    daemon_name: String,
    
    /// True if we started the daemon (must cleanup)
    /// False if daemon was already running (don't touch it)
    started_by_us: bool,

    /// Base URL of the daemon
    base_url: String,

    /// Process ID if we started it
    #[serde(skip_serializing_if = "Option::is_none")]
    pid: Option<u32>,
}

impl DaemonHandle {
    /// Create handle for daemon that was already running
    ///
    /// # Arguments
    /// * `daemon_name` - Name of the daemon
    /// * `base_url` - Base URL of the daemon
    pub fn already_running(daemon_name: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            started_by_us: false,
            base_url: base_url.into(),
            pid: None,
        }
    }

    /// Create handle for daemon that we just started
    ///
    /// # Arguments
    /// * `daemon_name` - Name of the daemon
    /// * `base_url` - Base URL of the daemon
    /// * `pid` - Process ID (if available)
    pub fn started_by_us(
        daemon_name: impl Into<String>,
        base_url: impl Into<String>,
        pid: Option<u32>,
    ) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            started_by_us: true,
            base_url: base_url.into(),
            pid,
        }
    }

    /// Check if we started the daemon (and should clean it up)
    pub const fn should_cleanup(&self) -> bool {
        self.started_by_us
    }

    /// Get the daemon's base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
    
    /// Get the daemon name
    pub fn daemon_name(&self) -> &str {
        &self.daemon_name
    }
    
    /// Get the process ID (if we started it)
    pub const fn pid(&self) -> Option<u32> {
        self.pid
    }
    
    /// Update the handle with discovered URL
    ///
    /// Service discovery - update URL after fetching from /v1/info
    pub fn with_discovered_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Keep the daemon alive (no shutdown after task)
    ///
    /// Daemon stays running for future tasks.
    pub fn shutdown(self) -> Result<()> {
        // Note: Actual shutdown logic is in lifecycle crates
        // This just indicates we're done with the handle
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_already_running() {
        let handle = DaemonHandle::already_running("test-daemon", "http://localhost:8080");
        assert!(!handle.should_cleanup());
        assert_eq!(handle.daemon_name(), "test-daemon");
        assert_eq!(handle.base_url(), "http://localhost:8080");
        assert_eq!(handle.pid(), None);
    }

    #[test]
    fn test_started_by_us() {
        let handle = DaemonHandle::started_by_us("test-daemon", "http://localhost:8080", Some(12345));
        assert!(handle.should_cleanup());
        assert_eq!(handle.daemon_name(), "test-daemon");
        assert_eq!(handle.base_url(), "http://localhost:8080");
        assert_eq!(handle.pid(), Some(12345));
    }

    #[test]
    fn test_with_discovered_url() {
        let handle = DaemonHandle::already_running("test-daemon", "http://localhost:8080")
            .with_discovered_url("http://192.168.1.100:8080");
        assert_eq!(handle.base_url(), "http://192.168.1.100:8080");
    }

    #[test]
    fn test_serialization() {
        let handle = DaemonHandle::started_by_us("test-daemon", "http://localhost:8080", Some(12345));
        let json = serde_json::to_string(&handle).unwrap();
        let deserialized: DaemonHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(handle.daemon_name(), deserialized.daemon_name());
        assert_eq!(handle.base_url(), deserialized.base_url());
        assert_eq!(handle.pid(), deserialized.pid());
    }
}
```

---

## Step 3: Implement Status Types (status.rs)

### Source to Copy From

**Location:** `bin/99_shared_crates/daemon-lifecycle/src/status.rs`

**Search Command:**
```bash
# Find status types
rg "pub struct Status" -A 10 bin/99_shared_crates/daemon-lifecycle/src/status.rs
```

### Implementation

**File:** `src/status.rs`

```rust
//! Daemon status types
//!
//! TEAM-314: Extracted from daemon-lifecycle

use serde::{Deserialize, Serialize};

/// Request to check daemon status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusRequest {
    /// ID of the daemon instance (e.g., alias, worker ID)
    pub id: String,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

/// Response from daemon status check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    /// ID of the daemon instance
    pub id: String,

    /// Whether the daemon is running
    pub is_running: bool,

    /// Health status (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_status: Option<String>,

    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_request_serialization() {
        let request = StatusRequest {
            id: "test-daemon".to_string(),
            job_id: Some("job-123".to_string()),
        };
        let json = serde_json::to_string(&request).unwrap();
        let deserialized: StatusRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(request.id, deserialized.id);
        assert_eq!(request.job_id, deserialized.job_id);
    }

    #[test]
    fn test_status_response_serialization() {
        let response = StatusResponse {
            id: "test-daemon".to_string(),
            is_running: true,
            health_status: Some("healthy".to_string()),
            metadata: None,
        };
        let json = serde_json::to_string(&response).unwrap();
        let deserialized: StatusResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(response.id, deserialized.id);
        assert_eq!(response.is_running, deserialized.is_running);
    }
}
```

---

## Step 4: Implement Install Types (install.rs)

### Source to Copy From

**Location:** `bin/99_shared_crates/daemon-lifecycle/src/install.rs`

**Search Command:**
```bash
# Find install types
rg "pub struct Install" -A 10 bin/99_shared_crates/daemon-lifecycle/src/install.rs
```

### Implementation

**File:** `src/install.rs`

```rust
//! Daemon installation types
//!
//! TEAM-314: Extracted from daemon-lifecycle

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Configuration for daemon installation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallConfig {
    /// Name of the daemon binary (e.g., "rbee-hive", "vllm-worker")
    pub binary_name: String,

    /// Optional path to binary (auto-detects if None)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,

    /// Optional target installation path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_path: Option<String>,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

/// Result of daemon installation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallResult {
    /// Path to the installed binary
    pub binary_path: String,

    /// Installation timestamp
    #[serde(with = "systemtime_serde")]
    pub install_time: SystemTime,
}

/// Configuration for daemon uninstallation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UninstallConfig {
    /// Name of the daemon (e.g., "queen-rbee", "rbee-hive")
    pub daemon_name: String,

    /// Installation path
    pub install_path: String,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

// Helper module for SystemTime serialization
mod systemtime_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap();
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + std::time::Duration::from_secs(secs))
    }
}
```

---

## Step 5: Implement Lifecycle Types (lifecycle.rs)

### Source to Copy From

**Location:** `bin/99_shared_crates/daemon-lifecycle/src/lifecycle.rs`

**Search Command:**
```bash
# Find lifecycle config
rg "pub struct HttpDaemonConfig" -A 10 bin/99_shared_crates/daemon-lifecycle/src/lifecycle.rs
```

### Implementation

**File:** `src/lifecycle.rs`

```rust
//! Daemon lifecycle configuration types
//!
//! TEAM-314: Extracted from daemon-lifecycle

use serde::{Deserialize, Serialize};

/// Configuration for HTTP-based daemons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpDaemonConfig {
    /// Daemon name for narration (e.g., "queen-rbee", "rbee-hive")
    pub daemon_name: String,

    /// Health check URL (e.g., "http://localhost:7833/health")
    pub health_url: String,

    /// Optional shutdown endpoint (e.g., "/v1/shutdown")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shutdown_endpoint: Option<String>,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}
```

---

## Step 6: Implement Shutdown Types (shutdown.rs)

### Source to Copy From

**Location:** `bin/99_shared_crates/daemon-lifecycle/src/shutdown.rs`

**Search Command:**
```bash
# Find shutdown config
rg "pub struct ShutdownConfig" -A 10 bin/99_shared_crates/daemon-lifecycle/src/shutdown.rs
```

### Implementation

**File:** `src/shutdown.rs`

```rust
//! Daemon shutdown configuration types
//!
//! TEAM-314: Extracted from daemon-lifecycle

use serde::{Deserialize, Serialize};

/// Configuration for graceful shutdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownConfig {
    /// Daemon name for narration
    pub daemon_name: String,

    /// Process ID to shut down
    pub pid: u32,

    /// Timeout for graceful shutdown (seconds)
    #[serde(default = "default_graceful_timeout")]
    pub graceful_timeout_secs: u64,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
}

fn default_graceful_timeout() -> u64 {
    5
}
```

---

## Step 7: Main Library File (lib.rs)

**File:** `src/lib.rs`

```rust
//! daemon-contract
//!
//! TEAM-314: Generic daemon lifecycle contracts
//!
//! # Purpose
//!
//! This crate provides the foundation types for daemon lifecycle management
//! across the rbee ecosystem. All daemons (queen, hive, workers) use these
//! contracts for consistent lifecycle management.
//!
//! # Components
//!
//! - **DaemonHandle** - Generic handle for all daemons
//! - **Status Types** - Status check protocol
//! - **Install Types** - Installation protocol
//! - **Lifecycle Types** - HTTP daemon configuration
//! - **Shutdown Types** - Shutdown configuration

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Generic daemon handle
pub mod handle;

/// Status types
pub mod status;

/// Installation types
pub mod install;

/// Lifecycle configuration
pub mod lifecycle;

/// Shutdown configuration
pub mod shutdown;

// Re-export main types
pub use handle::DaemonHandle;
pub use install::{InstallConfig, InstallResult, UninstallConfig};
pub use lifecycle::HttpDaemonConfig;
pub use shutdown::ShutdownConfig;
pub use status::{StatusRequest, StatusResponse};
```

---

## Step 8: Create README.md

**File:** `README.md`

```markdown
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

\`\`\`rust
use daemon_contract::DaemonHandle;

// Daemon already running
let handle = DaemonHandle::already_running("queen-rbee", "http://localhost:7833");

// We started the daemon
let handle = DaemonHandle::started_by_us("rbee-hive", "http://localhost:7835", Some(12345));

// Check if we should cleanup
if handle.should_cleanup() {
    // Shutdown daemon
}
\`\`\`

### Status Types

Protocol for checking daemon status:

\`\`\`rust
use daemon_contract::{StatusRequest, StatusResponse};

let request = StatusRequest {
    id: "workstation".to_string(),
    job_id: None,
};
\`\`\`

### Install Types

Protocol for daemon installation:

\`\`\`rust
use daemon_contract::InstallConfig;

let config = InstallConfig {
    binary_name: "rbee-hive".to_string(),
    binary_path: None,
    target_path: Some("/usr/local/bin".to_string()),
    job_id: None,
};
\`\`\`

## Usage

Add to your `Cargo.toml`:

\`\`\`toml
[dependencies]
daemon-contract = { path = "../97_contracts/daemon-contract" }
\`\`\`

## Type Aliases

Lifecycle crates can create type aliases for clarity:

\`\`\`rust
// queen-lifecycle
pub type QueenHandle = daemon_contract::DaemonHandle;

// hive-lifecycle
pub type HiveHandle = daemon_contract::DaemonHandle;

// worker-lifecycle
pub type WorkerHandle = daemon_contract::DaemonHandle;
\`\`\`

## License

GPL-3.0-or-later
```

---

## Step 9: Migration Guide

### Update queen-lifecycle

**File:** `bin/05_rbee_keeper_crates/queen-lifecycle/src/types.rs`

**Before:**
```rust
pub struct QueenHandle {
    started_by_us: bool,
    base_url: String,
    pid: Option<u32>,
}
```

**After:**
```rust
// TEAM-314: Use generic DaemonHandle from contract
pub use daemon_contract::DaemonHandle as QueenHandle;
```

### Add to hive-lifecycle

**File:** `bin/05_rbee_keeper_crates/hive-lifecycle/src/lib.rs`

**Add:**
```rust
// TEAM-314: Add HiveHandle
pub use daemon_contract::DaemonHandle as HiveHandle;
```

---

## Step 10: Testing

```bash
# Build the contract
cd bin/97_contracts/daemon-contract
cargo build

# Run tests
cargo test

# Check documentation
cargo doc --open

# Test with queen-lifecycle
cd ../../05_rbee_keeper_crates/queen-lifecycle
cargo build

# Test with hive-lifecycle
cd ../hive-lifecycle
cargo build

# Full integration test
cd ../../../
cargo build --bin rbee-keeper
```

---

## Verification Checklist

- [ ] daemon-contract crate created
- [ ] All types implemented
- [ ] All tests pass
- [ ] Documentation complete
- [ ] QueenHandle migrated
- [ ] HiveHandle added
- [ ] No breaking changes
- [ ] Full build passes

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27  
**Status:** PLAN 📋
