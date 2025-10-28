# TEAM-314: ssh-contract Implementation

**Priority:** 🟡 HIGH  
**Estimated Time:** 1 day  
**Status:** 📋 PLAN

---

## Goal

Create SSH contract to eliminate duplication of `SshTarget` type.

---

## Step 1: Create Crate Structure

```bash
cd /home/vince/Projects/llama-orch/bin/97_contracts
cargo new --lib ssh-contract
cd ssh-contract
```

### Cargo.toml

```toml
[package]
name = "ssh-contract"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"
description = "SSH-related contracts for rbee ecosystem"

[dependencies]
serde = { version = "1.0", features = ["derive"] }

[dev-dependencies]
serde_json = "1.0"

[lints]
workspace = true
```

---

## Step 2: Find Current Implementation

### Search Commands

```bash
# Find SshTarget definition
rg "pub struct SshTarget" -A 15 bin/05_rbee_keeper_crates/ssh-config/src/lib.rs

# Find SshTargetStatus definition
rg "pub enum SshTargetStatus" -A 5 bin/05_rbee_keeper_crates/ssh-config/src/lib.rs

# Find duplication in tauri_commands
rg "pub struct SshTarget" bin/00_rbee_keeper/src/tauri_commands.rs
```

---

## Step 3: Implement SshTarget (target.rs)

### Source to Copy From

**Location:** `bin/05_rbee_keeper_crates/ssh-config/src/lib.rs`

### Implementation

**File:** `src/target.rs`

```rust
//! SSH target types
//!
//! TEAM-314: Extracted from ssh-config to eliminate duplication

use serde::{Deserialize, Serialize};

/// SSH target from ~/.ssh/config
///
/// Represents a host entry from SSH configuration with connection details.
///
/// # Example
///
/// ```rust
/// use ssh_contract::{SshTarget, SshTargetStatus};
///
/// let target = SshTarget {
///     host: "workstation".to_string(),
///     host_subtitle: None,
///     hostname: "192.168.1.100".to_string(),
///     user: "vince".to_string(),
///     port: 22,
///     status: SshTargetStatus::Unknown,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SshTarget {
    /// Host alias from SSH config (first word)
    pub host: String,
    
    /// Host subtitle (second word, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_subtitle: Option<String>,
    
    /// Hostname (IP or domain)
    pub hostname: String,
    
    /// SSH username
    pub user: String,
    
    /// SSH port
    pub port: u16,
    
    /// Connection status
    pub status: SshTargetStatus,
}

impl SshTarget {
    /// Create a new SSH target
    pub fn new(
        host: impl Into<String>,
        hostname: impl Into<String>,
        user: impl Into<String>,
        port: u16,
    ) -> Self {
        Self {
            host: host.into(),
            host_subtitle: None,
            hostname: hostname.into(),
            user: user.into(),
            port,
            status: SshTargetStatus::Unknown,
        }
    }

    /// Create with subtitle
    pub fn with_subtitle(mut self, subtitle: impl Into<String>) -> Self {
        self.host_subtitle = Some(subtitle.into());
        self
    }

    /// Update connection status
    pub fn with_status(mut self, status: SshTargetStatus) -> Self {
        self.status = status;
        self
    }

    /// Get full SSH connection string (user@hostname:port)
    pub fn connection_string(&self) -> String {
        format!("{}@{}:{}", self.user, self.hostname, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssh_target_creation() {
        let target = SshTarget::new("workstation", "192.168.1.100", "vince", 22);
        assert_eq!(target.host, "workstation");
        assert_eq!(target.hostname, "192.168.1.100");
        assert_eq!(target.user, "vince");
        assert_eq!(target.port, 22);
        assert_eq!(target.status, SshTargetStatus::Unknown);
    }

    #[test]
    fn test_connection_string() {
        let target = SshTarget::new("workstation", "192.168.1.100", "vince", 2222);
        assert_eq!(target.connection_string(), "vince@192.168.1.100:2222");
    }

    #[test]
    fn test_serialization() {
        let target = SshTarget::new("workstation", "192.168.1.100", "vince", 22)
            .with_status(SshTargetStatus::Online);
        
        let json = serde_json::to_string(&target).unwrap();
        let deserialized: SshTarget = serde_json::from_str(&json).unwrap();
        
        assert_eq!(target, deserialized);
    }
}
```

---

## Step 4: Implement SshTargetStatus (status.rs)

**File:** `src/status.rs`

```rust
//! SSH target connection status
//!
//! TEAM-314: Extracted from ssh-config

use serde::{Deserialize, Serialize};

/// SSH target connection status
///
/// Indicates whether an SSH host is reachable.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SshTargetStatus {
    /// Host is reachable and responding
    Online,
    
    /// Host is unreachable or not responding
    Offline,
    
    /// Status has not been checked yet
    Unknown,
}

impl Default for SshTargetStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

impl SshTargetStatus {
    /// Check if host is online
    pub const fn is_online(&self) -> bool {
        matches!(self, Self::Online)
    }

    /// Check if host is offline
    pub const fn is_offline(&self) -> bool {
        matches!(self, Self::Offline)
    }

    /// Check if status is unknown
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// Get status as string
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Online => "online",
            Self::Offline => "offline",
            Self::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_checks() {
        assert!(SshTargetStatus::Online.is_online());
        assert!(SshTargetStatus::Offline.is_offline());
        assert!(SshTargetStatus::Unknown.is_unknown());
    }

    #[test]
    fn test_status_as_str() {
        assert_eq!(SshTargetStatus::Online.as_str(), "online");
        assert_eq!(SshTargetStatus::Offline.as_str(), "offline");
        assert_eq!(SshTargetStatus::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_serialization() {
        let status = SshTargetStatus::Online;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"online\"");
        
        let deserialized: SshTargetStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, deserialized);
    }
}
```

---

## Step 5: Main Library File (lib.rs)

**File:** `src/lib.rs`

```rust
//! ssh-contract
//!
//! TEAM-314: SSH-related contracts for rbee ecosystem
//!
//! # Purpose
//!
//! This crate provides SSH-related types used across the rbee ecosystem.
//! It eliminates duplication of `SshTarget` between ssh-config and tauri_commands.
//!
//! # Components
//!
//! - **SshTarget** - SSH host information from ~/.ssh/config
//! - **SshTargetStatus** - Connection status (online/offline/unknown)

#![warn(missing_docs)]
#![warn(clippy::all)]

/// SSH target types
pub mod target;

/// SSH target status
pub mod status;

// Re-export main types
pub use status::SshTargetStatus;
pub use target::SshTarget;
```

---

## Step 6: Create README.md

**File:** `README.md`

```markdown
# ssh-contract

SSH-related contracts for the rbee ecosystem.

## Purpose

This crate provides SSH-related types used across rbee components:
- ssh-config (parser)
- rbee-keeper (CLI)
- Tauri UI (frontend)

## Components

### SshTarget

Represents an SSH host from `~/.ssh/config`:

\`\`\`rust
use ssh_contract::{SshTarget, SshTargetStatus};

let target = SshTarget::new("workstation", "192.168.1.100", "vince", 22)
    .with_status(SshTargetStatus::Online);

println!("Connect to: {}", target.connection_string());
// Output: Connect to: vince@192.168.1.100:22
\`\`\`

### SshTargetStatus

Connection status:

\`\`\`rust
use ssh_contract::SshTargetStatus;

let status = SshTargetStatus::Online;
assert!(status.is_online());
\`\`\`

## Usage

Add to your `Cargo.toml`:

\`\`\`toml
[dependencies]
ssh-contract = { path = "../97_contracts/ssh-contract" }
\`\`\`

## License

GPL-3.0-or-later
```

---

## Step 7: Migration Guide

### Update ssh-config

**File:** `bin/05_rbee_keeper_crates/ssh-config/src/lib.rs`

**Before:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshTarget {
    pub host: String,
    // ...
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SshTargetStatus {
    Online,
    Offline,
    Unknown,
}
```

**After:**
```rust
// TEAM-314: Use types from ssh-contract
pub use ssh_contract::{SshTarget, SshTargetStatus};
```

### Update tauri_commands

**File:** `bin/00_rbee_keeper/src/tauri_commands.rs`

**Before:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SshTarget {
    pub host: String,
    // ... (duplicate definition)
}
```

**After:**
```rust
// TEAM-314: Use SshTarget from contract, add Tauri Type derive
use ssh_contract::SshTarget as SshTargetContract;
use specta::Type;

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SshTarget {
    #[serde(flatten)]
    inner: SshTargetContract,
}

// Or if Tauri supports it, just:
pub use ssh_contract::SshTarget;
```

---

## Step 8: Testing

```bash
# Build the contract
cd bin/97_contracts/ssh-contract
cargo build

# Run tests
cargo test

# Check documentation
cargo doc --open

# Test with ssh-config
cd ../../05_rbee_keeper_crates/ssh-config
cargo build

# Test with rbee-keeper
cd ../../00_rbee_keeper
cargo build

# Full integration test
cd ../../
cargo build --bin rbee-keeper
```

---

## Verification Checklist

- [ ] ssh-contract crate created
- [ ] SshTarget implemented
- [ ] SshTargetStatus implemented
- [ ] All tests pass
- [ ] Documentation complete
- [ ] ssh-config updated
- [ ] tauri_commands updated
- [ ] No duplication remains
- [ ] Full build passes

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27  
**Status:** PLAN 📋
