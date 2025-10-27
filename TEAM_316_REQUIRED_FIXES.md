# TEAM-316: Required Fixes for Contract Implementation

**Date:** 2025-10-27  
**Priority:** CRITICAL  
**Estimated Time:** 4-5 hours

---

## Overview

TEAM-315 created contracts but did NOT complete the migration. This document lists all required fixes.

---

## 🔴 FIX 1: Migrate daemon-lifecycle to use daemon-contract (CRITICAL)

**Priority:** CRITICAL  
**Time:** 2-3 hours  
**Impact:** HIGH - Affects all daemons

### Problem

daemon-lifecycle has duplicate types that should come from daemon-contract:
- StatusRequest
- StatusResponse
- InstallConfig
- InstallResult
- HttpDaemonConfig
- ShutdownConfig

### Files to Modify

1. `/bin/99_shared_crates/daemon-lifecycle/Cargo.toml`
2. `/bin/99_shared_crates/daemon-lifecycle/src/status.rs`
3. `/bin/99_shared_crates/daemon-lifecycle/src/install.rs`
4. `/bin/99_shared_crates/daemon-lifecycle/src/lifecycle.rs`
5. `/bin/99_shared_crates/daemon-lifecycle/src/shutdown.rs`

### Step-by-Step Instructions

#### Step 1: Add daemon-contract dependency

```toml
# File: daemon-lifecycle/Cargo.toml
[dependencies]
# ... existing dependencies ...
daemon-contract = { path = "../../97_contracts/daemon-contract" }
```

#### Step 2: Update status.rs

```rust
// File: daemon-lifecycle/src/status.rs
// TEAM-316: Use types from daemon-contract
pub use daemon_contract::{StatusRequest, StatusResponse};

// DELETE these duplicate definitions:
// pub struct StatusRequest { ... }
// pub struct StatusResponse { ... }

// UPDATE function signature:
pub async fn check_daemon_status(
    id: &str,
    health_url: &str,
    daemon_type: Option<&str>,
    job_id: Option<&str>,
) -> Result<StatusResponse> {
    // Create contract request
    let request = StatusRequest {
        id: id.to_string(),
        job_id: job_id.map(String::from),
    };
    
    // ... implementation ...
    
    Ok(StatusResponse {
        id: request.id,
        is_running: running,
        health_status: Some(health_url.to_string()),
        metadata: None,
    })
}
```

#### Step 3: Update install.rs

```rust
// File: daemon-lifecycle/src/install.rs
// TEAM-316: Use types from daemon-contract
pub use daemon_contract::{InstallConfig, InstallResult};

// DELETE these duplicate definitions:
// pub struct InstallConfig { ... }
// pub struct InstallResult { ... }

// UPDATE function to use contract types
pub async fn install_daemon(config: InstallConfig) -> Result<InstallResult> {
    // ... implementation ...
    
    Ok(InstallResult {
        binary_path: path.to_string_lossy().to_string(),
        install_time: std::time::SystemTime::now(),
    })
}
```

**NOTE:** daemon-contract's InstallResult has `install_time` but daemon-lifecycle has `found_in_target`. Decision needed:
- Option A: Add `found_in_target` to contract
- Option B: Return it separately from function

#### Step 4: Update lifecycle.rs

```rust
// File: daemon-lifecycle/src/lifecycle.rs
// TEAM-316: Use types from daemon-contract
pub use daemon_contract::HttpDaemonConfig;

// DELETE this duplicate definition:
// pub struct HttpDaemonConfig { ... }
```

#### Step 5: Update shutdown.rs

```rust
// File: daemon-lifecycle/src/shutdown.rs
// TEAM-316: Use types from daemon-contract
pub use daemon_contract::ShutdownConfig;

// DELETE this duplicate definition:
// pub struct ShutdownConfig { ... }
```

#### Step 6: Run tests

```bash
cargo test --package daemon-lifecycle
cargo test --workspace  # Ensure nothing breaks
```

### Type Mismatches to Resolve

#### StatusRequest

**Current daemon-lifecycle:**
```rust
pub struct StatusRequest {
    pub id: String,
    pub health_url: String,        // Extra field
    pub daemon_type: Option<String>, // Extra field
}
```

**Contract version:**
```rust
pub struct StatusRequest {
    pub id: String,
    pub job_id: Option<String>,
}
```

**Resolution:** Make `health_url` and `daemon_type` function parameters, not struct fields.

#### StatusResponse

**Current daemon-lifecycle:**
```rust
pub struct StatusResponse {
    pub id: String,
    pub running: bool,           // Field name mismatch
    pub health_url: String,      // Extra field
}
```

**Contract version:**
```rust
pub struct StatusResponse {
    pub id: String,
    pub is_running: bool,
    pub health_status: Option<String>,
    pub metadata: Option<serde_json::Value>,
}
```

**Resolution:** Use contract version (`is_running`), put health_url in `health_status`.

#### InstallResult

**Current daemon-lifecycle:**
```rust
pub struct InstallResult {
    pub binary_path: String,
    pub found_in_target: bool,
}
```

**Contract version:**
```rust
pub struct InstallResult {
    pub binary_path: String,
    pub install_time: SystemTime,
}
```

**Resolution:** Either:
1. Add `found_in_target: bool` to contract, OR
2. Return `found_in_target` separately from function

### Verification

```bash
# Check no duplicate types remain
rg "pub struct StatusRequest" bin/99_shared_crates/daemon-lifecycle/src/  # Should be 0 results
rg "pub struct StatusResponse" bin/99_shared_crates/daemon-lifecycle/src/  # Should be 0 results
rg "pub struct InstallConfig" bin/99_shared_crates/daemon-lifecycle/src/   # Should be 0 results
rg "pub struct InstallResult" bin/99_shared_crates/daemon-lifecycle/src/   # Should be 0 results
rg "pub struct HttpDaemonConfig" bin/99_shared_crates/daemon-lifecycle/src/ # Should be 0 results
rg "pub struct ShutdownConfig" bin/99_shared_crates/daemon-lifecycle/src/   # Should be 0 results

# Check imports are correct
rg "daemon_contract::" bin/99_shared_crates/daemon-lifecycle/src/  # Should have imports
```

---

## 🟡 FIX 2: Remove SshTarget duplication from tauri_commands (HIGH)

**Priority:** HIGH  
**Time:** 1 hour  
**Impact:** MEDIUM - Eliminates duplication

### Problem

`tauri_commands.rs` has its own SshTarget/SshTargetStatus types instead of using ssh-contract.

### Files to Modify

1. `/bin/00_rbee_keeper/src/tauri_commands.rs`
2. `/bin/97_contracts/ssh-contract/src/target.rs` (add Tauri derive if needed)

### Step-by-Step Instructions

#### Step 1: Check if ssh-contract needs Tauri derive

```bash
# Check what derives tauri_commands uses
grep -A 1 "pub struct SshTarget" bin/00_rbee_keeper/src/tauri_commands.rs
```

Output:
```rust
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SshTarget {
```

The `Type` derive is from `tauri-specta`. This might need to be added to ssh-contract.

**Option A:** Add `Type` derive to ssh-contract (if Tauri-specific)
**Option B:** Create a wrapper in tauri_commands that adds `Type` (cleaner)

#### Step 2: Update tauri_commands.rs

```rust
// File: bin/00_rbee_keeper/src/tauri_commands.rs

// TEAM-316: Use SSH types from contract
use ssh_contract::{SshTarget as SshTargetContract, SshTargetStatus as SshTargetStatusContract};

// TEAM-316: Tauri-specific wrapper with Type derive
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct SshTarget {
    #[serde(flatten)]
    inner: SshTargetContract,
}

impl From<SshTargetContract> for SshTarget {
    fn from(target: SshTargetContract) -> Self {
        Self { inner: target }
    }
}

// DELETE these duplicate definitions (lines 43-66):
// pub struct SshTarget { ... }
// pub enum SshTargetStatus { ... }

// UPDATE hive_list function:
#[tauri::command]
#[specta::specta]
pub async fn hive_list() -> Result<Vec<SshTarget>, String> {
    use observability_narration_core::n;
    
    n!("hive_list_start", "Reading SSH config");
    
    let config_path = shellexpand::tilde("~/.ssh/config");
    let config_path = Path::new(config_path.as_ref());
    
    let targets = ssh_config::parse_ssh_config(config_path)
        .map_err(|e| format!("Failed to parse SSH config: {}", e))?;
    
    n!("hive_list_parsed", "Found {} SSH targets", targets.len());
    
    // Convert to Tauri wrapper
    let converted: Vec<SshTarget> = targets
        .into_iter()
        .map(|t| t.into())
        .collect();
    
    Ok(converted)
}
```

**Alternative (simpler):** If Tauri's `Type` can be derived on ssh-contract:

```rust
// File: bin/00_rbee_keeper/src/tauri_commands.rs

// TEAM-316: Use SSH types from contract directly
pub use ssh_contract::{SshTarget, SshTargetStatus};

// DELETE duplicate definitions (lines 43-66)

// No conversion needed anymore!
```

Then add to ssh-contract:

```rust
// File: bin/97_contracts/ssh-contract/Cargo.toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
tauri-specta = { version = "2.0.0-rc", optional = true }  # Optional for Tauri

[features]
tauri = ["tauri-specta"]
```

```rust
// File: bin/97_contracts/ssh-contract/src/target.rs
#[cfg(feature = "tauri")]
use tauri_specta::Type;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "tauri", derive(Type))]
pub struct SshTarget {
    // ... fields ...
}
```

#### Step 3: Run tests

```bash
cargo test --package rbee-keeper
```

### Verification

```bash
# Check no duplicate SshTarget in tauri_commands
rg "pub struct SshTarget" bin/00_rbee_keeper/src/tauri_commands.rs  # Should be 0 or 1 (wrapper)
rg "pub enum SshTargetStatus" bin/00_rbee_keeper/src/tauri_commands.rs  # Should be 0
```

---

## 🟢 FIX 3: Add daemon-contract to hive-lifecycle (MEDIUM)

**Priority:** MEDIUM  
**Time:** 30 minutes  
**Impact:** MEDIUM - Consistency

### Problem

hive-lifecycle doesn't use daemon-contract, so no HiveHandle type alias exists.

### Files to Modify

1. `/bin/05_rbee_keeper_crates/hive-lifecycle/Cargo.toml`
2. `/bin/05_rbee_keeper_crates/hive-lifecycle/src/lib.rs`

### Step-by-Step Instructions

#### Step 1: Add daemon-contract dependency

```toml
# File: hive-lifecycle/Cargo.toml
[dependencies]
# ... existing dependencies ...
daemon-contract = { path = "../../97_contracts/daemon-contract" }  # TEAM-316: Generic daemon handle
```

#### Step 2: Add HiveHandle type alias

```rust
// File: hive-lifecycle/src/lib.rs

// Add after line 42:

// TEAM-316: Generic daemon handle
pub use daemon_contract::DaemonHandle;

/// Type alias for hive daemon handle
/// 
/// This is a specialization of the generic DaemonHandle for hive operations.
pub type HiveHandle = DaemonHandle;
```

#### Step 3: Run tests

```bash
cargo test --package hive-lifecycle
```

### Optional: Use daemon-contract types in operations

If daemon-lifecycle is fixed (FIX 1), hive-lifecycle will automatically use contract types through daemon-lifecycle.

### Verification

```bash
# Check HiveHandle exists
rg "pub type HiveHandle" bin/05_rbee_keeper_crates/hive-lifecycle/src/lib.rs

# Check dependency added
rg "daemon-contract" bin/05_rbee_keeper_crates/hive-lifecycle/Cargo.toml
```

---

## Summary Checklist

### FIX 1: daemon-lifecycle → daemon-contract
- [ ] Add daemon-contract dependency to Cargo.toml
- [ ] Update status.rs (use contract types)
- [ ] Update install.rs (use contract types)
- [ ] Update lifecycle.rs (use contract types)
- [ ] Update shutdown.rs (use contract types)
- [ ] Resolve type mismatches
- [ ] Run tests
- [ ] Verify no duplicates remain

### FIX 2: tauri_commands → ssh-contract
- [ ] Decide on Tauri derive strategy (wrapper vs feature)
- [ ] Remove duplicate SshTarget from tauri_commands
- [ ] Update hive_list function
- [ ] Run tests
- [ ] Verify no duplicates remain

### FIX 3: hive-lifecycle → daemon-contract
- [ ] Add daemon-contract dependency to Cargo.toml
- [ ] Add HiveHandle type alias
- [ ] Run tests
- [ ] Verify HiveHandle exists

---

## Final Verification

After all fixes:

```bash
# 1. All contract tests pass
cargo test --package daemon-contract
cargo test --package ssh-contract
cargo test --package keeper-config-contract

# 2. All affected crates compile
cargo check --package daemon-lifecycle
cargo check --package hive-lifecycle
cargo check --package rbee-keeper
cargo check --package queen-lifecycle

# 3. All tests pass
cargo test --workspace

# 4. No duplicates remain
rg "pub struct StatusRequest" --type rust bin/99_shared_crates/daemon-lifecycle/src/
rg "pub struct SshTarget" --type rust bin/00_rbee_keeper/src/tauri_commands.rs
# Both should return 0 results

# 5. Contracts are used
rg "use daemon_contract" --type rust bin/99_shared_crates/daemon-lifecycle/src/
rg "pub type HiveHandle" --type rust bin/05_rbee_keeper_crates/hive-lifecycle/src/
# Should have results
```

---

## Estimated Timeline

| Fix | Priority | Time | Complexity |
|-----|----------|------|------------|
| FIX 1: daemon-lifecycle | CRITICAL | 2-3h | HIGH (type mismatches) |
| FIX 2: tauri_commands | HIGH | 1h | MEDIUM (Tauri integration) |
| FIX 3: hive-lifecycle | MEDIUM | 30m | LOW (simple alias) |
| **TOTAL** | | **4-5h** | |

---

**Maintained by:** TEAM-316  
**Date:** 2025-10-27  
**Status:** ACTIONABLE FIXES
