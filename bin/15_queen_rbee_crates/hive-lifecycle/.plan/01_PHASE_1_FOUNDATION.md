# TEAM-210: Phase 1 - Foundation

**Assigned to:** TEAM-210  
**Depends on:** None  
**Blocks:** TEAM-211, TEAM-212, TEAM-213, TEAM-214  
**Estimated LOC:** ~150 lines

---

## Mission

Set up the foundational structure for hive-lifecycle crate:
- Request/Response types for all operations
- Module structure
- Validation helpers
- Dependencies

---

## Deliverables

### 1. Update Cargo.toml Dependencies

Add missing dependencies:
```toml
[dependencies]
anyhow = "1.0"
tokio = { version = "1.0", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
once_cell = "1.19"  # TEAM-209: Required for Lazy static in validation helper

# Internal dependencies
daemon-lifecycle = { path = "../../99_shared_crates/daemon-lifecycle" }
observability-narration-core = { path = "../../99_shared_crates/narration-core" }
timeout-enforcer = { path = "../../99_shared_crates/timeout-enforcer" }
rbee-config = { path = "../../99_shared_crates/rbee-config" }
queen-rbee-ssh-client = { path = "../ssh-client" }
```

### 2. Create Module Structure

**File:** `src/lib.rs`
```rust
// TEAM-210: Hive lifecycle management crate structure

pub mod types;
pub mod validation;
pub mod install;
pub mod uninstall;
pub mod start;
pub mod stop;
pub mod list;
pub mod get;
pub mod status;
pub mod capabilities;

// Re-export SSH test (already implemented)
pub use types::*;

// Re-export existing SSH test
mod ssh_test;
pub use ssh_test::{execute_ssh_test, SshTestRequest, SshTestResponse};
```

### 3. Create Request/Response Types

**File:** `src/types.rs`
```rust
// TEAM-210: Request/Response types for all hive operations

use serde::{Deserialize, Serialize};

// ============================================================================
// INSTALL
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveInstallRequest {
    pub alias: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveInstallResponse {
    pub success: bool,
    pub message: String,
    pub binary_path: Option<String>,
}

// ============================================================================
// UNINSTALL
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveUninstallRequest {
    pub alias: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveUninstallResponse {
    pub success: bool,
    pub message: String,
}

// ============================================================================
// START
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStartRequest {
    pub alias: String,
    pub job_id: String,  // CRITICAL: Required for SSE routing
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStartResponse {
    pub success: bool,
    pub message: String,
    pub endpoint: Option<String>,
}

// ============================================================================
// STOP
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStopRequest {
    pub alias: String,
    pub job_id: String,  // CRITICAL: Required for SSE routing
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStopResponse {
    pub success: bool,
    pub message: String,
}

// ============================================================================
// LIST
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveListRequest {
    // No parameters needed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveListResponse {
    pub hives: Vec<HiveInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveInfo {
    pub alias: String,
    pub hostname: String,
    pub hive_port: u16,
    pub binary_path: Option<String>,
}

// ============================================================================
// GET
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveGetRequest {
    pub alias: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveGetResponse {
    pub hive: HiveInfo,
}

// ============================================================================
// STATUS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStatusRequest {
    pub alias: String,
    pub job_id: String,  // CRITICAL: Required for SSE routing
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveStatusResponse {
    pub alias: String,
    pub running: bool,
    pub health_url: String,
}

// ============================================================================
// REFRESH CAPABILITIES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveRefreshCapabilitiesRequest {
    pub alias: String,
    pub job_id: String,  // CRITICAL: Required for SSE routing
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveRefreshCapabilitiesResponse {
    pub success: bool,
    pub device_count: usize,
    pub message: String,
}
```

### 4. Create Validation Helpers

**File:** `src/validation.rs`
```rust
// TEAM-210: Validation helpers for hive operations

use anyhow::Result;
use rbee_config::{RbeeConfig, HiveEntry};
use std::path::PathBuf;
use once_cell::sync::Lazy;

/// Validate that a hive alias exists in config
///
/// Returns helpful error message listing available hives if alias not found.
/// Special case: "localhost" always returns a default entry.
///
/// COPIED FROM: job_router.rs lines 98-160
pub fn validate_hive_exists<'a>(
    config: &'a RbeeConfig,
    alias: &str,
) -> Result<&'a HiveEntry> {
    if alias == "localhost" {
        // Localhost operations do not require configuration
        static LOCALHOST_ENTRY: Lazy<HiveEntry> = Lazy::new(|| HiveEntry {
            alias: "localhost".to_string(),
            hostname: "127.0.0.1".to_string(),
            ssh_port: 22,
            ssh_user: "user".to_string(),
            hive_port: 9000,
            binary_path: Some("target/debug/rbee-hive".to_string()),
        });
        return Ok(&LOCALHOST_ENTRY);
    }

    config.hives.get(alias).ok_or_else(|| {
        let available_hives = config.hives.all();
        let hive_list = if available_hives.is_empty() {
            "  (none configured)".to_string()
        } else {
            available_hives
                .iter()
                .map(|h| format!("  - {}", h.alias))
                .collect::<Vec<_>>()
                .join("\n")
        };

        // Check if hives.conf exists
        let config_dir = RbeeConfig::config_dir().unwrap_or_else(|_| PathBuf::from("~/.config/rbee"));
        let hives_conf_path = config_dir.join("hives.conf");
        if !hives_conf_path.exists() && alias != "localhost" {
            // Auto-generate a template for hives.conf
            let template_content = format!(
                "# hives.conf - rbee hive configuration\n\nHost {}\n  HostName <hostname or IP>\n  Port 22\n  User <username>\n  HivePort 8600\n  BinaryPath /path/to/rbee-hive\n",
                alias
            );
            let _ = std::fs::write(&hives_conf_path, &template_content);

            anyhow::anyhow!(
                "Hive alias '{}' not found in hives.conf.\n\nAvailable hives:\n{}\n\nA template hives.conf has been auto-generated at {}.\nPlease edit this file to configure access to '{}'.\n\nExample configuration:\n{}\n",
                alias,
                hive_list,
                hives_conf_path.display(),
                alias,
                template_content
            )
        } else {
            anyhow::anyhow!(
                "Hive alias '{}' not found in hives.conf.\n\nAvailable hives:\n{}\n\nAdd '{}' to ~/.config/rbee/hives.conf to use it.",
                alias,
                hive_list,
                alias
            )
        }
    })
}
```

### 5. Create Module Stubs

**File:** `src/install.rs`
```rust
// TEAM-210: Stub for TEAM-213
// TODO: TEAM-213 will implement this
```

**File:** `src/uninstall.rs`
```rust
// TEAM-210: Stub for TEAM-213
// TODO: TEAM-213 will implement this
```

**File:** `src/start.rs`
```rust
// TEAM-210: Stub for TEAM-212
// TODO: TEAM-212 will implement this
```

**File:** `src/stop.rs`
```rust
// TEAM-210: Stub for TEAM-212
// TODO: TEAM-212 will implement this
```

**File:** `src/list.rs`
```rust
// TEAM-210: Stub for TEAM-211
// TODO: TEAM-211 will implement this
```

**File:** `src/get.rs`
```rust
// TEAM-210: Stub for TEAM-211
// TODO: TEAM-211 will implement this
```

**File:** `src/status.rs`
```rust
// TEAM-210: Stub for TEAM-211
// TODO: TEAM-211 will implement this
```

**File:** `src/capabilities.rs`
```rust
// TEAM-210: Stub for TEAM-214
// TODO: TEAM-214 will implement this
```

### 6. Move SSH Test to Dedicated Module

**File:** `src/ssh_test.rs`
```rust
// TEAM-210: Moved from lib.rs for better organization

use anyhow::Result;
use observability_narration_core::Narration;
use queen_rbee_ssh_client::{test_ssh_connection, SshConfig};

// Actor and action constants
const ACTOR_HIVE_LIFECYCLE: &str = "🐝 hive-lifecycle";
const ACTION_SSH_TEST: &str = "ssh_test";

// Request/Response types (keep existing)
#[derive(Debug, Clone)]
pub struct SshTestRequest {
    pub ssh_host: String,
    pub ssh_port: u16,
    pub ssh_user: String,
}

#[derive(Debug, Clone)]
pub struct SshTestResponse {
    pub success: bool,
    pub error: Option<String>,
    pub test_output: Option<String>,
}

// Keep existing execute_ssh_test implementation
pub async fn execute_ssh_test(request: SshTestRequest) -> Result<SshTestResponse> {
    // ... existing implementation ...
}
```

---

## Acceptance Criteria

- [ ] Cargo.toml updated with all dependencies
- [ ] Module structure created (lib.rs with 10 modules)
- [ ] All request/response types defined in types.rs
- [ ] Validation helper implemented and tested
- [ ] All module stubs created with TEAM-XXX markers
- [ ] SSH test moved to dedicated module
- [ ] Crate compiles: `cargo check -p queen-rbee-hive-lifecycle`
- [ ] No TODO markers in TEAM-210 code
- [ ] All code has TEAM-210 signatures

---

## Handoff to TEAM-211

**What's Ready:**
- ✅ Foundation structure complete
- ✅ Types defined for all operations
- ✅ Validation helper ready to use
- ✅ Dependencies configured

**Next Steps for TEAM-211:**
1. Read `02_PHASE_2_SIMPLE_OPERATIONS.md`
2. Implement list.rs, get.rs, status.rs
3. Use types from `src/types.rs`
4. Use validation from `src/validation.rs`

---

## Notes

- All narration MUST include `.job_id(&job_id)` for SSE routing (see MEMORY)
- Follow daemon-lifecycle patterns for process management
- Preserve exact error messages from job_router.rs
- Use NarrationFactory pattern: `const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");`

---

## TEAM-209 CRITICAL FINDINGS

### ⚠️  ARCHITECTURAL GAP: device-detection Dependency

**The plans are missing a critical piece of the architecture:**

```
queen-rbee (this crate)          rbee-hive                device-detection
     ↓                                ↓                         ↓
fetch_hive_capabilities()  →  GET /capabilities  →  detect_gpus()
     ↓                                ↓                         ↓
JSON response           ←  HiveDevice JSON     ←  GpuInfo structs
```

**What's missing:**
- Phase 3 & 5 talk about "fetching capabilities" but don't explain HOW hive generates them
- `rbee-hive` uses `rbee-hive-device-detection` crate (in `bin/25_rbee_hive_crates/`)
- This crate calls `nvidia-smi`, parses output, and returns device info
- **Location:** `bin/20_rbee_hive/src/main.rs:156`

**Impact:**
- TEAM-212 (Phase 3) needs to understand this flow when implementing capabilities fetch
- TEAM-214 (Phase 5) needs to document the full chain: queen → hive → device-detection
- Error handling must account for device detection failures (nvidia-smi not found, parse errors, etc.)

**Action Required:**
- Update Phase 3 & 5 documentation to explain the full architectural flow
- Add error handling notes for device detection failures
- Reference device-detection README: `bin/25_rbee_hive_crates/device-detection/README.md`
