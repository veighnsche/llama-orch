# TEAM-211: Phase 2 - Simple Operations

**Assigned to:** TEAM-211  
**Depends on:** TEAM-210 (Foundation)  
**Blocks:** None (can run in parallel with TEAM-212, 213, 214)  
**Estimated LOC:** ~100 lines

---

## Mission

Implement read-only and simple hive operations:
- HiveList - List all configured hives
- HiveGet - Get details for a single hive
- HiveStatus - Check if hive is running

These are the easiest operations (no process management, no side effects).

---

## Source Code Reference

**From:** `job_router.rs` lines 821-921

### HiveList (lines 821-863)
```rust
Operation::HiveList => {
    NARRATE.action("hive_list").human("📊 Listing all hives").job_id(&job_id).emit();
    
    let hives: Vec<_> = state.config.hives.all().iter().map(|h| (&h.alias, *h)).collect();
    
    if hives.is_empty() {
        NARRATE.action("hive_empty").job_id(&job_id).human("No hives registered...").emit();
        return Ok(());
    }
    
    // Convert to JSON and display as table
    let hives_json: Vec<serde_json::Value> = hives.iter().map(...).collect();
    NARRATE.action("hive_result").job_id(&job_id).table(&serde_json::Value::Array(hives_json)).emit();
}
```

### HiveGet (lines 864-877)
```rust
Operation::HiveGet { alias } => {
    let hive_config = validate_hive_exists(&state.config, &alias)?;
    
    NARRATE.action("hive_get").context(&alias).human("Hive '{}' details:").job_id(&job_id).emit();
    
    println!("Alias: {}", alias);
    println!("Host: {}", hive_config.hostname);
    println!("Port: {}", hive_config.hive_port);
    if let Some(ref bp) = hive_config.binary_path {
        println!("Binary: {}", bp);
    }
}
```

### HiveStatus (lines 878-921)
```rust
Operation::HiveStatus { alias } => {
    let hive_config = validate_hive_exists(&state.config, &alias)?;
    
    let health_url = format!("http://{}:{}/health", hive_config.hostname, hive_config.hive_port);
    
    NARRATE.action("hive_check").job_id(&job_id).context(&health_url).human("Checking hive status at {}").emit();
    
    let client = reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(5)).build()?;
    
    match client.get(&health_url).send().await {
        Ok(response) if response.status().is_success() => {
            NARRATE.action("hive_check").job_id(&job_id).human("✅ Hive is running").emit();
        }
        Ok(response) => {
            NARRATE.action("hive_check").job_id(&job_id).human("⚠️  Hive responded with status: {}").emit();
        }
        Err(_) => {
            NARRATE.action("hive_check").job_id(&job_id).human("❌ Hive is not running").emit();
        }
    }
}
```

---

## Deliverables

### 1. Implement HiveList

**File:** `src/list.rs`
```rust
// TEAM-211: List all configured hives

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveListRequest, HiveListResponse, HiveInfo};

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// List all configured hives
///
/// Returns all hives from config (hives.conf).
/// If no hives configured, returns empty list.
///
/// # Arguments
/// * `request` - List request (no parameters)
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveListResponse)` - List of hives
/// * `Err` - Configuration error
pub async fn execute_hive_list(
    _request: HiveListRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveListResponse> {
    NARRATE
        .action("hive_list")
        .job_id(job_id)
        .human("📊 Listing all hives")
        .emit();

    let hives: Vec<HiveInfo> = config
        .hives
        .all()
        .iter()
        .map(|h| HiveInfo {
            alias: h.alias.clone(),
            hostname: h.hostname.clone(),
            hive_port: h.hive_port,
            binary_path: h.binary_path.clone(),
        })
        .collect();

    if hives.is_empty() {
        NARRATE
            .action("hive_empty")
            .job_id(job_id)
            .human(
                "No hives registered.\n\
                 \n\
                 To install a hive:\n\
                 \n\
                   ./rbee hive install",
            )
            .emit();
    } else {
        // Convert to JSON for table display
        let hives_json: Vec<serde_json::Value> = hives
            .iter()
            .map(|h| {
                serde_json::json!({
                    "alias": h.alias,
                    "host": h.hostname,
                    "port": h.hive_port,
                    "binary_path": h.binary_path.as_ref().unwrap_or(&"-".to_string()),
                })
            })
            .collect();

        NARRATE
            .action("hive_result")
            .job_id(job_id)
            .context(hives.len().to_string())
            .human("Found {} hive(s):")
            .table(&serde_json::Value::Array(hives_json))
            .emit();
    }

    Ok(HiveListResponse { hives })
}
```

### 2. Implement HiveGet

**File:** `src/get.rs`
```rust
// TEAM-211: Get details for a single hive

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveGetRequest, HiveGetResponse, HiveInfo};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Get details for a single hive
///
/// Returns hive configuration from hives.conf.
/// Validates that hive exists.
///
/// # Arguments
/// * `request` - Get request with hive alias
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveGetResponse)` - Hive details
/// * `Err` - Hive not found or configuration error
pub async fn execute_hive_get(
    request: HiveGetRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveGetResponse> {
    let hive_config = validate_hive_exists(&config, &request.alias)?;

    NARRATE
        .action("hive_get")
        .job_id(job_id)
        .context(&request.alias)
        .human("Hive '{}' details:")
        .emit();

    // Log details (matches original behavior)
    println!("Alias: {}", request.alias);
    println!("Host: {}", hive_config.hostname);
    println!("Port: {}", hive_config.hive_port);
    if let Some(ref bp) = hive_config.binary_path {
        println!("Binary: {}", bp);
    }

    let hive = HiveInfo {
        alias: request.alias.clone(),
        hostname: hive_config.hostname.clone(),
        hive_port: hive_config.hive_port,
        binary_path: hive_config.binary_path.clone(),
    };

    Ok(HiveGetResponse { hive })
}
```

### 3. Implement HiveStatus

**File:** `src/status.rs`
```rust
// TEAM-211: Check if hive is running

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;
use tokio::time::Duration;

use crate::types::{HiveStatusRequest, HiveStatusResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Check if hive is running
///
/// Performs HTTP health check to hive endpoint.
/// Timeout: 5 seconds.
///
/// # Arguments
/// * `request` - Status request with hive alias
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveStatusResponse)` - Health check result
/// * `Err` - Configuration error
pub async fn execute_hive_status(
    request: HiveStatusRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveStatusResponse> {
    let hive_config = validate_hive_exists(&config, &request.alias)?;

    let health_url = format!(
        "http://{}:{}/health",
        hive_config.hostname, hive_config.hive_port
    );

    NARRATE
        .action("hive_check")
        .job_id(job_id)
        .context(&health_url)
        .human("Checking hive status at {}")
        .emit();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    let running = match client.get(&health_url).send().await {
        Ok(response) if response.status().is_success() => {
            NARRATE
                .action("hive_check")
                .job_id(job_id)
                .context(&request.alias)
                .context(&health_url)
                .human("✅ Hive '{0}' is running on {1}")
                .emit();
            true
        }
        Ok(response) => {
            NARRATE
                .action("hive_check")
                .job_id(job_id)
                .context(&request.alias)
                .context(response.status().to_string())
                .human("⚠️  Hive '{0}' responded with status: {1}")
                .emit();
            false
        }
        Err(_) => {
            NARRATE
                .action("hive_check")
                .job_id(job_id)
                .context(&request.alias)
                .context(&health_url)
                .human("❌ Hive '{0}' is not running on {1}")
                .emit();
            false
        }
    };

    Ok(HiveStatusResponse {
        alias: request.alias,
        running,
        health_url,
    })
}
```

### 4. Update lib.rs Exports

**File:** `src/lib.rs` (add exports)
```rust
// TEAM-211: Export simple operations
pub use list::execute_hive_list;
pub use get::execute_hive_get;
pub use status::execute_hive_status;
```

---

## Acceptance Criteria

- [ ] `src/list.rs` implemented with execute_hive_list()
- [ ] `src/get.rs` implemented with execute_hive_get()
- [ ] `src/status.rs` implemented with execute_hive_status()
- [ ] All functions use NarrationFactory pattern
- [ ] All narration includes `.job_id(job_id)` for SSE routing
- [ ] Error messages match original exactly
- [ ] Crate compiles: `cargo check -p queen-rbee-hive-lifecycle`
- [ ] No TODO markers in TEAM-211 code
- [ ] All code has TEAM-211 signatures

---

## Testing

```bash
# Compile check
cargo check -p queen-rbee-hive-lifecycle

# Run tests (if any)
cargo test -p queen-rbee-hive-lifecycle

# Verify exports
cargo doc -p queen-rbee-hive-lifecycle --open
```

---

## Handoff to Next Teams

**What's Ready:**
- ✅ Simple operations complete (list, get, status)
- ✅ Validation helper tested
- ✅ Narration patterns established

**Parallel Work:**
- TEAM-212 can start on start/stop operations
- TEAM-213 can start on install/uninstall operations
- TEAM-214 can start on capabilities refresh

---

## Notes

- These operations are READ-ONLY (no side effects)
- No process management needed
- Simple HTTP health checks only
- Preserve exact error messages from job_router.rs
- All narration MUST include `.job_id(job_id)` for SSE routing
