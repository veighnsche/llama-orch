# TEAM-212: Phase 3 - Lifecycle Core (Start/Stop)

**Assigned to:** TEAM-212  
**Depends on:** TEAM-210 (Foundation)  
**Blocks:** TEAM-215 (Integration)  
**Estimated LOC:** ~350 lines

---

## Mission

Implement the core lifecycle operations (most complex):
- HiveStart - Binary resolution, spawning, health polling, capabilities caching
- HiveStop - Graceful shutdown with SIGTERM/SIGKILL fallback

These are the most critical operations with process management.

---

## Source Code Reference

**From:** `job_router.rs` lines 485-820

### HiveStart (lines 485-717) - 232 LOC
Key steps:
1. Check if already running (health check)
2. Get binary path from config or find in target/
3. Spawn daemon using daemon-lifecycle
4. Poll health until ready (10 attempts, exponential backoff)
5. Fetch and cache capabilities (with TimeoutEnforcer)
6. Display device information

### HiveStop (lines 718-820) - 102 LOC
Key steps:
1. Check if running (health check)
2. Send SIGTERM (graceful shutdown)
3. Wait 5 seconds for graceful shutdown
4. If still running, send SIGKILL (force kill)

---

## Deliverables

### 1. Implement HiveStart

**File:** `src/start.rs`
```rust
// TEAM-212: Start hive daemon

use anyhow::{Context, Result};
use daemon_lifecycle::DaemonManager;
use observability_narration_core::NarrationFactory;
use rbee_config::{RbeeConfig, HiveCapabilities, DeviceType};
use std::sync::Arc;
use std::time::Duration;
use timeout_enforcer::TimeoutEnforcer;
use tokio::time::sleep;

use crate::types::{HiveStartRequest, HiveStartResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Start hive daemon
///
/// COPIED FROM: job_router.rs lines 485-717
///
/// Steps:
/// 1. Check if already running
/// 2. Resolve binary path
/// 3. Spawn daemon process
/// 4. Poll health until ready
/// 5. Fetch and cache capabilities
///
/// # Arguments
/// * `request` - Start request with alias and job_id
/// * `config` - RbeeConfig with hive configuration
///
/// # Returns
/// * `Ok(HiveStartResponse)` - Success with endpoint
/// * `Err` - Failed to start or timeout
pub async fn execute_hive_start(
    request: HiveStartRequest,
    config: Arc<RbeeConfig>,
) -> Result<HiveStartResponse> {
    let job_id = &request.job_id;
    let alias = &request.alias;
    
    let hive_config = validate_hive_exists(&config, alias)?;

    NARRATE
        .action("hive_start")
        .job_id(job_id)
        .context(alias)
        .human("🚀 Starting hive '{}'")
        .emit();

    // Step 1: Check if already running
    NARRATE
        .action("hive_check")
        .job_id(job_id)
        .human("📋 Checking if hive is already running...")
        .emit();

    let health_url = format!(
        "http://{}:{}/health",
        hive_config.hostname, hive_config.hive_port
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;

    if let Ok(response) = client.get(&health_url).send().await {
        if response.status().is_success() {
            NARRATE
                .action("hive_running")
                .job_id(job_id)
                .context(alias)
                .context(&health_url)
                .human("✅ Hive '{0}' is already running on {1}")
                .emit();

            // Check cache and return early
            return handle_capabilities_cache(alias, &hive_config, &config, job_id).await;
        }
    }

    // Step 2: Get binary path
    let binary_path = resolve_binary_path(&hive_config, job_id)?;

    NARRATE
        .action("hive_spawn")
        .job_id(job_id)
        .context(&binary_path)
        .human("🔧 Spawning hive daemon: {}")
        .emit();

    // Step 3: Spawn the hive daemon
    let manager = DaemonManager::new(
        std::path::PathBuf::from(&binary_path),
        vec!["--port".to_string(), hive_config.hive_port.to_string()],
    );

    let _child = manager.spawn().await?;

    // Step 4: Wait for health check
    NARRATE
        .action("hive_health")
        .job_id(job_id)
        .human("⏳ Waiting for hive to be healthy...")
        .emit();

    // TEAM-206: Check first, THEN sleep (avoid unnecessary delay)
    for attempt in 1..=10 {
        if let Ok(response) = client.get(&health_url).send().await {
            if response.status().is_success() {
                NARRATE
                    .action("hive_success")
                    .job_id(job_id)
                    .context(alias)
                    .context(&health_url)
                    .human("✅ Hive '{0}' started successfully on {1}")
                    .emit();

                // Step 5: Fetch and cache capabilities
                return fetch_and_cache_capabilities(alias, &hive_config, &config, job_id).await;
            }
        }

        // Sleep before next attempt (but not after last)
        if attempt < 10 {
            sleep(Duration::from_millis(200 * attempt)).await;
        }
    }

    NARRATE
        .action("hive_timeout")
        .job_id(job_id)
        .human(
            "⚠️  Hive started but health check timed out.\n\
             Check if it's running:\n\
             \n\
               ./rbee hive status",
        )
        .emit();

    anyhow::bail!("Hive health check timed out")
}

/// Resolve binary path from config or find in target/
fn resolve_binary_path(
    hive_config: &rbee_config::HiveEntry,
    job_id: &str,
) -> Result<String> {
    if let Some(provided_path) = &hive_config.binary_path {
        NARRATE
            .action("hive_binary")
            .job_id(job_id)
            .context(provided_path)
            .human("📁 Using provided binary path: {}")
            .emit();

        let path = std::path::Path::new(provided_path);
        if !path.exists() {
            NARRATE
                .action("hive_bin_err")
                .job_id(job_id)
                .context(provided_path)
                .human("❌ Binary not found at: {}")
                .emit();
            anyhow::bail!("Binary not found: {}", provided_path);
        }

        NARRATE
            .action("hive_binary")
            .job_id(job_id)
            .human("✅ Binary found")
            .emit();

        Ok(provided_path.clone())
    } else {
        // Find binary in target directory
        NARRATE
            .action("hive_binary")
            .job_id(job_id)
            .human("🔍 Looking for rbee-hive binary in target/debug...")
            .emit();

        let debug_path = std::path::PathBuf::from("target/debug/rbee-hive");
        let release_path = std::path::PathBuf::from("target/release/rbee-hive");

        if debug_path.exists() {
            NARRATE
                .action("hive_binary")
                .job_id(job_id)
                .context(debug_path.display().to_string())
                .human("✅ Found binary at: {}")
                .emit();
            Ok(debug_path.display().to_string())
        } else if release_path.exists() {
            NARRATE
                .action("hive_binary")
                .job_id(job_id)
                .context(release_path.display().to_string())
                .human("✅ Found binary at: {}")
                .emit();
            Ok(release_path.display().to_string())
        } else {
            NARRATE
                .action("hive_bin_err")
                .job_id(job_id)
                .human(
                    "❌ rbee-hive binary not found.\n\
                     \n\
                     Please build it first:\n\
                     \n\
                       cargo build --bin rbee-hive\n\
                     \n\
                     Or provide a binary path:\n\
                     \n\
                       ./rbee hive install --binary-path /path/to/rbee-hive",
                )
                .emit();
            anyhow::bail!(
                "rbee-hive binary not found. Build it with: cargo build --bin rbee-hive"
            )
        }
    }
}

/// Handle capabilities cache (hive already running)
async fn handle_capabilities_cache(
    alias: &str,
    hive_config: &rbee_config::HiveEntry,
    config: &Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveStartResponse> {
    NARRATE
        .action("hive_cache_chk")
        .job_id(job_id)
        .human("💾 Checking capabilities cache...")
        .emit();

    if config.capabilities.contains(alias) {
        NARRATE
            .action("hive_cache_hit")
            .job_id(job_id)
            .human("✅ Using cached capabilities (use 'rbee hive refresh' to update)")
            .emit();

        // Display cached devices
        if let Some(caps) = config.capabilities.get(alias) {
            display_devices(&caps.devices, job_id);
        }

        let endpoint = format!("http://{}:{}", hive_config.hostname, hive_config.hive_port);
        return Ok(HiveStartResponse {
            success: true,
            message: format!("Hive '{}' is already running", alias),
            endpoint: Some(endpoint),
        });
    }

    // Cache miss - fetch fresh
    fetch_and_cache_capabilities(alias, hive_config, config, job_id).await
}

/// Fetch and cache capabilities
async fn fetch_and_cache_capabilities(
    alias: &str,
    hive_config: &rbee_config::HiveEntry,
    config: &Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveStartResponse> {
    let endpoint = format!("http://{}:{}", hive_config.hostname, hive_config.hive_port);

    NARRATE
        .action("hive_cache_miss")
        .job_id(job_id)
        .human("ℹ️  No cached capabilities, fetching fresh...")
        .emit();

    NARRATE
        .action("hive_caps")
        .job_id(job_id)
        .human("📊 Fetching device capabilities from hive...")
        .emit();

    // TEAM-207: Wrap in TimeoutEnforcer for visible timeout
    let caps_result = TimeoutEnforcer::new(Duration::from_secs(15))
        .with_label("Fetching device capabilities")
        .with_job_id(job_id) // CRITICAL for SSE routing!
        .with_countdown()
        .enforce(async {
            NARRATE
                .action("hive_caps_http")
                .job_id(job_id)
                .context(&format!("{}/capabilities", endpoint))
                .human("🌐 GET {}")
                .emit();

            use crate::hive_client::fetch_hive_capabilities;
            fetch_hive_capabilities(&endpoint).await
        })
        .await;

    match caps_result {
        Ok(devices) => {
            NARRATE
                .action("hive_caps_ok")
                .job_id(job_id)
                .context(devices.len().to_string())
                .human("✅ Discovered {} device(s)")
                .emit();

            display_devices(&devices, job_id);

            // Update capabilities cache
            NARRATE
                .action("hive_cache")
                .job_id(job_id)
                .human("💾 Updating capabilities cache...")
                .emit();

            let caps = HiveCapabilities::new(alias.to_string(), devices, endpoint.clone());

            let mut config_mut = (**config).clone();
            config_mut.capabilities.update_hive(alias, caps);
            if let Err(e) = config_mut.capabilities.save() {
                NARRATE
                    .action("hive_cache_error")
                    .job_id(job_id)
                    .context(e.to_string())
                    .human("⚠️  Failed to save capabilities cache: {}")
                    .emit();
            } else {
                NARRATE
                    .action("hive_cache_saved")
                    .job_id(job_id)
                    .human("✅ Capabilities cached")
                    .emit();
            }

            Ok(HiveStartResponse {
                success: true,
                message: format!("Hive '{}' started successfully", alias),
                endpoint: Some(endpoint),
            })
        }
        Err(e) => {
            NARRATE
                .action("hive_caps_err")
                .job_id(job_id)
                .context(e.to_string())
                .human("⚠️  Failed to fetch capabilities: {}")
                .emit();

            Ok(HiveStartResponse {
                success: true,
                message: format!("Hive '{}' started but capabilities fetch failed", alias),
                endpoint: Some(endpoint),
            })
        }
    }
}

/// Display device information
fn display_devices(devices: &[rbee_config::DeviceInfo], job_id: &str) {
    for device in devices {
        let device_info = match device.device_type {
            DeviceType::Gpu => {
                format!(
                    "  🎮 {} - {} (VRAM: {} GB, Compute: {})",
                    device.id,
                    device.name,
                    device.vram_gb,
                    device.compute_capability.as_deref().unwrap_or("unknown")
                )
            }
            DeviceType::Cpu => {
                format!("  🖥️  {} - {}", device.id, device.name)
            }
        };

        NARRATE
            .action("hive_device")
            .job_id(job_id)
            .context(&device_info)
            .human("{}")
            .emit();
    }
}
```

### TEAM-209 FINDING: Binary Path Resolution Issue

**Current implementation in job_router.rs lines 512-516:**
```rust
let binary_path = hive_config
    .binary_path
    .as_ref()
    .ok_or_else(|| anyhow::anyhow!("Hive '{}' has no binary_path configured", alias))?;
```

**This DIFFERS from the plan's `resolve_binary_path()` function!**

The plan implements fallback logic (search target/debug, target/release), but **actual code requires binary_path to be set**.

**Decision needed:**
- **Option A:** Keep current behavior (require binary_path in config)
- **Option B:** Implement the plan's fallback logic (search target/)

See HiveInstall operation (lines 280-401) for the fallback logic that COULD be reused.

---

### 2. Implement HiveStop

**File:** `src/stop.rs`
```rust
// TEAM-212: Stop hive daemon

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use crate::types::{HiveStopRequest, HiveStopResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Stop hive daemon
///
/// COPIED FROM: job_router.rs lines 718-820
///
/// Steps:
/// 1. Check if running
/// 2. Send SIGTERM (graceful shutdown)
/// 3. Wait 5 seconds for graceful shutdown
/// 4. If still running, send SIGKILL (force kill)
///
/// # Arguments
/// * `request` - Stop request with alias and job_id
/// * `config` - RbeeConfig with hive configuration
///
/// # Returns
/// * `Ok(HiveStopResponse)` - Success message
/// * `Err` - Failed to stop
pub async fn execute_hive_stop(
    request: HiveStopRequest,
    config: Arc<RbeeConfig>,
) -> Result<HiveStopResponse> {
    let job_id = &request.job_id;
    let alias = &request.alias;
    
    let hive_config = validate_hive_exists(&config, alias)?;

    NARRATE
        .action("hive_stop")
        .job_id(job_id)
        .context(alias)
        .human("🛑 Stopping hive '{}'")
        .emit();

    // Check if it's running
    NARRATE
        .action("hive_check")
        .job_id(job_id)
        .human("📋 Checking if hive is running...")
        .emit();

    let health_url = format!(
        "http://{}:{}/health",
        hive_config.hostname, hive_config.hive_port
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;

    if let Ok(response) = client.get(&health_url).send().await {
        if !response.status().is_success() {
            NARRATE
                .action("hive_not_run")
                .job_id(job_id)
                .context(alias)
                .human("⚠️  Hive '{}' is not running")
                .emit();
            return Ok(HiveStopResponse {
                success: true,
                message: format!("Hive '{}' is not running", alias),
            });
        }
    } else {
        NARRATE
            .action("hive_not_run")
            .job_id(job_id)
            .context(alias)
            .human("⚠️  Hive '{}' is not running")
            .emit();
        return Ok(HiveStopResponse {
            success: true,
            message: format!("Hive '{}' is not running", alias),
        });
    }

    // Stop the hive process
    NARRATE
        .action("hive_sigterm")
        .job_id(job_id)
        .human("📤 Sending SIGTERM (graceful shutdown)...")
        .emit();

    // Use pkill to stop the hive by binary name
    let binary_path = hive_config
        .binary_path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Hive '{}' has no binary_path configured", alias))?;

    let binary_name = std::path::Path::new(&binary_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("rbee-hive");

    // Send SIGTERM
    let output = tokio::process::Command::new("pkill")
        .args(&["-TERM", binary_name])
        .output()
        .await?;

    if !output.status.success() {
        NARRATE
            .action("hive_not_found")
            .job_id(job_id)
            .context(binary_name)
            .human("⚠️  No running process found for '{}'")
            .emit();
        return Ok(HiveStopResponse {
            success: true,
            message: format!("No running process found for '{}'", binary_name),
        });
    }

    // Wait for graceful shutdown
    NARRATE
        .action("hive_wait")
        .job_id(job_id)
        .human("⏳ Waiting for graceful shutdown (5s)...")
        .emit();

    for attempt in 1..=5 {
        sleep(Duration::from_secs(1)).await;

        if let Err(_) = client.get(&health_url).send().await {
            // Health check failed - hive stopped
            NARRATE
                .action("hive_success")
                .job_id(job_id)
                .context(alias)
                .human("✅ Hive '{}' stopped successfully")
                .emit();
            return Ok(HiveStopResponse {
                success: true,
                message: format!("Hive '{}' stopped successfully", alias),
            });
        }

        if attempt == 5 {
            // Timeout - force kill
            NARRATE
                .action("hive_sigkill")
                .job_id(job_id)
                .human("⚠️  Graceful shutdown timed out, sending SIGKILL...")
                .emit();

            tokio::process::Command::new("pkill")
                .args(&["-KILL", binary_name])
                .output()
                .await?;

            sleep(Duration::from_millis(500)).await;

            NARRATE
                .action("hive_forced")
                .job_id(job_id)
                .context(alias)
                .human("✅ Hive '{}' force-stopped")
                .emit();
        }
    }

    Ok(HiveStopResponse {
        success: true,
        message: format!("Hive '{}' stopped", alias),
    })
}
```

### 3. Update lib.rs Exports

```rust
// TEAM-212: Export lifecycle operations
pub use start::execute_hive_start;
pub use stop::execute_hive_stop;
```

---

## Acceptance Criteria

- [ ] `src/start.rs` implemented (complete with capabilities)
- [ ] `src/stop.rs` implemented (graceful + force kill)
- [ ] Binary path resolution working
- [ ] Health polling with exponential backoff
- [ ] Capabilities fetching with TimeoutEnforcer
- [ ] All narration includes `.job_id(job_id)` for SSE routing
- [ ] Error messages match original exactly
- [ ] Crate compiles: `cargo check -p queen-rbee-hive-lifecycle`
- [ ] No TODO markers in TEAM-212 code
- [ ] All code has TEAM-212 signatures

---

## Notes

- **CRITICAL:** All narration MUST include `.job_id(job_id)` for SSE routing
- Use `daemon-lifecycle::DaemonManager` for process spawning
- Use `timeout-enforcer::TimeoutEnforcer` for capabilities fetch
- Preserve exact error messages from job_router.rs
- Handle both debug and release binary paths
- Graceful shutdown (SIGTERM) before force kill (SIGKILL)

---

## TEAM-209 CRITICAL FINDINGS

### ⚠️  Architecture Flow: Capabilities Fetching

**The complete chain for device capabilities:**

```
1. queen-rbee (HiveStart)
   ↓
   fetch_hive_capabilities(&endpoint) [hive_client module]
   ↓
   GET http://127.0.0.1:9000/capabilities
   ↓
2. rbee-hive (/capabilities endpoint)
   ↓
   rbee_hive_device_detection::detect_gpus()
   ↓
   nvidia-smi --query-gpu=... (system call)
   ↓
   Parse CSV output → GpuInfo structs
   ↓
   Map to HiveDevice JSON
   ↓
   Add CPU-0 fallback
   ↓
   Return JSON response
   ↓
3. queen-rbee (receives JSON)
   ↓
   Parse into Vec<DeviceInfo>
   ↓
   Cache in config.capabilities
```

**Key Points:**
- `fetch_hive_capabilities()` is in `job_router.rs` as part of `hive_client` module
- It does NOT need to be moved to hive-lifecycle (it's just an HTTP client call)
- The actual device detection happens **in rbee-hive**, not queen-rbee
- device-detection crate is used by rbee-hive, NOT by hive-lifecycle

**Implementation Notes for TEAM-212:**
- Import `fetch_hive_capabilities` from parent module (job_router's hive_client)
- OR: Extract it to hive-lifecycle as a helper function
- Handle HTTP errors (connection refused, timeout, parse errors)
- Handle device detection errors (nvidia-smi not found, parse failures)
- Timeout already handled by TimeoutEnforcer (15 seconds)

**Error Scenarios to Handle:**
1. Hive not running → Connection refused
2. Hive running but device detection fails → Empty devices array
3. Hive timeout (GPU detection takes >15s) → TimeoutError
4. Invalid JSON response → Parse error
5. No GPUs found → CPU-only mode (not an error)
