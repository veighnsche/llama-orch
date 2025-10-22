# TEAM-214: Phase 5 - Capabilities Refresh

**Assigned to:** TEAM-214  
**Depends on:** TEAM-210 (Foundation)  
**Blocks:** TEAM-215 (Integration)  
**Estimated LOC:** ~100 lines

---

## Mission

Implement capabilities refresh operation:
- HiveRefreshCapabilities - Fetch fresh device capabilities and update cache

This operation is similar to the capabilities fetch in HiveStart, but can be run independently.

---

## Source Code Reference

**From:** `job_router.rs` lines 922-1011 (89 LOC)

### HiveRefreshCapabilities
Key steps:
1. Validate hive exists
2. Check if hive is running (health check)
3. Fetch fresh capabilities from hive
4. Display device information
5. Update capabilities cache

---

## Deliverables

### 1. Implement HiveRefreshCapabilities

**File:** `src/capabilities.rs`
```rust
// TEAM-214: Refresh hive capabilities

use anyhow::{Context, Result};
use observability_narration_core::NarrationFactory;
use rbee_config::{RbeeConfig, HiveCapabilities, DeviceType};
use std::sync::Arc;

use crate::types::{HiveRefreshCapabilitiesRequest, HiveRefreshCapabilitiesResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Refresh device capabilities for a running hive
///
/// COPIED FROM: job_router.rs lines 922-1011
///
/// Steps:
/// 1. Validate hive exists
/// 2. Check if hive is running
/// 3. Fetch fresh capabilities
/// 4. Display devices
/// 5. Update cache
///
/// # Arguments
/// * `request` - Refresh request with alias and job_id
/// * `config` - RbeeConfig with hive configuration
///
/// # Returns
/// * `Ok(HiveRefreshCapabilitiesResponse)` - Success with device count
/// * `Err` - Hive not running or fetch failed
pub async fn execute_hive_refresh_capabilities(
    request: HiveRefreshCapabilitiesRequest,
    config: Arc<RbeeConfig>,
) -> Result<HiveRefreshCapabilitiesResponse> {
    let job_id = &request.job_id;
    let alias = &request.alias;

    NARRATE
        .action("hive_refresh")
        .job_id(job_id)
        .context(alias)
        .human("🔄 Refreshing capabilities for '{}'")
        .emit();

    // Get hive config
    let hive_config = validate_hive_exists(&config, alias)?;

    // Check if hive is running
    let endpoint = format!("http://{}:{}", hive_config.hostname, hive_config.hive_port);

    NARRATE
        .action("hive_health_check")
        .job_id(job_id)
        .human("📋 Checking if hive is running...")
        .emit();

    match check_hive_health(&endpoint).await {
        Ok(true) => {
            NARRATE
                .action("hive_healthy")
                .job_id(job_id)
                .human("✅ Hive is running")
                .emit();
        }
        Ok(false) => {
            return Err(anyhow::anyhow!(
                "Hive '{}' is not healthy. Start it first with:\n\
                 \n\
                   ./rbee hive start -h {}",
                alias,
                alias
            ));
        }
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Failed to connect to hive '{}': {}\n\
                 \n\
                 Start it first with:\n\
                 \n\
                   ./rbee hive start -h {}",
                alias,
                e,
                alias
            ));
        }
    }

    // Fetch fresh capabilities
    NARRATE
        .action("hive_caps")
        .job_id(job_id)
        .human("📊 Fetching device capabilities...")
        .emit();

    let devices = fetch_hive_capabilities(&endpoint)
        .await
        .context("Failed to fetch capabilities")?;

    NARRATE
        .action("hive_caps_ok")
        .job_id(job_id)
        .context(devices.len().to_string())
        .human("✅ Discovered {} device(s)")
        .emit();

    // Log discovered devices
    for device in &devices {
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

    // Update cache
    NARRATE
        .action("hive_cache")
        .job_id(job_id)
        .human("💾 Updating capabilities cache...")
        .emit();

    let caps = HiveCapabilities::new(alias.to_string(), devices.clone(), endpoint.clone());

    let mut config_mut = (**config).clone();
    config_mut.capabilities.update_hive(alias, caps);
    config_mut.capabilities.save()?;

    NARRATE
        .action("hive_refresh_complete")
        .job_id(job_id)
        .context(alias)
        .human("✅ Capabilities refreshed for '{}'")
        .emit();

    Ok(HiveRefreshCapabilitiesResponse {
        success: true,
        device_count: devices.len(),
        message: format!("Capabilities refreshed for '{}'", alias),
    })
}

/// Check if hive is healthy
///
/// Performs HTTP health check to hive endpoint.
///
/// # Arguments
/// * `endpoint` - Hive base URL (e.g., "http://localhost:9000")
///
/// # Returns
/// * `Ok(true)` - Hive is healthy
/// * `Ok(false)` - Hive responded but not healthy
/// * `Err` - Connection failed
async fn check_hive_health(endpoint: &str) -> Result<bool> {
    let health_url = format!("{}/health", endpoint);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;

    match client.get(&health_url).send().await {
        Ok(response) => Ok(response.status().is_success()),
        Err(e) => Err(e.into()),
    }
}

/// Fetch device capabilities from hive
///
/// Calls GET /capabilities endpoint on hive.
///
/// # Arguments
/// * `endpoint` - Hive base URL (e.g., "http://localhost:9000")
///
/// # Returns
/// * `Ok(Vec<DeviceInfo>)` - List of devices
/// * `Err` - HTTP error or parse error
async fn fetch_hive_capabilities(endpoint: &str) -> Result<Vec<rbee_config::DeviceInfo>> {
    let caps_url = format!("{}/capabilities", endpoint);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;

    let response = client.get(&caps_url).send().await?;
    let devices: Vec<rbee_config::DeviceInfo> = response.json().await?;

    Ok(devices)
}
```

### 2. Update lib.rs Exports

```rust
// TEAM-214: Export capabilities operation
pub use capabilities::execute_hive_refresh_capabilities;
```

---

## Acceptance Criteria

- [ ] `src/capabilities.rs` implemented
- [ ] Health check helper function working
- [ ] Capabilities fetch helper function working
- [ ] Device display logic working
- [ ] Cache update working
- [ ] All narration includes `.job_id(job_id)` for SSE routing
- [ ] Error messages match original exactly
- [ ] Crate compiles: `cargo check -p queen-rbee-hive-lifecycle`
- [ ] No TODO markers in TEAM-214 code
- [ ] All code has TEAM-214 signatures

---

## Notes

- This operation requires hive to be running (health check enforced)
- Fetches fresh capabilities (ignores cache)
- Updates cache after successful fetch
- Similar to capabilities fetch in HiveStart, but standalone
- All narration MUST include `.job_id(job_id)` for SSE routing
- Preserve exact error messages from job_router.rs

---

## Integration Notes for TEAM-215

The helper functions `check_hive_health()` and `fetch_hive_capabilities()` are currently in `job_router.rs` as part of the `hive_client` module. TEAM-214 implements them here in the capabilities module.

TEAM-215 (Integration) will need to:
1. Remove these functions from `job_router.rs`
2. Use the versions from `hive-lifecycle` crate
3. Update imports in `job_router.rs`

---

## TEAM-209 CRITICAL FINDINGS

### ⚠️  Missing Architecture Documentation: device-detection Flow

**This phase talks about "fetching capabilities" but omits critical details:**

```
Phase 5 Implementation (HiveRefreshCapabilities)
        ↓
fetch_hive_capabilities(&endpoint)
        ↓
GET http://127.0.0.1:9000/capabilities
        ↓
[BLACK BOX - NOT DOCUMENTED IN PLAN]
        ↓
JSON response with devices
```

**What happens inside the black box:**

```
rbee-hive receives GET /capabilities
        ↓
get_capabilities() handler
        ↓
rbee_hive_device_detection::detect_gpus()
        ↓
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap,pci.bus_id
        ↓
Parse CSV output:
  "0, NVIDIA GeForce RTX 4090, 24576, 23456, 8.9, 0000:01:00.0"
        ↓
GpuInfo { index: 0, name: "NVIDIA GeForce RTX 4090", ... }
        ↓
Map to HiveDevice JSON
        ↓
Add CPU-0 fallback device
        ↓
Serialize to JSON
        ↓
Return CapabilitiesResponse
```

**Why this matters:**

1. **Error Handling**: Must handle nvidia-smi failures
   - Command not found → No GPUs detected, CPU-only mode
   - Parse errors → Invalid output format
   - Permission denied → Can't access GPU

2. **Narration**: rbee-hive emits narration during detection (TEAM-206 added this)
   - But queen can't see it (no job_id propagation to hive)
   - User only sees: "Fetching device capabilities..." then result
   - NO visibility into GPU detection process

3. **Timeout Behavior**: 15-second timeout includes:
   - HTTP request time (~10-50ms)
   - nvidia-smi execution (~50-200ms)
   - JSON serialization (~1-5ms)
   - Total: Usually <500ms, but can timeout if GPU hangs

4. **Device Detection Crate**: Lives in `bin/25_rbee_hive_crates/device-detection/`
   - NOT a shared crate (only used by rbee-hive)
   - Has its own README with detection strategy
   - Supports multiple device types (GPU, CPU, future: Metal, ROCm)

**Recommendations for TEAM-214:**

1. **Document the full chain** in capabilities.rs comments
2. **Add error handling notes** for nvidia-smi failures
3. **Reference device-detection crate** in module docs
4. **Clarify timeout expectations** (usually fast, can timeout on GPU issues)
5. **Consider caching strategy**: When to refresh? How often?

**See Also:**
- `bin/25_rbee_hive_crates/device-detection/README.md`
- `bin/20_rbee_hive/src/main.rs:142-200` (get_capabilities handler)
- `bin/TEAM_206_HIVE_START_NARRATION_ANALYSIS.md` (narration gap analysis)
