// TEAM-214: Refresh hive capabilities
//
// COPIED FROM: bin/10_queen_rbee/src/job_router.rs lines 922-1011
//
// This module implements the HiveRefreshCapabilities operation:
// 1. Validate hive exists
// 2. Check if hive is running (health check)
// 3. Fetch fresh capabilities from hive
// 4. Display device information
// 5. Update capabilities cache

use anyhow::{Context, Result};
use observability_narration_core::NarrationFactory;
use rbee_config::{HiveCapabilities, DeviceType};
use std::sync::Arc;

use crate::types::{HiveRefreshCapabilitiesRequest, HiveRefreshCapabilitiesResponse};
use crate::validation::validate_hive_exists;
use crate::hive_client::{check_hive_health, fetch_hive_capabilities};

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
    config: Arc<rbee_config::RbeeConfig>,
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

    let device_count = devices.len();

    NARRATE
        .action("hive_caps_ok")
        .job_id(job_id)
        .context(device_count.to_string())
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

    let caps = HiveCapabilities::new(alias.clone(), devices, endpoint.clone());

    let mut config_mut = (*config).clone();
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
        device_count,
        message: format!("Capabilities refreshed for '{}'", alias),
    })
}
