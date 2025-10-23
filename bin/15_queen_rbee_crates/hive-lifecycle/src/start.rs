// TEAM-212: Start hive daemon
// TEAM-220: Investigated - 5-step lifecycle + exponential backoff documented

use anyhow::Result;
use daemon_lifecycle::DaemonManager;
use observability_narration_core::NarrationFactory;
use rbee_config::{DeviceType, HiveCapabilities, RbeeConfig};
use std::sync::Arc;
use std::time::Duration;
use timeout_enforcer::TimeoutEnforcer;
use tokio::time::sleep;

use crate::hive_client::fetch_hive_capabilities;
use crate::ssh_helper::{get_remote_binary_path, is_remote_hive, ssh_exec};
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

    let health_url = format!("http://{}:{}/health", hive_config.hostname, hive_config.hive_port);
    let client = reqwest::Client::builder().timeout(Duration::from_secs(2)).build()?;

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
            return handle_capabilities_cache(alias, hive_config, &config, job_id).await;
        }
    }

    // Step 2: Check if remote or local
    let is_remote = is_remote_hive(hive_config);

    if is_remote {
        // REMOTE START: Use SSH
        let binary_path = get_remote_binary_path(hive_config);

        NARRATE
            .action("hive_mode")
            .job_id(job_id)
            .context(&format!("{}@{}", hive_config.ssh_user, hive_config.hostname))
            .human("🌐 Remote start: {}")
            .emit();

        // Start hive daemon on remote host
        let start_cmd = format!(
            "nohup {} --port {} > /dev/null 2>&1 & echo $!",
            binary_path, hive_config.hive_port
        );

        let pid_output = ssh_exec(
            hive_config,
            &start_cmd,
            job_id,
            "hive_spawn",
            &format!("Starting remote hive: {}", binary_path),
        )
        .await?;

        let pid = pid_output.trim();
        NARRATE
            .action("hive_spawned")
            .job_id(job_id)
            .context(pid)
            .human("✅ Remote hive started with PID: {}")
            .emit();
    } else {
        // LOCAL START: Use DaemonManager
        let binary_path = resolve_binary_path(hive_config, job_id)?;

        NARRATE
            .action("hive_spawn")
            .job_id(job_id)
            .context(&binary_path)
            .human("🔧 Spawning local hive daemon: {}")
            .emit();

        let manager = DaemonManager::new(
            std::path::PathBuf::from(&binary_path),
            vec!["--port".to_string(), hive_config.hive_port.to_string()],
        );

        let _child = manager.spawn().await?;
    }

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
                return fetch_and_cache_capabilities(alias, hive_config, &config, job_id).await;
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
fn resolve_binary_path(hive_config: &rbee_config::HiveConfig, job_id: &str) -> Result<String> {
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

        NARRATE.action("hive_binary").job_id(job_id).human("✅ Binary found").emit();

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
            anyhow::bail!("rbee-hive binary not found. Build it with: cargo build --bin rbee-hive")
        }
    }
}

/// Handle capabilities cache (hive already running)
async fn handle_capabilities_cache(
    alias: &str,
    hive_config: &rbee_config::HiveConfig,
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
    hive_config: &rbee_config::HiveConfig,
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
                .context(format!("{}/capabilities", endpoint))
                .human("🌐 GET {}")
                .emit();

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

        NARRATE.action("hive_device").job_id(job_id).context(&device_info).human("{}").emit();
    }
}
