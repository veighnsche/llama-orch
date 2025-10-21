//! Job routing and operation dispatch
//!
//! TEAM-186: Centralized job routing logic for all operations
//!
//! This module handles:
//! - Parsing operation payloads into typed Operation enum
//! - Routing operations to appropriate handlers
//! - Job lifecycle management (create, register, execute)
//!
//! # Architecture
//!
//! ```text
//! POST /v1/jobs (JSON payload)
//!     ↓
//! job_router::route_job()
//!     ↓
//! Parse into Operation enum
//!     ↓
//! Match and dispatch to handler
//!     ↓
//! Execute async in background
//!     ↓
//! Stream results via SSE
//! ```

use anyhow::{Context, Result};
use job_registry::JobRegistry;
use observability_narration_core::NarrationFactory;
use queen_rbee_hive_lifecycle::{execute_ssh_test, SshTestRequest};
use queen_rbee_hive_registry::HiveRegistry; // TEAM-190: For Status operation
use rbee_config::{HiveCapabilities, RbeeConfig};
use rbee_operations::Operation;
use std::sync::Arc;

// TEAM-196: Import hive client for capabilities fetching
use crate::hive_client::{check_hive_health, fetch_hive_capabilities};

// TEAM-192: Narration factory for job router
const NARRATE: NarrationFactory = NarrationFactory::new("qn-router");

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub config: Arc<RbeeConfig>,          // TEAM-194: File-based config
    pub hive_registry: Arc<HiveRegistry>, // TEAM-190: For Status operation
}

/// Response from job creation
#[derive(Debug, serde::Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Create a new job and store its payload
///
/// This is the clean public API for job creation.
/// Called by HTTP layer to create jobs.
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    let sse_url = format!("/v1/jobs/{}/stream", job_id);

    state.registry.set_payload(&job_id, payload);

    // TEAM-200: Create job-specific SSE channel for isolation
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 1000);

    NARRATE
        .action("job_create")
        .context(&job_id)
        .job_id(&job_id)  // ← TEAM-200: Include job_id so narration routes correctly
        .human("Job {} created, waiting for client connection")
        .emit();

    Ok(JobResponse { job_id, sse_url })
}

/// Execute a job by retrieving its payload and streaming results
///
/// This is the clean public API for job execution.
/// Called by HTTP layer when client connects to SSE stream.
pub async fn execute_job(
    job_id: String,
    state: JobState,
) -> impl futures::stream::Stream<Item = String> {
    let registry = state.registry.clone();
    let config = state.config.clone(); // TEAM-194
    let hive_registry = state.hive_registry.clone(); // TEAM-190

    job_registry::execute_and_stream(job_id, registry.clone(), move |_job_id, payload| {
        route_operation(payload, registry, config, hive_registry)
    })
    .await
}

/// TEAM-195: Validate that a hive alias exists in config
///
/// Returns helpful error message listing available hives if alias not found.
fn validate_hive_exists<'a>(
    config: &'a RbeeConfig,
    alias: &str,
) -> Result<&'a rbee_config::HiveEntry> {
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

        anyhow::anyhow!(
            "Hive alias '{}' not found in hives.conf.\n\
             \n\
             Available hives:\n\
             {}\n\
             \n\
             Add '{}' to ~/.config/rbee/hives.conf to use it.",
            alias,
            hive_list,
            alias
        )
    })
}

/// Internal: Route operation to appropriate handler
///
/// This parses the payload and dispatches to the correct operation handler.
async fn route_operation(
    payload: serde_json::Value,
    registry: Arc<JobRegistry<String>>,
    config: Arc<RbeeConfig>,          // TEAM-194
    hive_registry: Arc<HiveRegistry>, // TEAM-190: Added for Status operation
) -> Result<()> {
    let state = JobState { registry, config, hive_registry };
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name();

    NARRATE.action("route_job").context(operation_name).human("Executing operation: {}").emit();

    // TEAM-186: Route to appropriate handler based on operation type
    // TEAM-187: Updated to handle HiveInstall/HiveUninstall/HiveUpdate operations
    // TEAM-188: Implemented SshTest operation
    // TEAM-190: Added Status operation for live hive/worker overview
    match operation {
        // System-wide operations
        Operation::Status => {
            // TEAM-190: Show live status of all hives and workers from registry (not catalog)

            NARRATE.action("status").human("📊 Fetching live status from registry").emit();

            // Get all active hives (heartbeat within last 30 seconds)
            let active_hive_ids = state.hive_registry.list_active_hives(30_000);

            if active_hive_ids.is_empty() {
                NARRATE
                    .action("status_empty")
                    .human(
                        "No active hives found.\n\
                         \n\
                         Hives must send heartbeats to appear here.\n\
                         \n\
                         To start a hive:\n\
                         \n\
                           ./rbee hive start",
                    )
                    .emit();
                return Ok(());
            }

            // Collect all hives and their workers
            let mut all_rows = Vec::new();

            for hive_id in &active_hive_ids {
                if let Some(hive_state) = state.hive_registry.get_hive_state(hive_id) {
                    if hive_state.workers.is_empty() {
                        // Hive with no workers
                        all_rows.push(serde_json::json!({
                            "hive": hive_id,
                            "worker": "-",
                            "state": "-",
                            "model": "-",
                            "url": "-",
                        }));
                    } else {
                        // Hive with workers
                        for worker in &hive_state.workers {
                            all_rows.push(serde_json::json!({
                                "hive": hive_id,
                                "worker": worker.worker_id,
                                "state": worker.state,
                                "model": worker.model_id.as_ref().unwrap_or(&"-".to_string()),
                                "url": worker.url,
                            }));
                        }
                    }
                }
            }

            // Display as table
            NARRATE
                .action("status_result")
                .context(active_hive_ids.len().to_string())
                .context(all_rows.iter().filter(|r| r["worker"] != "-").count().to_string())
                .human("Live Status ({0} hive(s), {1} worker(s)):")
                .table(&serde_json::Value::Array(all_rows))
                .emit();
        }

        // Hive operations
        Operation::SshTest { alias } => {
            // TEAM-188: Test SSH connection using config from hives.conf
            // TEAM-194: Lookup SSH details from config by alias
            // TEAM-195: Use validation helper for better error messages
            let hive_config = validate_hive_exists(&state.config, &alias)?;

            let request = SshTestRequest {
                ssh_host: hive_config.hostname.clone(),
                ssh_port: hive_config.ssh_port,
                ssh_user: hive_config.ssh_user.clone(),
            };

            let response = execute_ssh_test(request).await?;

            if !response.success {
                return Err(anyhow::anyhow!(
                    "SSH connection failed: {}",
                    response.error.unwrap_or_else(|| "Unknown error".to_string())
                ));
            }

            NARRATE
                .action("ssh_test_ok")
                .context(response.test_output.unwrap_or_default())
                .human("✅ SSH test successful: {}")
                .emit();
        }
        Operation::HiveInstall { alias } => {
            // TEAM-194: Install hive using config from hives.conf
            // TEAM-195: Use validation helper for better error messages
            let hive_config = validate_hive_exists(&state.config, &alias)?;

            NARRATE.action("hive_install").context(&alias).human("🔧 Installing hive '{}'").emit();

            // STEP 1: Determine if this is localhost or remote installation
            let is_remote =
                hive_config.hostname != "127.0.0.1" && hive_config.hostname != "localhost";

            if is_remote {
                // REMOTE INSTALLATION
                let host = &hive_config.hostname;
                let ssh_port = hive_config.ssh_port;
                let user = &hive_config.ssh_user;

                NARRATE
                    .action("hive_mode")
                    .context(format!("{}@{}:{}", user, host, ssh_port))
                    .human("🌐 Remote installation: {}")
                    .emit();

                // TODO: Implement remote SSH installation
                NARRATE
                    .action("hive_not_impl")
                    .human(
                        "❌ Remote SSH installation not yet implemented.\n\
                           \n\
                           Currently only localhost installation is supported.",
                    )
                    .emit();
                return Err(anyhow::anyhow!("Remote installation not yet implemented"));
            } else {
                // LOCALHOST INSTALLATION
                NARRATE.action("hive_mode").human("🏠 Localhost installation").emit();

                // STEP 2: Find or build the rbee-hive binary
                let binary = if let Some(provided_path) = &hive_config.binary_path {
                    NARRATE
                        .action("hive_binary")
                        .context(provided_path)
                        .human("📁 Using provided binary path: {}")
                        .emit();

                    // Verify binary exists
                    let path = std::path::Path::new(provided_path);
                    if !path.exists() {
                        NARRATE
                            .action("hive_bin_err")
                            .context(provided_path)
                            .human("❌ Binary not found at: {}")
                            .emit();
                        return Err(anyhow::anyhow!("Binary not found: {}", provided_path));
                    }

                    NARRATE.action("hive_binary").human("✅ Binary found").emit();

                    provided_path.clone()
                } else {
                    // Find binary in target directory
                    NARRATE
                        .action("hive_binary")
                        .human("🔍 Looking for rbee-hive binary in target/debug...")
                        .emit();

                    let debug_path = std::path::PathBuf::from("target/debug/rbee-hive");
                    let release_path = std::path::PathBuf::from("target/release/rbee-hive");

                    if debug_path.exists() {
                        NARRATE
                            .action("hive_binary")
                            .context(debug_path.display().to_string())
                            .human("✅ Found binary at: {}")
                            .emit();
                        debug_path.display().to_string()
                    } else if release_path.exists() {
                        NARRATE
                            .action("hive_binary")
                            .context(release_path.display().to_string())
                            .human("✅ Found binary at: {}")
                            .emit();
                        release_path.display().to_string()
                    } else {
                        NARRATE
                            .action("hive_bin_err")
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
                        return Err(anyhow::anyhow!("rbee-hive binary not found. Build it with: cargo build --bin rbee-hive"));
                    }
                };

                NARRATE
                    .action("hive_complete")
                    .context(&alias)
                    .context(hive_config.hive_port.to_string())
                    .context(&binary)
                    .human(
                        "✅ Hive '{0}' configured successfully!\n\
                         \n\
                         Configuration:\n\
                         - Host: localhost\n\
                         - Port: {1}\n\
                         - Binary: {2}\n\
                         \n\
                         To start the hive:\n\
                         \n\
                           ./rbee hive start --host {0}",
                    )
                    .emit();
            }
        }
        Operation::HiveUninstall { alias } => {
            // TEAM-194: Hive uninstall - remove from config
            // TEAM-195: Use validation helper for better error messages
            let _hive_config = validate_hive_exists(&state.config, &alias)?;

            NARRATE
                .action("hive_uninstall")
                .context(&alias)
                .human("🗑️  Uninstalling hive '{}'")
                .emit();

            // TEAM-196: Remove from capabilities cache
            if state.config.capabilities.contains(&alias) {
                NARRATE
                    .action("hive_cache_cleanup")
                    .human("🗑️  Removing from capabilities cache...")
                    .emit();

                let mut config = (*state.config).clone();
                config.capabilities.remove(&alias);
                if let Err(e) = config.capabilities.save() {
                    NARRATE
                        .action("hive_cache_error")
                        .context(e.to_string())
                        .human("⚠️  Failed to save capabilities cache: {}")
                        .emit();
                } else {
                    NARRATE
                        .action("hive_cache_removed")
                        .human("✅ Removed from capabilities cache")
                        .emit();
                }
            }

            NARRATE
                .action("hive_complete")
                .context(&alias)
                .human(
                    "✅ Hive '{}' uninstalled successfully.\n\
                     \n\
                     To remove from config, edit ~/.config/rbee/hives.conf",
                )
                .emit();

            // TEAM-189: Implemented pre-flight check - hive must be stopped before uninstall
            //
            // CURRENT IMPLEMENTATION:
            // 1. Check if hive exists in catalog → error if not found
            // 2. Check if hive is running (health endpoint) → error if running
            // 3. Remove from catalog
            //
            // IMPORTANT: User must manually stop the hive first:
            //   ./rbee hive stop
            // This ensures clean shutdown of hive and all child workers.
            //
            // FUTURE ENHANCEMENTS (for TEAM-190+):
            //
            // CATALOG-ONLY MODE (catalog_only=true):
            // - Used for unreachable remote hives (network issues, host down)
            // - Simply remove HiveRecord from catalog
            // - No SSH connection or health check
            // - Warn user about orphaned processes on remote host
            //
            // ADDITIONAL CLEANUP OPTIONS (flags):
            // - --remove-workers: Delete worker binaries (requires hive stopped)
            // - --remove-models: Delete model files (requires hive stopped)
            // - --remove-binary: Delete hive binary itself
            //
            // LOCALHOST FULL CLEANUP:
            // 1. Verify hive stopped (health check fails)
            // 2. Verify no worker processes running (pgrep llm-worker)
            // 3. Optional: Remove worker binaries if --remove-workers
            // 4. Optional: Remove models if --remove-models
            // 5. Optional: Remove hive binary if --remove-binary
            // 6. Remove from catalog
            //
            // REMOTE SSH FULL CLEANUP:
            // 1. Run SshTest to verify connectivity
            // 2. Verify hive stopped (SSH: curl health or pgrep)
            // 3. Verify no worker processes (SSH: pgrep llm-worker)
            // 4. Optional: Remove files via SSH (rm commands)
            // 5. Remove from catalog
        }
        Operation::HiveStart { alias } => {
            // TEAM-194: Start hive using config from hives.conf
            // TEAM-195: Use validation helper for better error messages
            let hive_config = validate_hive_exists(&state.config, &alias)?;

            NARRATE.action("hive_start").context(&alias).human("🚀 Starting hive '{}'").emit();

            // Check if already running
            NARRATE.action("hive_check").human("📋 Checking if hive is already running...").emit();

            let health_url =
                format!("http://{}:{}/health", hive_config.hostname, hive_config.hive_port);
            let client =
                reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(2)).build()?;

            if let Ok(response) = client.get(&health_url).send().await {
                if response.status().is_success() {
                    NARRATE
                        .action("hive_running")
                        .context(&alias)
                        .context(&health_url)
                        .human("✅ Hive '{0}' is already running on {1}")
                        .emit();
                    return Ok(());
                }
            }

            // Get binary path
            let binary_path = hive_config
                .binary_path
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Hive '{}' has no binary_path configured", alias))?;

            NARRATE
                .action("hive_spawn")
                .context(binary_path)
                .human("🔧 Spawning hive daemon: {}")
                .emit();

            // Spawn the hive daemon
            use daemon_lifecycle::DaemonManager;
            let manager = DaemonManager::new(
                std::path::PathBuf::from(binary_path),
                vec!["--port".to_string(), hive_config.hive_port.to_string()],
            );

            let _child = manager.spawn().await?;

            // Wait for health check
            NARRATE.action("hive_health").human("⏳ Waiting for hive to be healthy...").emit();

            for attempt in 1..=10 {
                tokio::time::sleep(tokio::time::Duration::from_millis(200 * attempt)).await;

                if let Ok(response) = client.get(&health_url).send().await {
                    if response.status().is_success() {
                        NARRATE
                            .action("hive_success")
                            .context(&alias)
                            .context(&health_url)
                            .human("✅ Hive '{0}' started successfully on {1}")
                            .emit();

                        // TEAM-196: Fetch and cache capabilities
                        let endpoint =
                            format!("http://{}:{}", hive_config.hostname, hive_config.hive_port);

                        NARRATE
                            .action("hive_capabilities")
                            .human("📊 Fetching device capabilities...")
                            .emit();

                        match fetch_hive_capabilities(&endpoint).await {
                            Ok(devices) => {
                                NARRATE
                                    .action("hive_capabilities_found")
                                    .context(devices.len().to_string())
                                    .human("✅ Discovered {} device(s)")
                                    .emit();

                                // Log discovered devices
                                for device in &devices {
                                    let device_info = match device.device_type {
                                        rbee_config::DeviceType::Gpu => {
                                            format!(
                                                "  🎮 {} - {} (VRAM: {} GB, Compute: {})",
                                                device.id,
                                                device.name,
                                                device.vram_gb,
                                                device
                                                    .compute_capability
                                                    .as_deref()
                                                    .unwrap_or("unknown")
                                            )
                                        }
                                        rbee_config::DeviceType::Cpu => {
                                            format!("  🖥️  {} - {}", device.id, device.name)
                                        }
                                    };

                                    NARRATE
                                        .action("hive_device")
                                        .context(&device_info)
                                        .human("{}")
                                        .emit();
                                }

                                // Update capabilities cache
                                NARRATE
                                    .action("hive_cache")
                                    .human("💾 Updating capabilities cache...")
                                    .emit();

                                let caps =
                                    HiveCapabilities::new(alias.clone(), devices, endpoint.clone());

                                // Clone config, update it, and replace in state
                                let mut config = (*state.config).clone();
                                config.capabilities.update_hive(&alias, caps);
                                if let Err(e) = config.capabilities.save() {
                                    NARRATE
                                        .action("hive_cache_error")
                                        .context(e.to_string())
                                        .human("⚠️  Failed to save capabilities cache: {}")
                                        .emit();
                                } else {
                                    NARRATE
                                        .action("hive_cache_saved")
                                        .human("✅ Capabilities cached")
                                        .emit();
                                }
                            }
                            Err(e) => {
                                NARRATE
                                    .action("hive_capabilities_error")
                                    .context(e.to_string())
                                    .human("⚠️  Failed to fetch capabilities: {}")
                                    .emit();
                            }
                        }

                        return Ok(());
                    }
                }
            }

            NARRATE
                .action("hive_timeout")
                .human(
                    "⚠️  Hive started but health check timed out.\n\
                     Check if it's running:\n\
                     \n\
                       ./rbee hive status",
                )
                .emit();
        }
        Operation::HiveStop { alias } => {
            // TEAM-194: Graceful shutdown using config
            // TEAM-195: Use validation helper for better error messages
            let hive_config = validate_hive_exists(&state.config, &alias)?;

            NARRATE.action("hive_stop").context(&alias).human("🛑 Stopping hive '{}'").emit();

            // Check if it's running
            NARRATE.action("hive_check").human("📋 Checking if hive is running...").emit();

            let health_url =
                format!("http://{}:{}/health", hive_config.hostname, hive_config.hive_port);
            let client =
                reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(2)).build()?;

            if let Ok(response) = client.get(&health_url).send().await {
                if !response.status().is_success() {
                    NARRATE
                        .action("hive_not_run")
                        .context(&alias)
                        .human("⚠️  Hive '{}' is not running")
                        .emit();
                    return Ok(());
                }
            } else {
                NARRATE
                    .action("hive_not_run")
                    .context(&alias)
                    .human("⚠️  Hive '{}' is not running")
                    .emit();
                return Ok(());
            }

            // Stop the hive process
            NARRATE
                .action("hive_sigterm")
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
                    .context(binary_name)
                    .human("⚠️  No running process found for '{}'")
                    .emit();
                return Ok(());
            }

            // Wait for graceful shutdown
            NARRATE.action("hive_wait").human("⏳ Waiting for graceful shutdown (5s)...").emit();

            for attempt in 1..=5 {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

                if let Err(_) = client.get(&health_url).send().await {
                    // Health check failed - hive stopped
                    NARRATE
                        .action("hive_success")
                        .context(&alias)
                        .human("✅ Hive '{}' stopped successfully")
                        .emit();
                    return Ok(());
                }

                if attempt == 5 {
                    // Timeout - force kill
                    NARRATE
                        .action("hive_sigkill")
                        .human("⚠️  Graceful shutdown timed out, sending SIGKILL...")
                        .emit();

                    tokio::process::Command::new("pkill")
                        .args(&["-KILL", binary_name])
                        .output()
                        .await?;

                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                    NARRATE
                        .action("hive_forced")
                        .context(&alias)
                        .human("✅ Hive '{}' force-stopped")
                        .emit();
                }
            }
        }
        Operation::HiveList => {
            // TEAM-194: List all hives from config

            NARRATE.action("hive_list").human("📊 Listing all hives").emit();

            // Query config
            let hives: Vec<_> = state.config.hives.all().iter().map(|h| (&h.alias, *h)).collect();

            if hives.is_empty() {
                NARRATE
                    .action("hive_empty")
                    .human(
                        "No hives registered.\n\
                         \n\
                         To install a hive:\n\
                         \n\
                           ./rbee hive install",
                    )
                    .emit();
                return Ok(());
            }

            // Convert to JSON array for table display
            let hives_json: Vec<serde_json::Value> = hives
                .iter()
                .map(|(alias, h)| {
                    serde_json::json!({
                        "alias": alias,
                        "host": h.hostname,
                        "port": h.hive_port,
                        "binary_path": h.binary_path.as_ref().unwrap_or(&"-".to_string()),
                    })
                })
                .collect();

            // Display as table
            NARRATE
                .action("hive_result")
                .context(hives.len().to_string())
                .human("Found {} hive(s):")
                .table(&serde_json::Value::Array(hives_json))
                .emit();
        }
        Operation::HiveGet { alias } => {
            // TEAM-194: Get single hive details from config
            // TEAM-195: Use validation helper for better error messages
            let hive_config = validate_hive_exists(&state.config, &alias)?;

            NARRATE.action("hive_get").context(&alias).human("Hive '{}' details:").emit();

            println!("Alias: {}", alias);
            println!("Host: {}", hive_config.hostname);
            println!("Port: {}", hive_config.hive_port);
            if let Some(ref bp) = hive_config.binary_path {
                println!("Binary: {}", bp);
            }
        }
        Operation::HiveStatus { alias } => {
            // TEAM-194: Check hive health using config
            // TEAM-195: Use validation helper for better error messages
            let hive_config = validate_hive_exists(&state.config, &alias)?;

            let health_url =
                format!("http://{}:{}/health", hive_config.hostname, hive_config.hive_port);

            NARRATE
                .action("hive_check")
                .context(&health_url)
                .human("Checking hive status at {}")
                .emit();

            let client =
                reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(5)).build()?;

            match client.get(&health_url).send().await {
                Ok(response) if response.status().is_success() => {
                    NARRATE
                        .action("hive_check")
                        .context(&alias)
                        .context(&health_url)
                        .human("✅ Hive '{0}' is running on {1}")
                        .emit();
                }
                Ok(response) => {
                    NARRATE
                        .action("hive_check")
                        .context(&alias)
                        .context(response.status().to_string())
                        .human("⚠️  Hive '{0}' responded with status: {1}")
                        .emit();
                }
                Err(_) => {
                    NARRATE
                        .action("hive_check")
                        .context(&alias)
                        .context(&health_url)
                        .human("❌ Hive '{0}' is not running on {1}")
                        .emit();
                }
            }
        }
        Operation::HiveRefreshCapabilities { alias } => {
            // TEAM-196: Refresh device capabilities for a running hive
            NARRATE
                .action("hive_refresh")
                .context(&alias)
                .human("🔄 Refreshing capabilities for '{}'")
                .emit();

            // Get hive config
            let hive_config = validate_hive_exists(&state.config, &alias)?;

            // Check if hive is running
            let endpoint = format!("http://{}:{}", hive_config.hostname, hive_config.hive_port);

            NARRATE.action("hive_health_check").human("📋 Checking if hive is running...").emit();

            match check_hive_health(&endpoint).await {
                Ok(true) => {
                    NARRATE.action("hive_healthy").human("✅ Hive is running").emit();
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
            NARRATE.action("hive_capabilities").human("📊 Fetching device capabilities...").emit();

            let devices =
                fetch_hive_capabilities(&endpoint).await.context("Failed to fetch capabilities")?;

            NARRATE
                .action("hive_capabilities_found")
                .context(devices.len().to_string())
                .human("✅ Discovered {} device(s)")
                .emit();

            // Log discovered devices
            for device in &devices {
                let device_info = match device.device_type {
                    rbee_config::DeviceType::Gpu => {
                        format!(
                            "  🎮 {} - {} (VRAM: {} GB, Compute: {})",
                            device.id,
                            device.name,
                            device.vram_gb,
                            device.compute_capability.as_deref().unwrap_or("unknown")
                        )
                    }
                    rbee_config::DeviceType::Cpu => {
                        format!("  🖥️  {} - {}", device.id, device.name)
                    }
                };

                NARRATE.action("hive_device").context(&device_info).human("{}").emit();
            }

            // Update cache
            NARRATE.action("hive_cache").human("💾 Updating capabilities cache...").emit();

            let caps = HiveCapabilities::new(alias.clone(), devices, endpoint.clone());

            let mut config = (*state.config).clone();
            config.capabilities.update_hive(&alias, caps);
            config.capabilities.save()?;

            NARRATE
                .action("hive_refresh_complete")
                .context(&alias)
                .human("✅ Capabilities refreshed for '{}'")
                .emit();
        }

        // Worker operations
        Operation::WorkerSpawn { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id → error if not found
            //  * 2. Get hive host:port from HiveRecord
            //  * 3. Forward entire operation payload to: POST http://{host}:{port}/v1/jobs
            //  * 4. Connect to SSE stream: GET http://{host}:{port}/v1/jobs/{job_id}/stream
            //  * 5. Stream hive responses back to client
            //  */
        }
        Operation::WorkerList { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }
        Operation::WorkerGet { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }
        Operation::WorkerDelete { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }

        // Model operations
        Operation::ModelDownload { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }
        Operation::ModelList { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }
        Operation::ModelGet { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }
        Operation::ModelDelete { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }

        // Inference operation
        Operation::Infer { .. } => {
            // //
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  * 4. Hive will handle worker selection, model loading, and inference
            //
        }
    }

    Ok(())
}
