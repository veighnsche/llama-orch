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

use anyhow::Result;
use job_registry::JobRegistry;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use queen_rbee_hive_lifecycle::{execute_ssh_test, SshTestRequest};
use queen_rbee_hive_registry::HiveRegistry; // TEAM-190: For Status operation
use rbee_operations::Operation;
use std::sync::Arc;

// TEAM-192: Narration factory for job router
const NARRATE: NarrationFactory = NarrationFactory::new("qn-router");

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub config: Arc<RbeeConfig>, // TEAM-194: File-based config
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

    NARRATE
        .action("job_create")
        .context(&job_id)
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

/// Internal: Route operation to appropriate handler
///
/// This parses the payload and dispatches to the correct operation handler.
async fn route_operation(
    payload: serde_json::Value,
    registry: Arc<JobRegistry<String>>,
    config: Arc<RbeeConfig>, // TEAM-194
    hive_registry: Arc<HiveRegistry>, // TEAM-190: Added for Status operation
) -> Result<()> {
    let state = JobState { registry, config, hive_registry };
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name();

    NARRATE
        .action("route_job")
        .context(operation_name)
        .human("Executing operation: {}")
        .emit();

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
                .table(serde_json::Value::Array(all_rows))
                .emit();
        }

        // Hive operations
        Operation::SshTest { ssh_host, ssh_port, ssh_user } => {
            // TEAM-188: Test SSH connection to remote host
            let request = SshTestRequest { ssh_host, ssh_port, ssh_user };

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
        Operation::HiveInstall { hive_id, ssh_host, ssh_port, ssh_user, port, binary_path } => {
            // TEAM-189: Comprehensive hive installation with pre-flight checks and narration
            use queen_rbee_hive_catalog::HiveRecord;

            NARRATE
                .action("hive_install")
                .context(&hive_id)
                .human("🔧 Installing hive '{}'")
                .emit();

            // STEP 1: Pre-flight check - is it already installed?
            NARRATE
                .action("hive_preflight")
                .human("📋 Checking if hive is already installed...")
                .emit();

            if state.hive_catalog.hive_exists(&hive_id).await? {
                NARRATE
                    .action("hive_exists")
                    .context(&hive_id)
                    .context(&hive_id)
                    .human("⚠️  Hive '{0}' is already installed.\n\
                         \n\
                         To reinstall, first uninstall it:\n\
                         \n\
                           ./rbee hive uninstall --id {1}")
                    .emit();
                return Err(anyhow::anyhow!(
                    "Hive '{}' already exists. Uninstall first to reinstall.",
                    hive_id
                ));
            }

            NARRATE
                .action("hive_preflight")
                .human("✅ Hive not found in catalog - proceeding with installation")
                .emit();

            // STEP 2: Determine if this is localhost or remote installation
            let is_remote = ssh_host.is_some();

            if is_remote {
                // REMOTE INSTALLATION
                let host = ssh_host.as_ref().unwrap();
                let ssh_port = ssh_port.unwrap_or(22);
                let user = ssh_user.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--ssh-user required for remote installation")
                })?;

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
                NARRATE
                    .action("hive_mode")
                    .human("🏠 Localhost installation")
                    .emit();

                // STEP 3: Find or build the rbee-hive binary
                let binary = if let Some(provided_path) = binary_path {
                    NARRATE
                        .action("hive_binary")
                        .context(&provided_path)
                        .human("📁 Using provided binary path: {}")
                        .emit();

                    // Verify binary exists
                    let path = std::path::Path::new(&provided_path);
                    if !path.exists() {
                        NARRATE
                            .action("hive_bin_err")
                            .context(&provided_path)
                            .human("❌ Binary not found at: {}")
                            .emit();
                        return Err(anyhow::anyhow!("Binary not found: {}", provided_path));
                    }

                    NARRATE.action("hive_binary").human("✅ Binary found").emit();

                    provided_path
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

                // STEP 4: Register in catalog
                NARRATE
                    .action("hive_register")
                    .human("📝 Registering hive in catalog...")
                    .emit();

                let now_ms = chrono::Utc::now().timestamp_millis();
                let record = HiveRecord {
                    id: hive_id.clone(),
                    host: "localhost".to_string(),
                    port,
                    ssh_host: None,
                    ssh_port: None,
                    ssh_user: None,
                    binary_path: Some(binary.clone()),
                    devices: None, // Will be populated later via refresh_capabilities
                    created_at_ms: now_ms,
                    updated_at_ms: now_ms,
                };

                if let Err(e) = state.hive_catalog.add_hive(record).await {
                    NARRATE
                        .action("hive_error")
                        .context(e.to_string())
                        .human("❌ Failed to add hive to catalog: {}")
                        .emit();
                    return Err(e);
                }

                NARRATE
                    .action("hive_complete")
                    .context(&hive_id)
                    .context(port.to_string())
                    .context(&binary)
                    .human("✅ Hive '{0}' installed successfully!\n\
                         \n\
                         Configuration:\n\
                         - Host: localhost\n\
                         - Port: {1}\n\
                         - Binary: {2}\n\
                         \n\
                         To start the hive:\n\
                         \n\
                           ./rbee hive start")
                    .emit();
            }
        }
        Operation::HiveUninstall { hive_id, catalog_only: _ } => {
            // TEAM-189: Hive uninstall with pre-flight check to ensure hive is stopped
            NARRATE
                .action("hive_uninstall")
                .context(&hive_id)
                .human("🗑️  Uninstalling hive '{}'")
                .emit();

            // Check if hive exists
            let hive = state.hive_catalog.get_hive(&hive_id).await?;
            if hive.is_none() {
                NARRATE
                    .action("hive_not_found")
                    .context(&hive_id)
                    .human("❌ Hive '{}' is not installed.\n\
                         \n\
                         Nothing to uninstall.\n\
                         \n\
                         To see installed hives:\n\
                         \n\
                           ./rbee hive list")
                    .emit();
                return Err(anyhow::anyhow!("Hive '{}' is not installed", hive_id));
            }

            let hive = hive.unwrap();

            // TEAM-189: Pre-flight check - ensure hive is stopped before uninstalling
            NARRATE
                .action("hive_preflight")
                .human("📋 Checking if hive is running...")
                .emit();

            let health_url = format!("http://{}:{}/health", hive.host, hive.port);
            let client =
                reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(2)).build()?;

            if let Ok(response) = client.get(&health_url).send().await {
                if response.status().is_success() {
                    NARRATE
                        .action("hive_running")
                        .context(&hive_id)
                        .human("❌ Cannot uninstall hive '{}' while it's running.\n\
                             \n\
                             Please stop the hive first:\n\
                             \n\
                               ./rbee hive stop")
                        .emit();
                    return Err(anyhow::anyhow!(
                        "Hive '{}' is still running. Stop it first with: ./rbee hive stop",
                        hive_id
                    ));
                }
            }

            NARRATE
                .action("hive_preflight")
                .human("✅ Hive is stopped - proceeding with uninstall")
                .emit();

            // Remove from catalog
            NARRATE
                .action("hive_remove")
                .human("📝 Removing hive from catalog...")
                .emit();

            state.hive_catalog.remove_hive(&hive_id).await?;

            NARRATE
                .action("hive_complete")
                .context(&hive_id)
                .human("✅ Hive '{}' uninstalled successfully")
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
        Operation::HiveUpdate { .. } => {
            // /**
            //  * Update hive configuration and optionally refresh capabilities
            //  *
            //  * COMMON FLOW:
            //  * 1. Lookup hive in catalog → error if not found
            //  * 2. Update SSH connection details if provided:
            //  *    - ssh_host, ssh_port, ssh_user
            //  *    - Update HiveRecord in catalog
            //  *
            //  * CAPABILITY REFRESH (refresh_capabilities=true):
            //  *
            //  * LOCALHOST:
            //  * 1. Check if hive is running (ping health endpoint)
            //  * 2. If not running: error (hive must be running for capability detection)
            //  * 3. Call hive API: GET /v1/devices
            //  * 4. Parse DeviceCapabilities response (CPU, GPUs)
            //  * 5. Update HiveRecord.devices in catalog
            //  *
            //  * REMOTE SSH:
            //  * 1. Run SshTest if SSH details changed
            //  * 2. Check if hive is running (over SSH or health endpoint)
            //  * 3. If not running: error
            //  * 4. Call hive API: GET /v1/devices (via SSH tunnel or direct)
            //  * 5. Parse DeviceCapabilities response
            //  * 6. Update HiveRecord.devices in catalog
            //  *
            //  * TODO: Implement hive health check endpoint
            //  * TODO: Implement device detection API in rbee-hive
            //  */
        }
        Operation::HiveStart { hive_id } => {
            // TEAM-189: Spawn hive daemon with health check polling
            NARRATE
                .action("hive_start")
                .context(&hive_id)
                .human("🚀 Starting hive '{}'")
                .emit();

            // Get hive from catalog
            let hive = state
                .hive_catalog
                .get_hive(&hive_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Hive '{}' not found in catalog", hive_id))?;

            // Check if already running
            NARRATE
                .action("hive_check")
                .human("📋 Checking if hive is already running...")
                .emit();

            let health_url = format!("http://{}:{}/health", hive.host, hive.port);
            let client =
                reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(2)).build()?;

            if let Ok(response) = client.get(&health_url).send().await {
                if response.status().is_success() {
                    NARRATE
                        .action("hive_running")
                        .context(&hive_id)
                        .context(&health_url)
                        .human("✅ Hive '{0}' is already running on {1}")
                        .emit();
                    return Ok(());
                }
            }

            // Get binary path
            let binary_path = hive.binary_path.ok_or_else(|| {
                anyhow::anyhow!("Hive '{}' has no binary_path configured", hive_id)
            })?;

            NARRATE
                .action("hive_spawn")
                .context(&binary_path)
                .human("🔧 Spawning hive daemon: {}")
                .emit();

            // Spawn the hive daemon
            use daemon_lifecycle::DaemonManager;
            let manager = DaemonManager::new(
                std::path::PathBuf::from(&binary_path),
                vec!["--port".to_string(), hive.port.to_string()],
            );

            let _child = manager.spawn().await?;

            // Wait for health check
            NARRATE
                .action("hive_health")
                .human("⏳ Waiting for hive to be healthy...")
                .emit();

            for attempt in 1..=10 {
                tokio::time::sleep(tokio::time::Duration::from_millis(200 * attempt)).await;

                if let Ok(response) = client.get(&health_url).send().await {
                    if response.status().is_success() {
                        NARRATE
                            .action("hive_success")
                            .context(&hive_id)
                            .context(&health_url)
                            .human("✅ Hive '{0}' started successfully on {1}")
                            .emit();
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
        Operation::HiveStop { hive_id } => {
            // TEAM-189: Graceful shutdown (SIGTERM) with SIGKILL fallback
            NARRATE
                .action("hive_stop")
                .context(&hive_id)
                .human("🛑 Stopping hive '{}'")
                .emit();

            // Get hive from catalog
            let hive = state
                .hive_catalog
                .get_hive(&hive_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Hive '{}' not found in catalog", hive_id))?;

            // Check if it's running
            NARRATE
                .action("hive_check")
                .human("📋 Checking if hive is running...")
                .emit();

            let health_url = format!("http://{}:{}/health", hive.host, hive.port);
            let client =
                reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(2)).build()?;

            if let Ok(response) = client.get(&health_url).send().await {
                if !response.status().is_success() {
                    NARRATE
                        .action("hive_not_run")
                        .context(&hive_id)
                        .human("⚠️  Hive '{}' is not running")
                        .emit();
                    return Ok(());
                }
            } else {
                NARRATE
                    .action("hive_not_run")
                    .context(&hive_id)
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
            let binary_path = hive.binary_path.ok_or_else(|| {
                anyhow::anyhow!("Hive '{}' has no binary_path configured", hive_id)
            })?;

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
            NARRATE
                .action("hive_wait")
                .human("⏳ Waiting for graceful shutdown (5s)...")
                .emit();

            for attempt in 1..=5 {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

                if let Err(_) = client.get(&health_url).send().await {
                    // Health check failed - hive stopped
                    NARRATE
                        .action("hive_success")
                        .context(&hive_id)
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
                        .context(&hive_id)
                        .human("✅ Hive '{}' force-stopped")
                        .emit();
                }
            }
        }
        Operation::HiveList => {
            // TEAM-190: List all hives from catalog with table output

            NARRATE.action("hive_list").human("📊 Listing all hives").emit();

            // Query catalog
            let hives = state.hive_catalog.list_hives().await?;

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
                .map(|h| {
                    serde_json::json!({
                        "id": h.id,
                        "host": h.host,
                        "port": h.port,
                        "binary_path": h.binary_path.as_ref().unwrap_or(&"-".to_string()),
                    })
                })
                .collect();

            // Display as table
            NARRATE
                .action("hive_result")
                .context(hives.len().to_string())
                .human("Found {} hive(s):")
                .table(serde_json::Value::Array(hives_json))
                .emit();
        }
        Operation::HiveGet { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Get single hive details from catalog
            //  * - Query HiveCatalog.get(hive_id)
            //  * - Return HiveRecord or 404
            //  */
        }
        Operation::HiveStatus { hive_id } => {
            // TEAM-189: Check hive health endpoint with friendly error messages
            // Check hive status by pinging its health endpoint
            // Similar to queen status pattern

            // TEAM-189: Friendly error handling for missing hive
            let hive = match state.hive_catalog.get_hive(&hive_id).await {
                Ok(Some(h)) => h,
                Ok(None) => {
                    // Hive not found in catalog
                    let install_cmd = if hive_id == "localhost" {
                        "  ./rbee hive install".to_string()
                    } else {
                        format!("  ./rbee hive install --id {}", hive_id)
                    };

                    NARRATE
                        .action("hive_check")
                        .context(&hive_id)
                        .context(&install_cmd)
                        .human("❌ Hive '{0}' not found in catalog.\n\
                             \n\
                             To install the hive:\n\
                             \n\
                             {1}")
                        .emit();
                    return Err(anyhow::anyhow!(
                        "Hive '{}' not found. Use 'rbee hive install' to add it.",
                        hive_id
                    ));
                }
                Err(e) if e.to_string().contains("no column") => {
                    // Database schema issue OR hive not installed (treat same for users)
                    let install_cmd = if hive_id == "localhost" {
                        "  ./rbee hive install".to_string()
                    } else {
                        format!("  ./rbee hive install --id {}", hive_id)
                    };

                    NARRATE
                        .action("hive_check")
                        .context(&hive_id)
                        .context(&install_cmd)
                        .human("❌ Hive '{0}' not found.\n\
                             \n\
                             Please install the hive first:\n\
                             \n\
                             {1}")
                        .emit();
                    return Err(anyhow::anyhow!(
                        "Hive '{}' not installed. Use 'rbee hive install' to add it.",
                        hive_id
                    ));
                }
                Err(e) => {
                    // Other database error
                    NARRATE
                        .action("hive_check")
                        .context(e.to_string())
                        .human("❌ Database error: {}")
                        .emit();
                    return Err(e);
                }
            };

            let health_url = format!("http://{}:{}/health", hive.host, hive.port);

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
                        .context(&hive_id)
                        .context(&health_url)
                        .human("✅ Hive '{0}' is running on {1}")
                        .emit();
                }
                Ok(response) => {
                    NARRATE
                        .action("hive_check")
                        .context(&hive_id)
                        .context(response.status().to_string())
                        .human("⚠️  Hive '{0}' responded with status: {1}")
                        .emit();
                }
                Err(_) => {
                    NARRATE
                        .action("hive_check")
                        .context(&hive_id)
                        .context(&health_url)
                        .human("❌ Hive '{0}' is not running on {1}")
                        .emit();
                }
            }
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
