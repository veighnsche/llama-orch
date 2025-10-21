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
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::HiveCatalog;
use queen_rbee_hive_lifecycle::{execute_ssh_test, SshTestRequest};
use rbee_operations::Operation;
use std::sync::Arc;

const ACTOR_QUEEN_ROUTER: &str = "👑 queen-router";
const ACTION_ROUTE_JOB: &str = "route_job";
const ACTION_PARSE_OPERATION: &str = "parse_operation";

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub hive_catalog: Arc<HiveCatalog>,
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

    Narration::new(ACTOR_QUEEN_ROUTER, "job_create", &job_id)
        .human(format!("Job {} created, waiting for client connection", job_id))
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
    let hive_catalog = state.hive_catalog.clone();

    job_registry::execute_and_stream(job_id, registry.clone(), move |_job_id, payload| {
        route_operation(payload, registry, hive_catalog)
    })
    .await
}

/// Internal: Route operation to appropriate handler
///
/// This parses the payload and dispatches to the correct operation handler.
async fn route_operation(
    payload: serde_json::Value,
    registry: Arc<JobRegistry<String>>,
    hive_catalog: Arc<HiveCatalog>,
) -> Result<()> {
    let state = JobState { registry, hive_catalog };
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name();

    Narration::new(ACTOR_QUEEN_ROUTER, ACTION_ROUTE_JOB, operation_name)
        .human(format!("Executing operation: {}", operation_name))
        .emit();

    // TEAM-186: Route to appropriate handler based on operation type
    // TEAM-187: Updated to handle HiveInstall/HiveUninstall/HiveUpdate operations
    // TEAM-188: Implemented SshTest operation
    match operation {
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

            Narration::new(ACTOR_QUEEN_ROUTER, "ssh_test_complete", "success")
                .human(format!(
                    "✅ SSH test successful: {}",
                    response.test_output.unwrap_or_default()
                ))
                .emit();
        }
        Operation::HiveInstall { hive_id, ssh_host, ssh_port, ssh_user, port, binary_path } => {
            // TEAM-189: Comprehensive hive installation with pre-flight checks and narration
            use queen_rbee_hive_catalog::HiveRecord;

            Narration::new(ACTOR_QUEEN_ROUTER, "hive_install", &hive_id)
                .human(format!("🔧 Installing hive '{}'", hive_id))
                .emit();

            // STEP 1: Pre-flight check - is it already installed?
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_preflight", &hive_id)
                .human("📋 Checking if hive is already installed...")
                .emit();

            if state.hive_catalog.hive_exists(&hive_id).await? {
                Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_exists", &hive_id)
                    .human(format!(
                        "⚠️  Hive '{}' is already installed.\n\
                         \n\
                         To reinstall, first uninstall it:\n\
                         \n\
                           ./rbee hive uninstall --id {}",
                        hive_id, hive_id
                    ))
                    .emit();
                return Err(anyhow::anyhow!(
                    "Hive '{}' already exists. Uninstall first to reinstall.",
                    hive_id
                ));
            }

            Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_preflight", &hive_id)
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

                Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_mode", &hive_id)
                    .human(format!("🌐 Remote installation: {}@{}:{}", user, host, ssh_port))
                    .emit();

                // TODO: Implement remote SSH installation
                Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_not_implemented", &hive_id)
                    .human(
                        "❌ Remote SSH installation not yet implemented.\n\
                           \n\
                           Currently only localhost installation is supported.",
                    )
                    .emit();
                return Err(anyhow::anyhow!("Remote installation not yet implemented"));
            } else {
                // LOCALHOST INSTALLATION
                Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_mode", &hive_id)
                    .human("🏠 Localhost installation")
                    .emit();

                // STEP 3: Find or build the rbee-hive binary
                let binary = if let Some(provided_path) = binary_path {
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_binary", &hive_id)
                        .human(format!("📁 Using provided binary path: {}", provided_path))
                        .emit();

                    // Verify binary exists
                    let path = std::path::Path::new(&provided_path);
                    if !path.exists() {
                        Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_binary_error", &hive_id)
                            .human(format!("❌ Binary not found at: {}", provided_path))
                            .emit();
                        return Err(anyhow::anyhow!("Binary not found: {}", provided_path));
                    }

                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_binary", &hive_id)
                        .human("✅ Binary found")
                        .emit();

                    provided_path
                } else {
                    // Find binary in target directory
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_binary", &hive_id)
                        .human("🔍 Looking for rbee-hive binary in target/debug...")
                        .emit();

                    let debug_path = std::path::PathBuf::from("target/debug/rbee-hive");
                    let release_path = std::path::PathBuf::from("target/release/rbee-hive");

                    if debug_path.exists() {
                        Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_binary", &hive_id)
                            .human(format!("✅ Found binary at: {}", debug_path.display()))
                            .emit();
                        debug_path.display().to_string()
                    } else if release_path.exists() {
                        Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_binary", &hive_id)
                            .human(format!("✅ Found binary at: {}", release_path.display()))
                            .emit();
                        release_path.display().to_string()
                    } else {
                        Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_binary_error", &hive_id)
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
                Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_register", &hive_id)
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
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_error", &hive_id)
                        .human(format!("❌ Failed to add hive to catalog: {}", e))
                        .emit();
                    return Err(e);
                }

                Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_complete", &hive_id)
                    .human(format!(
                        "✅ Hive '{}' installed successfully!\n\
                         \n\
                         Configuration:\n\
                         - Host: localhost\n\
                         - Port: {}\n\
                         - Binary: {}\n\
                         \n\
                         To start the hive:\n\
                         \n\
                           ./rbee hive start",
                        hive_id, port, binary
                    ))
                    .emit();
            }
        }
        Operation::HiveUninstall { hive_id, catalog_only: _ } => {
            // TEAM-189: Hive uninstall with pre-flight check to ensure hive is stopped
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_uninstall", &hive_id)
                .human(format!("🗑️  Uninstalling hive '{}'", hive_id))
                .emit();

            // Check if hive exists
            let hive = state.hive_catalog.get_hive(&hive_id).await?;
            if hive.is_none() {
                Narration::new(ACTOR_QUEEN_ROUTER, "hive_uninstall_not_found", &hive_id)
                    .human(format!(
                        "❌ Hive '{}' is not installed.\n\
                         \n\
                         Nothing to uninstall.\n\
                         \n\
                         To see installed hives:\n\
                         \n\
                           ./rbee hive list",
                        hive_id
                    ))
                    .emit();
                return Err(anyhow::anyhow!("Hive '{}' is not installed", hive_id));
            }
            
            let hive = hive.unwrap();

            // TEAM-189: Pre-flight check - ensure hive is stopped before uninstalling
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_uninstall_preflight", &hive_id)
                .human("📋 Checking if hive is running...")
                .emit();

            let health_url = format!("http://{}:{}/health", hive.host, hive.port);
            let client =
                reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(2)).build()?;

            if let Ok(response) = client.get(&health_url).send().await {
                if response.status().is_success() {
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_uninstall_running", &hive_id)
                        .human(format!(
                            "❌ Cannot uninstall hive '{}' while it's running.\n\
                             \n\
                             Please stop the hive first:\n\
                             \n\
                               ./rbee hive stop",
                            hive_id
                        ))
                        .emit();
                    return Err(anyhow::anyhow!(
                        "Hive '{}' is still running. Stop it first with: ./rbee hive stop",
                        hive_id
                    ));
                }
            }

            Narration::new(ACTOR_QUEEN_ROUTER, "hive_uninstall_preflight", &hive_id)
                .human("✅ Hive is stopped - proceeding with uninstall")
                .emit();

            // Remove from catalog
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_uninstall_remove", &hive_id)
                .human("📝 Removing hive from catalog...")
                .emit();

            state.hive_catalog.remove_hive(&hive_id).await?;

            Narration::new(ACTOR_QUEEN_ROUTER, "hive_uninstall_complete", &hive_id)
                .human(format!("✅ Hive '{}' uninstalled successfully", hive_id))
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
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_start", &hive_id)
                .human(format!("🚀 Starting hive '{}'", hive_id))
                .emit();

            // Get hive from catalog
            let hive = state
                .hive_catalog
                .get_hive(&hive_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Hive '{}' not found in catalog", hive_id))?;

            // Check if already running
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_start_check", &hive_id)
                .human("📋 Checking if hive is already running...")
                .emit();

            let health_url = format!("http://{}:{}/health", hive.host, hive.port);
            let client =
                reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(2)).build()?;

            if let Ok(response) = client.get(&health_url).send().await {
                if response.status().is_success() {
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_start_already_running", &hive_id)
                        .human(format!(
                            "✅ Hive '{}' is already running on {}",
                            hive_id, health_url
                        ))
                        .emit();
                    return Ok(());
                }
            }

            // Get binary path
            let binary_path = hive.binary_path.ok_or_else(|| {
                anyhow::anyhow!("Hive '{}' has no binary_path configured", hive_id)
            })?;

            Narration::new(ACTOR_QUEEN_ROUTER, "hive_start_spawn", &hive_id)
                .human(format!("🔧 Spawning hive daemon: {}", binary_path))
                .emit();

            // Spawn the hive daemon
            use daemon_lifecycle::DaemonManager;
            let manager = DaemonManager::new(
                std::path::PathBuf::from(&binary_path),
                vec!["--port".to_string(), hive.port.to_string()],
            );

            let _child = manager.spawn().await?;

            // Wait for health check
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_start_health", &hive_id)
                .human("⏳ Waiting for hive to be healthy...")
                .emit();

            for attempt in 1..=10 {
                tokio::time::sleep(tokio::time::Duration::from_millis(200 * attempt)).await;

                if let Ok(response) = client.get(&health_url).send().await {
                    if response.status().is_success() {
                        Narration::new(ACTOR_QUEEN_ROUTER, "hive_start_success", &hive_id)
                            .human(format!(
                                "✅ Hive '{}' started successfully on {}",
                                hive_id, health_url
                            ))
                            .emit();
                        return Ok(());
                    }
                }
            }

            Narration::new(ACTOR_QUEEN_ROUTER, "hive_start_timeout", &hive_id)
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
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop", &hive_id)
                .human(format!("🛑 Stopping hive '{}'", hive_id))
                .emit();

            // Get hive from catalog
            let hive = state
                .hive_catalog
                .get_hive(&hive_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Hive '{}' not found in catalog", hive_id))?;

            // Check if it's running
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop_check", &hive_id)
                .human("📋 Checking if hive is running...")
                .emit();

            let health_url = format!("http://{}:{}/health", hive.host, hive.port);
            let client =
                reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(2)).build()?;

            if let Ok(response) = client.get(&health_url).send().await {
                if !response.status().is_success() {
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop_not_running", &hive_id)
                        .human(format!("⚠️  Hive '{}' is not running", hive_id))
                        .emit();
                    return Ok(());
                }
            } else {
                Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop_not_running", &hive_id)
                    .human(format!("⚠️  Hive '{}' is not running", hive_id))
                    .emit();
                return Ok(());
            }

            // Stop the hive process
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop_sigterm", &hive_id)
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
                Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop_not_found", &hive_id)
                    .human(format!("⚠️  No running process found for '{}'", binary_name))
                    .emit();
                return Ok(());
            }

            // Wait for graceful shutdown
            Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop_wait", &hive_id)
                .human("⏳ Waiting for graceful shutdown (5s)...")
                .emit();

            for attempt in 1..=5 {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

                if let Err(_) = client.get(&health_url).send().await {
                    // Health check failed - hive stopped
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop_success", &hive_id)
                        .human(format!("✅ Hive '{}' stopped successfully", hive_id))
                        .emit();
                    return Ok(());
                }

                if attempt == 5 {
                    // Timeout - force kill
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop_sigkill", &hive_id)
                        .human("⚠️  Graceful shutdown timed out, sending SIGKILL...")
                        .emit();

                    tokio::process::Command::new("pkill")
                        .args(&["-KILL", binary_name])
                        .output()
                        .await?;

                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop_forced", &hive_id)
                        .human(format!("✅ Hive '{}' force-stopped", hive_id))
                        .emit();
                }
            }
        }
        Operation::HiveList => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * List all hives from catalog
            //  * - Query HiveCatalog.list_all()
            //  * - Return array of HiveRecords
            //  */
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

                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_status_check", &hive_id)
                        .human(format!(
                            "❌ Hive '{}' not found in catalog.\n\
                             \n\
                             To install the hive:\n\
                             \n\
                             {}",
                            hive_id, install_cmd
                        ))
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

                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_status_check", &hive_id)
                        .human(format!(
                            "❌ Hive '{}' not found.\n\
                             \n\
                             Please install the hive first:\n\
                             \n\
                             {}",
                            hive_id, install_cmd
                        ))
                        .emit();
                    return Err(anyhow::anyhow!(
                        "Hive '{}' not installed. Use 'rbee hive install' to add it.",
                        hive_id
                    ));
                }
                Err(e) => {
                    // Other database error
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_status_check", "db_error")
                        .human(format!("❌ Database error: {}", e))
                        .emit();
                    return Err(e);
                }
            };

            let health_url = format!("http://{}:{}/health", hive.host, hive.port);

            Narration::new(ACTOR_QUEEN_ROUTER, "hive_status_check", &hive_id)
                .human(format!("Checking hive status at {}", health_url))
                .emit();

            let client =
                reqwest::Client::builder().timeout(tokio::time::Duration::from_secs(5)).build()?;

            match client.get(&health_url).send().await {
                Ok(response) if response.status().is_success() => {
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_status_check", &hive_id)
                        .human(format!("✅ Hive '{}' is running on {}", hive_id, health_url))
                        .emit();
                }
                Ok(response) => {
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_status_check", &hive_id)
                        .human(format!(
                            "⚠️  Hive '{}' responded with status: {}",
                            hive_id,
                            response.status()
                        ))
                        .emit();
                }
                Err(_) => {
                    Narration::new(ACTOR_QUEEN_ROUTER, "hive_status_check", &hive_id)
                        .human(format!("❌ Hive '{}' is not running on {}", hive_id, health_url))
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
