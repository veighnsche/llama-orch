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
        Operation::HiveInstall {  .. } => {
            // /**
            //  * Install hive binary and register in catalog
            //  *
            //  * LOCALHOST INSTALLATION:
            //  * 1. Check if hive_id already exists in catalog → error if exists
            //  * 2. Determine binary location:
            //  *    - If binary_path provided: validate path exists
            //  *    - Else: check catalog for previous install path
            //  *    - Else: default to cargo build (requires rustup)
            //  * 3. If building from source:
            //  *    - Verify rustup installed (fail fast if not)
            //  *    - Git clone veighsnche/llama-orch to temp dir
            //  *    - Run: cargo build --release --bin rbee-hive
            //  *    - Copy binary to standard location
            //  * 4. Verify binary is executable
            //  * 5. Add to catalog: HiveRecord { id, host: "localhost", port, binary_path, devices: None }
            //  *
            //  * REMOTE SSH INSTALLATION:
            //  * 1. Run SshTest operation first (fail fast on SSH issues)
            //  * 2. Check if hive_id already exists in catalog → error if exists
            //  * 3. Determine binary location on remote:
            //  *    - If binary_path provided: use that path
            //  *    - Else: default to git clone + cargo build on remote
            //  * 4. If building from source (over SSH):
            //  *    - Verify rustup installed on remote (ssh command: rustup --version)
            //  *    - Git clone veighsnche/llama-orch on remote
            //  *    - Run: cargo build --release --bin rbee-hive (over SSH)
            //  * 5. Verify binary exists on remote (ssh command: test -f <path>)
            //  * 6. Add to catalog: HiveRecord { id, host, port, ssh_*, binary_path, devices: None }
            //  *
            //  * NOTE: Capabilities (devices) are populated later via HiveUpdate with refresh_capabilities=true
            //  */
        }
        Operation::HiveUninstall {  .. } => {
            // /**
            //  * Uninstall hive and optionally clean up resources
            //  *
            //  * CATALOG-ONLY MODE (catalog_only=true):
            //  * - Used for unreachable remote hives
            //  * - Simply remove HiveRecord from catalog
            //  * - No SSH connection or binary cleanup
            //  * - Return success
            //  *
            //  * FULL UNINSTALL (catalog_only=false):
            //  *
            //  * LOCALHOST:
            //  * 1. Lookup hive in catalog → error if not found
            //  * 2. Get binary_path from HiveRecord
            //  * 3. Check if hive is running:
            //  *    - If running: SIGTERM first, wait 5s, then SIGKILL if needed
            //  *    - Kill all child workers (SIGKILL)
            //  * 4. Cleanup options (interactive or flags):
            //  *    - Remove workers? (delete worker binaries)
            //  *    - Remove models? (delete model files)
            //  * 5. Remove hive binary at binary_path
            //  * 6. Remove from catalog
            //  *
            //  * REMOTE SSH:
            //  * 1. Run SshTest operation first
            //  * 2. Lookup hive in catalog → error if not found
            //  * 3. Get binary_path from HiveRecord
            //  * 4. Check if hive is running (over SSH):
            //  *    - SSH: pkill -TERM rbee-hive, wait, pkill -KILL if needed
            //  * 5. Cleanup options (same as localhost, but over SSH)
            //  * 6. Remove hive binary on remote (ssh: rm <binary_path>)
            //  * 7. Remove from catalog
            //  *
            //  * TODO: Implement emergency stop (SIGKILL all workers)
            //  * TODO: Add interactive cleanup prompts or CLI flags
            //  */
        }
        Operation::HiveUpdate {  .. } => {
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
        Operation::HiveStart { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Start a hive daemon process
            //  * - Lookup binary_path from catalog
            //  * - Spawn hive process with proper config
            //  * - Wait for health check to confirm startup
            //  */
        }
        Operation::HiveStop { .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Stop a running hive daemon
            //  * - Send SIGTERM, wait for graceful shutdown
            //  * - SIGKILL if timeout exceeded
            //  */
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

        // Worker operations
        Operation::WorkerSpawn {  .. } => {
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
        Operation::WorkerList {  .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }
        Operation::WorkerGet {  .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }
        Operation::WorkerDelete {  .. } => {
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
        Operation::ModelDownload {  .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }
        Operation::ModelList {  .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }
        Operation::ModelGet {  .. } => {
            // /**
            //  * TODO: IMPLEMENT THIS
            //  *
            //  * Forward operation to hive using job-based architecture:
            //  * 1. Lookup hive in catalog by hive_id
            //  * 2. POST operation to http://{host}:{port}/v1/jobs
            //  * 3. Stream response from /v1/jobs/{job_id}/stream
            //  */
        }
        Operation::ModelDelete {  .. } => {
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
        Operation::Infer {  .. } => {
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
