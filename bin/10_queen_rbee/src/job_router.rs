//! Job routing and operation dispatch
//!
//! TEAM-186: Centralized job routing logic for all operations
//! TEAM-217: Investigated Oct 22, 2025 - Behavior inventory complete
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
use job_server::JobRegistry;
use observability_narration_core::NarrationFactory;
use queen_rbee_hive_lifecycle::{
    execute_hive_get, execute_hive_install, execute_hive_list, execute_hive_refresh_capabilities,
    execute_hive_start, execute_hive_status, execute_hive_stop, execute_hive_uninstall,
    execute_ssh_test, validate_hive_exists, HiveGetRequest, HiveInstallRequest, HiveListRequest,
    HiveRefreshCapabilitiesRequest, HiveStartRequest, HiveStatusRequest, HiveStopRequest,
    HiveUninstallRequest, SshTestRequest,
};
// TEAM-275: Removed unused import (using state.hive_registry which is already Arc<WorkerRegistry>)
use rbee_config::RbeeConfig;
use rbee_operations::Operation;
use std::sync::Arc;

use super::hive_forwarder; // TEAM-258: Generic forwarding for hive-managed operations

// TEAM-192: Narration factory for job router
const NARRATE: NarrationFactory = NarrationFactory::new("qn-router");

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub config: Arc<RbeeConfig>,                                     // TEAM-194: File-based config
    pub hive_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>, // TEAM-190/262/275: For Status and Infer operations
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
        .job_id(&job_id) // ← TEAM-200: Include job_id so narration routes correctly
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

    job_server::execute_and_stream(job_id, registry.clone(), move |job_id, payload| {
        route_operation(job_id, payload, registry, config, hive_registry)
    })
    .await
}

// TEAM-215: Validation moved to hive-lifecycle crate
// Use queen_rbee_hive_lifecycle::validate_hive_exists instead

/// Internal: Route operation to appropriate handler
///
/// This parses the payload and dispatches to the correct operation handler.
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    registry: Arc<JobRegistry<String>>,
    config: Arc<RbeeConfig>,                                          // TEAM-194
    hive_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>, // TEAM-190/262/275: For Status and Infer operations
) -> Result<()> {
    let state = JobState { registry, config, hive_registry };
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name();

    NARRATE
        .action("route_job")
        .context(operation_name)
        .job_id(&job_id)
        .human("Executing operation: {}")
        .emit();

    // ============================================================================
    // OPERATION ROUTING (Step 2 of 3-File Pattern)
    // ============================================================================
    //
    // When adding a new operation:
    // 1. Add Operation variant in rbee-operations/src/lib.rs (DONE)
    // 2. Add match arm HERE to route it to handler
    // 3. Add CLI command in rbee-keeper/src/main.rs (TODO)
    //
    // Pattern for hive operations:
    //   - Create request struct (e.g., HiveXxxRequest)
    //   - Call execute_hive_xxx() from hive-lifecycle crate
    //   - Pass job_id for SSE routing
    //
    // Pattern for worker/model/infer operations:
    //   - Forward to hive via HTTP (TODO: implement forwarding)
    //
    // TEAM-186: Route to appropriate handler based on operation type
    // TEAM-187: Updated to handle HiveInstall/HiveUninstall/HiveUpdate operations
    // TEAM-188: Implemented SshTest operation
    // TEAM-190: Added Status operation for live hive/worker overview
    match operation {
        // System-wide operations
        Operation::Status => {
            // TEAM-190/275: Show live status of workers from registry

            NARRATE
                .action("status")
                .job_id(&job_id)
                .human("📊 Fetching live status from worker registry")
                .emit();

            // Get all online workers (workers with recent heartbeats)
            let online_workers = state.hive_registry.list_online_workers();

            if online_workers.is_empty() {
                NARRATE
                    .action("status_empty")
                    .job_id(&job_id)
                    .human(
                        "No online workers found.\n\
                         \n\
                         Workers must send heartbeats to appear here.\n\
                         \n\
                         To spawn a worker:\n\
                         \n\
                           ./rbee worker spawn --model <model> --device <device>",
                    )
                    .emit();
                return Ok(());
            }

            // Display workers
            let mut all_rows = Vec::new();
            for worker in &online_workers {
                all_rows.push(serde_json::json!({
                    "worker_id": worker.id,
                    "model": worker.model_id,
                    "device": worker.device,
                    "port": worker.port,
                    "status": format!("{:?}", worker.status),
                }));
            }

            // Display as table
            NARRATE
                .action("status_result")
                .job_id(&job_id)
                .context(&online_workers.len().to_string())
                .human("Live Status ({} worker(s)):")
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
                .job_id(&job_id)
                .context(response.test_output.unwrap_or_default())
                .human("✅ SSH test successful: {}")
                .emit();
        }
        Operation::HiveInstall { alias } => {
            // TEAM-215: Delegate to hive-lifecycle crate
            let request = HiveInstallRequest { alias };
            execute_hive_install(request, state.config.clone(), &job_id).await?;
        }
        Operation::HiveUninstall { alias } => {
            // ============================================================
            // BUG FIX: TEAM-257 | Hive uninstall message not shown
            // ============================================================
            // SUSPICION:
            // - TEAM-256 suspected SSE stream was closing early
            // - TEAM-256 added response message in uninstall.rs but it wasn't shown
            //
            // INVESTIGATION:
            // - Added debug output to trace execution flow
            // - Discovered action name "hive_cache_cleanup" (18 chars) exceeded 15-char limit
            // - This caused panic in builder.rs:719, stopping all narration
            // - Panic only occurred when had_capabilities=true (first uninstall)
            // - When had_capabilities=false (second uninstall), no panic, message showed
            //
            // ROOT CAUSE:
            // - Action names in uninstall.rs exceeded 15-character limit:
            //   * "hive_cache_cleanup" (18 chars) → panic at line 58
            //   * "hive_cache_error" (16 chars) → would panic at line 67
            //   * "hive_cache_removed" (18 chars) → would panic at line 74
            // - Panic prevented function from completing and returning response
            // - User never saw any messages after initial "Uninstalling" narration
            //
            // FIX:
            // 1. Shortened action names in uninstall.rs to ≤15 chars:
            //    * "hive_cache_cleanup" → "hive_cache_rm" (13 chars)
            //    * "hive_cache_error" → "hive_cache_err" (14 chars)
            //    * "hive_cache_removed" → "hive_cache_ok" (13 chars)
            // 2. Capture response from execute_hive_uninstall() and emit in job_router
            //    This adds redundancy - user sees message from both uninstall.rs and job_router
            //
            // TESTING:
            // - Run `./rbee hive uninstall -a localhost` twice
            // - First run shows: "Removing from capabilities cache", "Removed from capabilities cache",
            //   "Hive 'localhost' uninstalled successfully" (from uninstall.rs + job_router)
            // - Second run shows: "Hive 'localhost' already uninstalled (no cached capabilities)"
            //   (from uninstall.rs + job_router)
            // - Verified all messages appear in output, no panics
            // ============================================================

            // TEAM-215: Delegate to hive-lifecycle crate
            let request = HiveUninstallRequest { alias };
            let response = execute_hive_uninstall(request, state.config.clone(), &job_id).await?;

            // TEAM-257: Emit final status message from response
            // This ensures the message is shown even if SSE stream closed early
            NARRATE
                .action("hive_uninst_ok")
                .job_id(&job_id)
                .context(&response.message)
                .human("{}")
                .emit();
        }
        Operation::HiveStart { alias } => {
            // TEAM-215: Delegate to hive-lifecycle crate
            let request = HiveStartRequest { alias, job_id: job_id.clone() };
            execute_hive_start(request, state.config.clone()).await?;
        }
        Operation::HiveStop { alias } => {
            // TEAM-215: Delegate to hive-lifecycle crate
            let request = HiveStopRequest { alias, job_id: job_id.clone() };
            execute_hive_stop(request, state.config.clone()).await?;
        }
        Operation::HiveList => {
            // TEAM-215: Delegate to hive-lifecycle crate
            let request = HiveListRequest {};
            execute_hive_list(request, state.config.clone(), &job_id).await?;
        }
        Operation::HiveGet { alias } => {
            // TEAM-215: Delegate to hive-lifecycle crate
            let request = HiveGetRequest { alias };
            execute_hive_get(request, state.config.clone(), &job_id).await?;
        }
        Operation::HiveStatus { alias } => {
            // TEAM-215: Delegate to hive-lifecycle crate
            let request = HiveStatusRequest { alias, job_id: job_id.clone() };
            execute_hive_status(request, state.config.clone(), &job_id).await?;
        }
        Operation::HiveRefreshCapabilities { alias } => {
            // TEAM-215: Delegate to hive-lifecycle crate
            let request = HiveRefreshCapabilitiesRequest { alias, job_id: job_id.clone() };
            execute_hive_refresh_capabilities(request, state.config.clone()).await?;
        }
        Operation::HiveImportSsh { ssh_config_path, default_hive_port } => {
            // Import SSH config into hives.conf
            NARRATE
                .action("ssh_import")
                .job_id(&job_id)
                .context(&ssh_config_path)
                .context(&default_hive_port.to_string())
                .human("📥 Importing SSH config from {} (HivePort: {})")
                .emit();

            use std::path::PathBuf;
            let ssh_path = PathBuf::from(&ssh_config_path);

            // Import from SSH config
            let imported = match rbee_config::HivesConfig::import_from_ssh_config(
                &ssh_path,
                default_hive_port,
            ) {
                Ok(config) => config,
                Err(e) => {
                    NARRATE
                        .action("ssh_fail")
                        .job_id(&job_id)
                        .context(&ssh_config_path)
                        .context(e.to_string())
                        .human("❌ Failed to import SSH config from {}: {}")
                        .error_kind("import_failed")
                        .emit();
                    return Err(anyhow::anyhow!("Failed to import SSH config: {}", e));
                }
            };

            NARRATE
                .action("ssh_parsed")
                .job_id(&job_id)
                .context(imported.len().to_string())
                .human("✅ Parsed {} host(s) from SSH config")
                .emit();

            // Load existing hives.conf
            let config_dir = match rbee_config::RbeeConfig::config_dir() {
                Ok(dir) => dir,
                Err(e) => {
                    NARRATE
                        .action("ssh_cfg_dir")
                        .job_id(&job_id)
                        .context(e.to_string())
                        .human("❌ Failed to get config directory: {}")
                        .error_kind("config_dir_failed")
                        .emit();
                    return Err(e.into());
                }
            };
            let hives_conf_path = config_dir.join("hives.conf");

            let mut existing = match rbee_config::HivesConfig::load(&hives_conf_path) {
                Ok(config) => config,
                Err(e) => {
                    NARRATE
                        .action("ssh_load_fail")
                        .job_id(&job_id)
                        .context(hives_conf_path.display().to_string())
                        .context(e.to_string())
                        .human("❌ Failed to load existing hives.conf from {}: {}")
                        .error_kind("load_failed")
                        .emit();
                    return Err(e.into());
                }
            };
            let existing_count = existing.len();

            // Merge (existing wins)
            existing.merge(imported);
            let new_count = existing.len() - existing_count;

            // Save
            if let Err(e) = existing.save(&hives_conf_path) {
                NARRATE
                    .action("ssh_save_fail")
                    .job_id(&job_id)
                    .context(hives_conf_path.display().to_string())
                    .context(e.to_string())
                    .human("❌ Failed to save hives.conf to {}: {}")
                    .error_kind("save_failed")
                    .emit();
                return Err(e.into());
            }

            NARRATE
                .action("ssh_complete")
                .job_id(&job_id)
                .context(&new_count.to_string())
                .context(&hives_conf_path.display().to_string())
                .human("✅ Imported {} new host(s) to {}")
                .emit();

            if new_count == 0 {
                NARRATE
                    .action("ssh_no_new")
                    .job_id(&job_id)
                    .human("ℹ️  All hosts already exist in hives.conf (no duplicates)")
                    .emit();
            }
        }

        // ========================================================================
        // INFERENCE ROUTING - CRITICAL ARCHITECTURE NOTE (TEAM-261)
        // ========================================================================
        //
        // ⚠️  UNINTUITIVE BUT CORRECT: Infer is handled in QUEEN, not forwarded to HIVE!
        //
        // Why?
        // - Queen needs direct control for scheduling/load balancing
        // - Hive only manages worker LIFECYCLE (spawn/stop/list)
        // - Queen → Worker is DIRECT HTTP (no job-server on worker side)
        // - This eliminates a hop and simplifies the inference hot path
        //
        // DO NOT use hive_forwarder::forward_to_hive() for Infer!
        // Queen circumvents hive for performance.
        //
        // See: bin/.plan/TEAM_261_ARCHITECTURE_CLARITY.md
        //
        Operation::Infer {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            ..
        } => {
            // TEAM-275: Use scheduler crate (pre-wired for M2 Rhai scheduler)
            use queen_rbee_scheduler::{JobRequest, JobScheduler, SimpleScheduler};

            NARRATE
                .action("infer_start")
                .job_id(&job_id)
                .context(&model)
                .human("🤖 Starting inference for model '{}'")
                .emit();

            let job_request = JobRequest {
                job_id: job_id.clone(),
                model: model.clone(),
                prompt: prompt.clone(),
                max_tokens,
                temperature,
                top_p,
                top_k,
            };

            // TEAM-275: Use SimpleScheduler (M0/M1)
            // TODO M2: Replace with RhaiScheduler for programmable routing
            let scheduler = SimpleScheduler::new(state.hive_registry.clone());

            // Schedule (find worker)
            let schedule_result = scheduler.schedule(job_request.clone()).await.map_err(|e| {
                anyhow::anyhow!("Scheduling failed: {}", e)
            })?;

            // Create line handler that emits to SSE
            let line_handler = |line: &str| -> Result<(), queen_rbee_scheduler::SchedulerError> {
                NARRATE
                    .action("infer_token")
                    .job_id(&job_id)
                    .human(line)
                    .emit();
                Ok(())
            };

            // Execute job and stream results
            scheduler.execute_job(schedule_result, job_request, line_handler).await.map_err(|e| {
                anyhow::anyhow!("Job execution failed: {}", e)
            })?;

            NARRATE
                .action("infer_success")
                .job_id(&job_id)
                .human("✅ Inference complete")
                .emit();
        }

        // TEAM-272: Active worker operations (query queen's registry)
        // These operations query the worker registry maintained by queen via heartbeats
        Operation::ActiveWorkerList => {
            NARRATE
                .action("active_worker_list_start")
                .job_id(&job_id)
                .human("📋 Listing active workers")
                .emit();

            // TODO: Query worker registry
            NARRATE
                .action("active_worker_list_empty")
                .job_id(&job_id)
                .human("No active workers found (worker registry not yet implemented)")
                .emit();

            // Future implementation:
            // let workers = state.worker_registry.list_active();
            // for worker in &workers {
            //     NARRATE
            //         .action("active_worker_entry")
            //         .job_id(&job_id)
            //         .context(&worker.id)
            //         .context(&worker.model_id)
            //         .context(&worker.hive_id)
            //         .human("  {} | {} | {}")
            //         .emit();
            // }
        }

        Operation::ActiveWorkerGet { worker_id } => {
            NARRATE
                .action("active_worker_get_start")
                .job_id(&job_id)
                .context(&worker_id)
                .human("🔍 Getting active worker '{}'")
                .emit();

            // TODO: Query worker registry
            return Err(anyhow::anyhow!(
                "ActiveWorkerGet not yet implemented (worker registry needed). Worker ID: {}",
                worker_id
            ));

            // Future implementation:
            // let worker = state.worker_registry.get(&worker_id)?;
            // let json = serde_json::to_string_pretty(&worker)?;
            // NARRATE
            //     .action("active_worker_found")
            //     .job_id(&job_id)
            //     .human(&json)
            //     .emit();
        }

        Operation::ActiveWorkerRetire { worker_id } => {
            NARRATE
                .action("active_worker_retire_start")
                .job_id(&job_id)
                .context(&worker_id)
                .human("🛑 Retiring active worker '{}'")
                .emit();

            // TODO: Mark worker as retired in registry
            return Err(anyhow::anyhow!(
                "ActiveWorkerRetire not yet implemented (worker registry needed). Worker ID: {}",
                worker_id
            ));

            // Future implementation:
            // state.worker_registry.retire(&worker_id)?;
            // NARRATE
            //     .action("active_worker_retired")
            //     .job_id(&job_id)
            //     .context(&worker_id)
            //     .human("✅ Worker '{}' retired")
            //     .emit();
        }

        // TEAM-258: All worker/model operations are forwarded to hive
        // This allows new operations to be added to rbee-hive without modifying queen-rbee
        op if op.should_forward_to_hive() => {
            hive_forwarder::forward_to_hive(&job_id, op, state.config.clone()).await?
        }

        // Catch-all for any unhandled operations
        op => {
            return Err(anyhow::anyhow!("Operation '{}' is not implemented", op.name()));
        }
    }

    Ok(())
}
