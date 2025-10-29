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
use observability_narration_core::{n, with_narration_context, NarrationContext};
// TEAM-278: DELETED execute_hive_install, execute_hive_uninstall, execute_ssh_test
// TEAM-285: DELETED execute_hive_start, execute_hive_stop (localhost-only, no lifecycle management)
// TEAM-290: DELETED hive_lifecycle imports (queen no longer manages hives, rbee-keeper does via SSH)
// TEAM-275: Removed unused import (using state.hive_registry which is already Arc<WorkerRegistry>)
// TEAM-284: DELETED HivesConfig import (no longer needed without daemon-sync)
// TEAM-290: DELETED rbee_config import (file-based config deprecated)
use operations_contract::Operation; // TEAM-284: Renamed from rbee_operations
use std::sync::Arc;

use super::hive_forwarder; // TEAM-258: Generic forwarding for hive-managed operations
                           // TEAM-284: DELETED daemon_sync import (SSH/remote operations removed)

// TEAM-312: Migrated to n!() macro

// TEAM-312: Queen-check handler module
#[path = "handlers/queen_check.rs"]
mod queen_check;

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    // TEAM-290: DELETED config field (file-based config deprecated)
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

    // TEAM-312: Use n!() macro - actor auto-detected from crate name
    n!("job_create", "Job {} created, waiting for client connection", job_id);

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
    // TEAM-290: DELETED config clone (file-based config deprecated)
    let hive_registry = state.hive_registry.clone(); // TEAM-190

    // TEAM-312: Pass None for timeout (no timeout needed for queen operations)
    job_server::execute_and_stream(
        job_id,
        registry.clone(),
        move |job_id, payload| route_operation(job_id, payload, registry, hive_registry),
        None,
    )
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
    // TEAM-290: DELETED config parameter (file-based config deprecated)
    hive_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>, // TEAM-190/262/275: For Status and Infer operations
) -> Result<()> {
    let state = JobState { registry, hive_registry };
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name();

    n!("route_job", "Executing operation: {}", operation_name);

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

            n!("status", "📊 Fetching live status from worker registry");

            // Get all online workers (workers with recent heartbeats)
            let online_workers = state.hive_registry.list_online_workers();

            if online_workers.is_empty() {
                n!(
                    "status_empty",
                    "No online workers found.\n\n\
                    Workers must send heartbeats to appear here.\n\n\
                    To spawn a worker:\n\n  \
                    ./rbee worker spawn --model <model> --device <device>"
                );
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
            n!("status_result", "Live Status ({} worker(s))", online_workers.len());
            // TEAM-312: Table display via println for now
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::Value::Array(all_rows)).unwrap()
            );
        }

        // TEAM-312: Queen-check operation for deep narration testing
        Operation::QueenCheck => {
            // TEAM-312: Set narration context so n!() calls route to SSE
            let ctx = NarrationContext::new().with_job_id(&job_id);
            with_narration_context(ctx, queen_check::handle_queen_check()).await?;
        }

        // TEAM-284: DELETED all Package operations (PackageSync, PackageStatus, PackageInstall, PackageUninstall, PackageValidate, PackageMigrate)
        // SSH and remote installation functionality removed
        // TEAM-278: DELETED Operation::SshTest
        // TEAM-285: DELETED Operation::HiveStart, Operation::HiveStop (localhost-only, no lifecycle management)
        // TEAM-290: DELETED HiveList, HiveGet, HiveStatus, HiveRefreshCapabilities (queen no longer manages hives, rbee-keeper does via SSH)

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
        // TEAM-285: Fixed to match TEAM-284's typed request pattern
        Operation::Infer(req) => {
            // TEAM-275: Use scheduler crate (pre-wired for M2 Rhai scheduler)
            use queen_rbee_scheduler::{JobRequest, JobScheduler, SimpleScheduler};

            n!("infer_start", "🤖 Starting inference: model={}, prompt={}", req.model, req.prompt);

            let job_request = JobRequest {
                job_id: job_id.clone(),
                model: req.model.clone(),
                prompt: req.prompt.clone(),
                max_tokens: req.max_tokens,
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: req.top_k,
            };

            // TEAM-275: Use SimpleScheduler (M0/M1)
            // TODO M2: Replace with RhaiScheduler for programmable routing
            let scheduler = SimpleScheduler::new(state.hive_registry.clone());

            // Schedule (find worker)
            let schedule_result = scheduler
                .schedule(job_request.clone())
                .await
                .map_err(|e| anyhow::anyhow!("Scheduling failed: {}", e))?;

            // Create line handler that emits to SSE
            let line_handler = |line: &str| -> Result<(), queen_rbee_scheduler::SchedulerError> {
                println!("{}", line); // TEAM-312: Direct output for streaming
                Ok(())
            };

            // Execute job and stream results
            scheduler
                .execute_job(schedule_result, job_request, line_handler)
                .await
                .map_err(|e| anyhow::anyhow!("Job execution failed: {}", e))?;

            n!("infer_success", "✅ Inference complete");
        }

        // TEAM-272: Active worker operations (query queen's registry)
        // These operations query the worker registry maintained by queen via heartbeats
        Operation::ActiveWorkerList => {
            n!("active_worker_list_start", "📋 Listing active workers");

            // TODO: Query worker registry
            n!(
                "active_worker_list_empty",
                "No active workers found (worker registry not yet implemented)"
            );

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
            n!("active_worker_get_start", "🔍 Getting active worker '{}'", worker_id);

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
            n!("active_worker_retire_start", "🛑 Retiring active worker '{}'", worker_id);

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

        // TEAM-258/CLEANUP: All worker/model operations are forwarded to hive
        // TEAM-CLEANUP: Updated to use target_server() instead of should_forward_to_hive()
        op if matches!(op.target_server(), operations_contract::TargetServer::Hive) => {
            hive_forwarder::forward_to_hive(&job_id, op).await?
        }

        // Catch-all for any unhandled operations
        op => {
            return Err(anyhow::anyhow!("Operation '{}' is not implemented", op.name()));
        }
    }

    Ok(())
}
