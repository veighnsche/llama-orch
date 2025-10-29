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
use operations_contract::Operation;
use std::sync::Arc;

use super::hive_forwarder;

// TEAM-312: Queen-check handler module
#[path = "handlers/queen_check.rs"]
mod queen_check;

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub hive_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>,
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
    let hive_registry = state.hive_registry.clone();

    job_server::execute_and_stream(
        job_id,
        registry.clone(),
        move |job_id, payload| route_operation(job_id, payload, registry, hive_registry),
        None,
    )
    .await
}

/// Internal: Route operation to appropriate handler
///
/// This parses the payload and dispatches to the correct operation handler.
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    registry: Arc<JobRegistry<String>>,
    hive_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>,
) -> Result<()> {
    let state = JobState { registry, hive_registry };
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name();

    n!("route_job", "Executing operation: {}", operation_name);

    match operation {
        // ═══════════════════════════════════════════════════════════════════════
        // QUEEN OPERATIONS
        // ═══════════════════════════════════════════════════════════════════════
        
        Operation::Status => {
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

            n!("status_result", "Live Status ({} worker(s))", online_workers.len());
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::Value::Array(all_rows)).unwrap()
            );
        }

        // ⚠️  CRITICAL: Infer is handled in QUEEN, not forwarded to HIVE!
        //
        // Why?
        // - Queen needs direct control for scheduling/load balancing
        // - Hive only manages worker LIFECYCLE (spawn/stop/list)
        // - Queen → Worker is DIRECT HTTP (no job-server on worker side)
        // - This eliminates a hop and simplifies the inference hot path
        Operation::Infer(req) => {
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

            let scheduler = SimpleScheduler::new(state.hive_registry.clone());

            let schedule_result = scheduler
                .schedule(job_request.clone())
                .await
                .map_err(|e| anyhow::anyhow!("Scheduling failed: {}", e))?;

            let line_handler = |line: &str| -> Result<(), queen_rbee_scheduler::SchedulerError> {
                println!("{}", line);
                Ok(())
            };

            scheduler
                .execute_job(schedule_result, job_request, line_handler)
                .await
                .map_err(|e| anyhow::anyhow!("Job execution failed: {}", e))?;

            n!("infer_success", "✅ Inference complete");
        }

        // ═══════════════════════════════════════════════════════════════════════
        // DIAGNOSTIC OPERATIONS
        // ═══════════════════════════════════════════════════════════════════════
        
        Operation::QueenCheck => {
            let ctx = NarrationContext::new().with_job_id(&job_id);
            with_narration_context(ctx, queen_check::handle_queen_check()).await?;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // RHAI SCRIPT OPERATIONS
        // ═══════════════════════════════════════════════════════════════════════
        // TEAM-350: RHAI operations now use config structs for #[with_job_id] macro
        Operation::RhaiScriptSave { name, content, id } => {
            let config = crate::rhai::RhaiSaveConfig {
                job_id: Some(job_id.clone()),
                name,
                content,
                id,
            };
            crate::rhai::execute_rhai_script_save(config).await?;
        }
        Operation::RhaiScriptTest { content } => {
            let config = crate::rhai::RhaiTestConfig {
                job_id: Some(job_id.clone()),
                content,
            };
            crate::rhai::execute_rhai_script_test(config).await?;
        }
        Operation::RhaiScriptGet { id } => {
            let config = crate::rhai::RhaiGetConfig {
                job_id: Some(job_id.clone()),
                id,
            };
            crate::rhai::execute_rhai_script_get(config).await?;
        }
        Operation::RhaiScriptList => {
            let config = crate::rhai::RhaiListConfig {
                job_id: Some(job_id.clone()),
            };
            crate::rhai::execute_rhai_script_list(config).await?;
        }
        Operation::RhaiScriptDelete { id } => {
            let config = crate::rhai::RhaiDeleteConfig {
                job_id: Some(job_id.clone()),
                id,
            };
            crate::rhai::execute_rhai_script_delete(config).await?;
        }

        // ═══════════════════════════════════════════════════════════════════════
        // FORWARDING TO HIVE
        // ═══════════════════════════════════════════════════════════════════════
        // All worker/model operations are forwarded to rbee-hive
        op if op.target_server() == operations_contract::TargetServer::Hive => {
            hive_forwarder::forward_to_hive(&job_id, op).await?;
        }

        op => {
            return Err(anyhow::anyhow!("Unhandled operation: '{}'", op.name()));
        }
    }

    Ok(())
}
