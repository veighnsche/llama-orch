//! Job routing and operation dispatch for rbee-hive
//!
//! TEAM-261: Mirrors queen-rbee pattern for consistency
//!
//! This module handles:
//! - Parsing operation payloads into typed Operation enum
//! - Routing operations to appropriate handlers
//! - Job lifecycle management (create, register, execute)
//!
//! # Architecture
//!
//! ```text
//! POST /v1/jobs (JSON payload from queen-rbee)
//!     ↓
//! job_router::route_job()
//!     ↓
//! Parse into Operation enum
//!     ↓
//! Match and dispatch to handler
//!     ↓
//! Execute async in background
//!     ↓
//! Stream results via SSE back to queen
//! ```

use anyhow::Result;
use job_server::JobRegistry;
use observability_narration_core::n;
use operations_contract::Operation; // TEAM-284: Renamed from rbee_operations
use rbee_hive_artifact_catalog::{Artifact, ArtifactCatalog}; // TEAM-273: Traits for catalog methods
use rbee_hive_model_catalog::ModelCatalog; // TEAM-268: Model catalog
use rbee_hive_worker_catalog::WorkerCatalog; // TEAM-274: Worker catalog
use std::sync::Arc;

// TEAM-314: All narration migrated to n!() macro

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>, // TEAM-268: Model catalog
    pub worker_catalog: Arc<WorkerCatalog>, // TEAM-274: Worker catalog
                                          // TODO: Add model_provisioner when TEAM-269 implements it
                                          // TODO: Add worker_registry when implemented
}

/// Response from job creation
#[derive(Debug, serde::Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Create a new job and store its payload
///
/// TEAM-261: Mirrors queen-rbee pattern
/// Called by HTTP layer to create jobs.
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    let sse_url = format!("/v1/jobs/{}/stream", job_id);

    state.registry.set_payload(&job_id, payload);

    // Create job-specific SSE channel for isolation
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 1000);

    n!("job_create", "Job {} created, waiting for client connection", job_id);

    Ok(JobResponse { job_id, sse_url })
}

/// Execute a job by retrieving its payload and streaming results
///
/// TEAM-261: Mirrors queen-rbee pattern
/// Called by HTTP layer when client connects to SSE stream.
pub async fn execute_job(
    job_id: String,
    state: JobState,
) -> impl futures::stream::Stream<Item = String> {
    let registry = state.registry.clone();
    let state_clone = state.clone(); // TEAM-268: Clone full state for closure

    // TEAM-312: Pass None for timeout (no timeout needed for hive operations)
    job_server::execute_and_stream(
        job_id,
        registry.clone(),
        move |job_id, payload| route_operation(job_id, payload, state_clone.clone()),
        None,
    )
    .await
}

/// Internal: Route operation to appropriate handler
///
/// TEAM-261: Parse payload and dispatch to worker/model handlers
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState, // TEAM-268: Changed from _registry to full state
) -> Result<()> {
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name();

    n!("route_job", "Executing operation: {}", operation_name);

    // ============================================================================
    // OPERATION ROUTING
    // ============================================================================
    //
    // Hive handles Worker and Model operations
    // (Hive operations like HiveStart/HiveStop are handled by queen-rbee)
    //
    match operation {
        // TEAM-313: HiveCheck - narration test through hive SSE
        // TEAM-314: Migrated to n!() macro
        Operation::HiveCheck { .. } => {
            use observability_narration_core::{with_narration_context, NarrationContext};

            n!("hive_check_start", "🔍 Starting hive narration check");

            // Set narration context so n!() calls route to SSE
            let ctx = NarrationContext::new().with_job_id(&job_id);
            with_narration_context(ctx, rbee_hive::hive_check::handle_hive_check()).await?;

            n!("hive_check_complete", "✅ Hive narration check complete");
        }

        // Worker operations
        // TEAM-284: Updated to use typed requests
        Operation::WorkerSpawn(request) => {
            // TEAM-272: Implemented worker spawning using worker-lifecycle
            use rbee_hive_worker_lifecycle::{start_worker, WorkerStartConfig};

            n!(
                "worker_spawn_start",
                "🚀 Spawning worker '{}' with model '{}' on device {}",
                request.worker,
                request.model,
                request.device
            );

            // Allocate port (simple sequential allocation for now)
            // TODO: Implement proper port allocation
            let port = 9000 + (rand::random::<u16>() % 1000);

            // Queen URL for heartbeat (hardcoded for now)
            // TODO: Get from config
            let queen_url = "http://localhost:7833".to_string();

            let config = WorkerStartConfig {
                worker_id: request.worker.clone(),
                model_id: request.model.clone(),
                device: request.device.to_string(),
                port,
                queen_url,
                job_id: job_id.clone(),
            };

            let result = start_worker(config).await?;

            n!(
                "worker_spawn_complete",
                "✅ Worker '{}' spawned (PID: {}, port: {})",
                result.worker_id,
                result.pid,
                result.port
            );
        }

        // TEAM-278: DELETED WorkerBinaryList, WorkerBinaryGet, WorkerBinaryDelete (~110 LOC)
        // Worker binary management is now handled by PackageSync in queen-rbee

        // TEAM-274: Worker process operations (local ps-based)
        Operation::WorkerProcessList(request) => {
            let hive_id = request.hive_id.clone();
            use rbee_hive_worker_lifecycle::list_workers;

            n!("worker_proc_list_start", "📋 Listing worker processes on hive '{}'", hive_id);

            let processes = list_workers(&job_id).await?;

            n!("worker_proc_list_result", "Found {} worker process(es)", processes.len());

            if processes.is_empty() {
                n!("worker_proc_list_empty", "No worker processes found");
            } else {
                for proc in &processes {
                    // TEAM-278: WorkerInfo only has pid, command, args
                    n!("worker_proc_list_entry", "  PID {} | {}", proc.pid, proc.command);
                }
            }
        }

        Operation::WorkerProcessGet(request) => {
            let hive_id = request.hive_id.clone();
            let pid = request.pid;
            use rbee_hive_worker_lifecycle::get_worker;

            n!(
                "worker_proc_get_start",
                "🔍 Getting worker process PID {} on hive '{}'",
                pid,
                hive_id
            );

            let proc_info = get_worker(&job_id, pid).await?;

            // TEAM-278: WorkerInfo only has pid, command, args
            n!("worker_proc_get_found", "✅ PID {}: {}", proc_info.pid, proc_info.command);

            // Emit process details as JSON
            let json = serde_json::to_string_pretty(&proc_info)
                .unwrap_or_else(|_| "Failed to serialize".to_string());

            n!("worker_proc_get_details", "{}", json);
        }

        Operation::WorkerProcessDelete(request) => {
            let hive_id = request.hive_id.clone();
            let pid = request.pid;
            use rbee_hive_worker_lifecycle::stop_worker;

            n!(
                "worker_proc_del_start",
                "🗑️  Deleting worker process PID {} on hive '{}'",
                pid,
                hive_id
            );

            // Worker ID is not known (hive is stateless), so use generic ID
            let worker_id = format!("pid-{}", pid);
            stop_worker(&job_id, &worker_id, pid).await?;

            n!("worker_proc_del_ok", "✅ Worker process PID {} deleted successfully", pid);
        }

        // Model operations
        Operation::ModelDownload(request) => {
            let hive_id = request.hive_id.clone();
            let model = request.model.clone();
            // TEAM-269: Implemented model download with provisioner
            n!("model_download_start", "📥 Downloading model '{}' on hive '{}'", model, hive_id);

            // Check if model already exists
            if state.model_catalog.contains(&model) {
                n!("model_download_exists", "⚠️  Model '{}' already exists in catalog", model);
                return Err(anyhow::anyhow!("Model '{}' already exists", model));
            }

            // TODO: TEAM-269 will implement model provisioner
            n!(
                "model_download_not_implemented",
                "⚠️  Model download not yet implemented (waiting for TEAM-269)"
            );

            return Err(anyhow::anyhow!(
                "Model download not yet implemented (waiting for TEAM-269)"
            ));
        }

        Operation::ModelList(request) => {
            let hive_id = request.hive_id.clone();
            // TEAM-268: Implemented model list
            n!("model_list_start", "📋 Listing models on hive '{}'", hive_id);

            let models = state.model_catalog.list();

            n!("model_list_result", "Found {} model(s)", models.len());

            // Format as JSON table
            if models.is_empty() {
                n!("model_list_empty", "No models found");
            } else {
                for model in &models {
                    let size_gb = model.size() as f64 / 1_000_000_000.0;
                    let status = match model.status() {
                        rbee_hive_artifact_catalog::ArtifactStatus::Available => "available",
                        rbee_hive_artifact_catalog::ArtifactStatus::Downloading => "downloading",
                        rbee_hive_artifact_catalog::ArtifactStatus::Failed { .. } => "failed",
                    };

                    n!(
                        "model_list_entry",
                        "  {} | {} | {:.2} GB | {}",
                        model.id(),
                        model.name(),
                        size_gb,
                        status
                    );
                }
            }
        }

        Operation::ModelGet(request) => {
            let hive_id = request.hive_id.clone();
            let id = request.id.clone();
            // TEAM-268: Implemented model get
            n!("model_get_start", "🔍 Getting model '{}' on hive '{}'", id, hive_id);

            match state.model_catalog.get(&id) {
                Ok(model) => {
                    n!(
                        "model_get_found",
                        "✅ Model: {} | Name: {} | Path: {}",
                        model.id(),
                        model.name(),
                        model.path().display()
                    );

                    // Emit model details as JSON
                    let json = serde_json::to_string_pretty(&model)
                        .unwrap_or_else(|_| "Failed to serialize".to_string());

                    n!("model_get_details", "{}", json);
                }
                Err(e) => {
                    n!("model_get_error", "❌ Model '{}' not found: {}", id, e);
                    return Err(e);
                }
            }
        }

        Operation::ModelDelete(request) => {
            let hive_id = request.hive_id.clone();
            let id = request.id.clone();
            // TEAM-268: Implemented model delete
            n!("model_delete_start", "🗑️  Deleting model '{}' on hive '{}'", id, hive_id);

            match state.model_catalog.remove(&id) {
                Ok(()) => {
                    n!("model_delete_complete", "✅ Model '{}' deleted successfully", id);
                }
                Err(e) => {
                    n!("model_delete_error", "❌ Failed to delete model '{}': {}", id, e);
                    return Err(e);
                }
            }
        }

        // ========================================================================
        // INFERENCE REJECTION - CRITICAL ARCHITECTURE NOTE (TEAM-261)
        // ========================================================================
        //
        // ⚠️  INFER SHOULD NOT BE IN HIVE!
        //
        // Why?
        // - Hive only manages worker LIFECYCLE (spawn/stop/list)
        // - Queen handles inference routing DIRECTLY to workers
        // - Queen → Worker is DIRECT HTTP (circumvents hive)
        // - This is INTENTIONAL for performance and simplicity
        //
        // If you see Infer here, something is wrong with the routing in queen-rbee!
        //
        // See: bin/.plan/TEAM_261_ARCHITECTURE_CLARITY.md
        //
        Operation::Infer { .. } => {
            return Err(anyhow::anyhow!(
                "Infer operation should NOT be routed to hive! \
                 Queen should route inference directly to workers. \
                 This indicates a routing bug in queen-rbee/src/job_router.rs. \
                 See bin/.plan/TEAM_261_ARCHITECTURE_CLARITY.md for details."
            ));
        }

        // Unsupported operations (handled by queen-rbee)
        _ => {
            return Err(anyhow::anyhow!(
                "Operation '{}' is not supported by rbee-hive (should be handled by queen-rbee)",
                operation_name
            ));
        }
    }

    Ok(())
}
