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
use rbee_hive_artifact_catalog::{Artifact, ArtifactCatalog, ArtifactProvisioner}; // TEAM-273: Traits for catalog methods
use rbee_hive_model_catalog::ModelCatalog; // TEAM-268: Model catalog
use rbee_hive_model_provisioner::ModelProvisioner; // Model provisioner for downloads
use rbee_hive_worker_catalog::WorkerCatalog; // TEAM-274: Worker catalog
use std::sync::Arc;

// TEAM-314: All narration migrated to n!() macro

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>, // TEAM-268: Model catalog
    pub model_provisioner: Arc<ModelProvisioner>, // Model provisioner for downloads
    pub worker_catalog: Arc<WorkerCatalog>, // TEAM-274: Worker catalog
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
/// TEAM-381: Set narration context once for ALL operations (not just HiveCheck)
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState, // TEAM-268: Changed from _registry to full state
) -> Result<()> {
    use observability_narration_core::{with_narration_context, NarrationContext};
    
    // TEAM-381: Set narration context for ALL operations so n!() calls route to SSE
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async move {
        // Parse payload into typed Operation enum
        let operation: Operation = serde_json::from_value(payload)
            .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

        let operation_name = operation.name().to_string();

        n!("route_job", "Executing operation: {}", operation_name);
        
        execute_operation(operation, operation_name, job_id, state).await
    }).await
}

/// TEAM-381: Execute the actual operation logic (extracted for cleaner context wrapping)
async fn execute_operation(operation: Operation, operation_name: String, job_id: String, state: JobState) -> Result<()> {

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
        // TEAM-381: Narration context now set at router level, no need to set here
        Operation::HiveCheck { .. } => {
            n!("hive_check_start", "🔍 Starting hive narration check");
            rbee_hive::hive_check::handle_hive_check().await?;
            n!("hive_check_complete", "✅ Hive narration check complete");
        }

        // ========================================================================
        // WORKER OPERATIONS
        // ========================================================================
        //
        // TEAM-334: Worker lifecycle uses daemon-lifecycle directly
        //
        // IMPORTANT: Worker binary installation/uninstallation is NOT handled here!
        // - Worker binaries are managed by worker-catalog
        // - Hive only spawns/stops worker PROCESSES
        // - There are unlimited types of workers (cpu, cuda, metal, vulkan, etc.)
        // - Each worker type is a separate binary in the catalog
        //
        // Worker operations:
        // - WorkerSpawn: Start a worker process (assumes binary exists in catalog)
        // - WorkerProcessList/Get/Delete: Manage running processes (ps/kill)
        //
        Operation::WorkerSpawn(request) => {
            use lifecycle_local::{start_daemon, HttpDaemonConfig, StartConfig};
            use rbee_hive_worker_catalog::{Platform, WorkerType};

            n!(
                "worker_spawn_start",
                "🚀 Spawning worker '{}' with model '{}' on device {}",
                request.worker,
                request.model,
                request.device
            );

            // Determine worker type from worker string (e.g., "cpu", "cuda")
            // NOTE: This assumes the worker binary is already in the catalog!
            // Worker installation is handled by worker-catalog, not hive.
            let worker_type = match request.worker.as_str() {
                "cuda" => WorkerType::CudaLlm,
                "cpu" => WorkerType::CpuLlm,
                "metal" => WorkerType::MetalLlm,
                _ => return Err(anyhow::anyhow!("Unsupported worker type: {}", request.worker)),
            };

            // Find worker binary in catalog
            // NOTE: If binary not found, it means worker-catalog needs to install it first!
            // Hive does NOT install worker binaries - that's worker-catalog's job.
            let _worker_binary = state
                .worker_catalog
                .find_by_type_and_platform(worker_type, Platform::current())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Worker binary not found for {:?}. \
                     Worker binaries must be installed via worker-catalog first!",
                        worker_type
                    )
                })?;

            // Allocate port
            let port = 9000 + (rand::random::<u16>() % 1000);
            let queen_url = "http://localhost:7833".to_string();
            let worker_id = format!("worker-{}-{}", request.worker, port);

            // Build worker arguments
            let args = vec![
                "--worker-id".to_string(),
                worker_id.clone(),
                "--model".to_string(),
                request.model.clone(),
                "--device".to_string(),
                request.device.to_string(),
                "--port".to_string(),
                port.to_string(),
                "--queen-url".to_string(),
                queen_url.clone(),
            ];

            // TEAM-359: Start worker with monitoring (cgroup on Linux)
            // TEAM-372: Fixed API - use with_monitor_group + with_monitor_instance
            let base_url = format!("http://localhost:{}", port);
            let daemon_config = HttpDaemonConfig::new(&worker_id, &base_url)
                .with_args(args)
                .with_monitor_group("llm")
                .with_monitor_instance(port.to_string());
            
            let config = StartConfig {
                daemon_config,
                job_id: Some(job_id.clone()),
            };

            let pid = start_daemon(config).await?;

            n!(
                "worker_spawn_complete",
                "✅ Worker '{}' spawned (PID: {}, port: {})",
                worker_id,
                pid,
                port
            );
        }

        // ========================================================================
        // WORKER BINARY MANAGEMENT - NOT IMPLEMENTED HERE
        // ========================================================================
        //
        // Worker binary installation/uninstallation is the responsibility of:
        // - worker-catalog (manages the catalog of available worker binaries)
        // - queen-rbee's PackageSync (distributes binaries to hives)
        //
        // Hive ONLY manages worker PROCESSES, not binaries!
        //
        // There are unlimited types of workers:
        // - cpu-llm-worker-rbee
        // - cuda-llm-worker-rbee
        // - metal-llm-worker-rbee
        // - vulkan-llm-worker-rbee (future)
        // - rocm-llm-worker-rbee (future)
        // - etc.
        //
        // Each worker type is a separate binary that must be:
        // 1. Built (cargo build)
        // 2. Added to worker-catalog
        // 3. Distributed to hives (via PackageSync)
        //
        // TEAM-278: DELETED WorkerBinaryList, WorkerBinaryGet, WorkerBinaryDelete (~110 LOC)
        //
        // ========================================================================

        // TEAM-334: Worker process operations using ps command
        Operation::WorkerProcessList(request) => {
            let hive_id = request.hive_id.clone();

            n!("worker_proc_list_start", "📋 Listing worker processes on hive '{}'", hive_id);

            // Use ps to list worker processes
            let output = tokio::process::Command::new("ps")
                .args(&["aux"])
                .output()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to run ps: {}", e))?;

            let stdout = String::from_utf8_lossy(&output.stdout);
            let worker_lines: Vec<_> = stdout
                .lines()
                .filter(|line| line.contains("llm-worker") || line.contains("worker-rbee"))
                .collect();

            n!("worker_proc_list_result", "Found {} worker process(es)", worker_lines.len());

            if worker_lines.is_empty() {
                n!("worker_proc_list_empty", "No worker processes found");
            } else {
                for line in worker_lines {
                    n!("worker_proc_list_entry", "  {}", line);
                }
            }
        }

        Operation::WorkerProcessGet(request) => {
            let hive_id = request.hive_id.clone();
            let pid = request.pid;

            n!(
                "worker_proc_get_start",
                "🔍 Getting worker process PID {} on hive '{}'",
                pid,
                hive_id
            );

            // Use ps to get specific process
            let output = tokio::process::Command::new("ps")
                .args(&["-p", &pid.to_string(), "-o", "pid,command"])
                .output()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to run ps: {}", e))?;

            if !output.status.success() {
                return Err(anyhow::anyhow!("Process {} not found", pid));
            }

            let stdout = String::from_utf8_lossy(&output.stdout);
            n!("worker_proc_get_found", "✅ PID {}: {}", pid, stdout.trim());
        }

        Operation::WorkerProcessDelete(request) => {
            let hive_id = request.hive_id.clone();
            let pid = request.pid;

            n!(
                "worker_proc_del_start",
                "🗑️  Deleting worker process PID {} on hive '{}'",
                pid,
                hive_id
            );

            // Kill process using SIGTERM
            #[cfg(unix)]
            {
                use nix::sys::signal::{kill, Signal};
                use nix::unistd::Pid;

                let pid_nix = Pid::from_raw(pid as i32);
                match kill(pid_nix, Signal::SIGTERM) {
                    Ok(_) => {
                        n!("worker_proc_del_sigterm", "Sent SIGTERM to PID {}", pid);
                        // Wait briefly for graceful shutdown
                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                        // Try SIGKILL if still alive
                        let _ = kill(pid_nix, Signal::SIGKILL);
                    }
                    Err(_) => {
                        n!("worker_proc_del_already_dead", "Process {} may already be dead", pid);
                    }
                }
            }

            #[cfg(not(unix))]
            {
                return Err(anyhow::anyhow!("Process killing not supported on this platform"));
            }

            n!("worker_proc_del_ok", "✅ Worker process PID {} deleted successfully", pid);
        }

        // Model operations
        Operation::ModelDownload(request) => {
            let hive_id = request.hive_id.clone();
            let model = request.model.clone();
            n!("model_download_start", "📥 Downloading model '{}' on hive '{}'", model, hive_id);

            // Check if model already exists
            if state.model_catalog.contains(&model) {
                n!("model_download_exists", "⚠️  Model '{}' already exists in catalog", model);
                return Err(anyhow::anyhow!("Model '{}' already exists", model));
            }

            // Provision model from HuggingFace
            let model_entry = state.model_provisioner.provision(&model, &job_id).await?;

            // Add to catalog
            state.model_catalog.add(model_entry)?;

            n!("model_download_complete", "✅ Model '{}' downloaded and added to catalog", model);
        }

        Operation::ModelList(request) => {
            let hive_id = request.hive_id.clone();
            // TEAM-268: Implemented model list
            n!("model_list_start", "📋 Listing models on hive '{}'", hive_id);

            let models = state.model_catalog.list();

            n!("model_list_result", "Found {} model(s)", models.len());

            // Emit JSON response for frontend consumption
            let json = serde_json::to_string(&models)
                .unwrap_or_else(|_| "[]".to_string());
            n!("model_list_json", "{}", json);
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

        Operation::ModelLoad(request) => {
            let _hive_id = request.hive_id.clone();
            let id = request.id.clone();
            let device = request.device.clone();
            n!("model_load_start", "🚀 Loading model '{}' to RAM on device '{}'", id, device);

            // TODO: Implement actual model loading to RAM
            // This will spawn a worker with the model loaded
            // For now, just narrate the intent
            n!("model_load_progress", "📦 Allocating memory for model '{}'", id);
            n!("model_load_progress", "🔄 Loading model weights into VRAM/RAM");
            n!("model_load_complete", "✅ Model '{}' loaded to RAM on device '{}'", id, device);
        }

        Operation::ModelUnload(request) => {
            let _hive_id = request.hive_id.clone();
            let id = request.id.clone();
            n!("model_unload_start", "🔽 Unloading model '{}' from RAM", id);

            // TODO: Implement actual model unloading
            // This will kill the worker process
            // For now, just narrate the intent
            n!("model_unload_progress", "🧹 Freeing memory for model '{}'", id);
            n!("model_unload_complete", "✅ Model '{}' unloaded from RAM", id);
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
