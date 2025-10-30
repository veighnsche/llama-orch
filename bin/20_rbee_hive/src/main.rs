// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-189: Implemented basic HTTP server with /health endpoint
// TEAM-190: Added hive heartbeat task to send status to queen every 5 seconds
// TEAM-202: Replaced println!() with narration for job-scoped SSE visibility
// TEAM-218: Investigated Oct 22, 2025 - Behavior inventory complete
// Purpose: rbee-hive binary entry point (DAEMON ONLY - NO CLI!)
// Status: Basic HTTP daemon with heartbeat and narration - ready for worker management implementation

//! rbee-hive
//!
//! Daemon for managing LLM worker instances on a single machine

mod heartbeat; // TEAM-292: Re-enabled hive heartbeat
mod http;
mod job_router;

use observability_narration_core::n;

use axum::{
    extract::{Query, State}, // TEAM-365: Query for queen_url parameter, State for HiveState
    routing::{delete, get, post}, // TEAM-305-FIX: Added delete for cancel endpoint
    Json,
    Router,
};
use clap::Parser;
use job_server::JobRegistry;
use rbee_hive_artifact_catalog::ArtifactCatalog; // TEAM-273: Trait for catalog methods
use rbee_hive_model_catalog::ModelCatalog; // TEAM-268: Model catalog
use rbee_hive_worker_catalog::WorkerCatalog; // TEAM-274: Worker catalog
use serde::{Deserialize, Serialize}; // TEAM-365: Deserialize for CapabilitiesQuery
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering}; // TEAM-365: Atomic for heartbeat control
use std::sync::Arc;
use tokio::sync::RwLock; // TEAM-365: RwLock for dynamic queen_url

#[derive(Parser, Debug)]
#[command(name = "rbee-hive")]
#[command(about = "rbee Hive Daemon - Worker and model management")]
#[command(version)]
struct Args {
    /// HTTP server port
    #[arg(short, long, default_value = "7835")]
    port: u16,

    /// Queen URL for heartbeat reporting
    /// TEAM-292: Added to enable hive heartbeat
    #[arg(long, default_value = "http://localhost:7833")]
    queen_url: String,

    /// Hive ID (alias)
    /// TEAM-292: Added to identify this hive
    #[arg(long, default_value = "localhost")]
    hive_id: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // TEAM-202: Use narration instead of println!()
    // This automatically goes through job-scoped SSE (if in job context)
    // and uses centralized formatting (TEAM-201)
    // TEAM-261: Simplified - no hive heartbeat (workers send to queen directly)
    // TEAM-340: Migrated to n!() macro
    n!("startup", "🐝 Starting rbee-hive on port {}", args.port);

    // TEAM-261: Initialize job registry for dual-call pattern
    let job_registry: Arc<JobRegistry<String>> = Arc::new(JobRegistry::new());

    // TEAM-268: Initialize model catalog
    let model_catalog = Arc::new(ModelCatalog::new().expect("Failed to initialize model catalog"));

    // TEAM-340: Migrated to n!() macro
    n!("catalog_init", "📚 Model catalog initialized ({} models)", model_catalog.len());

    // TEAM-274: Initialize worker catalog
    let worker_catalog =
        Arc::new(WorkerCatalog::new().expect("Failed to initialize worker catalog"));

    // TEAM-340: Migrated to n!() macro
    n!("worker_cat_init", "🔧 Worker catalog initialized ({} binaries)", worker_catalog.len());

    // TODO: TEAM-269 will add model provisioner initialization here

    // TEAM-365: Create HiveInfo for heartbeat
    let hive_info = hive_contract::HiveInfo {
        id: args.hive_id.clone(),
        hostname: "127.0.0.1".to_string(),
        port: args.port,
        operational_status: hive_contract::OperationalStatus::Ready,
        health_status: hive_contract::HealthStatus::Healthy,
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    // TEAM-365: Create shared HiveState for dynamic queen_url and heartbeat control
    let hive_state = Arc::new(HiveState {
        job_registry: job_registry.clone(),
        model_catalog: model_catalog.clone(),
        worker_catalog: worker_catalog.clone(),
        queen_url: Arc::new(RwLock::new(Some(args.queen_url.clone()))), // TEAM-365: Dynamic queen URL
        heartbeat_running: Arc::new(AtomicBool::new(false)), // TEAM-365: Heartbeat control
        hive_info: hive_info.clone(),
    });

    // TEAM-261: Create HTTP state for job endpoints (for backwards compatibility)
    // TEAM-268: Added model_catalog to state
    // TEAM-274: Added worker_catalog to state
    let job_state = http::jobs::HiveState { registry: job_registry, model_catalog, worker_catalog };

    // ============================================================
    // BUG FIX: TEAM-291 | Fixed Axum routing panic on startup
    // ============================================================
    // SUSPICION:
    // - TEAM-290 reported hive crashes immediately after spawn
    // - Error: "Path segments must not start with `:`. For capture groups, use `{capture}`"
    //
    // INVESTIGATION:
    // - Checked Axum version in Cargo.toml - using 0.7.x
    // - Found line 92 using old Axum 0.6 syntax `:job_id`
    // - Axum 0.7+ requires new syntax `{job_id}`
    //
    // ROOT CAUSE:
    // - Route pattern used old Axum 0.6 syntax (`:job_id`)
    // - Axum 0.7+ requires curly braces (`{job_id}`)
    // - This caused panic on router creation, before HTTP server started
    //
    // FIX:
    // - Changed `:job_id` to `{job_id}` in route pattern
    // - Now compatible with Axum 0.7+
    //
    // TESTING:
    // - ./rbee hive start - SUCCESS (no crash)
    // - pgrep -f rbee-hive - SUCCESS (process running)
    // - curl http://localhost:9000/health - SUCCESS (returns "ok")
    // ============================================================

    // TEAM-365: Create router with two different states
    // - HiveState for capabilities endpoint (needs queen_url)
    // - JobState for job endpoints (existing pattern)
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/capabilities", get(get_capabilities))
        .with_state(hive_state.clone()) // TEAM-365: HiveState for capabilities
        .route("/v1/shutdown", post(http::handle_shutdown)) // TEAM-339: Graceful shutdown endpoint
        .route("/v1/jobs", post(http::jobs::handle_create_job))
        .route("/v1/jobs/{job_id}/stream", get(http::jobs::handle_stream_job)) // TEAM-291: Fixed :job_id → {job_id}
        .route("/v1/jobs/{job_id}", delete(http::jobs::handle_cancel_job)) // TEAM-305-FIX: Cancel job endpoint
        .with_state(job_state);

    // TEAM-335: Bind to 0.0.0.0 to allow remote access (needed for remote hives)
    // Localhost-only binding (127.0.0.1) would prevent health checks from remote machines
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));

    // TEAM-202: Narrate listen address
    // TEAM-340: Migrated to n!() macro
    n!("listen", "✅ Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // TEAM-202: Narrate ready state
    // TEAM-340: Migrated to n!() macro
    n!("ready", "✅ Hive ready");

    // TEAM-365: Start heartbeat task with discovery (exponential backoff)
    // This will be implemented in Phase 4
    let _heartbeat_handle = heartbeat::start_heartbeat_task(hive_info, args.queen_url.clone());

    // TEAM-340: Migrated to n!() macro
    n!("heartbeat_start", "💓 Heartbeat task started (sending to {})", args.queen_url);

    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check endpoint
/// TEAM-189: Returns "ok" - used by hive start/stop/status operations
async fn health_check() -> &'static str {
    "ok"
}

/// TEAM-205: Device information for capabilities response
#[derive(Debug, Serialize)]
struct HiveDevice {
    id: String,
    name: String,
    device_type: String,
    vram_gb: Option<u32>,
    compute_capability: Option<String>,
}

/// TEAM-205: Capabilities response
#[derive(Debug, Serialize)]
struct CapabilitiesResponse {
    devices: Vec<HiveDevice>,
}

/// TEAM-365: Query parameters for capabilities endpoint
#[derive(Debug, Deserialize)]
struct CapabilitiesQuery {
    /// Queen URL for bidirectional discovery
    queen_url: Option<String>,
}

/// TEAM-365: Shared state for dynamic Queen URL and heartbeat control
#[derive(Clone)]
pub struct HiveState {
    pub job_registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,
    pub worker_catalog: Arc<WorkerCatalog>,
    pub queen_url: Arc<RwLock<Option<String>>>,  // TEAM-365: Dynamic queen URL
    pub heartbeat_running: Arc<AtomicBool>,      // TEAM-365: Prevent duplicate tasks
    pub hive_info: hive_contract::HiveInfo,      // TEAM-365: For heartbeat
}

impl HiveState {
    /// TEAM-365: Store queen URL dynamically
    pub async fn set_queen_url(&self, url: String) {
        *self.queen_url.write().await = Some(url);
    }
    
    /// TEAM-365: Start heartbeat task (idempotent - only starts once)
    pub async fn start_heartbeat_task(&self, queen_url: String) {
        // Only start if not already running
        if self.heartbeat_running.swap(true, Ordering::SeqCst) {
            n!("heartbeat_skip", "💓 Heartbeat already running, skipping");
            return;
        }
        
        n!("heartbeat_start", "💓 Starting heartbeat task to {}", queen_url);
        let hive_info = self.hive_info.clone();
        heartbeat::start_heartbeat_task(hive_info, queen_url);
    }
}

/// TEAM-205: Capabilities endpoint - returns detected GPU/CPU devices
/// TEAM-206: Added comprehensive narration for device detection visibility
/// TEAM-365: Enhanced with queen_url parameter for bidirectional discovery
async fn get_capabilities(
    Query(params): Query<CapabilitiesQuery>,
    State(state): State<Arc<HiveState>>,
) -> Json<CapabilitiesResponse> {
    // TEAM-206: Narrate incoming request
    // TEAM-340: Migrated to n!() macro
    n!("caps_request", "📡 Received capabilities request from queen");
    
    // TEAM-365: Handle queen_url parameter for discovery
    if let Some(queen_url) = params.queen_url {
        n!("caps_queen_url", "🔗 Queen URL received: {}", queen_url);
        state.set_queen_url(queen_url.clone()).await;
        state.start_heartbeat_task(queen_url).await;
    }

    // TEAM-206: Narrate GPU detection attempt
    // TEAM-340: Migrated to n!() macro
    n!("caps_gpu_check", "🔍 Detecting GPUs via nvidia-smi...");

    // Detect GPUs
    let gpu_info = rbee_hive_device_detection::detect_gpus();

    // TEAM-206: Narrate GPU detection results
    // TEAM-340: Migrated to n!() macro
    if gpu_info.count > 0 {
        n!("caps_gpu_found", "✅ Found {} GPU(s)", gpu_info.count);
    } else {
        n!("caps_gpu_none", "ℹ️  No GPUs detected, using CPU only");
    }

    let mut devices: Vec<HiveDevice> = gpu_info
        .devices
        .iter()
        .map(|gpu| HiveDevice {
            id: format!("GPU-{}", gpu.index),
            name: gpu.name.clone(),
            device_type: "gpu".to_string(),
            vram_gb: Some(gpu.vram_total_gb() as u32),
            compute_capability: Some(format!(
                "{}.{}",
                gpu.compute_capability.0, gpu.compute_capability.1
            )),
        })
        .collect();

    // TEAM-209: Get actual CPU system information
    let cpu_cores = rbee_hive_device_detection::get_cpu_cores();
    let system_ram_gb = rbee_hive_device_detection::get_system_ram_gb();

    // TEAM-206: Narrate CPU fallback
    // TEAM-340: Migrated to n!() macro
    n!("caps_cpu_add", "🖥️  Adding CPU-0: {} cores, {} GB RAM", cpu_cores, system_ram_gb);

    // Add CPU device (always available) with actual system info
    devices.push(HiveDevice {
        id: "CPU-0".to_string(),
        name: format!("CPU ({} cores)", cpu_cores),
        device_type: "cpu".to_string(),
        vram_gb: Some(system_ram_gb), // System RAM for CPU device
        compute_capability: None,
    });

    // TEAM-206: Narrate response being sent
    // TEAM-340: Migrated to n!() macro
    n!("caps_response", "📤 Sending capabilities response ({} device(s))", devices.len());

    Json(CapabilitiesResponse { devices })
}
