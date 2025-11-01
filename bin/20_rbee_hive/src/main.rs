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

// TEAM-XXX: Build metadata via shadow-rs
use shadow_rs::shadow;
shadow!(build);

use observability_narration_core::n;
use tower_http::cors::{Any, CorsLayer}; // TEAM-374: CORS support for web UI

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
use rbee_hive_model_provisioner::ModelProvisioner; // Model provisioner for downloads
use rbee_hive_worker_catalog::WorkerCatalog; // TEAM-274: Worker catalog
use serde::{Deserialize, Serialize}; // TEAM-365: Deserialize for CapabilitiesQuery
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering}; // TEAM-365: Atomic for heartbeat control
use std::sync::Arc;
use tokio::sync::RwLock; // TEAM-365: RwLock for dynamic queen_url
use tokio::sync::broadcast; // TEAM-372: For SSE broadcast channel

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

    /// Print build information and exit
    #[arg(long, hide = true)]
    build_info: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Handle --build-info flag
    if args.build_info {
        println!("{}", build::BUILD_RUST_CHANNEL);
        std::process::exit(0);
    }

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

    // Initialize model provisioner
    let model_provisioner = Arc::new(
        ModelProvisioner::new().expect("Failed to initialize model provisioner")
    );
    n!("provisioner_init", "📥 Model provisioner initialized (HuggingFace)");

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

    // TEAM-372: Create broadcast channel for SSE heartbeat stream
    // Capacity of 100 events - if clients are slow, old events are dropped
    let (heartbeat_tx, _) = broadcast::channel::<http::heartbeat_stream::HiveHeartbeatEvent>(100);

    let heartbeat_stream_state = Arc::new(http::heartbeat_stream::HeartbeatStreamState {
        hive_info: hive_info.clone(),
        event_tx: heartbeat_tx.clone(),
    });

    // TEAM-372: Start telemetry broadcaster (replaces POST loop)
    let _broadcaster_handle = http::heartbeat_stream::start_telemetry_broadcaster(
        hive_info.clone(),
        heartbeat_tx,
    );

    tracing::info!("SSE telemetry broadcaster started");

    // TEAM-261: Create HTTP state for job endpoints (for backwards compatibility)
    // TEAM-268: Added model_catalog to state
    // TEAM-274: Added worker_catalog to state
    let job_state = http::jobs::HiveState {
        registry: job_registry,
        model_catalog,
        model_provisioner,
        worker_catalog,
    };

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

    // TEAM-374: Log debug mode status
    #[cfg(debug_assertions)]
    {
        eprintln!("🔧 [HIVE] Running in DEBUG mode");
        eprintln!("   - /dev/{{*path}} → Proxy to Vite dev server (port 7836)");
        eprintln!("   - / → Static files (if built)");
    }
    #[cfg(not(debug_assertions))]
    {
        eprintln!("🚀 [HIVE] Running in RELEASE mode");
        eprintln!("   - / → Embedded static files (production)");
    }

    // TEAM-365: Create router with two different states
    // - HiveState for capabilities endpoint (needs queen_url)
    // - JobState for job endpoints (existing pattern)
    // TEAM-381: Moved /capabilities to /v1/capabilities for API consistency
    let mut app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/capabilities", get(get_capabilities))
        .with_state(hive_state.clone()) // TEAM-365: HiveState for capabilities
        // TEAM-372: SSE heartbeat stream (for Queen and Hive SDK)
        .route("/v1/heartbeats/stream", get(http::handle_heartbeat_stream))
        .with_state(heartbeat_stream_state)
        .route("/v1/shutdown", post(http::handle_shutdown)) // TEAM-339: Graceful shutdown endpoint
        .route("/v1/jobs", post(http::jobs::handle_create_job))
        .route("/v1/jobs/{job_id}/stream", get(http::jobs::handle_stream_job)) // TEAM-291: Fixed :job_id → {job_id}
        .route("/v1/jobs/{job_id}", delete(http::jobs::handle_cancel_job)) // TEAM-305-FIX: Cancel job endpoint
        .with_state(job_state);

    // TEAM-378: Development proxy - Extract Vite dev server URL from queen_url
    // For remote hives, proxy to the dev machine (where queen is running)
    // CRITICAL: Axum requires {*path} syntax for wildcard capture, NOT *path
    // Using *path will panic: "Path segments must not start with `*`"
    // CRITICAL: Must be added BEFORE static router merge, otherwise static fallback catches it!
    // CRITICAL: Need both /dev and /dev/{*path} to handle root and subpaths
    
    // Extract hostname from queen_url (e.g., "http://localhost:7833" → "localhost")
    let vite_host = if let Ok(queen_uri) = args.queen_url.parse::<axum::http::Uri>() {
        queen_uri.host().unwrap_or("localhost").to_string()
    } else {
        "localhost".to_string()
    };
    let vite_url = format!("http://{}:7836", vite_host);
    
    n!("dev_proxy_config", "🔧 Dev proxy configured: /dev → {}", vite_url);
    
    let dev_proxy_state = http::DevProxyState {
        vite_url: std::sync::Arc::new(vite_url),
    };
    
    // Create dev proxy router with its own state
    let dev_router = Router::new()
        .route("/dev", get(http::dev_proxy_handler))
        .route("/dev/", get(http::dev_proxy_handler))
        .route("/dev/{*path}", get(http::dev_proxy_handler))
        .with_state(dev_proxy_state);
    
    // Merge dev router into main app
    app = app.merge(dev_router);

    // TEAM-378: Add static file serving for production
    // Merge static router AFTER API routes so API takes priority
    // Static router has fallback handler for SPA routing
    let static_router = http::create_static_router();
    app = app.merge(static_router);

    // TEAM-381: Add CORS layer AFTER all router merges to ensure it applies to ALL routes
    let cors = CorsLayer::new()
        .allow_origin(Any) // Allow any origin for development
        .allow_methods(Any) // Allow any HTTP method
        .allow_headers(Any); // Allow any headers

    app = app.layer(cors);

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
    // TEAM-366: EDGE CASE #1 - Pass running_flag to prevent duplicate tasks
    // TEAM-367: EDGE CASE #2 FIX - queen_url is optional (standalone mode)
    let _heartbeat_handle = heartbeat::start_heartbeat_task(
        hive_info.clone(),
        Some(args.queen_url.clone()),
        hive_state.heartbeat_running.clone(),
    );

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
    /// TEAM-366: EDGE CASE #2 - Validate URL before storing
    pub async fn set_queen_url(&self, url: String) -> Result<(), String> {
        // TEAM-366: Validate URL
        if url.is_empty() {
            return Err("Cannot set empty queen_url".to_string());
        }
        
        if let Err(e) = url::Url::parse(&url) {
            return Err(format!("Invalid queen_url '{}': {}", url, e));
        }
        
        *self.queen_url.write().await = Some(url);
        Ok(())
    }
    
    /// TEAM-365: Start heartbeat task (idempotent - only starts once)
    /// TEAM-366: EDGE CASE #3 - Handle Queen URL changes
    /// TEAM-367: EDGE CASE #2 FIX - queen_url is optional
    pub async fn start_heartbeat_task(&self, queen_url: Option<String>) {
        // TEAM-367: Handle None case (standalone mode)
        let url = match queen_url {
            None => {
                n!("heartbeat_skip", "ℹ️  No queen_url provided, skipping heartbeat (standalone mode)");
                return;
            }
            Some(u) => u,
        };
        
        // TEAM-366: EDGE CASE #3 - Check if URL changed
        let current_url = self.queen_url.read().await.clone();
        if let Some(existing) = current_url {
            if existing != url {
                n!("heartbeat_url_changed", "⚠️  Queen URL changed: {} → {}. Heartbeat will continue to old URL.", existing, url);
                // TODO: In future, implement graceful task restart for URL changes
                // For now, keep sending to original Queen (prevents thrashing)
            }
        }
        
        // Only start if not already running
        if self.heartbeat_running.swap(true, Ordering::SeqCst) {
            n!("heartbeat_skip", "💓 Heartbeat already running, skipping");
            return;
        }
        
        n!("heartbeat_start", "💓 Starting heartbeat task to {}", url);
        let hive_info = self.hive_info.clone();
        heartbeat::start_heartbeat_task(
            hive_info,
            Some(url),
            self.heartbeat_running.clone(),
        );
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
    // TEAM-366: EDGE CASE #2 - Validate queen_url before using
    if let Some(queen_url) = params.queen_url {
        n!("caps_queen_url", "🔗 Queen URL received: {}", queen_url);
        
        // TEAM-366: Validate and store URL
        // TEAM-367: Pass Some(url) for optional parameter
        match state.set_queen_url(queen_url.clone()).await {
            Ok(_) => {
                state.start_heartbeat_task(Some(queen_url)).await;
            }
            Err(e) => {
                n!("caps_invalid_url", "❌ Invalid queen_url rejected: {}", e);
            }
        }
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
