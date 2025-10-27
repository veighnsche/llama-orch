//! queen-rbee - Orchestrator Daemon
//!
//! TEAM-151: Migrated from old.queen-rbee
//! TEAM-151: Cleaned up main.rs (97 → 77 lines)
//! TEAM-151: Wired up src/http/ folder with health endpoint
//! TEAM-151: Health endpoint active on GET /health
//! TEAM-152: Replaced tracing with narration for observability
//! TEAM-217: Investigated Oct 22, 2025 - Behavior inventory complete
//!
//! # Happy Flow
//! "The queen bee wakes up and immediately starts the http server."
//! Port 7833 (default) - rbee-keeper checks GET /health to see if queen is running

// TEAM-164: Migrated endpoints to dedicated modules/files
mod hive_forwarder; // TEAM-258: Generic forwarding for hive-managed operations
mod http;
mod job_router; // TEAM-186: Job routing and operation dispatch
mod narration; // TEAM-188: Narration constants
               // TEAM-188: operations module doesn't exist yet
               // mod operations;

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{delete, get, post}, // TEAM-305-FIX: Added delete for cancel endpoint
};
use tower_http::cors::{CorsLayer, Any}; // TEAM-288: CORS support for web UI
use anyhow::Result; // TEAM-288: Import Result for main function
use clap::Parser;
use job_server::JobRegistry;
use observability_narration_core::NarrationFactory;
// TEAM-290: DELETED rbee_config import (file-based config deprecated)
use std::net::SocketAddr;
use std::sync::Arc;

// TEAM-192: Local narration factory for main.rs
const NARRATE: NarrationFactory = NarrationFactory::new("queen");
// TEAM-188: operations module doesn't exist yet
// use crate::operations::*;

#[derive(Parser, Debug)]
#[command(name = "queen-rbee")]
#[command(about = "rbee Orchestrator Daemon - Job scheduling and hive management")]
#[command(version)]
struct Args {
    /// HTTP server port
    #[arg(short, long, default_value = "7833")]
    port: u16,
    // TEAM-290: DELETED config and config_dir args (file-based config deprecated)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // TEAM-164: Initialize SSE sink for distributed narration
    // TEAM-204: Removed init() - no global channel, job channels created on-demand

    NARRATE
        .action("start")
        .context(args.port.to_string())
        .human("Queen-rbee starting on port {} (localhost-only mode)")
        .emit();

    // TEAM-290: No config loading (file-based config deprecated)

    // TEAM-155: Initialize job registry for dual-call pattern
    // Generic over String for now (will stream text tokens)
    let job_server: Arc<JobRegistry<String>> = Arc::new(JobRegistry::new());

    // TEAM-188: Initialize worker registry (RAM) for runtime state
    // TEAM-262: Renamed from hive_registry to worker_registry
    let worker_registry = Arc::new(queen_rbee_worker_registry::WorkerRegistry::new());

    // TODO: Initialize other registries when migrated
    // - beehive_registry (SQLite catalog + RAM registry)
    // - worker_registry (RAM)
    // TEAM-290: No config loading (file-based config deprecated)

    let app = create_router(job_server, worker_registry);
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    NARRATE.action("listen").context(addr.to_string()).human("Listening on http://{}").emit();

    NARRATE.action("ready").human("Ready to accept connections").emit();

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await.map_err(|e| {
        NARRATE
            .action("error")
            .context(e.to_string())
            .human("Server error: {}")
            .error_kind("server_failed")
            .emit();
        anyhow::anyhow!("Server failed: {}", e)
    })
}

/// Create HTTP router
///
/// TEAM-155: Added job endpoints for dual-call pattern
/// TEAM-156: Added hive catalog to state
/// TEAM-158: Added heartbeat endpoint for hive health monitoring
/// TEAM-290: Removed config parameter (file-based config deprecated)
/// Currently health, shutdown, job, and heartbeat endpoints are active.
/// TODO: Uncomment http::routes::create_router() when registries are migrated
fn create_router(
    job_server: Arc<JobRegistry<String>>,
    worker_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>, // TEAM-262: Renamed
) -> axum::Router {
    // TEAM-164: Create states for HTTP endpoints
    // TEAM-190: Added hive_registry to job_state for Status operation
    // TEAM-290: Removed config from job_state (file-based config deprecated)
    let job_state = http::SchedulerState {
        registry: job_server,
        hive_registry: worker_registry.clone(), // TEAM-262: Still named hive_registry in struct
    };

    // TEAM-284: Initialize both worker and hive registries
    let hive_registry = Arc::new(queen_rbee_hive_registry::HiveRegistry::new());
    
    // TEAM-288: Create broadcast channel for real-time heartbeat events
    // Capacity of 100 events - if clients are slow, old events are dropped
    let (event_tx, _) = tokio::sync::broadcast::channel(100);
    
    let heartbeat_state = http::HeartbeatState {
        worker_registry: worker_registry.clone(),
        hive_registry,
        event_tx, // TEAM-288: Broadcast channel for real-time events
    };

    // TEAM-288: Add CORS layer to allow web UI access
    let cors = CorsLayer::new()
        .allow_origin(Any) // Allow any origin for development
        .allow_methods(Any) // Allow any HTTP method
        .allow_headers(Any); // Allow any headers

    // TEAM-293: Create API router first (takes priority over static files)
    let api_router = axum::Router::new()
        // Health check (no /v1 prefix for compatibility)
        .route("/health", get(http::handle_health))
        // TEAM-186: V1 API endpoints (matches API_REFERENCE.md)
        // TEAM-327: Removed /v1/shutdown (use signal-based shutdown: SIGTERM/SIGKILL)
        .route("/v1/build-info", get(http::handle_build_info)) // TEAM-262: Build information
        .route("/v1/info", get(http::handle_info)) // TEAM-292: Queen info for service discovery
        // TEAM-275: Removed /v1/heartbeat endpoint (deprecated, use /v1/worker-heartbeat instead)
        .route("/v1/worker-heartbeat", post(http::handle_worker_heartbeat)) // TEAM-261: Workers send heartbeats directly to queen
        .route("/v1/hive-heartbeat", post(http::handle_hive_heartbeat)) // TEAM-284/285: Hives send heartbeats directly to queen
        .route("/v1/heartbeats/stream", get(http::handle_heartbeat_stream)) // TEAM-285: Live heartbeat streaming for web UI
        .with_state(heartbeat_state)
        .route("/v1/jobs", post(http::handle_create_job))
        .with_state(job_state.clone())
        .route("/v1/jobs/{job_id}/stream", get(http::handle_stream_job)) // TEAM-288: Fixed path syntax for axum 0.8
        .with_state(job_state.clone())
        .route("/v1/jobs/{job_id}", delete(http::handle_cancel_job)) // TEAM-305-FIX: Cancel job endpoint
        .with_state(job_state);

    // TEAM-293: Merge API routes with static file serving
    // API routes take priority - static files are fallback
    api_router
        .merge(http::create_static_router())
        .layer(cors) // TEAM-288: Apply CORS layer to all routes
}

// TEAM-327: Removed handle_shutdown() - use signal-based shutdown (SIGTERM/SIGKILL) instead
