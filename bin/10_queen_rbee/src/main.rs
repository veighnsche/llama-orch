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

// TEAM-164: Use library modules instead of redeclaring them
// TEAM-XXX: Fixed - main.rs should use the library, not shadow it with mod declarations
use queen_rbee::http;

mod discovery; // TEAM-365: Hive discovery module

// TEAM-XXX: Build metadata via shadow-rs
use shadow_rs::shadow;
shadow!(build);

use anyhow::Result; // TEAM-288: Import Result for main function
use axum::{
    routing::{delete, get, post}, // TEAM-305-FIX: Added delete for cancel endpoint
};
use clap::Parser;
use job_server::JobRegistry;
use observability_narration_core::n;
use tower_http::cors::{Any, CorsLayer}; // TEAM-288: CORS support for web UI
                                        // TEAM-290: DELETED rbee_config import (file-based config deprecated)
use std::net::SocketAddr;
use std::sync::Arc;

// TEAM-192: Local narration factory for main.rs
// TEAM-340: Migrated to n!() macro
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

    /// Print build information and exit
    #[arg(long, hide = true)]
    build_info: bool,
    // TEAM-290: DELETED config and config_dir args (file-based config deprecated)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Handle --build-info flag
    if args.build_info {
        println!("{}", build::BUILD_RUST_CHANNEL);
        std::process::exit(0);
    }

    // TEAM-164: Initialize SSE sink for distributed narration
    // TEAM-204: Removed init() - no global channel, job channels created on-demand

    n!("start", "Queen-rbee starting on port {} (localhost-only mode)", args.port);

    // TEAM-290: No config loading (file-based config deprecated)

    // TEAM-155: Initialize job registry for dual-call pattern
    // Generic over String for now (will stream text tokens)
    let job_server: Arc<JobRegistry<String>> = Arc::new(JobRegistry::new());

    // TEAM-188: Initialize telemetry registry (RAM) for runtime state
    // TEAM-374: Renamed from worker_registry to telemetry (stores hives + workers)
    let telemetry = Arc::new(queen_rbee_telemetry_registry::TelemetryRegistry::new());

    // TODO: Initialize other registries when migrated
    // - beehive_registry (SQLite catalog + RAM registry)
    // - model_registry (SQLite catalog + RAM registry)

    // TEAM-365: Start hive discovery after 5s delay
    let queen_url = format!("http://localhost:{}", args.port);
    tokio::spawn(async move {
        if let Err(e) = discovery::discover_hives_on_startup(&queen_url).await {
            n!("discovery_error", "❌ Hive discovery failed: {}", e);
        }
    });

    // TEAM-152: Start HTTP server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let router = create_router(job_server, telemetry);

    n!("listen", "Listening on http://{}", addr);

    n!("ready", "Ready to accept connections");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await.map_err(|e| {
        n!("error", "Server error: {}", e);
        anyhow::anyhow!("Server failed: {}", e)
    })
}

/// Create HTTP router
///
/// TEAM-155: Added job endpoints for dual-call pattern
/// TEAM-156: Added hive catalog to state
/// TEAM-158: Added heartbeat endpoint for hive health monitoring
/// TEAM-290: Removed config parameter (file-based config deprecated)
/// TEAM-374: Updated to use TelemetryRegistry
/// Currently health, shutdown, job, and heartbeat endpoints are active.
/// TODO: Uncomment http::routes::create_router() when registries are migrated
fn create_router(
    job_server: Arc<JobRegistry<String>>,
    telemetry: Arc<queen_rbee_telemetry_registry::TelemetryRegistry>, // TEAM-374: Renamed
) -> axum::Router {
    // TEAM-164: Create states for HTTP endpoints
    // TEAM-190: Added hive_registry to job_state for Status operation
    // TEAM-290: Removed config from job_state (file-based config deprecated)
    // TEAM-374: Updated to use telemetry registry
    let job_state = http::SchedulerState {
        registry: job_server,
        hive_registry: telemetry.clone(), // TEAM-374: Now uses TelemetryRegistry
    };

    // TEAM-288: Create broadcast channel for real-time heartbeat events
    // Capacity of 100 events - if clients are slow, old events are dropped
    let (event_tx, _) = tokio::sync::broadcast::channel(100);

    let heartbeat_state = http::HeartbeatState {
        worker_registry: telemetry.clone(), // TEAM-374: Now uses TelemetryRegistry
        hive_registry: telemetry.clone(), // TEAM-374: Now uses TelemetryRegistry
        event_tx, // TEAM-288: Broadcast channel for real-time events
    };

    // TEAM-377: DELETED cleanup task - not needed
    // Hives are removed immediately when SSE connection closes
    // No stale entries exist in connection-based tracking

    // TEAM-288: Add CORS layer to allow web UI access
    let cors = CorsLayer::new()
        .allow_origin(Any) // Allow any origin for development
        .allow_methods(Any) // Allow any HTTP method
        .allow_headers(Any); // Allow any headers

    // TEAM-350: Log build mode for debugging
    #[cfg(debug_assertions)]
    {
        eprintln!("🔧 [QUEEN] Running in DEBUG mode");
        eprintln!("   - /dev/{{*path}} → Proxy to Vite dev server (port 7834)");
        eprintln!("   - / → Embedded static files (may be stale, rebuild to update)");
    }
    
    #[cfg(not(debug_assertions))]
    {
        eprintln!("🚀 [QUEEN] Running in RELEASE mode");
        eprintln!("   - / → Embedded static files (production)");
        eprintln!("   - /dev/{{*path}} → Proxy to Vite (for development only)");
    }

    // TEAM-293: Create API router first (takes priority over static files)
    let api_router = axum::Router::new()
        // Health check (no /v1 prefix for compatibility)
        .route("/health", get(http::handle_health))
        // TEAM-186: V1 API endpoints (matches API_REFERENCE.md)
        .route("/v1/shutdown", post(http::handle_shutdown)) // TEAM-339: Graceful shutdown endpoint
        .route("/v1/info", get(http::handle_info)) // TEAM-292/CLEANUP: Queen info (service discovery + build info)
        // TEAM-374: DELETED /v1/hive-heartbeat route - replaced by SSE subscription
        // TEAM-373: Hive ready callback (discovery) - triggers SSE subscription
        .route("/v1/hive/ready", post(http::handle_hive_ready)) // TEAM-373: One-time discovery callback
        .route("/v1/heartbeats/stream", get(http::handle_heartbeat_stream)) // TEAM-285: Live heartbeat streaming for web UI
        .with_state(heartbeat_state)
        .route("/v1/jobs", post(http::handle_create_job))
        .with_state(job_state.clone())
        .route("/v1/jobs/{job_id}/stream", get(http::handle_stream_job)) // TEAM-288: Fixed path syntax for axum 0.8
        .with_state(job_state.clone())
        .route("/v1/jobs/{job_id}", delete(http::handle_cancel_job)) // TEAM-305-FIX: Cancel job endpoint
        .with_state(job_state);

    // TEAM-350: Development proxy - forwards /dev/* to Vite dev server (port 7834)
    // CRITICAL: Axum requires {*path} syntax for wildcard capture, NOT *path
    // Using *path will panic: "Path segments must not start with `*`"
    // CRITICAL: Must be added BEFORE static router merge, otherwise static fallback catches it!
    // CRITICAL: Need both /dev and /dev/{*path} to handle root and subpaths
    let api_router = api_router
        .route("/dev", get(http::dev_proxy_handler))
        .route("/dev/", get(http::dev_proxy_handler))
        .route("/dev/{*path}", get(http::dev_proxy_handler));

    // TEAM-293: Merge API routes with static file serving
    // API routes take priority - static files are fallback
    api_router.merge(http::create_static_router()).layer(cors) // TEAM-288: Apply CORS layer to all routes
}
