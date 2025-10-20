//! queen-rbee - Orchestrator Daemon
//!
//! TEAM-151: Migrated from old.queen-rbee
//! TEAM-151: Cleaned up main.rs (97 → 77 lines)
//! TEAM-151: Wired up src/http/ folder with health endpoint
//! TEAM-151: Health endpoint active on GET /health
//! TEAM-152: Replaced tracing with narration for observability
//!
//! # Happy Flow
//! "The queen bee wakes up and immediately starts the http server."
//! Port 8500 (default) - rbee-keeper checks GET /health to see if queen is running

mod http;

use anyhow::Result;
use axum::routing::{get, post};
use clap::Parser;
use job_registry::JobRegistry;
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::HiveCatalog;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

// Actor and action constants
// TEAM-155: Added emoji prefix for visual identification
const ACTOR_QUEEN_RBEE: &str = "👑 queen-rbee";
const ACTION_START: &str = "start";
const ACTION_LISTEN: &str = "listen";
const ACTION_READY: &str = "ready";
const ACTION_ERROR: &str = "error";

#[derive(Parser, Debug)]
#[command(name = "queen-rbee")]
#[command(about = "rbee Orchestrator Daemon - Job scheduling and hive management")]
#[command(version)]
struct Args {
    /// HTTP server port
    #[arg(short, long, default_value = "8500")]
    port: u16,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,

    /// Database path (SQLite) for hive catalog
    #[arg(short, long)]
    database: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    Narration::new(ACTOR_QUEEN_RBEE, ACTION_START, &args.port.to_string())
        .human(format!("Queen-rbee starting on port {}", args.port))
        .emit();

    // TEAM-156: Initialize hive catalog
    let catalog_path =
        args.database.map(PathBuf::from).unwrap_or_else(|| PathBuf::from("queen-hive-catalog.db"));

    let hive_catalog = Arc::new(
        HiveCatalog::new(&catalog_path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to initialize hive catalog: {}", e))?,
    );

    Narration::new(ACTOR_QUEEN_RBEE, ACTION_START, &catalog_path.display().to_string())
        .human(format!("Initialized hive catalog at {}", catalog_path.display()))
        .emit();

    // TEAM-155: Initialize job registry for dual-call pattern
    // Generic over String for now (will stream text tokens)
    let job_registry: Arc<JobRegistry<String>> = Arc::new(JobRegistry::new());

    // TODO: Initialize other registries when migrated
    // - beehive_registry (SQLite catalog + RAM registry)
    // - worker_registry (RAM)
    // - Load config from args.config

    let app = create_router(job_registry, hive_catalog);
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    Narration::new(ACTOR_QUEEN_RBEE, ACTION_LISTEN, &addr.to_string())
        .human(format!("Listening on http://{}", addr))
        .emit();

    Narration::new(ACTOR_QUEEN_RBEE, ACTION_READY, "http-server")
        .human("Ready to accept connections")
        .emit();

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await.map_err(|e| {
        Narration::new(ACTOR_QUEEN_RBEE, ACTION_ERROR, "http-server")
            .human(format!("Server error: {}", e))
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
/// Currently health, shutdown, job, and heartbeat endpoints are active.
/// TODO: Uncomment http::routes::create_router() when registries are migrated
fn create_router(
    job_registry: Arc<JobRegistry<String>>,
    hive_catalog: Arc<HiveCatalog>,
) -> axum::Router {
    // TEAM-156: Create job state for endpoints with hive catalog
    let job_state =
        http::jobs::QueenJobState { registry: job_registry, hive_catalog: hive_catalog.clone() };

    // TEAM-158: Create heartbeat state
    // TEAM-161: Add device detector for first heartbeat flow
    let device_detector = Arc::new(http::device_detector::HttpDeviceDetector::new());
    let heartbeat_state = http::heartbeat::HeartbeatState {
        hive_catalog: hive_catalog.clone(),
        device_detector,
    };

    // TEAM-160: Add hive endpoint state
    let add_hive_state = hive_catalog;

    axum::Router::new()
        .route("/health", get(http::health::handle_health))
        .route("/shutdown", post(http::shutdown::handle_shutdown))
        // TEAM-155: Job endpoints (dual-call pattern)
        .route("/jobs", post(http::jobs::handle_create_job))
        .route("/jobs/{job_id}/stream", get(http::jobs::handle_stream_job))
        .with_state(job_state)
        // TEAM-158: Heartbeat endpoint
        .route("/heartbeat", post(http::heartbeat::handle_heartbeat))
        .with_state(heartbeat_state)
        // TEAM-160: Add hive endpoint for E2E testing
        .route("/add-hive", post(http::add_hive::handle_add_hive))
        .with_state(add_hive_state)
}
