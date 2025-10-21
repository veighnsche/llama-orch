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

// TEAM-164: Migrated endpoints to dedicated modules/files
mod http;
mod job_router; // TEAM-186: Job routing and operation dispatch
mod narration; // TEAM-188: Narration constants
               // TEAM-188: operations module doesn't exist yet
               // mod operations;

use anyhow::Result;
use axum::routing::{get, post};
use clap::Parser;
use job_registry::JobRegistry;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::net::SocketAddr;
use std::path::PathBuf;
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
    #[arg(short, long, default_value = "8500")]
    port: u16,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,

    /// Config directory path (defaults to ~/.config/rbee/)
    #[arg(long)]
    config_dir: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // TEAM-164: Initialize SSE sink for distributed narration
    observability_narration_core::sse_sink::init(1000);

    NARRATE.action("start")
        .context(args.port.to_string())
        .human("Queen-rbee starting on port {}")
        .emit();

    // TEAM-194: Load file-based config from ~/.config/rbee/
    let config = if let Some(dir) = args.config_dir {
        RbeeConfig::load_from_dir(&PathBuf::from(dir))
            .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?
    } else {
        RbeeConfig::load()
            .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?
    };

    let config_dir = RbeeConfig::config_dir()?;

    NARRATE.action("start")
        .context(config_dir.display().to_string())
        .human("Loaded config from {}")
        .emit();

    // TEAM-155: Initialize job registry for dual-call pattern
    // Generic over String for now (will stream text tokens)
    let job_registry: Arc<JobRegistry<String>> = Arc::new(JobRegistry::new());

    // TEAM-188: Initialize hive registry (RAM) for runtime state
    let hive_registry = Arc::new(queen_rbee_hive_registry::HiveRegistry::new());

    // TODO: Initialize other registries when migrated
    // - beehive_registry (SQLite catalog + RAM registry)
    // - worker_registry (RAM)
    // - Load config from args.config

    let app = create_router(job_registry, Arc::new(config), hive_registry);
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    NARRATE.action("listen")
        .context(addr.to_string())
        .human("Listening on http://{}")
        .emit();

    NARRATE.action("ready")
        .human("Ready to accept connections")
        .emit();

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await.map_err(|e| {
        NARRATE.action("error")
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
/// Currently health, shutdown, job, and heartbeat endpoints are active.
/// TODO: Uncomment http::routes::create_router() when registries are migrated
fn create_router(
    job_registry: Arc<JobRegistry<String>>,
    config: Arc<RbeeConfig>,
    hive_registry: Arc<queen_rbee_hive_registry::HiveRegistry>,
) -> axum::Router {
    // TEAM-164: Create states for HTTP endpoints
    // TEAM-190: Added hive_registry to job_state for Status operation
    // TEAM-194: Replaced hive_catalog with config
    let job_state = http::SchedulerState {
        registry: job_registry,
        config: config.clone(),
        hive_registry: hive_registry.clone(),
    };

    let heartbeat_state = http::HeartbeatState { hive_registry };

    axum::Router::new()
        // Health check (no /v1 prefix for compatibility)
        .route("/health", get(http::handle_health))
        // TEAM-186: V1 API endpoints (matches API_REFERENCE.md)
        .route("/v1/shutdown", post(handle_shutdown))
        .route("/v1/heartbeat", post(http::handle_heartbeat))
        .with_state(heartbeat_state)
        .route("/v1/jobs", post(http::handle_create_job))
        .with_state(job_state.clone())
        .route("/v1/jobs/{job_id}/stream", get(http::handle_stream_job))
        .with_state(job_state.clone()) // TEAM-186: Pass full state for payload retrieval
}

/// POST /v1/shutdown - Graceful shutdown
///
/// TEAM-153: Created by TEAM-153
/// TEAM-164: Migrated from http.rs to main.rs
use axum::http::StatusCode;
async fn handle_shutdown() -> StatusCode {
    NARRATE.action("shutdown")
        .human("Received shutdown request, exiting gracefully")
        .emit();
    std::process::exit(0);
}
