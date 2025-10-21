// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-189: Implemented basic HTTP server with /health endpoint
// TEAM-190: Added hive heartbeat task to send status to queen every 5 seconds
// TEAM-202: Replaced println!() with narration for job-scoped SSE visibility
// Purpose: rbee-hive binary entry point (DAEMON ONLY - NO CLI!)
// Status: Basic HTTP daemon with heartbeat and narration - ready for worker management implementation

//! rbee-hive
//!
//! Daemon for managing LLM worker instances on a single machine

mod narration;
use narration::{NARRATE, ACTION_STARTUP, ACTION_HEARTBEAT, ACTION_LISTEN, ACTION_READY};

use axum::{routing::get, Router};
use clap::Parser;
use rbee_heartbeat::{
    start_hive_heartbeat_task, HiveHeartbeatConfig, WorkerState, WorkerStateProvider,
};
use std::net::SocketAddr;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "rbee-hive")]
#[command(about = "rbee Hive Daemon - Worker and model management")]
#[command(version)]
struct Args {
    /// HTTP server port
    #[arg(short, long, default_value = "8600")]
    port: u16,

    /// TEAM-190: Hive ID (defaults to "localhost")
    #[arg(long, default_value = "localhost")]
    hive_id: String,

    /// TEAM-190: Queen URL for heartbeat reporting
    #[arg(long, default_value = "http://localhost:8500")]
    queen_url: String,
}

/// TEAM-190: Worker state provider for heartbeat aggregation
/// Currently returns empty list - will be populated when worker registry is implemented
struct HiveWorkerProvider;

impl WorkerStateProvider for HiveWorkerProvider {
    fn get_worker_states(&self) -> Vec<WorkerState> {
        // TEAM-190: Return empty list for now
        // TODO: Query worker registry when implemented
        vec![]
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // TEAM-202: Use narration instead of println!
    // This automatically goes through job-scoped SSE (if in job context)
    // and uses centralized formatting (TEAM-201)
    NARRATE
        .action(ACTION_STARTUP)
        .context(&args.port.to_string())
        .context(&args.hive_id)
        .context(&args.queen_url)
        .human("🐝 Starting on port {}, hive_id: {}, queen: {}")
        .emit();

    // TEAM-190: Start heartbeat task (5 second interval)
    let heartbeat_config = HiveHeartbeatConfig::new(
        args.hive_id.clone(),
        args.queen_url.clone(),
        "".to_string(), // Empty auth token for now
    )
    .with_interval(5); // 5 seconds as requested

    let worker_provider = Arc::new(HiveWorkerProvider);
    let _heartbeat_handle = start_hive_heartbeat_task(heartbeat_config, worker_provider);

    // TEAM-202: Narrate heartbeat startup
    NARRATE
        .action(ACTION_HEARTBEAT)
        .context("5s")
        .human("💓 Heartbeat task started ({} interval)")
        .emit();

    // Create basic router with health endpoint
    let app = Router::new().route("/health", get(health_check));

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    // TEAM-202: Narrate listen address
    NARRATE
        .action(ACTION_LISTEN)
        .context(&format!("http://{}", addr))
        .human("✅ Listening on {}")
        .emit();

    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    // TEAM-202: Narrate ready state
    NARRATE
        .action(ACTION_READY)
        .human("✅ Hive ready")
        .emit();
    
    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check endpoint
/// TEAM-189: Returns "ok" - used by hive start/stop/status operations
async fn health_check() -> &'static str {
    "ok"
}
