// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-189: Implemented basic HTTP server with /health endpoint
// TEAM-190: Added hive heartbeat task to send status to queen every 5 seconds
// Purpose: rbee-hive binary entry point (DAEMON ONLY - NO CLI!)
// Status: Basic HTTP daemon with heartbeat - ready for worker management implementation

//! rbee-hive
//!
//! Daemon for managing LLM worker instances on a single machine

use axum::{routing::get, Router};
use clap::Parser;
use rbee_heartbeat::{start_hive_heartbeat_task, HiveHeartbeatConfig, WorkerStateProvider, WorkerState};
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

    println!("🐝 rbee-hive starting on port {}", args.port);
    println!("📡 Hive ID: {}", args.hive_id);
    println!("👑 Queen URL: {}", args.queen_url);

    // TEAM-190: Start heartbeat task (5 second interval)
    let heartbeat_config = HiveHeartbeatConfig::new(
        args.hive_id.clone(),
        args.queen_url.clone(),
        "".to_string(), // Empty auth token for now
    )
    .with_interval(5); // 5 seconds as requested

    let worker_provider = Arc::new(HiveWorkerProvider);
    let _heartbeat_handle = start_hive_heartbeat_task(heartbeat_config, worker_provider);

    println!("💓 Heartbeat task started (5s interval)");

    // Create basic router with health endpoint
    let app = Router::new().route("/health", get(health_check));

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    
    println!("✅ rbee-hive listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check endpoint
/// TEAM-189: Returns "ok" - used by hive start/stop/status operations
async fn health_check() -> &'static str {
    "ok"
}
