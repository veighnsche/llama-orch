//! queen-rbee - Orchestrator Daemon
//!
//! TEAM-151: Migrated from old.queen-rbee
//! TEAM-151: Cleaned up main.rs (97 → 77 lines)
//! TEAM-151: Wired up src/http/ folder with health endpoint
//! TEAM-151: Health endpoint active on GET /health
//!
//! # Happy Flow
//! "The queen bee wakes up and immediately starts the http server."
//! Port 8500 (default) - rbee-keeper checks GET /health to see if queen is running

mod http;

use anyhow::Result;
use axum::routing::get;
use clap::Parser;
use std::net::SocketAddr;
use tracing::{error, info};

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
    init_tracing();
    
    let args = Args::parse();

    info!("🐝 queen-rbee starting on port {}", args.port);

    // TODO: Initialize registries when migrated
    // - beehive_registry (SQLite catalog + RAM registry)
    // - worker_registry (RAM)
    // - Load config from args.config

    let app = create_router();
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));

    info!("✅ Listening on http://{}", addr);
    info!("🚀 Ready to accept connections");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await.map_err(|e| {
        error!("Server error: {}", e);
        anyhow::anyhow!("Server failed: {}", e)
    })
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(false)
        .with_level(true)
        .init();
}

/// Create HTTP router
///
/// Currently only health endpoint is active.
/// TODO: Uncomment http::routes::create_router() when registries are migrated
fn create_router() -> axum::Router {
    axum::Router::new()
        .route("/health", get(http::health::handle_health))
}
