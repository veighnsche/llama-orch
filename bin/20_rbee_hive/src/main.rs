// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-189: Implemented basic HTTP server with /health endpoint
// Purpose: rbee-hive binary entry point (DAEMON ONLY - NO CLI!)
// Status: Basic HTTP daemon - ready for worker management implementation

//! rbee-hive
//!
//! Daemon for managing LLM worker instances on a single machine

use axum::{routing::get, Router};
use clap::Parser;
use std::net::SocketAddr;

#[derive(Parser, Debug)]
#[command(name = "rbee-hive")]
#[command(about = "rbee Hive Daemon - Worker and model management")]
#[command(version)]
struct Args {
    /// HTTP server port
    #[arg(short, long, default_value = "8600")]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("🐝 rbee-hive starting on port {}", args.port);

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
