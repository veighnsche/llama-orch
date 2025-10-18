// queen-rbee - Orchestrator Daemon
// Milestone: M1+
// Purpose: Job scheduling, admission control, worker registry, SSE relay
//
// TEAM-030: Added shutdown handler scaffold for cascading shutdown
// TEAM-043: Implemented dual registry system (beehive + worker)

mod beehive_registry;
mod http;
mod ssh;
mod worker_registry;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(name = "queen-rbee")]
#[command(about = "rbee Orchestrator Daemon - Job scheduling and worker management")]
#[command(version)]
struct Args {
    /// HTTP server port
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,

    /// Database path (SQLite) for beehive registry
    #[arg(short, long)]
    database: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt().with_target(false).with_thread_ids(true).with_level(true).init();

    let args = Args::parse();

    info!("🐝 queen-rbee Orchestrator Daemon starting...");
    info!("Port: {}", args.port);

    // TEAM-043: Initialize dual registry system
    let db_path = args.database.map(std::path::PathBuf::from);
    let beehive_registry = beehive_registry::BeehiveRegistry::new(db_path).await?;
    info!("✅ Beehive registry initialized (SQLite)");

    let worker_registry = worker_registry::WorkerRegistry::new();
    info!("✅ Worker registry initialized (in-memory)");

    // TEAM-102: Load API token for authentication
    // TODO: Replace with secrets-management file-based loading
    let expected_token = std::env::var("LLORCH_API_TOKEN")
        .unwrap_or_else(|_| {
            info!("⚠️  LLORCH_API_TOKEN not set - using dev mode (no auth)");
            String::new()
        });
    
    if !expected_token.is_empty() {
        info!("✅ API token loaded (authentication enabled)");
    }

    // Create router with registries
    // TEAM-052: Updated to use refactored http module
    // TEAM-102: Added expected_token for authentication
    let app = http::create_router(
        Arc::new(beehive_registry), 
        Arc::new(worker_registry),
        expected_token,
    );

    // Start HTTP server
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("🚀 HTTP server listening on {}", addr);

    // TEAM-030: Setup shutdown handler
    let server = axum::serve(listener, app);

    tokio::select! {
        result = server => {
            if let Err(e) = result {
                error!("Server error: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Shutdown signal received");
        }
    }

    info!("👋 queen-rbee shutting down");
    Ok(())
}

// TEAM-031: Unit tests for queen-rbee
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
    }

    #[test]
    fn test_args_defaults() {
        let args = Args::parse_from(&["queen-rbee"]);
        assert_eq!(args.port, 8080);
        assert_eq!(args.database, None);
        assert_eq!(args.config, None);
    }

    #[test]
    fn test_args_custom_port() {
        let args = Args::parse_from(&["queen-rbee", "--port", "9090"]);
    }

    #[test]
    fn test_args_custom_database() {
        let args = Args::parse_from(&["queen-rbee", "--database", "custom.db"]);
        assert_eq!(args.database, Some("custom.db".to_string()));
    }

    #[test]
    fn test_args_with_config() {
        let args = Args::parse_from(&["queen-rbee", "--config", "/path/to/config.toml"]);
        assert_eq!(args.config, Some("/path/to/config.toml".to_string()));
    }

    #[test]
    fn test_args_all_options() {
        let args = Args::parse_from(&[
            "queen-rbee",
            "--port",
            "9090",
            "--database",
            "custom.db",
            "--config",
            "/path/to/config.toml",
        ]);
        assert_eq!(args.port, 9090);
        assert_eq!(args.database, Some("custom.db".to_string()));
        assert_eq!(args.config, Some("/path/to/config.toml".to_string()));
    }

    #[test]
    fn test_args_short_flags() {
        let args = Args::parse_from(&[
            "queen-rbee",
            "-p",
            "9090",
            "-d",
            "custom.db",
            "-c",
            "/path/to/config.toml",
        ]);
        assert_eq!(args.port, 9090);
        assert_eq!(args.database, Some("custom.db".to_string()));
        assert_eq!(args.config, Some("/path/to/config.toml".to_string()));
    }
}
