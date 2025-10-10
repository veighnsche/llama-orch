// queen-rbee - Orchestrator Daemon
// Milestone: M1+
// Purpose: Job scheduling, admission control, worker registry, SSE relay
//
// TEAM-030: Added shutdown handler scaffold for cascading shutdown

use anyhow::Result;
use clap::Parser;
use tracing::{info, error};

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

    /// Database path (SQLite)
    #[arg(short, long, default_value = "rbee-orchestrator.db")]
    database: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_target(false)
        .with_thread_ids(true)
        .with_level(true)
        .init();

    let args = Args::parse();

    info!("🐝 rbee Orchestrator Daemon starting...");
    info!("Port: {}", args.port);
    info!("Database: {}", args.database);

    // TEAM-030: Setup shutdown handler for future implementation
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Shutdown signal received");
        // TODO: M1 implementation - cascade shutdown to all hives
        // - Send shutdown signal to all connected hives
        // - Wait for acknowledgment
        // - Clean up resources
        std::process::exit(0);
    });

    // TODO: M1 implementation
    // - Remove SQLite database (use in-memory registry)
    // - Create in-memory worker registry
    // - Create job queue
    // - Start HTTP server
    // - Track hive connections for cascading shutdown

    error!("❌ Orchestrator daemon not yet implemented (M1 milestone)");
    error!("This is a scaffold for the rbee rebrand");
    
    Ok(())
}

// TEAM-031: Unit tests for queen-rbee
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        // Test default values
        let args = Args::parse_from(&["queen-rbee"]);
        assert_eq!(args.port, 8080);
        assert_eq!(args.database, "rbee-orchestrator.db");
        assert!(args.config.is_none());
    }

    #[test]
    fn test_args_custom_port() {
        let args = Args::parse_from(&["queen-rbee", "--port", "9090"]);
        assert_eq!(args.port, 9090);
    }

    #[test]
    fn test_args_custom_database() {
        let args = Args::parse_from(&["queen-rbee", "--database", "custom.db"]);
        assert_eq!(args.database, "custom.db");
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
            "--port", "9090",
            "--database", "custom.db",
            "--config", "/path/to/config.toml"
        ]);
        assert_eq!(args.port, 9090);
        assert_eq!(args.database, "custom.db");
        assert_eq!(args.config, Some("/path/to/config.toml".to_string()));
    }

    #[test]
    fn test_args_short_flags() {
        let args = Args::parse_from(&[
            "queen-rbee",
            "-p", "9090",
            "-d", "custom.db",
            "-c", "/path/to/config.toml"
        ]);
        assert_eq!(args.port, 9090);
        assert_eq!(args.database, "custom.db");
        assert_eq!(args.config, Some("/path/to/config.toml".to_string()));
    }
}
