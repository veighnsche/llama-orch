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
    let beehive_registry = Arc::new(beehive_registry::BeehiveRegistry::new(db_path).await?);
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
        Arc::clone(&beehive_registry), 
        Arc::new(worker_registry),
        expected_token,
    );

    // Start HTTP server
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("🚀 HTTP server listening on {}", addr);

    // TEAM-030: Setup shutdown handler
    // TEAM-105: Enhanced with cascading shutdown to all hives
    let server = axum::serve(listener, app);
    let beehive_registry_shutdown = Arc::clone(&beehive_registry);

    tokio::select! {
        result = server => {
            if let Err(e) = result {
                error!("Server error: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("TEAM-105: Shutdown signal received - initiating cascading shutdown");
            shutdown_all_hives(beehive_registry_shutdown).await;
        }
    }

    info!("👋 queen-rbee shutting down");
    Ok(())
}

/// TEAM-105: Shutdown all registered hives via SSH
/// Sends SIGTERM to rbee-hive daemon processes on all registered nodes
/// Implements 30s timeout and audit logging
async fn shutdown_all_hives(beehive_registry: Arc<beehive_registry::BeehiveRegistry>) {
    use std::time::Instant;
    
    let shutdown_start = Instant::now();
    info!("TEAM-105: Starting cascading shutdown to all hives (30s timeout)");
    
    let hives = match beehive_registry.list_nodes().await {
        Ok(nodes) => nodes,
        Err(e) => {
            error!("TEAM-105: Failed to list hives: {}", e);
            return;
        }
    };
    
    let total_hives = hives.len();
    info!("TEAM-105: Found {} hives to shutdown", total_hives);
    
    if total_hives == 0 {
        info!("No hives registered - shutdown complete");
        return;
    }
    
    // TEAM-105: Parallel shutdown of all hives with timeout
    let mut shutdown_tasks = Vec::new();
    
    for hive in hives {
        let task = tokio::spawn(async move {
            info!("TEAM-105: Sending shutdown to hive: {}", hive.node_name);
            
            // Find rbee-hive daemon PID and send SIGTERM
            let find_pid_cmd = "pgrep -f 'rbee-hive daemon'";
            
            match ssh::execute_remote_command(
                &hive.ssh_host,
                hive.ssh_port,
                &hive.ssh_user,
                hive.ssh_key_path.as_deref(),
                find_pid_cmd,
            ).await {
                Ok((success, stdout, stderr)) => {
                    if success && !stdout.trim().is_empty() {
                        let pid = stdout.trim();
                        info!("TEAM-105: Found rbee-hive daemon PID {} on {}", pid, hive.node_name);
                        
                        // Send SIGTERM to the daemon
                        let kill_cmd = format!("kill -TERM {}", pid);
                        match ssh::execute_remote_command(
                            &hive.ssh_host,
                            hive.ssh_port,
                            &hive.ssh_user,
                            hive.ssh_key_path.as_deref(),
                            &kill_cmd,
                        ).await {
                            Ok((kill_success, _, kill_stderr)) => {
                                if kill_success {
                                    info!("TEAM-105: Successfully sent SIGTERM to {} (PID: {})", hive.node_name, pid);
                                    true
                                } else {
                                    error!("TEAM-105: Failed to send SIGTERM to {}: {}", hive.node_name, kill_stderr);
                                    false
                                }
                            }
                            Err(e) => {
                                error!("TEAM-105: SSH kill command failed for {}: {}", hive.node_name, e);
                                false
                            }
                        }
                    } else {
                        info!("TEAM-105: No rbee-hive daemon running on {} (stderr: {})", hive.node_name, stderr);
                        true // Not an error - daemon not running
                    }
                }
                Err(e) => {
                    error!("TEAM-105: SSH connection failed for {}: {}", hive.node_name, e);
                    false
                }
            }
        });
        
        shutdown_tasks.push(task);
    }
    
    // TEAM-105: Wait for all shutdown tasks with 30s timeout
    let mut completed = 0;
    let mut success_count = 0;
    let mut failed_count = 0;
    let mut timeout_count = 0;
    
    for task in shutdown_tasks {
        let elapsed = shutdown_start.elapsed();
        let remaining = std::time::Duration::from_secs(30).saturating_sub(elapsed);
        
        if remaining.is_zero() {
            // TEAM-105: Timeout exceeded - abort remaining tasks
            error!(
                "TEAM-105: Hive shutdown timeout (30s) exceeded - aborting remaining hives"
            );
            timeout_count += 1;
            task.abort();
            continue;
        }
        
        match tokio::time::timeout(remaining, task).await {
            Ok(Ok(success)) => {
                completed += 1;
                if success {
                    success_count += 1;
                } else {
                    failed_count += 1;
                }
                info!(
                    "TEAM-105: Hive shutdown progress: {}/{} completed (success: {}, failed: {}, timeout: {})",
                    completed, total_hives, success_count, failed_count, timeout_count
                );
            }
            Ok(Err(e)) => {
                completed += 1;
                failed_count += 1;
                error!("TEAM-105: Hive shutdown task failed: {}", e);
            }
            Err(_) => {
                // Task timed out
                completed += 1;
                timeout_count += 1;
                error!("TEAM-105: Hive shutdown task timed out");
            }
        }
    }
    
    let total_duration = shutdown_start.elapsed();
    
    // TEAM-105: Audit logging for cascading shutdown completion
    info!(
        "TEAM-105: SHUTDOWN AUDIT - Total Hives: {}, Success: {}, Failed: {}, Timeout: {}, Duration: {:.2}s",
        total_hives, success_count, failed_count, timeout_count, total_duration.as_secs_f64()
    );
    
    if timeout_count > 0 {
        error!(
            "TEAM-105: {} hives exceeded shutdown timeout",
            timeout_count
        );
    }
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
