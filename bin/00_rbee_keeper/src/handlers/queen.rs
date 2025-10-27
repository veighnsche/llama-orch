//! Queen-rbee command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-276: Refactored to delegate to queen-lifecycle crate
//! TEAM-322: Consolidated status/info - both use check_queen_status with verbose flag
//! TEAM-324: Moved QueenAction enum here to eliminate duplication
//!
//! This module is now a thin wrapper that delegates all queen lifecycle
//! operations to the queen-lifecycle crate. All business logic lives there.

use anyhow::Result;
use clap::Subcommand;
use daemon_lifecycle::{
    HttpDaemonConfig, stop_http_daemon, rebuild::rebuild_with_hot_reload, rebuild::RebuildConfig,
    is_daemon_healthy,
};
use observability_narration_core::n;
use std::path::PathBuf;

#[derive(Subcommand)]
pub enum QueenAction {
    /// Start queen-rbee daemon
    Start,
    /// Stop queen-rbee daemon
    Stop,
    /// Check queen-rbee daemon status
    Status,
    /// Rebuild queen from source
    Rebuild,
    /// Install queen binary
    /// TEAM-262: Similar to hive install
    Install {
        /// Binary path (optional, auto-detect from target/)
        #[arg(short, long)]
        binary: Option<String>,
    },
    /// Uninstall queen binary
    /// TEAM-262: Similar to hive uninstall
    Uninstall,
}

/// Handle queen-rbee lifecycle commands
///
/// Delegates to queen-lifecycle crate for all operations.
pub async fn handle_queen(action: QueenAction, queen_url: &str) -> Result<()> {
    // TEAM-322: Extract port from queen_url (always localhost)
    // TEAM-327: Use next_back() instead of last() for efficiency (clippy::double_ended_iterator_last)
    let port: u16 = queen_url
        .split(':')
        .next_back()
        .and_then(|p| p.parse().ok())
        .unwrap_or(7833);
    
    match action {
        QueenAction::Start => {
            let base_url = format!("http://localhost:{}", port);
            let args = vec!["--port".to_string(), port.to_string()];
            // TEAM-327: Binary path auto-resolved from daemon_name inside start_http_daemon
            let config = daemon_lifecycle::HttpDaemonConfig::new("queen-rbee", &base_url)
                .with_args(args);
            // TEAM-327: start_http_daemon now returns PID (discard it - we use health checks for status)
            let _pid = daemon_lifecycle::start_http_daemon(config).await?;
            Ok(())
        }
        // TEAM-322: Use daemon-lifecycle directly
        QueenAction::Stop => {
            // TEAM-327: Binary path auto-resolved from daemon_name
            let config = HttpDaemonConfig::new("queen-rbee", queen_url.to_string());
            stop_http_daemon(config).await
        }
        // TEAM-328: Use is_daemon_healthy() directly
        QueenAction::Status => {
            let health_url = format!("{}/health", queen_url);
            let is_running = is_daemon_healthy(&health_url, None, None).await;
            
            if is_running {
                n!("queen_status", "✅ queen 'localhost' is running on {}", health_url);
            } else {
                n!("queen_status", "❌ queen 'localhost' is not running on {}", health_url);
            }
            Ok(())
        }
        // TEAM-328: Use rebuild_with_hot_reload for automatic state management
        QueenAction::Rebuild => {
            let rebuild_config = RebuildConfig::new("queen-rbee");
            let daemon_config = HttpDaemonConfig::new("queen-rbee", queen_url.to_string())
                .with_args(vec!["--port".to_string(), port.to_string()]);
            
            rebuild_with_hot_reload(rebuild_config, daemon_config).await?;
            Ok(())
        }
        QueenAction::Install { binary } => {
            daemon_lifecycle::install_to_local_bin("queen-rbee", binary, None).await?;
            Ok(())
        }
        QueenAction::Uninstall => {
            let home = std::env::var("HOME")?;
            let config = daemon_lifecycle::UninstallConfig {
                daemon_name: "queen-rbee".to_string(),
                install_path: format!("{}/.local/bin/queen-rbee", home),
                health_url: Some(queen_url.to_string()),
                health_timeout_secs: Some(2),
                job_id: None,
            };
            daemon_lifecycle::uninstall_daemon(config).await
        }
    }
}
