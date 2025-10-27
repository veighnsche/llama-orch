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
    check_daemon_health, rebuild_daemon, stop_daemon, HttpDaemonConfig, RebuildConfig,
};
use observability_narration_core::n;

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
            // TEAM-329: start_daemon now returns PID (discard it - we use health checks for status)
            let _pid = daemon_lifecycle::start_daemon(config).await?;
            Ok(())
        }
        // TEAM-329: Use daemon-lifecycle directly
        QueenAction::Stop => {
            // TEAM-329: Binary path auto-resolved from daemon_name
            let config = HttpDaemonConfig::new("queen-rbee", queen_url.to_string());
            stop_daemon(config).await
        }
        // TEAM-329: Use check_daemon_health() directly
        QueenAction::Status => {
            let is_running = check_daemon_health(queen_url, None, None).await;
            
            if is_running {
                n!("queen_status", "✅ queen 'localhost' is running on {}", queen_url);
            } else {
                n!("queen_status", "❌ queen 'localhost' is not running on {}", queen_url);
            }
            Ok(())
        }
        // TEAM-329: Use rebuild_daemon (renamed from rebuild_with_hot_reload)
        QueenAction::Rebuild => {
            let rebuild_config = RebuildConfig::new("queen-rbee");
            let daemon_config = HttpDaemonConfig::new("queen-rbee", queen_url.to_string())
                .with_args(vec!["--port".to_string(), port.to_string()]);
            
            rebuild_daemon(rebuild_config, daemon_config).await?;
            Ok(())
        }
        QueenAction::Install { binary } => {
            // TEAM-329: Use install_daemon (renamed from install_to_local_bin)
            daemon_lifecycle::install_daemon("queen-rbee", binary, None).await?;
            Ok(())
        }
        QueenAction::Uninstall => {
            // TEAM-329: Use UninstallConfig builder pattern
            let home = std::env::var("HOME")?;
            let config = daemon_lifecycle::UninstallConfig::new(
                "queen-rbee",
                format!("{}/.local/bin/queen-rbee", home),
            )
            .with_health_url(queen_url)
            .with_health_timeout_secs(2);
            
            daemon_lifecycle::uninstall_daemon(config).await
        }
    }
}
