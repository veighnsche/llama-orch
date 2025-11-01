//! Queen-rbee command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-276: Refactored to delegate to queen-lifecycle crate
//! TEAM-322: Consolidated status/info - both use check_queen_status with verbose flag
//! TEAM-324: Moved QueenAction enum here to eliminate duplication
//! TEAM-332: Use ssh_resolver middleware (queen is always localhost)
//! TEAM-365: Updated to use lifecycle-local (no SshConfig, localhost only)
//!
//! This module is now a thin wrapper that delegates all queen lifecycle
//! operations to the lifecycle-local crate. All business logic lives there.

use anyhow::Result;
use clap::Subcommand;
use lifecycle_local::{
    check_daemon_health, install_daemon, rebuild_daemon, start_daemon, stop_daemon,
    uninstall_daemon, HttpDaemonConfig, InstallConfig, RebuildConfig, StartConfig,
    StopConfig, UninstallConfig,
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
    let port: u16 = queen_url.split(':').next_back().and_then(|p| p.parse().ok()).unwrap_or(7833);

    match action {
        QueenAction::Start => {
            // TEAM-365: Queen is always localhost - lifecycle-local has no SSH
            // TEAM-341: BUG FIX - Must pass health_url with /health path, not base_url
            let base_url = format!("http://localhost:{}", port);
            let health_url = format!("{}/health", base_url);
            let args = vec!["--port".to_string(), port.to_string()];
            let daemon_config = HttpDaemonConfig::new("queen-rbee", &health_url).with_args(args);
            let config = StartConfig { daemon_config, job_id: None };
            let _pid = start_daemon(config).await?;
            Ok(())
        }
        QueenAction::Stop => {
            // TEAM-365: Queen is always localhost - lifecycle-local has no SSH
            let shutdown_url = format!("{}/v1/shutdown", queen_url);
            let health_url = format!("{}/health", queen_url);
            let config = StopConfig {
                daemon_name: "queen-rbee".to_string(),
                shutdown_url,
                health_url,
                job_id: None,
            };
            stop_daemon(config).await
        }
        // TEAM-338: RULE ZERO - Updated to new check_daemon_health signature
        // TEAM-365: lifecycle-local has no SSH (localhost only)
        QueenAction::Status => {
            let health_url = format!("{}/health", queen_url);
            let status = check_daemon_health(&health_url, "queen-rbee").await;

            if status.is_running {
                n!("queen_status", "✅ queen 'localhost' is running on {}", queen_url);
            } else {
                n!("queen_status", "❌ queen 'localhost' is not running on {}", queen_url);
            }
            Ok(())
        }
        QueenAction::Rebuild => {
            // TEAM-365: Queen is always localhost - lifecycle-local has no SSH
            let daemon_config = HttpDaemonConfig::new("queen-rbee", queen_url.to_string())
                .with_args(vec!["--port".to_string(), port.to_string()]);
            let config = RebuildConfig {
                daemon_name: "queen-rbee".to_string(),
                daemon_config,
                job_id: None,
            };
            rebuild_daemon(config).await
        }
        QueenAction::Install { binary } => {
            // TEAM-365: Queen is always localhost - lifecycle-local has no SSH
            let config = InstallConfig {
                daemon_name: "queen-rbee".to_string(),
                local_binary_path: binary.map(std::path::PathBuf::from),
                job_id: None,
                force_reinstall: false, // TEAM-373: Normal install should check if already exists
            };
            install_daemon(config).await
        }
        QueenAction::Uninstall => {
            // TEAM-365: Queen is always localhost - lifecycle-local has no SSH
            let config = UninstallConfig {
                daemon_name: "queen-rbee".to_string(),
                health_url: Some(queen_url.to_string()),
                health_timeout_secs: Some(2),
                job_id: None,
            };
            uninstall_daemon(config).await
        }
    }
}
