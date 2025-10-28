//! Queen-rbee command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-276: Refactored to delegate to queen-lifecycle crate
//! TEAM-322: Consolidated status/info - both use check_queen_status with verbose flag
//! TEAM-324: Moved QueenAction enum here to eliminate duplication
//! TEAM-332: Use ssh_resolver middleware (queen is always localhost)
//!
//! This module is now a thin wrapper that delegates all queen lifecycle
//! operations to the queen-lifecycle crate. All business logic lives there.

use anyhow::Result;
use clap::Subcommand;
use daemon_lifecycle::{
    check_daemon_health, install_daemon, rebuild_daemon, start_daemon, stop_daemon,
    uninstall_daemon, HttpDaemonConfig, InstallConfig, RebuildConfig, SshConfig,
    StartConfig, StopConfig, UninstallConfig,
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
            // TEAM-333: Queen is always localhost - use SshConfig::localhost() directly
            let base_url = format!("http://localhost:{}", port);
            let args = vec!["--port".to_string(), port.to_string()];
            let daemon_config = HttpDaemonConfig::new("queen-rbee", &base_url).with_args(args);
            let config = StartConfig {
                ssh_config: SshConfig::localhost(),
                daemon_config,
                job_id: None,
            };
            let _pid = start_daemon(config).await?;
            Ok(())
        }
        QueenAction::Stop => {
            // TEAM-333: Queen is always localhost - use SshConfig::localhost() directly
            let shutdown_url = format!("{}/v1/shutdown", queen_url);
            let health_url = format!("{}/health", queen_url);
            let config = StopConfig {
                daemon_name: "queen-rbee".to_string(),
                shutdown_url,
                health_url,
                ssh_config: SshConfig::localhost(),
                job_id: None,
            };
            stop_daemon(config).await
        }
        // TEAM-338: RULE ZERO - Updated to new check_daemon_health signature
        QueenAction::Status => {
            let health_url = format!("{}/health", queen_url);
            let ssh_config = SshConfig::localhost(); // Queen is always localhost
            let status = check_daemon_health(&health_url, "queen-rbee", &ssh_config).await;
            
            if status.is_running {
                n!("queen_status", "✅ queen 'localhost' is running on {}", queen_url);
            } else {
                n!("queen_status", "❌ queen 'localhost' is not running on {}", queen_url);
            }
            Ok(())
        }
        QueenAction::Rebuild => {
            // TEAM-333: Queen is always localhost - use SshConfig::localhost() directly
            let daemon_config = HttpDaemonConfig::new("queen-rbee", queen_url.to_string())
                .with_args(vec!["--port".to_string(), port.to_string()]);
            let config = RebuildConfig {
                daemon_name: "queen-rbee".to_string(),
                ssh_config: SshConfig::localhost(),
                daemon_config,
                job_id: None,
            };
            rebuild_daemon(config).await
        }
        QueenAction::Install { binary } => {
            // TEAM-333: Queen is always localhost - use SshConfig::localhost() directly
            let config = InstallConfig {
                daemon_name: "queen-rbee".to_string(),
                ssh_config: SshConfig::localhost(),
                local_binary_path: binary.map(std::path::PathBuf::from),
                job_id: None,
            };
            install_daemon(config).await
        }
        QueenAction::Uninstall => {
            // TEAM-333: Queen is always localhost - use SshConfig::localhost() directly
            let config = UninstallConfig {
                daemon_name: "queen-rbee".to_string(),
                ssh_config: SshConfig::localhost(),
                health_url: Some(queen_url.to_string()),
                health_timeout_secs: Some(2),
                job_id: None,
            };
            uninstall_daemon(config).await
        }
    }
}
