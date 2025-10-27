//! Queen-rbee command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-276: Refactored to delegate to queen-lifecycle crate
//! TEAM-322: Consolidated status/info - both use check_queen_status with verbose flag
//!
//! This module is now a thin wrapper that delegates all queen lifecycle
//! operations to the queen-lifecycle crate. All business logic lives there.

use anyhow::Result;
use daemon_lifecycle::{
    HttpDaemonConfig, stop_http_daemon, rebuild::build_daemon_local, rebuild::RebuildConfig,
    check_daemon_status, install_to_local_bin,
};
use std::path::PathBuf;

use crate::cli::QueenAction;

/// Handle queen-rbee lifecycle commands
///
/// Delegates to queen-lifecycle crate for all operations.
pub async fn handle_queen(action: QueenAction, queen_url: &str) -> Result<()> {
    // TEAM-322: Extract port from queen_url (always localhost)
    let port: u16 = queen_url
        .split(':')
        .last()
        .and_then(|p| p.parse().ok())
        .unwrap_or(7833);
    
    match action {
        QueenAction::Start => {
            let binary = daemon_lifecycle::DaemonManager::find_binary("queen-rbee")?;
            let base_url = format!("http://localhost:{}", port);
            let args = vec!["--port".to_string(), port.to_string()];
            let config = daemon_lifecycle::HttpDaemonConfig::new("queen-rbee", binary, &base_url)
                .with_args(args);
            daemon_lifecycle::start_http_daemon(config).await
        }
        // TEAM-322: Use daemon-lifecycle directly
        QueenAction::Stop => {
            let config = HttpDaemonConfig::new(
                "queen-rbee",
                PathBuf::from("~/.local/bin/queen-rbee"),
                queen_url.to_string(),
            );
            stop_http_daemon(config).await
        }
        // TEAM-323: Use daemon-lifecycle directly (same as hive)
        QueenAction::Status => {
            check_daemon_status("localhost", &format!("{}/health", queen_url), Some("queen"), None).await?;
            Ok(())
        }
        // TEAM-322: Use daemon-lifecycle directly (removed local-hive feature complexity)
        QueenAction::Rebuild => {
            let config = RebuildConfig::new("queen-rbee");
            build_daemon_local(config).await?;
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
