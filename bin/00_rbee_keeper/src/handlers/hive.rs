//! Hive command handlers (localhost-only)
//!
//! TEAM-322: Removed all SSH/remote functionality (RULE ZERO - delete complexity)
//! TEAM-324: Moved HiveAction enum here to eliminate duplication

use anyhow::Result;
use clap::Subcommand;
use daemon_lifecycle::{
    check_daemon_status, rebuild::rebuild_with_hot_reload, rebuild::RebuildConfig, stop_http_daemon,
    HttpDaemonConfig,
};
use operations_contract::Operation;
use std::path::PathBuf;

use crate::job_client::{submit_and_stream_job, submit_and_stream_job_to_hive};

#[derive(Subcommand)]
pub enum HiveAction {
    /// Start rbee-hive locally
    Start {
        /// HTTP port (default: 7835)
        #[arg(short = 'p', long = "port")]
        port: Option<u16>,
    },
    /// Stop rbee-hive locally
    Stop {
        /// HTTP port (default: 7835)
        #[arg(short = 'p', long = "port")]
        port: Option<u16>,
    },
    /// Get hive details
    Get {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Check hive status
    Status {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Refresh device capabilities for a hive
    RefreshCapabilities {
        /// Hive alias (only "localhost" supported)
        #[arg(short = 'a', long = "host")]
        alias: String,
    },
    /// Run hive-check (narration test through hive SSE)
    Check {
        /// Hive alias (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Install rbee-hive binary locally
    Install {
        /// Hive alias (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Uninstall rbee-hive binary locally
    Uninstall {
        /// Hive alias (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Rebuild rbee-hive binary locally (cargo build)
    Rebuild {
        /// Hive alias (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
}

pub async fn handle_hive(action: HiveAction, queen_url: &str) -> Result<()> {
    match action {
        // TEAM-323: Use daemon-lifecycle directly (no wrapper needed)
        HiveAction::Start { port } => {
            let port = port.unwrap_or(7835);
            let base_url = format!("http://localhost:{}", port);
            let args = vec![
                "--port".to_string(),
                port.to_string(),
                "--queen-url".to_string(),
                queen_url.to_string(),
                "--hive-id".to_string(),
                "localhost".to_string(),
            ];
            // TEAM-327: Binary path auto-resolved from daemon_name inside start_http_daemon
            let config = daemon_lifecycle::HttpDaemonConfig::new("rbee-hive", &base_url)
                .with_args(args);
            // TEAM-327: start_http_daemon now returns PID (discard it - we use health checks for status)
            let _pid = daemon_lifecycle::start_http_daemon(config).await?;
            Ok(())
        }
        // TEAM-322: Use daemon-lifecycle directly
        HiveAction::Stop { port } => {
            let base_url = format!("http://localhost:{}", port.unwrap_or(7835));
            // TEAM-327: Binary path auto-resolved from daemon_name
            let config = HttpDaemonConfig::new("rbee-hive", base_url);
            stop_http_daemon(config).await
        }

        // TEAM-323: Status uses daemon-lifecycle directly (same as queen)
        HiveAction::Status { alias: _ } => {
            let hive_url = "http://localhost:7835";
            check_daemon_status("localhost", &format!("{}/health", hive_url), Some("hive"), None)
                .await?;
            Ok(())
        }

        // Operations that go through queen
        HiveAction::Get { alias } => {
            let operation = Operation::HiveGet { alias };
            submit_and_stream_job(queen_url, operation).await
        }
        HiveAction::RefreshCapabilities { alias } => {
            let operation = Operation::HiveRefreshCapabilities { alias };
            submit_and_stream_job(queen_url, operation).await
        }
        HiveAction::Check { alias } => {
            let hive_url = format!("http://localhost:7835");
            let operation = Operation::HiveCheck { alias };
            submit_and_stream_job_to_hive(&hive_url, operation).await
        }

        // TEAM-323: Hive lifecycle operations (use daemon-lifecycle directly, same as queen)
        HiveAction::Install { alias: _ } => {
            // TEAM-323: Use daemon-lifecycle directly (same pattern as queen)
            daemon_lifecycle::install_to_local_bin("rbee-hive", None, None).await?;
            Ok(())
        }
        HiveAction::Uninstall { alias: _ } => {
            // TEAM-323: Use daemon-lifecycle directly (same pattern as queen)
            let home = std::env::var("HOME")?;
            let config = daemon_lifecycle::UninstallConfig {
                daemon_name: "rbee-hive".to_string(),
                install_path: format!("{}/.local/bin/rbee-hive", home),
                health_url: Some("http://localhost:7835".to_string()),
                health_timeout_secs: Some(2),
                job_id: None,
            };
            daemon_lifecycle::uninstall_daemon(config).await
        }
        HiveAction::Rebuild { alias: _ } => {
            // TEAM-328: Use rebuild_with_hot_reload for automatic state management
            let rebuild_config = RebuildConfig::new("rbee-hive");
            let daemon_config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835".to_string())
                .with_args(vec!["--port".to_string(), "7835".to_string()]);
            
            rebuild_with_hot_reload(rebuild_config, daemon_config).await?;
            Ok(())
        }
    }
}
