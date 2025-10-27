//! Hive command handlers (localhost-only)
//!
//! TEAM-322: Removed all SSH/remote functionality (RULE ZERO - delete complexity)
//! TEAM-324: Moved HiveAction enum here to eliminate duplication

use anyhow::Result;
use clap::Subcommand;
use daemon_lifecycle::{
    check_daemon_health, rebuild_daemon, stop_daemon, HttpDaemonConfig, RebuildConfig,
};
use observability_narration_core::n;
use operations_contract::Operation;

use crate::job_client::submit_and_stream_job;

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
            // TEAM-329: start_daemon now returns PID (discard it - we use health checks for status)
            let _pid = daemon_lifecycle::start_daemon(config).await?;
            Ok(())
        }
        // TEAM-322: Use daemon-lifecycle directly
        HiveAction::Stop { port } => {
            let base_url = format!("http://localhost:{}", port.unwrap_or(7835));
            // TEAM-329: Binary path auto-resolved from daemon_name
            let config = HttpDaemonConfig::new("rbee-hive", base_url);
            stop_daemon(config).await
        }

        // TEAM-329: Use check_daemon_health() directly
        HiveAction::Status { alias: _ } => {
            let health_url = "http://localhost:7835";
            let is_running = check_daemon_health(health_url, None, None).await;
            
            if is_running {
                n!("hive_status", "✅ hive 'localhost' is running on {}", health_url);
            } else {
                n!("hive_status", "❌ hive 'localhost' is not running on {}", health_url);
            }
            Ok(())
        }

        // TEAM-329: Removed Get and Check handlers (user request)
        HiveAction::RefreshCapabilities { alias } => {
            let operation = Operation::HiveRefreshCapabilities { alias };
            submit_and_stream_job(queen_url, operation).await
        }

        // TEAM-323: Hive lifecycle operations (use daemon-lifecycle directly, same as queen)
        HiveAction::Install { alias: _ } => {
            // TEAM-329: Use install_daemon (renamed from install_to_local_bin)
            daemon_lifecycle::install_daemon("rbee-hive", None, None).await?;
            Ok(())
        }
        HiveAction::Uninstall { alias: _ } => {
            // TEAM-329: Use UninstallConfig builder pattern
            let home = std::env::var("HOME")?;
            let config = daemon_lifecycle::UninstallConfig::new(
                "rbee-hive",
                format!("{}/.local/bin/rbee-hive", home),
            )
            .with_health_url("http://localhost:7835")
            .with_health_timeout_secs(2);
            
            daemon_lifecycle::uninstall_daemon(config).await
        }
        HiveAction::Rebuild { alias: _ } => {
            // TEAM-329: Use rebuild_daemon (renamed from rebuild_with_hot_reload)
            let rebuild_config = RebuildConfig::new("rbee-hive");
            let daemon_config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835")
                .with_args(vec!["--port".to_string(), "7835".to_string()]);
            
            rebuild_daemon(rebuild_config, daemon_config).await?;
            Ok(())
        }
    }
}
