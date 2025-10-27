//! Hive command handlers (localhost-only)
//!
//! TEAM-322: Removed all SSH/remote functionality (RULE ZERO - delete complexity)
//! TEAM-324: Moved HiveAction enum here to eliminate duplication

use anyhow::Result;
use clap::Subcommand;
use daemon_lifecycle::{
    check_daemon_health, install_daemon, rebuild_daemon, start_daemon, stop_daemon,
    uninstall_daemon, HttpDaemonConfig, InstallConfig, RebuildConfig, SshConfig,
    StartConfig, StopConfig, UninstallConfig,
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
            // TEAM-330: Use SSH to localhost (consistent with remote operations)
            let ssh = SshConfig::new("localhost".to_string(), whoami::username(), 22);
            let daemon_config = HttpDaemonConfig::new("rbee-hive", &base_url).with_args(args);
            let config = StartConfig {
                ssh_config: ssh,
                daemon_config,
                job_id: None,
            };
            let _pid = start_daemon(config).await?;
            Ok(())
        }
        // TEAM-330: Use SSH to localhost (consistent with remote operations)
        HiveAction::Stop { port } => {
            let ssh = SshConfig::new("localhost".to_string(), whoami::username(), 22);
            let shutdown_url = format!("http://localhost:{}/v1/shutdown", port.unwrap_or(7835));
            let health_url = format!("http://localhost:{}/health", port.unwrap_or(7835));
            let config = StopConfig {
                daemon_name: "rbee-hive".to_string(),
                shutdown_url,
                health_url,
                ssh_config: ssh,
                job_id: None,
            };
            stop_daemon(config).await
        }

        // TEAM-330: Use check_daemon_health() directly (takes only URL)
        HiveAction::Status { alias: _ } => {
            let health_url = "http://localhost:7835/health";
            let is_running = check_daemon_health(health_url).await;
            
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

        // TEAM-330: Use SSH to localhost (consistent with remote operations)
        HiveAction::Install { alias: _ } => {
            let ssh = SshConfig::new("localhost".to_string(), whoami::username(), 22);
            let config = InstallConfig {
                daemon_name: "rbee-hive".to_string(),
                ssh_config: ssh,
                local_binary_path: None,
                job_id: None,
            };
            install_daemon(config).await
        }
        HiveAction::Uninstall { alias: _ } => {
            let ssh = SshConfig::new("localhost".to_string(), whoami::username(), 22);
            let config = UninstallConfig {
                daemon_name: "rbee-hive".to_string(),
                ssh_config: ssh,
                health_url: Some("http://localhost:7835".to_string()),
                health_timeout_secs: Some(2),
                job_id: None,
            };
            uninstall_daemon(config).await
        }
        HiveAction::Rebuild { alias: _ } => {
            let ssh = SshConfig::new("localhost".to_string(), whoami::username(), 22);
            let daemon_config = HttpDaemonConfig::new("rbee-hive", "http://localhost:7835")
                .with_args(vec!["--port".to_string(), "7835".to_string()]);
            let config = RebuildConfig {
                daemon_name: "rbee-hive".to_string(),
                ssh_config: ssh,
                daemon_config,
                job_id: None,
            };
            rebuild_daemon(config).await
        }
    }
}
