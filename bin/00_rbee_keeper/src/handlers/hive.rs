//! Hive command handlers (localhost + remote SSH)
//!
//! TEAM-322: Removed all SSH/remote functionality (RULE ZERO - delete complexity)
//! TEAM-324: Moved HiveAction enum here to eliminate duplication
//! TEAM-332: Added SSH config resolver middleware (eliminates repeated SshConfig::localhost())

use anyhow::Result;
use clap::Subcommand;
use daemon_lifecycle::{
    check_daemon_health, install_daemon, rebuild_daemon, start_daemon, stop_daemon,
    uninstall_daemon, HttpDaemonConfig, InstallConfig, RebuildConfig,
    StartConfig, StopConfig, UninstallConfig,
};
use observability_narration_core::n;
use operations_contract::Operation;

use crate::job_client::submit_and_stream_job;
use crate::ssh_resolver::resolve_ssh_config; // TEAM-332: SSH config middleware

#[derive(Subcommand)]
pub enum HiveAction {
    /// Start rbee-hive
    Start {
        /// Host alias (default: localhost, or use SSH config entry)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
        /// HTTP port (default: 7835)
        #[arg(short = 'p', long = "port")]
        port: Option<u16>,
    },
    /// Stop rbee-hive
    Stop {
        /// Host alias (default: localhost, or use SSH config entry)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
        /// HTTP port (default: 7835)
        #[arg(short = 'p', long = "port")]
        port: Option<u16>,
    },
    /// Check hive status
    Status {
        /// Host alias (default: localhost, or use SSH config entry)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Refresh device capabilities for a hive
    RefreshCapabilities {
        /// Host alias (default: localhost, or use SSH config entry)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Install rbee-hive binary
    Install {
        /// Host alias (default: localhost, or use SSH config entry)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Uninstall rbee-hive binary
    Uninstall {
        /// Host alias (default: localhost, or use SSH config entry)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Rebuild rbee-hive binary (cargo build)
    Rebuild {
        /// Host alias (default: localhost, or use SSH config entry)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
}

pub async fn handle_hive(action: HiveAction, queen_url: &str) -> Result<()> {
    match action {
        HiveAction::Start { alias, port } => {
            // TEAM-332: Resolve SSH config from alias (localhost or ~/.ssh/config)
            let ssh = resolve_ssh_config(&alias)?;
            let port = port.unwrap_or(7835);
            let base_url = format!("http://{}:{}", ssh.hostname, port);
            let health_url = format!("{}/health", base_url);
            let args = vec![
                "--port".to_string(),
                port.to_string(),
                "--queen-url".to_string(),
                queen_url.to_string(),
                "--hive-id".to_string(),
                alias.clone(),
            ];
            let daemon_config = HttpDaemonConfig::new("rbee-hive", &health_url).with_args(args);
            let config = StartConfig {
                ssh_config: ssh,
                daemon_config,
                job_id: None,
            };
            let _pid = start_daemon(config).await?;
            Ok(())
        }
        HiveAction::Stop { alias, port } => {
            // TEAM-332: Resolve SSH config from alias (localhost or ~/.ssh/config)
            let ssh = resolve_ssh_config(&alias)?;
            let port = port.unwrap_or(7835);
            let shutdown_url = format!("http://{}:{}/v1/shutdown", ssh.hostname, port);
            let health_url = format!("http://{}:{}/health", ssh.hostname, port);
            let config = StopConfig {
                daemon_name: "rbee-hive".to_string(),
                shutdown_url,
                health_url,
                ssh_config: ssh,
                job_id: None,
            };
            stop_daemon(config).await
        }

        HiveAction::Status { alias } => {
            // TEAM-332: Resolve SSH config from alias (localhost or ~/.ssh/config)
            let ssh = resolve_ssh_config(&alias)?;
            let health_url = format!("http://{}:7835/health", ssh.hostname);
            let is_running = check_daemon_health(&health_url).await;
            
            if is_running {
                n!("hive_status", "✅ hive '{}' is running on {}", alias, health_url);
            } else {
                n!("hive_status", "❌ hive '{}' is not running on {}", alias, health_url);
            }
            Ok(())
        }

        // TEAM-329: Removed Get and Check handlers (user request)
        HiveAction::RefreshCapabilities { alias } => {
            let operation = Operation::HiveRefreshCapabilities { alias };
            submit_and_stream_job(queen_url, operation).await
        }

        HiveAction::Install { alias } => {
            // TEAM-332: Resolve SSH config from alias (localhost or ~/.ssh/config)
            let ssh = resolve_ssh_config(&alias)?;
            let config = InstallConfig {
                daemon_name: "rbee-hive".to_string(),
                ssh_config: ssh,
                local_binary_path: None,
                job_id: None,
            };
            install_daemon(config).await
        }
        HiveAction::Uninstall { alias } => {
            // TEAM-332: Resolve SSH config from alias (localhost or ~/.ssh/config)
            let ssh = resolve_ssh_config(&alias)?;
            let health_url = format!("http://{}:7835", ssh.hostname);
            let config = UninstallConfig {
                daemon_name: "rbee-hive".to_string(),
                ssh_config: ssh,
                health_url: Some(health_url),
                health_timeout_secs: Some(2),
                job_id: None,
            };
            uninstall_daemon(config).await
        }
        HiveAction::Rebuild { alias } => {
            // TEAM-332: Resolve SSH config from alias (localhost or ~/.ssh/config)
            let ssh = resolve_ssh_config(&alias)?;
            let base_url = format!("http://{}:7835", ssh.hostname);
            let daemon_config = HttpDaemonConfig::new("rbee-hive", &base_url)
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
