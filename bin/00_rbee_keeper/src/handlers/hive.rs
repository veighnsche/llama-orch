//! Hive command handlers (localhost + remote SSH)
//!
//! TEAM-322: Removed all SSH/remote functionality (RULE ZERO - delete complexity)
//! TEAM-324: Moved HiveAction enum here to eliminate duplication
//! TEAM-332: Added SSH config resolver middleware (eliminates repeated SshConfig::localhost())
//! TEAM-365: Use lifecycle-local for localhost, lifecycle-ssh for remote (conditional dispatch)

use anyhow::Result;
use clap::Subcommand;
use observability_narration_core::n;

use crate::ssh_resolver::resolve_ssh_config; // TEAM-332: SSH config middleware

// TEAM-365: Import both lifecycle crates for conditional dispatch
use lifecycle_local;
use lifecycle_ssh;

// TEAM-368: Get local IP for remote hive queen_url
use local_ip_address::local_ip;

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
            let port = port.unwrap_or(7835);
            
            // TEAM-365: Conditional dispatch - localhost uses lifecycle-local, remote uses lifecycle-ssh
            if alias == "localhost" {
                // TEAM-365: Localhost - use lifecycle-local (no SSH)
                let base_url = format!("http://localhost:{}", port);
                let health_url = format!("{}/health", base_url);
                let args = vec![
                    "--port".to_string(),
                    port.to_string(),
                    "--queen-url".to_string(),
                    queen_url.to_string(),
                    "--hive-id".to_string(),
                    alias.clone(),
                ];
                let daemon_config = lifecycle_local::HttpDaemonConfig::new("rbee-hive", &health_url).with_args(args);
                let config = lifecycle_local::StartConfig { daemon_config, job_id: None };
                let _pid = lifecycle_local::start_daemon(config).await?;
            } else {
                // TEAM-365: Remote - use lifecycle-ssh
                // TEAM-368: Remote hive needs Queen's network address, not localhost
                let ssh = resolve_ssh_config(&alias)?;
                let base_url = format!("http://{}:{}", ssh.hostname, port);
                let health_url = format!("{}/health", base_url);
                
                // TEAM-368: Get keeper's local IP (Queen is on same machine)
                let local_ip = local_ip().map_err(|e| anyhow::anyhow!("Failed to get local IP: {}", e))?;
                let queen_port = queen_url.split(':').last()
                    .and_then(|p| p.parse::<u16>().ok())
                    .unwrap_or(7833);
                let network_queen_url = format!("http://{}:{}", local_ip, queen_port);
                
                n!("remote_hive_queen_url", "🌐 Remote hive will use Queen at: {}", network_queen_url);
                
                let args = vec![
                    "--port".to_string(),
                    port.to_string(),
                    "--queen-url".to_string(),
                    network_queen_url,  // TEAM-368: Use network address, not localhost
                    "--hive-id".to_string(),
                    alias.clone(),
                ];
                let daemon_config = lifecycle_ssh::HttpDaemonConfig::new("rbee-hive", &health_url).with_args(args);
                let config = lifecycle_ssh::StartConfig { ssh_config: ssh, daemon_config, job_id: None };
                let _pid = lifecycle_ssh::start_daemon(config).await?;
            }
            Ok(())
        }
        HiveAction::Stop { alias, port } => {
            let port = port.unwrap_or(7835);
            
            // TEAM-365: Conditional dispatch - localhost uses lifecycle-local, remote uses lifecycle-ssh
            if alias == "localhost" {
                // TEAM-365: Localhost - use lifecycle-local (no SSH)
                let shutdown_url = format!("http://localhost:{}/v1/shutdown", port);
                let health_url = format!("http://localhost:{}/health", port);
                let config = lifecycle_local::StopConfig {
                    daemon_name: "rbee-hive".to_string(),
                    shutdown_url,
                    health_url,
                    job_id: None,
                };
                lifecycle_local::stop_daemon(config).await
            } else {
                // TEAM-365: Remote - use lifecycle-ssh
                let ssh = resolve_ssh_config(&alias)?;
                let shutdown_url = format!("http://{}:{}/v1/shutdown", ssh.hostname, port);
                let health_url = format!("http://{}:{}/health", ssh.hostname, port);
                let config = lifecycle_ssh::StopConfig {
                    daemon_name: "rbee-hive".to_string(),
                    shutdown_url,
                    health_url,
                    ssh_config: ssh,
                    job_id: None,
                };
                lifecycle_ssh::stop_daemon(config).await
            }
        }

        HiveAction::Status { alias } => {
            // TEAM-365: Conditional dispatch - localhost uses lifecycle-local, remote uses lifecycle-ssh
            if alias == "localhost" {
                // TEAM-365: Localhost - use lifecycle-local (no SSH)
                let health_url = "http://localhost:7835/health";
                let status = lifecycle_local::check_daemon_health(health_url, "rbee-hive").await;

                if status.is_running {
                    n!("hive_status", "✅ hive '{}' is running on {}", alias, health_url);
                } else {
                    n!("hive_status", "❌ hive '{}' is not running on {}", alias, health_url);
                }
            } else {
                // TEAM-365: Remote - use lifecycle-ssh
                let ssh = resolve_ssh_config(&alias)?;
                let health_url = format!("http://{}:7835/health", ssh.hostname);
                let status = lifecycle_ssh::check_daemon_health(&health_url, "rbee-hive", &ssh).await;

                if status.is_running {
                    n!("hive_status", "✅ hive '{}' is running on {}", alias, health_url);
                } else {
                    n!("hive_status", "❌ hive '{}' is not running on {}", alias, health_url);
                }
            }
            Ok(())
        }

        // TEAM-329: Removed Get and Check handlers (user request)
        HiveAction::Install { alias } => {
            // TEAM-365: Conditional dispatch - localhost uses lifecycle-local, remote uses lifecycle-ssh
            if alias == "localhost" {
                // TEAM-365: Localhost - use lifecycle-local (no SSH)
                let config = lifecycle_local::InstallConfig {
                    daemon_name: "rbee-hive".to_string(),
                    local_binary_path: None,
                    job_id: None,
                };
                lifecycle_local::install_daemon(config).await
            } else {
                // TEAM-365: Remote - use lifecycle-ssh
                let ssh = resolve_ssh_config(&alias)?;
                let config = lifecycle_ssh::InstallConfig {
                    daemon_name: "rbee-hive".to_string(),
                    ssh_config: ssh,
                    local_binary_path: None,
                    job_id: None,
                };
                lifecycle_ssh::install_daemon(config).await
            }
        }
        HiveAction::Uninstall { alias } => {
            // TEAM-365: Conditional dispatch - localhost uses lifecycle-local, remote uses lifecycle-ssh
            if alias == "localhost" {
                // TEAM-365: Localhost - use lifecycle-local (no SSH)
                let config = lifecycle_local::UninstallConfig {
                    daemon_name: "rbee-hive".to_string(),
                    health_url: Some("http://localhost:7835".to_string()),
                    health_timeout_secs: Some(2),
                    job_id: None,
                };
                lifecycle_local::uninstall_daemon(config).await
            } else {
                // TEAM-365: Remote - use lifecycle-ssh
                let ssh = resolve_ssh_config(&alias)?;
                let health_url = format!("http://{}:7835", ssh.hostname);
                let config = lifecycle_ssh::UninstallConfig {
                    daemon_name: "rbee-hive".to_string(),
                    ssh_config: ssh,
                    health_url: Some(health_url),
                    health_timeout_secs: Some(2),
                    job_id: None,
                };
                lifecycle_ssh::uninstall_daemon(config).await
            }
        }
        HiveAction::Rebuild { alias } => {
            // TEAM-365: Conditional dispatch - localhost uses lifecycle-local, remote uses lifecycle-ssh
            if alias == "localhost" {
                // TEAM-365: Localhost - use lifecycle-local (no SSH)
                let daemon_config = lifecycle_local::HttpDaemonConfig::new("rbee-hive", "http://localhost:7835")
                    .with_args(vec!["--port".to_string(), "7835".to_string()]);
                let config = lifecycle_local::RebuildConfig {
                    daemon_name: "rbee-hive".to_string(),
                    daemon_config,
                    job_id: None,
                };
                lifecycle_local::rebuild_daemon(config).await
            } else {
                // TEAM-365: Remote - use lifecycle-ssh
                let ssh = resolve_ssh_config(&alias)?;
                let base_url = format!("http://{}:7835", ssh.hostname);
                let daemon_config = lifecycle_ssh::HttpDaemonConfig::new("rbee-hive", &base_url)
                    .with_args(vec!["--port".to_string(), "7835".to_string()]);
                let config = lifecycle_ssh::RebuildConfig {
                    daemon_name: "rbee-hive".to_string(),
                    ssh_config: ssh,
                    daemon_config,
                    job_id: None,
                };
                lifecycle_ssh::rebuild_daemon(config).await
            }
        }
    }
}
