//! Hive command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-290: Local and remote operations via hive-lifecycle
//! TEAM-196: Added RefreshCapabilities command
//! TEAM-263: Added smart prompt for localhost hive install
//! TEAM-294: HiveList now reads from ~/.ssh/config instead of forwarding to queen

use anyhow::{Context, Result};
use hive_lifecycle::{install_hive, uninstall_hive, start_hive, stop_hive, rebuild_hive, DEFAULT_INSTALL_DIR};
use observability_narration_core::n;
use operations_contract::Operation; // TEAM-284: Renamed from rbee_operations
use ssh_config::parse_ssh_config;

use crate::cli::HiveAction;
use crate::job_client::{submit_and_stream_job, submit_and_stream_job_to_hive};

pub async fn handle_hive(action: HiveAction, queen_url: &str) -> Result<()> {
    match action {
        // TEAM-290: RESTORED Install, Uninstall, Start, Stop (now with SSH support)
        // TEAM-314: Added build_remote flag for optional on-site builds
        HiveAction::Install { host, binary, install_dir, build_remote } => {
            install_hive(&host, binary, install_dir, build_remote).await
        }
        HiveAction::Uninstall { host, install_dir } => {
            let install_dir = install_dir.unwrap_or_else(|| {
                if host == "localhost" || host == "127.0.0.1" {
                    format!("{}/.local/bin", std::env::var("HOME").unwrap_or_default())
                } else {
                    "/usr/local/bin".to_string()
                }
            });
            uninstall_hive(&host, &install_dir).await
        }
        HiveAction::Start { host, install_dir, port } => {
            // TEAM-314: Use constant default install dir (matches install.rs)
            let install_dir = install_dir.unwrap_or_else(|| DEFAULT_INSTALL_DIR.to_string());
            
            // TEAM-314: Keeper manages hive lifecycle directly, no queen involvement
            // For localhost: pass queen_url for heartbeats
            // For remote: pass None (remote hives don't send heartbeats yet)
            let queen_url_opt = if host == "localhost" || host == "127.0.0.1" {
                Some(queen_url)
            } else {
                None
            };
            start_hive(&host, &install_dir, port, queen_url_opt).await
        }
        HiveAction::Stop { host } => {
            stop_hive(&host).await
        }
        // TEAM-294: Read from ~/.ssh/config instead of forwarding to queen
        HiveAction::List => {
            let ssh_config_path = dirs::home_dir()
                .context("Failed to get home directory")?
                .join(".ssh/config");

            let targets = parse_ssh_config(&ssh_config_path)?;

            if targets.is_empty() {
                n!("hive_list", "⚠️  No SSH hosts found in ~/.ssh/config");
                return Ok(());
            }

            // Display as table
            let json_value = serde_json::to_value(&targets)
                .context("Failed to serialize SSH targets")?;
            
            n!("hive_list", "Found {} SSH target(s)", targets.len());
            // Print table as JSON for now (n!() doesn't support .table() yet)
            println!("{}", serde_json::to_string_pretty(&json_value)?);

            Ok(())
        }
        HiveAction::Get { alias } => {
            let operation = Operation::HiveGet { alias };
            submit_and_stream_job(queen_url, operation).await
        }
        // TEAM-314: Status checks hive directly, not through queen
        HiveAction::Status { alias } => {
            // Determine hive URL based on alias
            let hive_url = if alias == "localhost" || alias == "127.0.0.1" {
                "http://localhost:7835".to_string()
            } else {
                // For remote hives, we need to resolve the SSH host to a URL
                // For now, assume standard port 7835
                // TODO: Read port from hive config or SSH tunnel
                anyhow::bail!("Remote hive status not yet implemented. Use 'hive check' for localhost.")
            };
            
            // TEAM-314: Connect directly to hive (not through queen)
            let operation = Operation::HiveStatus { alias };
            submit_and_stream_job_to_hive(&hive_url, operation).await
        }
        HiveAction::RefreshCapabilities { alias } => {
            let operation = Operation::HiveRefreshCapabilities { alias };
            submit_and_stream_job(queen_url, operation).await
        }
        // TEAM-313: HiveCheck - narration test through hive SSE (DIRECT to hive, not through queen)
        // TEAM-314: Added remote hive support
        HiveAction::Check { alias } => {
            // Keeper talks DIRECTLY to hive, not through queen
            let hive_url = if alias == "localhost" || alias == "127.0.0.1" {
                "http://localhost:7835".to_string()
            } else {
                // TEAM-314: For remote hives, resolve SSH host to HTTP URL
                // Try to get the actual hostname from SSH config
                let ssh_config_path = dirs::home_dir()
                    .context("Failed to get home directory")?
                    .join(".ssh/config");
                
                let targets = parse_ssh_config(&ssh_config_path)?;
                
                // Find the target host (field is 'host' not 'alias')
                let target = targets.iter()
                    .find(|t| t.host == alias)
                    .ok_or_else(|| anyhow::anyhow!("SSH host '{}' not found in ~/.ssh/config", alias))?;
                
                // Use the hostname from SSH config
                n!("hive_check_remote", "🔍 Connecting to remote hive at {}:7835", target.hostname);
                format!("http://{}:7835", target.hostname)
            };
            
            // TEAM-314: Connect directly to hive (not through queen)
            let operation = Operation::HiveCheck { alias };
            submit_and_stream_job_to_hive(&hive_url, operation).await
        }
        // TEAM-314: Rebuild - update hive from source (parity with queen)
        HiveAction::Rebuild { host, build_remote } => {
            rebuild_hive(&host, build_remote).await
        }
    }
}

// TEAM-278: DELETED check_local_hive_optimization() function
// This will be reimplemented for PackageSync/PackageInstall commands
// when they are added by TEAM-279
