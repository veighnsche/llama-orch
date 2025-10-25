//! Hive command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-290: Local and remote operations via hive-lifecycle
//! TEAM-196: Added RefreshCapabilities command
//! TEAM-263: Added smart prompt for localhost hive install
//! TEAM-294: HiveList now reads from ~/.ssh/config instead of forwarding to queen

use anyhow::{Context, Result};
use hive_lifecycle::{install_hive, uninstall_hive, start_hive, stop_hive};
use observability_narration_core::NarrationFactory;
use operations_contract::Operation; // TEAM-284: Renamed from rbee_operations
use ssh_config::parse_ssh_config;

use crate::cli::HiveAction;
use crate::job_client::submit_and_stream_job;

// TEAM-294: Local narration factory for hive handlers
const NARRATE: NarrationFactory = NarrationFactory::new("keeper");

pub async fn handle_hive(action: HiveAction, queen_url: &str) -> Result<()> {
    match action {
        // TEAM-290: RESTORED Install, Uninstall, Start, Stop (now with SSH support)
        HiveAction::Install { host, binary, install_dir } => {
            install_hive(&host, binary, install_dir).await
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
            let install_dir = install_dir.unwrap_or_else(|| {
                if host == "localhost" || host == "127.0.0.1" {
                    format!("{}/.local/bin", std::env::var("HOME").unwrap_or_default())
                } else {
                    "/usr/local/bin".to_string()
                }
            });
            // TEAM-292: Pass queen_url to hive so it knows where to send heartbeats
            start_hive(&host, &install_dir, port, Some(queen_url)).await
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
                NARRATE
                    .action("hive_list")
                    .human("⚠️  No SSH hosts found in ~/.ssh/config")
                    .emit();
                return Ok(());
            }

            // Display as table
            let json_value = serde_json::to_value(&targets)
                .context("Failed to serialize SSH targets")?;
            
            NARRATE
                .action("hive_list")
                .human(format!("Found {} SSH target(s)", targets.len()))
                .table(&json_value)
                .emit();

            Ok(())
        }
        HiveAction::Get { alias } => {
            let operation = Operation::HiveGet { alias };
            submit_and_stream_job(queen_url, operation).await
        }
        HiveAction::Status { alias } => {
            let operation = Operation::HiveStatus { alias };
            submit_and_stream_job(queen_url, operation).await
        }
        HiveAction::RefreshCapabilities { alias } => {
            let operation = Operation::HiveRefreshCapabilities { alias };
            submit_and_stream_job(queen_url, operation).await
        }
    }
}

// TEAM-278: DELETED check_local_hive_optimization() function
// This will be reimplemented for PackageSync/PackageInstall commands
// when they are added by TEAM-279
