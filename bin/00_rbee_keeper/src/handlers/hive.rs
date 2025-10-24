//! Hive command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-290: Local and remote operations via hive-lifecycle
//! TEAM-196: Added RefreshCapabilities command
//! TEAM-263: Added smart prompt for localhost hive install

use anyhow::Result;
use hive_lifecycle::{install_hive, uninstall_hive, start_hive, stop_hive};
use operations_contract::Operation; // TEAM-284: Renamed from rbee_operations

use crate::cli::HiveAction;
use crate::job_client::submit_and_stream_job;

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
        // Operations forwarded to queen
        HiveAction::List => {
            let operation = Operation::HiveList;
            submit_and_stream_job(queen_url, operation).await
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
