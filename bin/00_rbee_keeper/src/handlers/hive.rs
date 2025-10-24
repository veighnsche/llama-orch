//! Hive command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-194: Use alias-based operations (config from hives.conf)
//! TEAM-196: Added RefreshCapabilities command
//! TEAM-263: Added smart prompt for localhost hive install

use anyhow::Result;
use operations_contract::Operation; // TEAM-284: Renamed from rbee_operations

use crate::cli::HiveAction;
use crate::job_client::submit_and_stream_job;

pub async fn handle_hive(action: HiveAction, queen_url: &str) -> Result<()> {
    // TEAM-278: DELETED special handling for HiveInstall - operation no longer exists
    // TEAM-278: DELETED check_local_hive_optimization() - will be reimplemented for PackageSync

    let operation = match action {
        // TEAM-278: DELETED SshTest, Install, Uninstall, ImportSsh
        // TEAM-285: DELETED Start, Stop (localhost-only, no lifecycle management)
        HiveAction::List => Operation::HiveList,
        HiveAction::Get { alias } => Operation::HiveGet { alias },
        HiveAction::Status { alias } => Operation::HiveStatus { alias },
        HiveAction::RefreshCapabilities { alias } => Operation::HiveRefreshCapabilities { alias },
    };
    submit_and_stream_job(queen_url, operation).await
}

// TEAM-278: DELETED check_local_hive_optimization() function
// This will be reimplemented for PackageSync/PackageInstall commands
// when they are added by TEAM-279
