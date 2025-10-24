//! State query - detect what's actually installed on remote hosts
//!
//! Created by: TEAM-281 (fixing TEAM-280's TODO)
//!
//! This module queries remote hosts via SSH to detect:
//! - Which hives are installed (rbee-hive binary exists)
//! - Which workers are installed (rbee-worker-* binaries exist)
//!
//! This enables idempotency - sync can detect "already installed" state
//! and skip reinstallation.

use anyhow::{Context, Result};
use observability_narration_core::NarrationFactory;
use queen_rbee_ssh_client::RbeeSSHClient;
use rbee_config::declarative::HiveConfig;

const NARRATE: NarrationFactory = NarrationFactory::new("pkg-query");

/// Query which hives are installed
///
/// TEAM-281: Fixes TODO at sync.rs:119
///
/// Connects to each hive via SSH and checks if rbee-hive binary exists.
///
/// # Arguments
/// * `hives` - List of hive configurations to query
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(Vec<String>)` - List of installed hive aliases
/// * `Err` - Query failed
pub async fn query_installed_hives(hives: &[HiveConfig], job_id: &str) -> Result<Vec<String>> {
    NARRATE
        .action("query_hives")
        .job_id(job_id)
        .context(hives.len().to_string())
        .human("🔍 Querying {} hives for installed binaries")
        .emit();

    let mut installed = Vec::new();

    for hive in hives {
        match query_single_hive(hive, job_id).await {
            Ok(true) => {
                installed.push(hive.alias.clone());
                NARRATE
                    .action("hive_found")
                    .job_id(job_id)
                    .context(&hive.alias)
                    .human("✅ Hive '{}' is installed")
                    .emit();
            }
            Ok(false) => {
                NARRATE
                    .action("hive_not_found")
                    .job_id(job_id)
                    .context(&hive.alias)
                    .human("📦 Hive '{}' not installed")
                    .emit();
            }
            Err(e) => {
                NARRATE
                    .action("hive_query_error")
                    .job_id(job_id)
                    .context(&hive.alias)
                    .context(e.to_string())
                    .human("⚠️  Failed to query hive '{}': {}")
                    .emit();
                // Continue querying other hives even if one fails
            }
        }
    }

    NARRATE
        .action("query_complete")
        .job_id(job_id)
        .context(installed.len().to_string())
        .context(hives.len().to_string())
        .human("📊 Found {} of {} hives installed")
        .emit();

    Ok(installed)
}

/// Query which workers are installed on each hive
///
/// TEAM-281: Fixes TODO at sync.rs:119
///
/// Connects to each hive via SSH and checks which worker binaries exist.
///
/// # Arguments
/// * `hives` - List of hive configurations to query
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(Vec<(String, Vec<String>)>)` - List of (hive_alias, worker_types)
/// * `Err` - Query failed
pub async fn query_installed_workers(
    hives: &[HiveConfig],
    job_id: &str,
) -> Result<Vec<(String, Vec<String>)>> {
    NARRATE
        .action("query_workers")
        .job_id(job_id)
        .context(hives.len().to_string())
        .human("🔍 Querying workers on {} hives")
        .emit();

    let mut results = Vec::new();

    for hive in hives {
        match query_hive_workers(hive, job_id).await {
            Ok(workers) => {
                if !workers.is_empty() {
                    NARRATE
                        .action("workers_found")
                        .job_id(job_id)
                        .context(&hive.alias)
                        .context(workers.len().to_string())
                        .human("✅ Hive '{}' has {} workers installed")
                        .emit();
                }
                results.push((hive.alias.clone(), workers));
            }
            Err(e) => {
                NARRATE
                    .action("worker_query_error")
                    .job_id(job_id)
                    .context(&hive.alias)
                    .context(e.to_string())
                    .human("⚠️  Failed to query workers on '{}': {}")
                    .emit();
                // Continue querying other hives
                results.push((hive.alias.clone(), Vec::new()));
            }
        }
    }

    Ok(results)
}

/// Query if a single hive is installed
///
/// TEAM-281: Helper function
///
/// # Arguments
/// * `hive` - Hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(true)` - Hive is installed
/// * `Ok(false)` - Hive is not installed
/// * `Err` - Query failed (SSH connection, etc.)
async fn query_single_hive(hive: &HiveConfig, _job_id: &str) -> Result<bool> {
    // Connect via SSH
    let mut client = RbeeSSHClient::connect(&hive.hostname, hive.ssh_port, &hive.ssh_user)
        .await
        .context(format!("Failed to connect to {}", hive.hostname))?;

    // Check if rbee-hive binary exists and is executable
    let check_cmd = "~/.local/bin/rbee-hive --version";
    let (stdout, _, exit_code) = client.exec(check_cmd).await?;

    client.close().await?;

    // If command succeeded and output contains "rbee-hive", it's installed
    Ok(exit_code == 0 && stdout.contains("rbee-hive"))
}

/// Query which workers are installed on a single hive
///
/// TEAM-281: Helper function
///
/// # Arguments
/// * `hive` - Hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(Vec<String>)` - List of installed worker types
/// * `Err` - Query failed
async fn query_hive_workers(hive: &HiveConfig, _job_id: &str) -> Result<Vec<String>> {
    // Connect via SSH
    let mut client = RbeeSSHClient::connect(&hive.hostname, hive.ssh_port, &hive.ssh_user)
        .await
        .context(format!("Failed to connect to {}", hive.hostname))?;

    // List all rbee-worker-* binaries in ~/.local/bin
    let list_cmd =
        "ls -1 ~/.local/bin/rbee-worker-* 2>/dev/null | xargs -n1 basename 2>/dev/null || true";
    let (stdout, _, _) = client.exec(list_cmd).await?;

    client.close().await?;

    // Parse output - each line is a worker binary name
    let workers: Vec<String> = stdout
        .lines()
        .filter(|line| !line.is_empty())
        .filter(|line| line.starts_with("rbee-worker-"))
        .map(|line| {
            // Extract worker type from binary name
            // rbee-worker-cuda -> cuda
            line.strip_prefix("rbee-worker-").unwrap_or(line).to_string()
        })
        .collect();

    Ok(workers)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_worker_name_parsing() {
        let binary_name = "rbee-worker-cuda";
        let worker_type = binary_name.strip_prefix("rbee-worker-").unwrap();
        assert_eq!(worker_type, "cuda");

        let binary_name = "rbee-worker-cpu";
        let worker_type = binary_name.strip_prefix("rbee-worker-").unwrap();
        assert_eq!(worker_type, "cpu");
    }
}
