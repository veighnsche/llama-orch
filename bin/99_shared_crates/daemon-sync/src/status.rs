//! Package status checking (drift detection)
//!
//! Created by: TEAM-280
//!
//! Check if actual state matches desired state without applying changes.

use super::diff::compute_diff;
use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::declarative::HivesConfig;
use serde::{Deserialize, Serialize};

const NARRATE: NarrationFactory = NarrationFactory::new("pkg-stat");

/// Status report
///
/// TEAM-280: Result of status check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusReport {
    /// Overall status (ok, drift_detected)
    pub status: String,

    /// Total number of hives in config
    pub total_hives: usize,

    /// Number of hives installed correctly
    pub hives_ok: usize,

    /// Number of hives missing
    pub hives_missing: usize,

    /// Number of extra hives (not in config)
    pub hives_extra: usize,

    /// Number of workers missing
    pub workers_missing: usize,

    /// Number of extra workers (not in config)
    pub workers_extra: usize,

    /// Detailed drift information (if verbose)
    pub drift: Vec<DriftItem>,
}

/// Single drift item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftItem {
    /// Hive alias
    pub hive: String,

    /// Issue type (missing_hive, extra_hive, missing_worker, extra_worker)
    pub issue: String,

    /// Details
    pub details: String,
}

/// Check hive orchestration status (drift detection)
///
/// TEAM-280: Main status entry point
///
/// # Arguments
/// * `config` - Desired state from config file
/// * `verbose` - Include detailed drift information
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(StatusReport)` - Status check completed
/// * `Err` - Status check failed
pub async fn check_status(
    config: HivesConfig,
    verbose: bool,
    job_id: &str,
) -> Result<StatusReport> {
    NARRATE
        .action("status_check")
        .job_id(job_id)
        .context(&config.hives.len().to_string())
        .human("🔍 Checking status for {} hives")
        .emit();

    // Query actual state
    // TODO: Implement actual state query (for now, assume nothing installed)
    let actual_hives: Vec<String> = Vec::new();
    let actual_workers: Vec<(String, Vec<String>)> = Vec::new();

    // Compute diff
    let diff = compute_diff(&config.hives, &actual_hives, &actual_workers, true);

    // Build status report
    let mut drift = Vec::new();

    if verbose {
        // Add missing hives
        for hive in &diff.hives_to_install {
            drift.push(DriftItem {
                hive: hive.alias.clone(),
                issue: "missing_hive".to_string(),
                details: format!("Hive '{}' not installed", hive.alias),
            });
        }

        // Add extra hives
        for alias in &diff.hives_to_remove {
            drift.push(DriftItem {
                hive: alias.clone(),
                issue: "extra_hive".to_string(),
                details: format!("Hive '{}' not in config", alias),
            });
        }

        // Add missing workers
        for (hive_alias, workers) in &diff.workers_to_install {
            for worker in workers {
                drift.push(DriftItem {
                    hive: hive_alias.clone(),
                    issue: "missing_worker".to_string(),
                    details: format!("Worker '{}' not installed on '{}'", worker.worker_type, hive_alias),
                });
            }
        }

        // Add extra workers
        for (hive_alias, workers) in &diff.workers_to_remove {
            for worker in workers {
                drift.push(DriftItem {
                    hive: hive_alias.clone(),
                    issue: "extra_worker".to_string(),
                    details: format!("Worker '{}' not in config for '{}'", worker, hive_alias),
                });
            }
        }
    }

    let status = if diff.has_changes() {
        "drift_detected".to_string()
    } else {
        "ok".to_string()
    };

    let report = StatusReport {
        status: status.clone(),
        total_hives: config.hives.len(),
        hives_ok: diff.hives_already_installed.len(),
        hives_missing: diff.hives_to_install.len(),
        hives_extra: diff.hives_to_remove.len(),
        workers_missing: diff.workers_to_install.iter().map(|(_, w)| w.len()).sum(),
        workers_extra: diff.workers_to_remove.iter().map(|(_, w)| w.len()).sum(),
        drift,
    };

    NARRATE
        .action("status_result")
        .job_id(job_id)
        .context(&status)
        .human("📊 Status: {}")
        .emit();

    Ok(report)
}
