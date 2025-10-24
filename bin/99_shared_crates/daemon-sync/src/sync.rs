//! Sync orchestration - reconcile desired state with actual state
//!
//! Created by: TEAM-280
//!
//! This module orchestrates the sync operation:
//! 1. Query actual state
//! 2. Compute diff (desired vs actual)
//! 3. Apply changes concurrently
//! 4. Return sync report

use super::diff::{compute_diff, StateDiff};
use super::install::install_all;
use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::declarative::{HiveConfig, HivesConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const NARRATE: NarrationFactory = NarrationFactory::new("pkg-sync");

/// Sync options
///
/// TEAM-280: Configuration for sync operation
#[derive(Debug, Clone)]
pub struct SyncOptions {
    /// Dry run - show what would change without applying
    pub dry_run: bool,

    /// Remove components not in config
    pub remove_extra: bool,

    /// Force reinstall even if already installed
    pub force: bool,
}

/// Sync report
///
/// TEAM-280: Result of sync operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncReport {
    /// Number of hives installed
    pub hives_installed: usize,

    /// Number of hives already installed
    pub hives_already_installed: usize,

    /// Number of workers installed
    pub workers_installed: usize,

    /// Number of workers already installed
    pub workers_already_installed: usize,

    /// Number of hives removed
    pub hives_removed: usize,

    /// Number of workers removed
    pub workers_removed: usize,

    /// Total duration in seconds
    pub duration_seconds: f64,

    /// Detailed results per hive
    pub hive_results: Vec<HiveResult>,
}

/// Result for a single hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveResult {
    /// Hive alias
    pub alias: String,

    /// Status (installed, already_installed, failed)
    pub status: String,

    /// Workers installed on this hive
    pub workers: Vec<String>,

    /// Duration in seconds
    pub duration_seconds: f64,

    /// Error message (if failed)
    pub error: Option<String>,
}

/// Sync all hives from config
///
/// TEAM-280: Main sync entry point
///
/// # Steps
/// 1. Query actual state
/// 2. Compute diff (desired vs actual)
/// 3. If dry_run, return early with report
/// 4. Apply changes concurrently using tokio::spawn
/// 5. Return sync report
///
/// # Arguments
/// * `config` - Desired state from config file
/// * `opts` - Sync options
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(SyncReport)` - Sync completed successfully
/// * `Err` - Sync failed
pub async fn sync_all_hives(
    config: HivesConfig,
    opts: SyncOptions,
    job_id: &str,
) -> Result<SyncReport> {
    let start_time = std::time::Instant::now();

    NARRATE
        .action("sync_start")
        .job_id(job_id)
        .context(config.hives.len().to_string())
        .human("🔄 Syncing {} hives")
        .emit();

    // Step 1: Query actual state
    // TEAM-281: Implemented state query (was TODO)
    let actual_hives = super::query::query_installed_hives(&config.hives, job_id).await?;
    let actual_workers = super::query::query_installed_workers(&config.hives, job_id).await?;

    // Step 2: Compute diff
    let diff = compute_diff(&config.hives, &actual_hives, &actual_workers, opts.remove_extra);

    NARRATE
        .action("sync_diff")
        .job_id(job_id)
        .context(diff.change_count().to_string())
        .human("📊 Found {} changes")
        .emit();

    // Step 3: If dry run, return early
    if opts.dry_run {
        NARRATE
            .action("sync_dry_run")
            .job_id(job_id)
            .human("🔍 Dry run complete (no changes applied)")
            .emit();

        return Ok(SyncReport::from_diff(&diff, start_time.elapsed().as_secs_f64()));
    }

    // Step 4: Apply changes concurrently
    let mut hive_results = Vec::new();

    if !diff.hives_to_install.is_empty() {
        NARRATE
            .action("sync_install")
            .job_id(job_id)
            .context(diff.hives_to_install.len().to_string())
            .human("📦 Installing {} binaries")
            .emit();

        // Spawn concurrent installation tasks
        let mut tasks = Vec::new();
        for hive in &diff.hives_to_install {
            let hive_arc = Arc::new(hive.clone());
            let job_id_clone = job_id.to_string();

            let task = tokio::spawn(async move { sync_single_hive(hive_arc, job_id_clone).await });
            tasks.push(task);
        }

        // Wait for all tasks to complete
        let results = futures::future::join_all(tasks).await;

        // Collect results
        for (idx, result) in results.into_iter().enumerate() {
            let hive = &diff.hives_to_install[idx];
            match result {
                Ok(Ok(hive_result)) => {
                    hive_results.push(hive_result);
                }
                Ok(Err(e)) => {
                    // TEAM-260: Emit error narration so it's visible in SSE stream
                    NARRATE
                        .action("hive_install_failed")
                        .job_id(job_id)
                        .context(&hive.alias)
                        .context(&e.to_string())
                        .human("❌ Hive '{}' installation failed: {}")
                        .error_kind("install_failed")
                        .emit();
                    
                    hive_results.push(HiveResult {
                        alias: hive.alias.clone(),
                        status: "failed".to_string(),
                        workers: Vec::new(),
                        duration_seconds: 0.0,
                        error: Some(e.to_string()),
                    });
                }
                Err(e) => {
                    // TEAM-260: Emit panic narration
                    NARRATE
                        .action("hive_task_panicked")
                        .job_id(job_id)
                        .context(&hive.alias)
                        .context(&format!("{}", e))
                        .human("❌ Hive '{}' task panicked: {}")
                        .error_kind("task_panic")
                        .emit();
                    
                    hive_results.push(HiveResult {
                        alias: hive.alias.clone(),
                        status: "failed".to_string(),
                        workers: Vec::new(),
                        duration_seconds: 0.0,
                        error: Some(format!("Task panicked: {}", e)),
                    });
                }
            }
        }
    }

    // Add already-installed hives to results
    for hive in &diff.hives_already_installed {
        hive_results.push(HiveResult {
            alias: hive.alias.clone(),
            status: "already_installed".to_string(),
            workers: hive.workers.iter().map(|w| w.worker_type.clone()).collect(),
            duration_seconds: 0.0,
            error: None,
        });
    }

    let duration = start_time.elapsed().as_secs_f64();

    // TEAM-260: Emit results for each hive (including errors)
    for result in &hive_results {
        if let Some(error) = &result.error {
            NARRATE
                .action("hive_result_error")
                .job_id(job_id)
                .context(&result.alias)
                .context(error)
                .human("❌ Hive '{}' failed: {}")
                .error_kind("hive_sync_failed")
                .emit();
        } else {
            NARRATE
                .action("hive_result_success")
                .job_id(job_id)
                .context(&result.alias)
                .context(&result.status)
                .human("✅ Hive '{}': {}")
                .emit();
        }
    }

    NARRATE
        .action("sync_complete")
        .job_id(job_id)
        .context(format!("{:.2}", duration))
        .human("✅ Sync completed in {}s")
        .emit();

    Ok(SyncReport {
        hives_installed: diff.hives_to_install.len(),
        hives_already_installed: diff.hives_already_installed.len(),
        workers_installed: diff.workers_to_install.iter().map(|(_, w)| w.len()).sum(),
        workers_already_installed: diff
            .workers_already_installed
            .iter()
            .map(|(_, w)| w.len())
            .sum(),
        hives_removed: diff.hives_to_remove.len(),
        workers_removed: diff.workers_to_remove.iter().map(|(_, w)| w.len()).sum(),
        duration_seconds: duration,
        hive_results,
    })
}

/// Sync a single hive (install hive + workers)
///
/// TEAM-280: Per-hive sync logic
///
/// # Arguments
/// * `hive` - Hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveResult)` - Hive synced successfully
/// * `Err` - Sync failed
pub async fn sync_single_hive(hive: Arc<HiveConfig>, job_id: String) -> Result<HiveResult> {
    let start_time = std::time::Instant::now();

    NARRATE
        .action("sync_hive")
        .job_id(&job_id)
        .context(&hive.alias)
        .human("🔄 Syncing hive '{}'")
        .emit();

    // Install hive + workers
    install_all(Arc::clone(&hive), job_id.clone()).await?;

    // TODO: Start hive if auto_start is enabled
    // if hive.auto_start {
    //     start_hive(&hive, &job_id).await?;
    // }

    let duration = start_time.elapsed().as_secs_f64();

    Ok(HiveResult {
        alias: hive.alias.clone(),
        status: "installed".to_string(),
        workers: hive.workers.iter().map(|w| w.worker_type.clone()).collect(),
        duration_seconds: duration,
        error: None,
    })
}

impl SyncReport {
    /// Create sync report from diff (for dry-run)
    pub fn from_diff(diff: &StateDiff, duration: f64) -> Self {
        Self {
            hives_installed: diff.hives_to_install.len(),
            hives_already_installed: diff.hives_already_installed.len(),
            workers_installed: diff.workers_to_install.iter().map(|(_, w)| w.len()).sum(),
            workers_already_installed: diff
                .workers_already_installed
                .iter()
                .map(|(_, w)| w.len())
                .sum(),
            hives_removed: diff.hives_to_remove.len(),
            workers_removed: diff.workers_to_remove.iter().map(|(_, w)| w.len()).sum(),
            duration_seconds: duration,
            hive_results: Vec::new(),
        }
    }
}
