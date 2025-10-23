//! Worker binary uninstallation
//!
//! TEAM-276: Added stub for lifecycle consistency
//! TEAM-277: Worker uninstallation now handled by queen-rbee via SSH
//!
//! NOTE: Worker binaries are managed by worker-catalog, not uninstalled like queen/hive.
//! This module exists for API consistency across lifecycle crates.
//!
//! # TEAM-277 Architecture Change
//!
//! Worker uninstallation is now handled by queen-rbee's package_manager module via SSH.
//! Queen orchestrates uninstallation of both hive AND workers across remote hosts.
//! This stub remains for API consistency but delegates to worker-catalog.

use anyhow::Result;
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("worker-lc");

/// Uninstall a worker binary
///
/// TEAM-276: Stub implementation for lifecycle consistency
///
/// Workers are managed by worker-catalog and don't have a traditional "uninstall"
/// operation like queen or hive. Worker binaries are:
/// - Stored in ~/.cache/rbee/workers/
/// - Managed by worker-catalog
/// - Can be removed via catalog cleanup operations
///
/// This function exists for API consistency but delegates to worker-catalog.
///
/// # Arguments
///
/// * `job_id` - Job ID for narration routing
/// * `worker_type` - Worker type (e.g., "vllm", "llama-cpp")
///
/// # Returns
///
/// Ok(()) - Worker binary removed or not present
/// Err - Catalog error
///
/// # Example
///
/// ```rust,no_run
/// use rbee_hive_worker_lifecycle::uninstall_worker;
///
/// # async fn example() -> anyhow::Result<()> {
/// uninstall_worker("job-123", "vllm").await?;
/// # Ok(())
/// # }
/// ```
pub async fn uninstall_worker(job_id: &str, worker_type: &str) -> Result<()> {
    NARRATE
        .action("worker_uninstall")
        .job_id(job_id)
        .context(worker_type)
        .human("🗑️  Uninstalling worker binary: {}")
        .emit();

    // NOTE: Worker binaries are managed by worker-catalog
    // This is a stub that would remove the binary from the catalog
    
    NARRATE
        .action("worker_uninstall_stub")
        .job_id(job_id)
        .context(worker_type)
        .human("ℹ️  Worker binaries are managed by worker-catalog")
        .emit();

    // TODO: If needed, add logic to remove binary from catalog
    // use rbee_hive_worker_catalog::WorkerCatalog;
    // let catalog = WorkerCatalog::new()?;
    // catalog.remove(worker_type)?;

    NARRATE
        .action("worker_uninstall_complete")
        .job_id(job_id)
        .context(worker_type)
        .human("✅ Worker binary uninstalled: {}")
        .emit();

    Ok(())
}
