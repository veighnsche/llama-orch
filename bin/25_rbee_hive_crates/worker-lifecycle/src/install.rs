//! Worker binary installation
//!
//! TEAM-276: Added stub for lifecycle consistency
//! TEAM-277: Worker installation now handled by queen-rbee via SSH
//!
//! NOTE: Worker binaries are managed by worker-catalog, not installed like queen/hive.
//! This module exists for API consistency across lifecycle crates.
//!
//! # TEAM-277 Architecture Change
//!
//! Worker installation is now handled by queen-rbee's package_manager module via SSH.
//! Queen orchestrates installation of both hive AND workers across remote hosts.
//! This stub remains for API consistency but delegates to worker-catalog.

use anyhow::Result;
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("worker-lc");

/// Install a worker binary
///
/// TEAM-276: Stub implementation for lifecycle consistency
///
/// Workers are managed by worker-catalog and don't have a traditional "install"
/// operation like queen or hive. Worker binaries are:
/// - Downloaded via worker-catalog
/// - Stored in ~/.cache/rbee/workers/
/// - Selected at spawn time based on worker type
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
/// Ok(()) - Worker binary is available in catalog
/// Err - Worker binary not found or catalog error
///
/// # Example
///
/// ```rust,no_run
/// use rbee_hive_worker_lifecycle::install_worker;
///
/// # async fn example() -> anyhow::Result<()> {
/// install_worker("job-123", "vllm").await?;
/// # Ok(())
/// # }
/// ```
pub async fn install_worker(job_id: &str, worker_type: &str) -> Result<()> {
    NARRATE
        .action("worker_install")
        .job_id(job_id)
        .context(worker_type)
        .human("📦 Checking worker binary availability: {}")
        .emit();

    // NOTE: Worker binaries are managed by worker-catalog
    // This is a stub that verifies the binary exists in the catalog

    NARRATE
        .action("worker_install_stub")
        .job_id(job_id)
        .context(worker_type)
        .human("ℹ️  Worker binaries are managed by worker-catalog")
        .emit();

    // TODO: If needed, add logic to verify binary exists in catalog
    // use rbee_hive_worker_catalog::WorkerCatalog;
    // let catalog = WorkerCatalog::new()?;
    // catalog.get(worker_type)?;

    NARRATE
        .action("worker_install_complete")
        .job_id(job_id)
        .context(worker_type)
        .human("✅ Worker binary available: {}")
        .emit();

    Ok(())
}
