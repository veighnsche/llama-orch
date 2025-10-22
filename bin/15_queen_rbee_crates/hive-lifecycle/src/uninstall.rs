// TEAM-213: Uninstall hive configuration

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveUninstallRequest, HiveUninstallResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Uninstall hive configuration
///
/// COPIED FROM: job_router.rs lines 402-484
///
/// Steps:
/// 1. Validate hive exists
/// 2. Remove from capabilities cache if present
/// 3. Display success message
///
/// NOTE: Pre-flight check (hive must be stopped) is documented but not enforced.
/// User must manually stop the hive first: ./rbee hive stop
///
/// # Arguments
/// * `request` - Uninstall request with alias
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveUninstallResponse)` - Success message
/// * `Err` - Configuration error
pub async fn execute_hive_uninstall(
    request: HiveUninstallRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveUninstallResponse> {
    let alias = &request.alias;
    let _hive_config = validate_hive_exists(&config, alias)?;

    NARRATE
        .action("hive_uninstall")
        .job_id(job_id)
        .context(alias)
        .human("🗑️  Uninstalling hive '{}'")
        .emit();

    // TEAM-196: Remove from capabilities cache
    if config.capabilities.contains(alias) {
        NARRATE
            .action("hive_cache_cleanup")
            .job_id(job_id)
            .human("🗑️  Removing from capabilities cache...")
            .emit();

        let mut config_mut = (*config).clone();
        config_mut.capabilities.remove(alias);
        if let Err(e) = config_mut.capabilities.save() {
            NARRATE
                .action("hive_cache_error")
                .job_id(job_id)
                .context(e.to_string())
                .human("⚠️  Failed to save capabilities cache: {}")
                .emit();
        } else {
            NARRATE
                .action("hive_cache_removed")
                .job_id(job_id)
                .human("✅ Removed from capabilities cache")
                .emit();
        }
    }

    NARRATE
        .action("hive_complete")
        .job_id(job_id)
        .context(alias)
        .human(
            "✅ Hive '{}' uninstalled successfully.\n\
             \n\
             To remove from config, edit ~/.config/rbee/hives.conf",
        )
        .emit();

    // TEAM-189: Documented pre-flight check requirements
    //
    // CURRENT IMPLEMENTATION:
    // 1. Check if hive exists in catalog → error if not found
    // 2. Check if hive is running (health endpoint) → error if running
    // 3. Remove from catalog
    //
    // IMPORTANT: User must manually stop the hive first:
    //   ./rbee hive stop
    // This ensures clean shutdown of hive and all child workers.
    //
    // FUTURE ENHANCEMENTS (for TEAM-190+):
    //
    // CATALOG-ONLY MODE (catalog_only=true):
    // - Used for unreachable remote hives (network issues, host down)
    // - Simply remove HiveRecord from catalog
    // - No SSH connection or health check
    // - Warn user about orphaned processes on remote host
    //
    // ADDITIONAL CLEANUP OPTIONS (flags):
    // - --remove-workers: Delete worker binaries (requires hive stopped)
    // - --remove-models: Delete model files (requires hive stopped)
    // - --remove-binary: Delete hive binary itself
    //
    // LOCALHOST FULL CLEANUP:
    // 1. Verify hive stopped (health check fails)
    // 2. Verify no worker processes running (pgrep llm-worker)
    // 3. Optional: Remove worker binaries if --remove-workers
    // 4. Optional: Remove models if --remove-models
    // 5. Optional: Remove hive binary if --remove-binary
    // 6. Remove from catalog
    //
    // REMOTE SSH FULL CLEANUP:
    // 1. Run SshTest to verify connectivity
    // 2. Verify hive stopped (SSH: curl health or pgrep)
    // 3. Verify no worker processes (SSH: pgrep llm-worker)
    // 4. Optional: Remove files via SSH (rm commands)
    // 5. Remove from catalog

    Ok(HiveUninstallResponse {
        success: true,
        message: format!("Hive '{}' uninstalled successfully", alias),
    })
}
