// TEAM-213: Uninstall hive configuration
// TEAM-220: Investigated - Cache cleanup + future enhancements documented
// TEAM-256: Fixed idempotency - reload config from disk to detect already-uninstalled state
// TEAM-257: Fixed action name length violations (15-char limit)

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveUninstallRequest, HiveUninstallResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Uninstall hive configuration
///
/// Removes cached capabilities for a hive. The hive configuration in hives.conf
/// must be manually removed by editing the file.
///
/// # Idempotency
/// Running uninstall multiple times is safe:
/// - First run: Removes capabilities cache, shows "uninstalled successfully"
/// - Subsequent runs: Shows "already uninstalled (no cached capabilities)"
///
/// # Pre-flight Requirements
/// The hive should be stopped before uninstalling:
/// ```bash
/// ./rbee hive stop --alias <name>
/// ./rbee hive uninstall --alias <name>
/// ```
///
/// # Arguments
/// * `request` - Uninstall request with hive alias
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveUninstallResponse)` - Success message with status
/// * `Err` - If hive doesn't exist in configuration
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

    // TEAM-196: Remove from capabilities cache if present
    // TEAM-256: Reload config from disk to check current state (daemon may have stale config)
    let fresh_config = match rbee_config::RbeeConfig::load() {
        Ok(cfg) => cfg,
        Err(_) => (*config).clone(),
    };
    let had_capabilities = fresh_config.capabilities.contains(alias);
    
    if had_capabilities {
        // TEAM-257: Action names must be ≤15 chars (narration system limit)
        NARRATE
            .action("hive_cache_rm")  // Was: hive_cache_cleanup (18 chars)
            .job_id(job_id)
            .human("🗑️  Removing from capabilities cache...")
            .emit();

        let mut config_mut = (*config).clone();
        config_mut.capabilities.remove(alias);
        if let Err(e) = config_mut.capabilities.save() {
            NARRATE
                .action("hive_cache_err")  // Was: hive_cache_error (16 chars)
                .job_id(job_id)
                .context(e.to_string())
                .human("⚠️  Failed to save capabilities cache: {}")
                .emit();
        } else {
            NARRATE
                .action("hive_cache_ok")  // Was: hive_cache_removed (18 chars)
                .job_id(job_id)
                .human("✅ Removed from capabilities cache")
                .emit();
        }
    }

    // TEAM-256/257: Show appropriate completion message based on whether capabilities were removed
    if had_capabilities {
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
    } else {
        NARRATE
            .action("hive_complete")
            .job_id(job_id)
            .context(alias)
            .human(
                "ℹ️  Hive '{}' already uninstalled (no cached capabilities).\n\
                 \n\
                 To remove from config, edit ~/.config/rbee/hives.conf",
            )
            .emit();
    }

    let message = if had_capabilities {
        format!("Hive '{}' uninstalled successfully", alias)
    } else {
        format!("Hive '{}' already uninstalled (no cached capabilities)", alias)
    };
    
    Ok(HiveUninstallResponse {
        success: true,
        message,
    })
}
