// TEAM-213: Install hive configuration

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveInstallRequest, HiveInstallResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Install hive configuration
///
/// COPIED FROM: job_router.rs lines 280-401
///
/// Steps:
/// 1. Validate hive exists in config
/// 2. Determine if localhost or remote
/// 3. For localhost: Find or verify binary path
/// 4. Display success message
///
/// NOTE: Remote SSH installation not yet implemented
///
/// # Arguments
/// * `request` - Install request with alias
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveInstallResponse)` - Success with binary path
/// * `Err` - Configuration error or binary not found
pub async fn execute_hive_install(
    request: HiveInstallRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveInstallResponse> {
    let alias = &request.alias;
    let hive_config = validate_hive_exists(&config, alias)?;

    NARRATE
        .action("hive_install")
        .job_id(job_id)
        .context(alias)
        .human("🔧 Installing hive '{}'")
        .emit();

    // STEP 1: Determine if this is localhost or remote installation
    let is_remote = hive_config.hostname != "127.0.0.1" && hive_config.hostname != "localhost";

    if is_remote {
        // REMOTE INSTALLATION
        let host = &hive_config.hostname;
        let ssh_port = hive_config.ssh_port;
        let user = &hive_config.ssh_user;

        NARRATE
            .action("hive_mode")
            .job_id(job_id)
            .context(format!("{}@{}:{}", user, host, ssh_port))
            .human("🌐 Remote installation: {}")
            .emit();

        // TODO: Implement remote SSH installation
        NARRATE
            .action("hive_not_impl")
            .job_id(job_id)
            .human(
                "❌ Remote SSH installation not yet implemented.\n\
                   \n\
                   Currently only localhost installation is supported.",
            )
            .emit();
        return Err(anyhow::anyhow!("Remote installation not yet implemented"));
    } else {
        // LOCALHOST INSTALLATION
        NARRATE
            .action("hive_mode")
            .job_id(job_id)
            .human("🏠 Localhost installation")
            .emit();

        // STEP 2: Find or build the rbee-hive binary
        let binary = if let Some(provided_path) = &hive_config.binary_path {
            NARRATE
                .action("hive_binary")
                .job_id(job_id)
                .context(provided_path)
                .human("📁 Using provided binary path: {}")
                .emit();

            // Verify binary exists
            let path = std::path::Path::new(provided_path);
            if !path.exists() {
                NARRATE
                    .action("hive_bin_err")
                    .job_id(job_id)
                    .context(provided_path)
                    .human("❌ Binary not found at: {}")
                    .emit();
                return Err(anyhow::anyhow!("Binary not found: {}", provided_path));
            }

            NARRATE
                .action("hive_binary")
                .job_id(job_id)
                .human("✅ Binary found")
                .emit();

            provided_path.clone()
        } else {
            // Find binary in target directory
            NARRATE
                .action("hive_binary")
                .job_id(job_id)
                .human("🔍 Looking for rbee-hive binary in target/debug...")
                .emit();

            let debug_path = std::path::PathBuf::from("target/debug/rbee-hive");
            let release_path = std::path::PathBuf::from("target/release/rbee-hive");

            if debug_path.exists() {
                NARRATE
                    .action("hive_binary")
                    .job_id(job_id)
                    .context(debug_path.display().to_string())
                    .human("✅ Found binary at: {}")
                    .emit();
                debug_path.display().to_string()
            } else if release_path.exists() {
                NARRATE
                    .action("hive_binary")
                    .job_id(job_id)
                    .context(release_path.display().to_string())
                    .human("✅ Found binary at: {}")
                    .emit();
                release_path.display().to_string()
            } else {
                NARRATE
                    .action("hive_bin_err")
                    .job_id(job_id)
                    .human(
                        "❌ rbee-hive binary not found.\n\
                         \n\
                         Please build it first:\n\
                         \n\
                           cargo build --bin rbee-hive\n\
                         \n\
                         Or provide a binary path:\n\
                         \n\
                           ./rbee hive install --binary-path /path/to/rbee-hive",
                    )
                    .emit();
                return Err(anyhow::anyhow!(
                    "rbee-hive binary not found. Build it with: cargo build --bin rbee-hive"
                ));
            }
        };

        NARRATE
            .action("hive_complete")
            .job_id(job_id)
            .context(alias)
            .context(hive_config.hive_port.to_string())
            .context(&binary)
            .human(
                "✅ Hive '{0}' configured successfully!\n\
                 \n\
                 Configuration:\n\
                 - Host: localhost\n\
                 - Port: {1}\n\
                 - Binary: {2}\n\
                 \n\
                 To start the hive:\n\
                 \n\
                   ./rbee hive start --host {0}",
            )
            .emit();

        Ok(HiveInstallResponse {
            success: true,
            message: format!("Hive '{}' configured successfully", alias),
            binary_path: Some(binary),
        })
    }
}
