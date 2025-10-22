// TEAM-213: Install hive configuration
// TEAM-220: Investigated - Binary resolution + localhost-only documented
// TEAM-256: Migrated shell SSH commands to russh-based helpers

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::ssh_helper::{scp_copy, ssh_exec};
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
            .context(&format!("{}@{}:{}", user, host, ssh_port))
            .human("🌐 Remote installation: {}")
            .emit();

        // STEP 1: Find local binary
        let local_binary = if let Some(provided_path) = &hive_config.binary_path {
            NARRATE
                .action("hive_binary")
                .job_id(job_id)
                .context(provided_path)
                .human("📁 Using provided binary path: {}")
                .emit();

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

            provided_path.clone()
        } else {
            // Find binary in target directory
            NARRATE
                .action("hive_binary")
                .job_id(job_id)
                .human("🔍 Looking for rbee-hive binary...")
                .emit();

            let debug_path = std::path::PathBuf::from("target/debug/rbee-hive");
            let release_path = std::path::PathBuf::from("target/release/rbee-hive");

            if debug_path.exists() {
                NARRATE
                    .action("hive_binary")
                    .job_id(job_id)
                    .context(&debug_path.display().to_string())
                    .human("✅ Found binary at: {}")
                    .emit();
                debug_path.display().to_string()
            } else if release_path.exists() {
                NARRATE
                    .action("hive_binary")
                    .job_id(job_id)
                    .context(&release_path.display().to_string())
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
                           cargo build --bin rbee-hive",
                    )
                    .emit();
                return Err(anyhow::anyhow!(
                    "rbee-hive binary not found. Build it with: cargo build --bin rbee-hive"
                ));
            }
        };

        // STEP 2: Ensure remote directory exists
        // TEAM-256: Use ssh_exec instead of shell command
        ssh_exec(
            hive_config,
            "mkdir -p ~/.local/bin",
            job_id,
            "hive_mkdir",
            "Creating remote directory",
        )
        .await?;

        // STEP 3: Copy binary to remote host via SFTP
        // TEAM-256: Use scp_copy (now SFTP-based) instead of shell command
        let remote_path = "~/.local/bin/rbee-hive";
        scp_copy(hive_config, &local_binary, remote_path, job_id).await?;

        // STEP 4: Make binary executable
        // TEAM-256: Use ssh_exec instead of shell command
        ssh_exec(
            hive_config,
            &format!("chmod +x {}", remote_path),
            job_id,
            "hive_chmod",
            "Making binary executable",
        )
        .await?;

        // STEP 5: Verify installation
        // TEAM-256: Use ssh_exec instead of shell command
        let version = ssh_exec(
            hive_config,
            &format!("{} --version", remote_path),
            job_id,
            "hive_verify",
            "Verifying installation",
        )
        .await?;
        NARRATE
            .action("hive_verified")
            .job_id(job_id)
            .context(&version.trim().to_string())
            .human("✅ Verified: {}")
            .emit();

        NARRATE
            .action("hive_complete")
            .job_id(job_id)
            .context(alias)
            .context(&hive_config.hive_port.to_string())
            .context(remote_path)
            .human(
                "✅ Hive '{0}' installed successfully!\n\
                 \n\
                 Configuration:\n\
                 - Host: {}@{}\n\
                 - Port: {1}\n\
                 - Binary: {2}\n\
                 \n\
                 To start the hive:\n\
                 \n\
                   ./rbee hive start -a {0}",
            )
            .emit();

        Ok(HiveInstallResponse {
            success: true,
            message: format!("Hive '{}' installed successfully", alias),
            binary_path: Some(remote_path.to_string()),
        })
    } else {
        // LOCALHOST INSTALLATION
        NARRATE.action("hive_mode").job_id(job_id).human("🏠 Localhost installation").emit();

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

            NARRATE.action("hive_binary").job_id(job_id).human("✅ Binary found").emit();

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
