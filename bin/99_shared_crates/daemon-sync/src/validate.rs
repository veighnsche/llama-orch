//! Config validation without applying changes
//!
//! Created by: TEAM-280
//!
//! Validate config file syntax, SSH connectivity, hostnames, etc.

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use queen_rbee_ssh_client::RbeeSSHClient;
use rbee_config::declarative::HivesConfig;
use serde::{Deserialize, Serialize};

const NARRATE: NarrationFactory = NarrationFactory::new("pkg-valid");

/// Validation report
///
/// TEAM-280: Result of validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Overall status (valid, invalid)
    pub status: String,

    /// Warnings (non-fatal issues)
    pub warnings: Vec<ValidationIssue>,

    /// Errors (fatal issues)
    pub errors: Vec<ValidationIssue>,
}

/// Single validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Hive alias (if applicable)
    pub hive: Option<String>,

    /// Issue type (ssh_key_missing, invalid_hostname, etc.)
    pub issue_type: String,

    /// Human-readable message
    pub message: String,
}

/// Validate config
///
/// TEAM-280: Main validation entry point
///
/// # Checks
/// - Config syntax (TOML parsing)
/// - Unique hive aliases
/// - Valid hostnames
/// - Valid ports
/// - SSH connectivity (optional)
/// - Worker types exist (optional)
///
/// # Arguments
/// * `config` - Config to validate
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(ValidationReport)` - Validation completed
/// * `Err` - Validation failed
pub async fn validate_config(config: HivesConfig, job_id: &str) -> Result<ValidationReport> {
    NARRATE
        .action("validate_start")
        .job_id(job_id)
        .context(&config.hives.len().to_string())
        .human("🔍 Validating config with {} hives")
        .emit();

    let mut warnings = Vec::new();
    let mut errors = Vec::new();

    // Use built-in validation
    if let Err(e) = config.validate() {
        errors.push(ValidationIssue {
            hive: None,
            issue_type: "config_invalid".to_string(),
            message: e.to_string(),
        });
    }

    // Check for empty config
    if config.hives.is_empty() {
        warnings.push(ValidationIssue {
            hive: None,
            issue_type: "empty_config".to_string(),
            message: "Config contains no hives".to_string(),
        });
    }

    // Check each hive
    for hive in &config.hives {
        // Check hostname
        if hive.hostname.is_empty() {
            errors.push(ValidationIssue {
                hive: Some(hive.alias.clone()),
                issue_type: "invalid_hostname".to_string(),
                message: format!("Hive '{}' has empty hostname", hive.alias),
            });
        }

        // Check SSH user
        if hive.ssh_user.is_empty() {
            errors.push(ValidationIssue {
                hive: Some(hive.alias.clone()),
                issue_type: "invalid_ssh_user".to_string(),
                message: format!("Hive '{}' has empty SSH user", hive.alias),
            });
        }

        // Check ports
        if hive.ssh_port == 0 {
            errors.push(ValidationIssue {
                hive: Some(hive.alias.clone()),
                issue_type: "invalid_port".to_string(),
                message: format!("Hive '{}' has invalid SSH port", hive.alias),
            });
        }

        if hive.hive_port == 0 {
            errors.push(ValidationIssue {
                hive: Some(hive.alias.clone()),
                issue_type: "invalid_port".to_string(),
                message: format!("Hive '{}' has invalid hive port", hive.alias),
            });
        }

        // Check workers
        if hive.workers.is_empty() {
            warnings.push(ValidationIssue {
                hive: Some(hive.alias.clone()),
                issue_type: "no_workers".to_string(),
                message: format!("Hive '{}' has no workers configured", hive.alias),
            });
        }

        for worker in &hive.workers {
            if worker.worker_type.is_empty() {
                errors.push(ValidationIssue {
                    hive: Some(hive.alias.clone()),
                    issue_type: "invalid_worker_type".to_string(),
                    message: format!("Hive '{}' has worker with empty type", hive.alias),
                });
            }

            if worker.version.is_empty() {
                errors.push(ValidationIssue {
                    hive: Some(hive.alias.clone()),
                    issue_type: "invalid_worker_version".to_string(),
                    message: format!(
                        "Hive '{}' has worker '{}' with empty version",
                        hive.alias, worker.worker_type
                    ),
                });
            }
        }

        // TEAM-280: SSH connectivity check
        if let Err(e) = check_ssh_connectivity(hive, job_id).await {
            warnings.push(ValidationIssue {
                hive: Some(hive.alias.clone()),
                issue_type: "ssh_connectivity".to_string(),
                message: format!("SSH connectivity check failed for '{}': {}", hive.alias, e),
            });
        }

        // TEAM-280: Worker type existence check
        for worker in &hive.workers {
            if let Err(e) = check_worker_binary_exists(hive, &worker.worker_type, job_id).await {
                warnings.push(ValidationIssue {
                    hive: Some(hive.alias.clone()),
                    issue_type: "worker_binary_missing".to_string(),
                    message: format!(
                        "Worker binary '{}' not found on '{}': {}",
                        worker.worker_type, hive.alias, e
                    ),
                });
            }
        }
    }

    let status = if errors.is_empty() { "valid" } else { "invalid" };

    NARRATE
        .action("validate_result")
        .job_id(job_id)
        .context(status)
        .context(&errors.len().to_string())
        .context(&warnings.len().to_string())
        .human("📊 Validation {}: {} errors, {} warnings")
        .emit();

    Ok(ValidationReport {
        status: status.to_string(),
        warnings,
        errors,
    })
}

/// Check SSH connectivity to a hive
///
/// TEAM-280: Verify we can connect via SSH
async fn check_ssh_connectivity(
    hive: &rbee_config::declarative::HiveConfig,
    job_id: &str,
) -> Result<()> {
    NARRATE
        .action("ssh_check")
        .job_id(job_id)
        .context(&hive.alias)
        .human("🔌 Checking SSH connectivity to '{}'")
        .emit();

    // Skip localhost
    if hive.hostname == "127.0.0.1" || hive.hostname == "localhost" {
        return Ok(());
    }

    // Try to connect
    let mut client = RbeeSSHClient::connect(&hive.hostname, hive.ssh_port, &hive.ssh_user)
        .await
        .map_err(|e| anyhow::anyhow!("Connection failed: {}", e))?;

    // Run a simple command to verify connectivity
    let (stdout, stderr, exit_code) = client.exec("echo 'SSH OK'").await?;

    client.close().await?;

    if exit_code != 0 {
        return Err(anyhow::anyhow!("Command failed: {}", stderr));
    }

    if !stdout.contains("SSH OK") {
        return Err(anyhow::anyhow!("Unexpected output: {}", stdout));
    }

    NARRATE
        .action("ssh_ok")
        .job_id(job_id)
        .context(&hive.alias)
        .human("✅ SSH connectivity OK for '{}'")
        .emit();

    Ok(())
}

/// Check if worker binary exists on remote host
///
/// TEAM-280: Verify worker binary is installed
async fn check_worker_binary_exists(
    hive: &rbee_config::declarative::HiveConfig,
    worker_type: &str,
    job_id: &str,
) -> Result<()> {
    NARRATE
        .action("worker_check")
        .job_id(job_id)
        .context(&hive.alias)
        .context(worker_type)
        .human("🔍 Checking worker '{}' on '{}'")
        .emit();

    // Skip localhost
    if hive.hostname == "127.0.0.1" || hive.hostname == "localhost" {
        // For localhost, check local filesystem
        let binary_name = format!("rbee-worker-{}", worker_type);
        let possible_paths = vec![
            format!("~/.local/share/rbee/workers/{}", binary_name),
            format!("/usr/local/bin/{}", binary_name),
            format!("./target/debug/{}", binary_name),
            format!("./target/release/{}", binary_name),
        ];

        for path in possible_paths {
            let expanded_path = shellexpand::tilde(&path).to_string();
            if std::path::Path::new(&expanded_path).exists() {
                NARRATE
                    .action("worker_found")
                    .job_id(job_id)
                    .context(worker_type)
                    .context(&hive.alias)
                    .human("✅ Worker '{}' found on '{}'")
                    .emit();
                return Ok(());
            }
        }

        return Err(anyhow::anyhow!("Binary not found in any standard location"));
    }

    // For remote hosts, check via SSH
    let mut client = RbeeSSHClient::connect(&hive.hostname, hive.ssh_port, &hive.ssh_user)
        .await
        .map_err(|e| anyhow::anyhow!("SSH connection failed: {}", e))?;

    let binary_name = format!("rbee-worker-{}", worker_type);
    
    // Check multiple possible locations
    let check_cmd = format!(
        "test -f ~/.local/share/rbee/workers/{} || test -f /usr/local/bin/{} || which {}",
        binary_name, binary_name, binary_name
    );

    let (_, stderr, exit_code) = client.exec(&check_cmd).await?;

    client.close().await?;

    if exit_code != 0 {
        return Err(anyhow::anyhow!("Binary not found: {}", stderr));
    }

    NARRATE
        .action("worker_found")
        .job_id(job_id)
        .context(worker_type)
        .context(&hive.alias)
        .human("✅ Worker '{}' found on '{}'")
        .emit();

    Ok(())
}
