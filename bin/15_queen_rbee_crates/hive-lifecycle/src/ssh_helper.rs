//! SSH helper utilities for remote hive operations
//!
//! TEAM-256: Migrated from shell commands to russh-based client
//!
//! Provides reusable SSH command execution patterns for remote hive management.

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use queen_rbee_ssh_client::RbeeSSHClient;
use rbee_config::HiveEntry;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Execute SSH command on remote host
///
/// TEAM-256: Rewritten to use russh instead of shell commands
///
/// # Arguments
/// * `hive_config` - Hive configuration with SSH details
/// * `command` - Command to execute on remote host
/// * `job_id` - Job ID for SSE routing
/// * `action` - Action name for narration (max 15 chars, static string)
/// * `description` - Human-readable description
///
/// # Returns
/// * `Ok(String)` - Command output (stdout)
/// * `Err` - SSH command failed
pub async fn ssh_exec(
    hive_config: &HiveEntry,
    command: &str,
    job_id: &str,
    action: &'static str,
    description: &str,
) -> Result<String> {
    NARRATE.action(action).job_id(job_id).context(description).human("🔧 {}").emit();

    // TEAM-256: Connect using russh
    let mut client = match RbeeSSHClient::connect(
        &hive_config.hostname,
        hive_config.ssh_port,
        &hive_config.ssh_user,
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            let error_msg = format!("SSH connection failed: {}", e);
            NARRATE
                .action("ssh_connect_err")
                .job_id(job_id)
                .context(&error_msg)
                .human("❌ {}")
                .error_kind("ssh_connect_failed")
                .emit();
            return Err(e);
        }
    };

    // TEAM-256: Execute command
    let (stdout, stderr, exit_code) = client.exec(command).await?;

    // TEAM-256: Close connection
    client.close().await?;

    if exit_code != 0 {
        let error_msg = if stderr.trim().is_empty() {
            format!("Exit code {}", exit_code)
        } else {
            stderr.trim().to_string()
        };

        NARRATE
            .action("ssh_err")
            .job_id(job_id)
            .context(command)
            .context(&error_msg)
            .human("❌ SSH command '{}' failed: {}")
            .error_kind("ssh_failed")
            .emit();
        return Err(anyhow::anyhow!("SSH command '{}' failed: {}", command, error_msg));
    }

    Ok(stdout)
}

/// Copy file to remote host via SFTP
///
/// TEAM-256: Rewritten to use russh SFTP instead of shell SCP
///
/// # Arguments
/// * `hive_config` - Hive configuration with SSH details
/// * `local_path` - Local file path
/// * `remote_path` - Remote file path
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(())` - File copied successfully
/// * `Err` - SFTP failed
pub async fn scp_copy(
    hive_config: &HiveEntry,
    local_path: &str,
    remote_path: &str,
    job_id: &str,
) -> Result<()> {
    let scp_target = format!("{}@{}:{}", hive_config.ssh_user, hive_config.hostname, remote_path);

    NARRATE
        .action("hive_scp")
        .job_id(job_id)
        .context(&scp_target)
        .human("📤 Copying to {}...")
        .emit();

    // TEAM-256: Connect using russh
    let mut client =
        RbeeSSHClient::connect(&hive_config.hostname, hive_config.ssh_port, &hive_config.ssh_user)
            .await?;

    // TEAM-256: Copy file via SFTP
    client.copy_file(local_path, remote_path).await?;

    // TEAM-256: Close connection
    client.close().await?;

    NARRATE.action("hive_scp").job_id(job_id).human("✅ File copied successfully").emit();

    Ok(())
}

/// Check if hive is remote (not localhost)
pub fn is_remote_hive(hive_config: &HiveEntry) -> bool {
    hive_config.hostname != "127.0.0.1" && hive_config.hostname != "localhost"
}

/// Get remote binary path (defaults to ~/.local/bin/rbee-hive)
pub fn get_remote_binary_path(hive_config: &HiveEntry) -> String {
    hive_config.binary_path.clone().unwrap_or_else(|| "~/.local/bin/rbee-hive".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rbee_config::HiveEntry;

    #[test]
    fn test_is_remote_hive() {
        let localhost = HiveEntry {
            alias: "local".to_string(),
            hostname: "127.0.0.1".to_string(),
            ssh_port: 22,
            ssh_user: "user".to_string(),
            hive_port: 8081,
            binary_path: None,
        };
        assert!(!is_remote_hive(&localhost));

        let remote = HiveEntry {
            alias: "remote".to_string(),
            hostname: "192.168.1.100".to_string(),
            ssh_port: 22,
            ssh_user: "user".to_string(),
            hive_port: 8081,
            binary_path: None,
        };
        assert!(is_remote_hive(&remote));
    }

    #[test]
    fn test_get_remote_binary_path() {
        let default = HiveEntry {
            alias: "test".to_string(),
            hostname: "192.168.1.100".to_string(),
            ssh_port: 22,
            ssh_user: "user".to_string(),
            hive_port: 8081,
            binary_path: None,
        };
        assert_eq!(get_remote_binary_path(&default), "~/.local/bin/rbee-hive");

        let custom = HiveEntry {
            alias: "test".to_string(),
            hostname: "192.168.1.100".to_string(),
            ssh_port: 22,
            ssh_user: "user".to_string(),
            hive_port: 8081,
            binary_path: Some("/custom/path/rbee-hive".to_string()),
        };
        assert_eq!(get_remote_binary_path(&custom), "/custom/path/rbee-hive");
    }
}
