//! Check rbee-hive status on remote host
//!
//! TEAM-290: Remote hive status check via SSH

use anyhow::Result;

use ssh_config::SshClient; // TEAM-314: Use shared SSH client

/// Check if rbee-hive is running on remote host
///
/// # Arguments
/// * `host` - SSH host alias
///
/// # Returns
/// * `Ok(true)` - Hive is running
/// * `Ok(false)` - Hive is not running
pub async fn is_hive_running(host: &str) -> Result<bool> {
    let client = SshClient::connect(host).await?;
    
    let is_running = client
        .execute("pgrep -f rbee-hive")
        .await
        .is_ok();

    Ok(is_running)
}

/// Get hive status on remote host
///
/// # Arguments
/// * `host` - SSH host alias
///
/// # Returns
/// * `Ok(String)` - Status message
pub async fn hive_status(host: &str) -> Result<String> {
    let is_running = is_hive_running(host).await?;
    
    if is_running {
        Ok(format!("Hive is running on '{}'", host))
    } else {
        Ok(format!("Hive is NOT running on '{}'", host))
    }
}
