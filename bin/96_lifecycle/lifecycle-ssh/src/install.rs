//! Install daemon binary on remote machine via SCP
//!
//! # Types/Utils Used (from daemon-lifecycle)
//! - utils::ssh::ssh_exec() - Execute SSH commands
//! - utils::ssh::scp_upload() - Upload files via SCP
//! - build::build_daemon() - Build binary (reuses existing function)
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon binary
//! - `ssh_config`: SSH connection details
//! - `local_binary_path`: Optional path to pre-built binary (if None, build it)
//!
//! ## Process
//! 1. Build or locate binary locally
//!    - If `local_binary_path` provided: use it
//!    - Else: call `build_daemon(daemon_name)` (environment-aware: debug or release)
//!
//! 2. Copy binary to remote machine (ONE scp call)
//!    - Use: `scp -P {port} {local_path} {user}@{hostname}:~/.local/bin/{daemon_name}`
//!    - Create ~/.local/bin directory if needed (via SSH)
//!
//! 3. Make binary executable (ONE ssh call)
//!    - Use: `chmod +x ~/.local/bin/{daemon_name}`
//!
//! ## SSH/SCP Calls
//! - Total: 1 SCP call + 1 SSH call
//!
//! ## Error Handling
//! - Local binary not found
//! - SCP failed (connection, permissions, disk space)
//! - chmod failed
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::{install_daemon, SshConfig};
//! use std::path::PathBuf;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
//!
//! // Option 1: Build and install
//! install_daemon("rbee-hive", ssh.clone(), None).await?;
//!
//! // Option 2: Install pre-built binary
//! // TEAM-341: Path depends on build mode (debug or release)
//! let binary = PathBuf::from("target/release/rbee-hive");  // or target/debug/rbee-hive
//! install_daemon("rbee-hive", ssh, Some(binary)).await?;
//! # Ok(())
//! # }
//! ```

use crate::build::{build_daemon, BuildConfig};
use crate::utils::binary::check_binary_installed;
use crate::utils::ssh::{scp_upload, ssh_exec};
use crate::SshConfig;
use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::with_job_id;
use std::path::PathBuf;
use timeout_enforcer::with_timeout;

/// Configuration for remote daemon installation
///
/// TEAM-330: Includes optional job_id for SSE narration routing
///
/// # Example
/// ```rust,ignore
/// use remote_daemon_lifecycle::{InstallConfig, SshConfig};
///
/// let ssh = SshConfig::new("192.168.1.100".to_string(), "vince".to_string(), 22);
/// let config = InstallConfig {
///     daemon_name: "llm-worker-rbee".to_string(),
///     ssh_config: ssh,
///     local_binary_path: None,
///     job_id: Some("job-123".to_string()),  // For SSE routing
/// };
/// ```
#[derive(Debug, Clone)]
pub struct InstallConfig {
    /// Name of the daemon binary
    pub daemon_name: String,

    /// SSH connection configuration
    pub ssh_config: SshConfig,

    /// Optional path to pre-built binary (if None, will build from source)
    pub local_binary_path: Option<PathBuf>,

    /// Optional job ID for SSE narration routing
    /// When set, all narration (including timeout countdown) goes through SSE
    pub job_id: Option<String>,
}

/// Install daemon binary on remote machine
///
/// TEAM-330: Enforces 5-minute timeout for entire installation process
///
/// # Implementation
/// 1. Build or locate binary locally (daemon-lifecycle, environment-aware)
/// 2. Create remote directory via SSH
/// 3. Copy binary via SCP
/// 4. Make executable via SSH
/// 5. Verify installation
///
/// # Build Mode (TEAM-341)
/// - Debug parent binary → builds child daemons in debug mode (target/debug/)
/// - Release parent binary → builds child daemons in release mode (target/release/)
/// - This ensures dev builds can proxy to Vite dev servers
///
/// # Timeout Strategy
/// - Total timeout: 5 minutes (covers build + transfer + setup)
/// - Build can take 2-3 minutes for large binaries
/// - Transfer depends on binary size and network speed
/// - SSH commands are fast (<1s each)
///
/// # Job ID Support (TEAM-330)
/// When called from hive daemon managing worker lifecycle, set job_id in config:
///
/// ```rust,ignore
/// let config = InstallConfig {
///     daemon_name: "llm-worker-rbee".to_string(),
///     ssh_config,
///     local_binary_path: None,
///     job_id: Some(job_id),  // ← Routes narration + countdown through SSE!
/// };
/// install_daemon(config).await?;
/// ```
///
/// The #[with_job_id] macro automatically wraps the function in NarrationContext,
/// routing ALL narration (including timeout countdown) through SSE!
#[with_job_id(config_param = "install_config")]
#[with_timeout(secs = 300, label = "Install daemon")]
pub async fn install_daemon(install_config: InstallConfig) -> Result<()> {
    let daemon_name = &install_config.daemon_name;
    let ssh_config = &install_config.ssh_config;

    n!(
        "install_start",
        "📦 Installing {} on {}@{}",
        daemon_name,
        ssh_config.user,
        ssh_config.hostname
    );

    // Step 0: Check if already installed
    // TEAM-338: Use check_binary_installed utility
    if check_binary_installed(daemon_name, ssh_config).await {
        n!(
            "already_installed",
            "⚠️  {} is already installed on {}@{}",
            daemon_name,
            ssh_config.user,
            ssh_config.hostname
        );
        anyhow::bail!("{} is already installed. Use rebuild to update.", daemon_name);
    }

    // Step 1: Build or locate binary locally
    let binary_path = if let Some(path) = install_config.local_binary_path {
        n!("verify_binary", "🔍 Verifying pre-built binary at: {}", path.display());
        if !path.exists() {
            n!("binary_not_found", "❌ Binary not found at: {}", path.display());
            anyhow::bail!("Binary not found at: {}", path.display());
        }
        n!("using_binary", "📦 Using pre-built binary: {}", path.display());
        path
    } else {
        // Use build_daemon() instead of duplicating build code
        let build_config = BuildConfig {
            daemon_name: daemon_name.to_string(),
            target: None,
            job_id: install_config.job_id.clone(),
        };

        build_daemon(build_config).await?
    };

    // Step 2: Create remote directory
    n!("create_dir", "📁 Creating ~/.local/bin on remote");
    let create_dir_cmd = format!("mkdir -p ~/.local/bin");
    ssh_exec(&ssh_config, &create_dir_cmd).await.context("Failed to create remote directory")?;

    // Step 3: Copy binary via SCP
    let remote_path = format!("~/.local/bin/{}", daemon_name);
    n!(
        "copying",
        "📤 Copying {} to {}@{}:{}",
        daemon_name,
        ssh_config.user,
        ssh_config.hostname,
        remote_path
    );

    scp_upload(&ssh_config, &binary_path, &remote_path)
        .await
        .context("Failed to copy binary to remote")?;

    // Step 4: Make executable
    n!("chmod", "🔐 Making binary executable");
    let chmod_cmd = format!("chmod +x ~/.local/bin/{}", daemon_name);
    ssh_exec(&ssh_config, &chmod_cmd).await.context("Failed to make binary executable")?;

    // Step 5: Verify installation
    n!("verify", "✅ Verifying installation");
    let verify_cmd = format!("test -x ~/.local/bin/{} && echo 'OK'", daemon_name);
    let output =
        ssh_exec(&ssh_config, &verify_cmd).await.context("Failed to verify installation")?;

    if !output.trim().contains("OK") {
        anyhow::bail!("Installation verification failed");
    }

    n!(
        "install_complete",
        "🎉 {} installed successfully on {}@{}",
        daemon_name,
        ssh_config.user,
        ssh_config.hostname
    );

    Ok(())
}

// TEAM-330: ssh_exec() and scp_upload() are in utils/ssh.rs
// See: src/utils/ssh.rs
