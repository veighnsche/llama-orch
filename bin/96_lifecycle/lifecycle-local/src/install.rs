//! Install daemon binary LOCALLY
//!
//! TEAM-358: Refactored to remove SSH code (lifecycle-local = LOCAL only)
//!
//! # Types/Utils Used
//! - utils::local::local_copy() - Copy files locally
//! - build::build_daemon() - Build binary (reuses existing function)
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon binary
//! - `local_binary_path`: Optional path to pre-built binary (if None, build it)
//!
//! ## Process
//! 1. Build or locate binary locally
//!    - If `local_binary_path` provided: use it
//!    - Else: call `build_daemon(daemon_name)` (environment-aware: debug or release)
//!
//! 2. Copy binary to ~/.local/bin/ (local copy)
//!    - Use: `local_copy()` to copy file
//!    - Create ~/.local/bin directory if needed
//!
//! 3. Make binary executable
//!    - Use: `chmod +x ~/.local/bin/{daemon_name}`
//!
//! ## Error Handling
//! - Local binary not found
//! - Copy failed (permissions, disk space)
//! - chmod failed
//!
//! ## Example
//! ```rust,no_run
//! use lifecycle_local::{install_daemon, InstallConfig};
//! use std::path::PathBuf;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Option 1: Build and install
//! let config = InstallConfig {
//!     daemon_name: "rbee-hive".to_string(),
//!     local_binary_path: None,
//!     job_id: None,
//! };
//! install_daemon(config).await?;
//!
//! // Option 2: Install pre-built binary
//! let binary = PathBuf::from("target/release/rbee-hive");
//! let config = InstallConfig {
//!     daemon_name: "rbee-hive".to_string(),
//!     local_binary_path: Some(binary),
//!     job_id: None,
//! };
//! install_daemon(config).await?;
//! # Ok(())
//! # }
//! ```

use lifecycle_shared::resolve_binary_path;
// TEAM-378: RULE ZERO - Use check_binary_exists with CheckMode::InstalledOnly
use crate::utils::{check_binary_exists, CheckMode};
use crate::utils::local::local_copy;
use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::with_job_id;
use std::path::PathBuf;
use timeout_enforcer::with_timeout;

/// Configuration for LOCAL daemon installation
///
/// TEAM-358: Removed ssh_config (lifecycle-local = LOCAL only)
#[derive(Debug, Clone)]
pub struct InstallConfig {
    /// Name of the daemon binary
    pub daemon_name: String,

    /// Optional path to pre-built binary (if None, will build from source)
    pub local_binary_path: Option<PathBuf>,

    /// Optional job ID for SSE narration routing
    /// When set, all narration (including timeout countdown) goes through SSE
    pub job_id: Option<String>,
}

/// Install daemon binary LOCALLY
///
/// TEAM-358: Refactored to remove SSH code, enforces 5-minute timeout
///
/// # Implementation
/// 1. Build or locate binary locally (environment-aware)
/// 2. Create ~/.local/bin directory if needed
/// 3. Copy binary to ~/.local/bin/
/// 4. Make executable
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
/// # Job ID Support
/// When called with job_id, narration routes through SSE
#[with_job_id(config_param = "install_config")]
#[with_timeout(secs = 300, label = "Install daemon")]
pub async fn install_daemon(install_config: InstallConfig) -> Result<()> {
    let daemon_name = &install_config.daemon_name;

    n!("install_start", "📦 Installing {} locally", daemon_name);

    // Step 0: Check if already installed
    // TEAM-378: RULE ZERO - Use CheckMode::InstalledOnly
    if check_binary_exists(daemon_name, CheckMode::InstalledOnly).await {
        n!("already_installed", "⚠️  {} is already installed in ~/.local/bin/", daemon_name);
        anyhow::bail!("{} is already installed in ~/.local/bin/. Use rebuild to update.", daemon_name);
    }

    // Step 1: Build or locate binary locally
    // TEAM-367: Use shared resolve_binary_path function
    let binary_path = resolve_binary_path(
        daemon_name,
        install_config.local_binary_path,
        install_config.job_id.clone(),
    )
    .await?;

    // Step 2: Create ~/.local/bin directory
    // TEAM-377: RULE ZERO - Use constant for install directory
    use lifecycle_shared::BINARY_INSTALL_DIR;
    
    let home = std::env::var("HOME").context("HOME env var not set")?;
    let local_bin_dir = std::path::PathBuf::from(&home).join(BINARY_INSTALL_DIR);
    
    n!("create_dir", "📁 Creating {}", local_bin_dir.display());
    std::fs::create_dir_all(&local_bin_dir)
        .with_context(|| format!("Failed to create ~/{}", BINARY_INSTALL_DIR))?;

    // Step 3: Copy binary to ~/.local/bin (ALWAYS - both debug and release)
    let dest_path = local_bin_dir.join(daemon_name);
    n!("copying", "📤 Copying {} to {}", daemon_name, dest_path.display());

    local_copy(&binary_path, &dest_path.to_string_lossy())
        .await
        .context("Failed to copy binary to ~/.local/bin")?;

    // Step 4: Make executable
    n!("chmod", "🔐 Making binary executable");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&dest_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&dest_path, perms).context("Failed to make binary executable")?;
    }

    // Step 5: Verify installation
    n!("verify", "✅ Verifying installation");
    if !dest_path.exists() {
        anyhow::bail!("Installation verification failed: binary not found at {}", dest_path.display());
    }

    n!("install_complete", "🎉 {} installed successfully at {}", daemon_name, dest_path.display());

    Ok(())
}

// TEAM-358: local_copy() is in utils/local.rs
