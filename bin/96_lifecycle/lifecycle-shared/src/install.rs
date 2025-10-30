//! Installation utilities shared between lifecycle-local and lifecycle-ssh
//!
//! TEAM-367: Extracted from install.rs in both lifecycle-local and lifecycle-ssh

use crate::build::{build_daemon, BuildConfig};
use anyhow::Result;
use observability_narration_core::n;
use std::path::PathBuf;

/// Resolve binary path: either verify pre-built binary or build from source
///
/// TEAM-367: Shared logic extracted from install.rs in both lifecycle-local and lifecycle-ssh
///
/// # Arguments
/// - `daemon_name`: Name of the daemon binary
/// - `local_binary_path`: Optional path to pre-built binary
/// - `job_id`: Optional job ID for SSE narration routing
///
/// # Returns
/// Path to the binary (either the provided path or newly built binary)
///
/// # Example
/// ```rust,ignore
/// // Option 1: Use pre-built binary
/// let path = resolve_binary_path("rbee-hive", Some(PathBuf::from("target/release/rbee-hive")), None).await?;
///
/// // Option 2: Build from source
/// let path = resolve_binary_path("rbee-hive", None, Some("job-123".to_string())).await?;
/// ```
pub async fn resolve_binary_path(
    daemon_name: &str,
    local_binary_path: Option<PathBuf>,
    job_id: Option<String>,
) -> Result<PathBuf> {
    if let Some(path) = local_binary_path {
        n!("verify_binary", "🔍 Verifying pre-built binary at: {}", path.display());
        if !path.exists() {
            n!("binary_not_found", "❌ Binary not found at: {}", path.display());
            anyhow::bail!("Binary not found at: {}", path.display());
        }
        n!("using_binary", "📦 Using pre-built binary: {}", path.display());
        Ok(path)
    } else {
        // Build from source
        let build_config = BuildConfig {
            daemon_name: daemon_name.to_string(),
            target: None,
            job_id,
            features: None,
        };
        build_daemon(build_config).await
    }
}
