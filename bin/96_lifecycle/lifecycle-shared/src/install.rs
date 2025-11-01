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
        // TEAM-379: Handle special keywords "release" and "debug"
        let (resolved_path, build_target) = if path.to_str() == Some("release") {
            (PathBuf::from(format!("target/release/{}", daemon_name)), Some("release"))
        } else if path.to_str() == Some("debug") {
            (PathBuf::from(format!("target/debug/{}", daemon_name)), Some("debug"))
        } else {
            (path, None)
        };
        
        n!("verify_binary", "🔍 Verifying pre-built binary at: {}", resolved_path.display());
        if !resolved_path.exists() {
            // TEAM-379: If using keyword and binary doesn't exist, build it!
            if let Some(target) = build_target {
                n!("building_missing", "🔨 Binary not found, building {} version...", target);
                let build_config = BuildConfig {
                    daemon_name: daemon_name.to_string(),
                    target: Some(target.to_string()),
                    job_id: job_id.clone(),
                    features: None,
                };
                return build_daemon(build_config).await;
            }
            // For explicit paths (not keywords), fail if not found
            n!("binary_not_found", "❌ Binary not found at: {}", resolved_path.display());
            anyhow::bail!("Binary not found at: {}", resolved_path.display());
        }
        n!("using_binary", "📦 Using pre-built binary: {}", resolved_path.display());
        Ok(resolved_path)
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
