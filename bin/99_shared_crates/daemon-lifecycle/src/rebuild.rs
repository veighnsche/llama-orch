//! Daemon rebuild utilities
//!
//! TEAM-316: Extracted common rebuild patterns from queen-lifecycle and hive-lifecycle
//!
//! Provides reusable functions for rebuilding daemons from source:
//! - Health check before rebuild (prevent rebuilding while running)
//! - Local cargo build execution
//! - Build output handling

use anyhow::Result;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use std::process::Command;

/// Configuration for daemon rebuild
#[derive(Debug, Clone)]
pub struct RebuildConfig {
    /// Name of the daemon binary (e.g., "queen-rbee", "rbee-hive")
    pub binary_name: String,

    /// Optional features to enable (e.g., "local-hive")
    pub features: Option<Vec<String>>,

    /// Optional job ID for narration routing
    pub job_id: Option<String>,
}

impl RebuildConfig {
    /// Create a new rebuild config
    pub fn new(binary_name: impl Into<String>) -> Self {
        Self {
            binary_name: binary_name.into(),
            features: None,
            job_id: None,
        }
    }

    /// Set features to enable
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = Some(features);
        self
    }

    /// Set job_id for narration
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }
}

/// Check if daemon is running before rebuild
///
/// TEAM-316: Extracted from queen-lifecycle and hive-lifecycle
///
/// Prevents rebuilding while daemon is running to avoid file conflicts.
///
/// # Arguments
/// * `daemon_name` - Name of the daemon for error messages
/// * `health_url` - Health check URL (e.g., "http://localhost:7833")
/// * `job_id` - Optional job ID for narration routing
///
/// # Returns
/// * `Ok(())` - Daemon is not running, safe to rebuild
/// * `Err` - Daemon is running, must stop first
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::rebuild::check_not_running_before_rebuild;
///
/// # async fn example() -> anyhow::Result<()> {
/// check_not_running_before_rebuild(
///     "queen-rbee",
///     "http://localhost:7833",
///     None,
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn check_not_running_before_rebuild(
    daemon_name: &str,
    health_url: &str,
    job_id: Option<&str>,
) -> Result<()> {
    // TEAM-316: Migrated to n!() macro
    let ctx = job_id.map(|jid| NarrationContext::new().with_job_id(jid));

    let check_impl = async {
        let is_running = crate::health::is_daemon_healthy(
            health_url,
            None, // Use default /health endpoint
            Some(std::time::Duration::from_secs(2)),
        )
        .await;

        if is_running {
            n!(
                "daemon_still_running",
                "⚠️  {} is currently running. Stop it first.",
                daemon_name
            );
            anyhow::bail!("{} is still running. Stop it first.", daemon_name);
        }

        Ok(())
    };

    // Execute with context if job_id provided
    if let Some(ctx) = ctx {
        with_narration_context(ctx, check_impl).await
    } else {
        check_impl.await
    }
}

/// Build a daemon locally using cargo
///
/// TEAM-316: Extracted from queen-lifecycle and hive-lifecycle
///
/// Runs `cargo build --release --bin <binary_name>` with optional features.
///
/// # Arguments
/// * `config` - Rebuild configuration
///
/// # Returns
/// * `Ok(String)` - Path to built binary (e.g., "target/release/queen-rbee")
/// * `Err` - Build failed
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::rebuild::{RebuildConfig, build_daemon_local};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = RebuildConfig::new("queen-rbee")
///     .with_features(vec!["local-hive".to_string()]);
///
/// let binary_path = build_daemon_local(config).await?;
/// println!("Built: {}", binary_path);
/// # Ok(())
/// # }
/// ```
pub async fn build_daemon_local(config: RebuildConfig) -> Result<String> {
    // TEAM-316: Migrated to n!() macro
    let ctx = config
        .job_id
        .as_ref()
        .map(|jid| NarrationContext::new().with_job_id(jid));

    let build_impl = async {
        n!(
            "build_start",
            "⏳ Running cargo build (this may take a few minutes)..."
        );

        // Build command
        let mut cmd = Command::new("cargo");
        cmd.arg("build")
            .arg("--release")
            .arg("--bin")
            .arg(&config.binary_name);

        // Add features if specified
        if let Some(features) = &config.features {
            if !features.is_empty() {
                n!(
                    "build_features",
                    "✨ Building with features: {}",
                    features.join(", ")
                );
                cmd.arg("--features").arg(features.join(","));
            }
        }

        // Execute build
        let output = cmd.output()?;

        if output.status.success() {
            n!("build_success", "✅ Build successful!");

            // Determine binary path
            let binary_path = format!("target/release/{}", config.binary_name);
            n!("binary_location", "📦 Binary available at: {}", binary_path);

            Ok(binary_path)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            n!("build_failed", "❌ Build failed: {}", stderr);
            anyhow::bail!("Build failed: {}", stderr)
        }
    };

    // Execute with context if job_id provided
    if let Some(ctx) = ctx {
        with_narration_context(ctx, build_impl).await
    } else {
        build_impl.await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rebuild_config_builder() {
        let config = RebuildConfig::new("test-daemon")
            .with_features(vec!["feature1".to_string(), "feature2".to_string()])
            .with_job_id("job-123");

        assert_eq!(config.binary_name, "test-daemon");
        assert_eq!(
            config.features,
            Some(vec!["feature1".to_string(), "feature2".to_string()])
        );
        assert_eq!(config.job_id, Some("job-123".to_string()));
    }

    #[test]
    fn test_rebuild_config_no_features() {
        let config = RebuildConfig::new("test-daemon");

        assert_eq!(config.binary_name, "test-daemon");
        assert_eq!(config.features, None);
        assert_eq!(config.job_id, None);
    }
}
