//! Daemon rebuild utilities
//!
//! TEAM-316: Extracted common rebuild patterns from queen-lifecycle and hive-lifecycle
//! TEAM-328: Added conditional hot reload behavior
//!
//! Provides reusable functions for rebuilding daemons from source:
//! - Conditional hot reload (running → stop → rebuild → start → running)
//! - Cold rebuild (stopped → rebuild → stopped)
//! - Local cargo build execution
//! - Build output handling

use anyhow::Result;
use daemon_contract::HttpDaemonConfig;
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

/// Rebuild daemon with conditional hot reload
///
/// TEAM-328: Implements conditional hot reload behavior:
/// - running → stop → rebuild → start → running (hot reload)
/// - stopped → rebuild → stopped (cold rebuild)
///
/// # Arguments
/// * `rebuild_config` - Rebuild configuration (binary name, features, job_id)
/// * `daemon_config` - HTTP daemon configuration (for start/stop operations)
///
/// # Returns
/// * `Ok(bool)` - true if daemon was restarted (hot reload), false if left stopped
/// * `Err` - Build or lifecycle operation failed
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::rebuild::{RebuildConfig, rebuild_with_hot_reload};
/// use daemon_contract::HttpDaemonConfig;
///
/// # async fn example() -> anyhow::Result<()> {
/// let rebuild_config = RebuildConfig::new("queen-rbee")
///     .with_features(vec!["local-hive".to_string()])
///     .with_job_id("job-123");
///
/// let daemon_config = HttpDaemonConfig {
///     daemon_name: "queen-rbee".to_string(),
///     binary_path: None, // Auto-resolve
///     health_url: "http://localhost:7833".to_string(),
///     args: vec![],
///     env: vec![],
///     job_id: Some("job-123".to_string()),
/// };
///
/// let was_restarted = rebuild_with_hot_reload(rebuild_config, daemon_config).await?;
/// if was_restarted {
///     println!("Hot reload complete - daemon restarted");
/// } else {
///     println!("Cold rebuild complete - daemon left stopped");
/// }
/// # Ok(())
/// # }
/// ```
pub async fn rebuild_with_hot_reload(
    rebuild_config: RebuildConfig,
    daemon_config: HttpDaemonConfig,
) -> Result<bool> {
    // TEAM-328: Migrated to n!() macro
    let ctx = rebuild_config
        .job_id
        .as_ref()
        .map(|jid| NarrationContext::new().with_job_id(jid));

    let rebuild_impl = async {
        // Step 1: Check if daemon is currently running
        let was_running = crate::health::is_daemon_healthy(
            &daemon_config.health_url,
            None, // Use default /health endpoint
            Some(std::time::Duration::from_secs(2)),
        ).await;

        if was_running {
            n!(
                "hot_reload_start",
                "🔄 Hot reload detected - {} is running, will restart after rebuild",
                rebuild_config.binary_name
            );

            // Step 2: Stop the running daemon
            n!("hot_reload_stop", "⏸️  Stopping {}...", rebuild_config.binary_name);
            crate::stop::stop_http_daemon(daemon_config.clone()).await?;
            n!("hot_reload_stopped", "✅ {} stopped", rebuild_config.binary_name);
        } else {
            n!(
                "cold_rebuild_start",
                "🔨 Cold rebuild - {} is not running, will rebuild only",
                rebuild_config.binary_name
            );
        }

        // Step 3: Build the daemon
        let binary_path = build_daemon_local(rebuild_config.clone()).await?;

        // Step 4: If it was running, start it again (hot reload)
        if was_running {
            n!("hot_reload_restart", "▶️  Restarting {}...", rebuild_config.binary_name);
            
            // Update daemon config with built binary path
            let mut start_config = daemon_config;
            start_config.binary_path = Some(binary_path.into());
            
            crate::start::start_http_daemon(start_config).await?;
            n!(
                "hot_reload_complete",
                "✅ Hot reload complete - {} is running with new binary",
                rebuild_config.binary_name
            );
            Ok(true)
        } else {
            n!(
                "cold_rebuild_complete",
                "✅ Cold rebuild complete - {} binary updated, daemon left stopped",
                rebuild_config.binary_name
            );
            Ok(false)
        }
    };

    // Execute with context if job_id provided
    if let Some(ctx) = ctx {
        with_narration_context(ctx, rebuild_impl).await
    } else {
        rebuild_impl.await
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
