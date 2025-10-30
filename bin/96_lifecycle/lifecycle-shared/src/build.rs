//! Build configuration and function shared between lifecycle-local and lifecycle-ssh
//!
//! TEAM-367: Extracted from lifecycle-local/src/build.rs and lifecycle-ssh/src/build.rs

use anyhow::{Context, Result};
use observability_narration_core::{n, process_capture::ProcessNarrationCapture};
use observability_narration_macros::with_job_id;
use std::path::PathBuf;
use tokio::process::Command;

/// Configuration for building daemon binary
///
/// TEAM-330: Includes optional job_id for SSE streaming of cargo output
/// TEAM-NARRATION-FIX: Added features support for worker binaries
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Name of the daemon binary
    pub daemon_name: String,

    /// Optional cross-compilation target
    pub target: Option<String>,

    /// Optional job ID for SSE narration routing
    /// When set, cargo build output streams through SSE!
    pub job_id: Option<String>,

    /// Optional features to enable (e.g., "cpu", "cuda", "metal")
    /// TEAM-NARRATION-FIX: For worker binaries with feature-gated backends
    pub features: Option<String>,
}

impl BuildConfig {
    /// Create new BuildConfig with daemon name
    pub fn new(daemon_name: impl Into<String>) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            target: None,
            job_id: None,
            features: None,
        }
    }

    /// Set cross-compilation target
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Set job ID for SSE streaming
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    /// Set features to enable
    pub fn with_features(mut self, features: impl Into<String>) -> Self {
        self.features = Some(features.into());
        self
    }
}

/// Build daemon binary locally
///
/// TEAM-367: Shared implementation used by both lifecycle-local and lifecycle-ssh
/// TEAM-330: Streams cargo build output through SSE when job_id is set
///
/// # Implementation
/// 1. Runs `cargo build --release --bin {daemon_name}` (or debug in dev mode)
/// 2. Optionally supports cross-compilation via `--target`
/// 3. Optionally supports features via `--features`
/// 4. Captures stdout/stderr and streams through SSE (via ProcessNarrationCapture)
/// 5. Returns path to built binary
///
/// # SSE Streaming
/// When called with a job_id in BuildConfig, cargo build output
/// streams in real-time through SSE to the web UI!
///
/// # Example
/// ```rust,ignore
/// use lifecycle_shared::{BuildConfig, build_daemon};
///
/// let config = BuildConfig::new("llm-worker-rbee")
///     .with_job_id("job-123")  // ← Cargo output streams through SSE!
///     .with_features("cuda");
/// 
/// let binary_path = build_daemon(config).await?;
/// ```
#[with_job_id(config_param = "build_config")]
pub async fn build_daemon(build_config: BuildConfig) -> Result<PathBuf> {
    let daemon_name = &build_config.daemon_name;
    let target = build_config.target.as_deref();

    n!("build_start", "🔨 Building {} from source...", daemon_name);

    // TEAM-330: Extract job_id from config for SSE streaming
    // The #[with_job_id] macro automatically wraps this in NarrationContext
    let job_id = build_config.job_id;

    // Create process capture - cargo output streams through SSE if job_id is set!
    let capture = ProcessNarrationCapture::new(job_id);

    // Build cargo command
    // TEAM-341: Environment-aware build mode
    // Debug builds (cargo build) → build child daemons in debug mode
    // Release builds (cargo build --release) → build child daemons in release mode
    // This ensures dev builds can proxy to Vite dev servers
    let mut command = Command::new("cargo");
    command.arg("build");
    
    #[cfg(debug_assertions)]
    {
        n!("build_mode", "🔧 Building in DEBUG mode (dev environment)");
        // No --release flag in debug mode
    }
    
    #[cfg(not(debug_assertions))]
    {
        n!("build_mode", "🚀 Building in RELEASE mode (production)");
        command.arg("--release");
    }
    
    command.arg("--bin").arg(daemon_name);

    // Add target if specified (for cross-compilation)
    if let Some(target_triple) = target {
        command.arg("--target").arg(target_triple);
        n!("build_target", "📦 Cross-compiling for target: {}", target_triple);
    }

    // TEAM-NARRATION-FIX: Add features if specified (for worker binaries)
    if let Some(features) = &build_config.features {
        command.arg("--features").arg(features);
        n!("build_features", "🎯 Building with features: {}", features);
    }

    n!("build_running", "⚙️  Running cargo build (output streaming via SSE)...");

    // Spawn with capture - stdout/stderr stream through SSE!
    let mut child = capture.spawn(command).await.context("Failed to spawn cargo build")?;

    // Wait for build to complete
    n!("build_waiting", "⏳ Waiting for cargo build to complete...");
    let status = child.wait().await.context("Failed to wait for cargo build")?;

    if !status.success() {
        n!("build_failed", "❌ Cargo build failed with exit code: {:?}", status.code());
        anyhow::bail!("Cargo build failed for {}", daemon_name);
    }

    // Determine binary path based on build mode
    // TEAM-341: Match build mode (debug vs release)
    #[cfg(debug_assertions)]
    let build_mode = "debug";
    #[cfg(not(debug_assertions))]
    let build_mode = "release";
    
    let binary_path = if let Some(target_triple) = target {
        PathBuf::from(format!("target/{}/{}/{}", target_triple, build_mode, daemon_name))
    } else {
        PathBuf::from(format!("target/{}/{}", build_mode, daemon_name))
    };

    // Verify binary exists
    n!("build_verify", "🔍 Verifying binary at: {}", binary_path.display());
    if !binary_path.exists() {
        n!("build_verify_failed", "❌ Binary not found at expected path");
        anyhow::bail!(
            "Binary not found at expected path: {}. Build may have failed.",
            binary_path.display()
        );
    }

    n!("build_complete", "✅ Build complete: {}", binary_path.display());

    Ok(binary_path)
}
