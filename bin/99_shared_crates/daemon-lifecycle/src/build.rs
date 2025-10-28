//! Build daemon binary locally for remote deployment
//!
//! # Types/Utils Used (from daemon-lifecycle)
//! - None (self-contained, uses ProcessNarrationCapture directly)
//!
//! # Requirements
//!
//! ## Input
//! - `daemon_name`: Name of daemon binary to build
//! - `target`: Optional build target (default: current architecture)
//!
//! ## Process
//! 1. Run cargo build (LOCAL, NO SSH)
//!    - Use: `cargo build --release --bin {daemon_name}`
//!    - If target specified: `cargo build --release --bin {daemon_name} --target {target}`
//!    - Wait for build to complete
//!
//! 2. Return path to built binary
//!    - Default: `target/release/{daemon_name}`
//!    - With target: `target/{target}/release/{daemon_name}`
//!
//! ## SSH Calls
//! - Total: 0 SSH calls (local build only)
//!
//! ## Error Handling
//! - Cargo build failed (compilation errors)
//! - Binary not found after build
//! - Invalid target specified
//!
//! ## Example
//! ```rust,no_run
//! use remote_daemon_lifecycle::build_daemon;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Build for current architecture
//! let binary_path = build_daemon("rbee-hive", None).await?;
//! println!("Built at: {}", binary_path.display());
//!
//! // Build for specific target
//! let binary_path = build_daemon(
//!     "rbee-hive",
//!     Some("x86_64-unknown-linux-gnu")
//! ).await?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use observability_narration_core::{n, process_capture::ProcessNarrationCapture};
use observability_narration_macros::with_job_id;
use std::path::PathBuf;
use tokio::process::Command;

/// Configuration for building daemon binary
///
/// TEAM-330: Includes optional job_id for SSE streaming of cargo output
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Name of the daemon binary
    pub daemon_name: String,

    /// Optional cross-compilation target
    pub target: Option<String>,

    /// Optional job ID for SSE narration routing
    /// When set, cargo build output streams through SSE!
    pub job_id: Option<String>,
}

/// Build daemon binary locally for remote deployment
///
/// TEAM-330: Streams cargo build output through SSE when job_id is set
///
/// # Implementation
/// 1. Runs `cargo build --release --bin {daemon_name}`
/// 2. Optionally supports cross-compilation via `--target`
/// 3. Captures stdout/stderr and streams through SSE (via ProcessNarrationCapture)
/// 4. Returns path to built binary
///
/// # SSE Streaming
/// When called with a job_id in BuildConfig, cargo build output
/// streams in real-time through SSE to the web UI!
///
/// # Example
/// ```rust,ignore
/// use remote_daemon_lifecycle::{BuildConfig, build_daemon};
///
/// let config = BuildConfig {
///     daemon_name: "llm-worker-rbee".to_string(),
///     target: None,
///     job_id: Some("job-123".to_string()),  // ← Cargo output streams through SSE!
/// };
/// build_daemon(config).await?;
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
    let mut command = Command::new("cargo");
    command.arg("build").arg("--release").arg("--bin").arg(daemon_name);

    // Add target if specified (for cross-compilation)
    if let Some(target_triple) = target {
        command.arg("--target").arg(target_triple);
        n!("build_target", "📦 Cross-compiling for target: {}", target_triple);
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

    // Determine binary path
    let binary_path = if let Some(target_triple) = target {
        PathBuf::from(format!("target/{}/release/{}", target_triple, daemon_name))
    } else {
        PathBuf::from(format!("target/release/{}", daemon_name))
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
