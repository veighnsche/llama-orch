//! Start HTTP-based daemons
//!
//! TEAM-316: Extracted from lifecycle.rs (RULE ZERO - single responsibility)

use anyhow::Result;
use daemon_contract::HttpDaemonConfig;
use std::path::PathBuf;

use crate::health::{poll_until_healthy, HealthPollConfig};
use crate::manager::DaemonManager;

/// Get PID file path for a daemon
///
/// TEAM-327: Standard Unix pattern - PID files in ~/.local/var/run/
/// 
/// # Arguments
/// * `daemon_name` - Name of the daemon (e.g., "queen-rbee", "rbee-hive")
///
/// # Returns
/// Path to PID file (e.g., ~/.local/var/run/queen-rbee.pid)
fn get_pid_file_path(daemon_name: &str) -> Result<PathBuf> {
    let home = std::env::var("HOME")?;
    let run_dir = PathBuf::from(format!("{}/.local/var/run", home));
    
    // Create directory if it doesn't exist
    std::fs::create_dir_all(&run_dir)?;
    
    Ok(run_dir.join(format!("{}.pid", daemon_name)))
}

/// Start an HTTP-based daemon (spawn + health polling)
///
/// TEAM-276: High-level function that combines spawn and health polling
/// TEAM-316: Extracted from lifecycle.rs
/// TEAM-319: Detaches child process internally (no mem::forget needed)
/// TEAM-327: Returns PID for signal-based shutdown tracking
///
/// Steps:
/// 1. Resolve binary path (auto-resolve from daemon_name if not provided)
/// 2. Spawn the daemon process
/// 3. Extract PID before detaching
/// 4. Poll health endpoint until ready
/// 5. Write PID to file (~/.local/var/run/{daemon}.pid) for stateless shutdown
/// 6. Detach the child process (daemon keeps running)
/// 7. Return PID for tracking (optional - PID file is primary)
///
/// # Arguments
/// * `config` - HTTP daemon configuration
///
/// # Returns
/// * `Ok(pid)` - Daemon started, healthy, and detached. Returns PID for shutdown.
/// * `Err` - Failed to start or become healthy
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{start_http_daemon, HttpDaemonConfig};
///
/// # async fn example() -> anyhow::Result<()> {
/// // Binary path auto-resolved from daemon_name
/// let config = HttpDaemonConfig::new(
///     "queen-rbee",
///     "http://localhost:8500",
/// )
/// .with_args(vec!["--port".to_string(), "8500".to_string()])
/// .with_job_id("job-123");
///
/// let pid = start_http_daemon(config).await?;
/// println!("Daemon started with PID: {}", pid);
/// // Daemon is now running and healthy (detached)
/// # Ok(())
/// # }
/// ```
pub async fn start_http_daemon(config: HttpDaemonConfig) -> Result<u32> {
    // Step 1: Resolve binary path (auto-resolve from daemon_name if not provided)
    // TEAM-327: Binary resolution moved inside start_http_daemon
    let binary_path = match config.binary_path {
        Some(path) => path,
        None => DaemonManager::find_binary(&config.daemon_name)?,
    };
    
    // Step 2: Spawn the daemon
    let manager = DaemonManager::new(binary_path, config.args.clone());
    let child = manager.spawn().await?;
    
    // Step 3: Extract PID before detaching
    // TEAM-327: PID needed for signal-based shutdown
    let pid = child.id().ok_or_else(|| {
        anyhow::anyhow!("Failed to get PID from spawned daemon: {}", config.daemon_name)
    })?;

    // Step 4: Poll until healthy
    let mut health_config =
        HealthPollConfig::new(&config.health_url).with_daemon_name(&config.daemon_name);

    if let Some(attempts) = config.max_health_attempts {
        health_config = health_config.with_max_attempts(attempts);
    }

    if let Some(job_id) = config.job_id.as_deref() {
        health_config = health_config.with_job_id(job_id);
    }

    poll_until_healthy(health_config).await?;

    // Step 5: Write PID file for stateless shutdown
    // TEAM-327: Standard Unix pattern - enables stateless stop command
    let pid_file = get_pid_file_path(&config.daemon_name)?;
    std::fs::write(&pid_file, pid.to_string())?;

    // Step 6: Detach the child process
    // The daemon will keep running independently
    // We use ManuallyDrop to prevent Drop from running without mem::forget
    let _ = std::mem::ManuallyDrop::new(child);

    // Step 7: Return PID for tracking (optional - PID file is primary)
    Ok(pid)
}
