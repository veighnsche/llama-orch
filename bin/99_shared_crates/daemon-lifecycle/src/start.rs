//! Start HTTP-based daemons
//!
//! TEAM-316: Extracted from lifecycle.rs (RULE ZERO - single responsibility)
//! TEAM-329: Inlined spawn() logic to eliminate entropy (RULE ZERO)

use anyhow::{Context, Result};
use observability_narration_core::n;
use std::process::Stdio;
use tokio::process::Command;

use crate::types::start::HttpDaemonConfig; // TEAM-329: types/start.rs (PARITY)
use crate::types::status::HealthPollConfig; // TEAM-329: types/status.rs (PARITY)
use crate::utils::find::find_binary; // TEAM-329: utils/find.rs
use crate::utils::pid::write_pid_file; // TEAM-329: utils/pid.rs (centralized PID operations)
use crate::utils::poll::poll_daemon_health; // TEAM-329: utils/poll.rs

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
pub async fn start_daemon(config: HttpDaemonConfig) -> Result<u32> {
    // Step 1: Resolve binary path (auto-resolve from daemon_name if not provided)
    // TEAM-327: Binary resolution moved inside start_http_daemon
    // TEAM-329: Use standalone find_binary function
    let binary_path = match config.binary_path {
        Some(path) => path,
        None => find_binary(&config.daemon_name)?,
    };
    
    // Step 2: Spawn the daemon (inlined from manager.spawn())
    // TEAM-329: Inlined spawn() to eliminate entropy (RULE ZERO)
    n!("spawn", "Spawning daemon: {} with args: {:?}", binary_path.display(), config.args);
    
    // TEAM-164: Use Stdio::null() to prevent daemon from holding parent's pipes
    let mut cmd = Command::new(&binary_path);
    cmd.args(&config.args)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    
    // TEAM-189: Propagate SSH agent environment variables to daemon
    if let Ok(ssh_auth_sock) = std::env::var("SSH_AUTH_SOCK") {
        cmd.env("SSH_AUTH_SOCK", ssh_auth_sock);
    }
    
    let child = cmd
        .spawn()
        .context(format!("Failed to spawn daemon: {}", binary_path.display()))?;
    
    // Step 3: Extract PID before detaching
    // TEAM-327: PID needed for signal-based shutdown
    let pid = child.id().ok_or_else(|| {
        anyhow::anyhow!("Failed to get PID from spawned daemon: {}", config.daemon_name)
    })?;
    
    n!("spawned", "Daemon spawned with PID: {}", pid);

    // Step 4: Poll until healthy
    let mut health_config =
        HealthPollConfig::new(&config.health_url).with_daemon_name(&config.daemon_name);

    if let Some(attempts) = config.max_health_attempts {
        health_config = health_config.with_max_attempts(attempts);
    }

    if let Some(job_id) = config.job_id.as_deref() {
        health_config = health_config.with_job_id(job_id);
    }

    poll_daemon_health(health_config).await?;

    // Step 5: Write PID file for stateless shutdown
    // TEAM-327: Standard Unix pattern - enables stateless stop command
    // TEAM-329: Use centralized PID operations
    write_pid_file(&config.daemon_name, pid)?;

    // Step 6: Detach the child process
    // The daemon will keep running independently
    // We use ManuallyDrop to prevent Drop from running without mem::forget
    let _ = std::mem::ManuallyDrop::new(child);

    // Step 7: Return PID for tracking (optional - PID file is primary)
    Ok(pid)
}

