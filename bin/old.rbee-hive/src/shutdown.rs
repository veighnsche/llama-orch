//! Graceful shutdown orchestration with force-kill fallback
//!
//! Implements Week 4 Priority 1: Graceful Shutdown Completion
//! - 30s graceful shutdown timeout
//! - Force-kill fallback for hung workers
//! - Comprehensive metrics and logging
//!
//! Created by: TEAM-116

use crate::registry::WorkerRegistry;
use anyhow::Result;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

/// Shutdown configuration
#[derive(Debug, Clone)]
pub struct ShutdownConfig {
    /// Timeout for graceful shutdown before force-kill (default: 30s)
    pub graceful_timeout_secs: u64,
    /// Timeout for HTTP shutdown request (default: 5s)
    pub http_timeout_secs: u64,
    /// Wait time before checking if process still alive (default: 10s)
    pub force_kill_wait_secs: u64,
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        Self {
            graceful_timeout_secs: 30,
            http_timeout_secs: 5,
            force_kill_wait_secs: 10,
        }
    }
}

/// Shutdown metrics
#[derive(Debug, Default)]
pub struct ShutdownMetrics {
    pub total_workers: usize,
    pub graceful_shutdown: usize,
    pub force_killed: usize,
    pub timeout_exceeded: usize,
    pub duration_secs: f64,
}

/// Orchestrate graceful shutdown of all workers with force-kill fallback
///
/// # Shutdown Sequence
/// 1. Send HTTP shutdown to all workers (parallel)
/// 2. Wait up to 30 seconds for graceful shutdown
/// 3. Force-kill any remaining workers (SIGTERM → wait 10s → SIGKILL)
/// 4. Clean up worker registry
/// 5. Log shutdown metrics
///
/// # Arguments
/// * `registry` - Worker registry
/// * `config` - Shutdown configuration
///
/// # Returns
/// Shutdown metrics
pub async fn shutdown_all_workers(
    registry: Arc<WorkerRegistry>,
    config: ShutdownConfig,
) -> ShutdownMetrics {
    let shutdown_start = Instant::now();
    let workers = registry.list().await;
    let total_workers = workers.len();

    info!(
        "TEAM-116: Starting graceful shutdown of {} workers ({}s timeout)",
        total_workers, config.graceful_timeout_secs
    );

    if total_workers == 0 {
        info!("No workers to shutdown");
        return ShutdownMetrics::default();
    }

    // Phase 1: Parallel graceful shutdown via HTTP
    let mut shutdown_tasks = Vec::new();

    for worker in workers {
        let worker_id = worker.id.clone();
        let worker_url = worker.url.clone();
        let worker_pid = worker.pid;
        let http_timeout = Duration::from_secs(config.http_timeout_secs);
        let force_kill_wait = Duration::from_secs(config.force_kill_wait_secs);

        let task = tokio::spawn(async move {
            info!("Initiating graceful shutdown for worker {}", worker_id);

            // Try graceful HTTP shutdown
            let graceful_success = match shutdown_worker_http(&worker_url, http_timeout).await {
                Ok(_) => {
                    info!("Worker {} acknowledged graceful shutdown", worker_id);
                    true
                }
                Err(e) => {
                    warn!("Worker {} graceful shutdown failed: {}", worker_id, e);
                    false
                }
            };

            // Phase 2: Force-kill if needed
            let force_killed = if let Some(pid) = worker_pid {
                if !graceful_success || !wait_for_process_exit(pid, force_kill_wait).await {
                    match force_kill_worker(pid, &worker_id).await {
                        Ok(killed) => killed,
                        Err(e) => {
                            error!("Force-kill failed for worker {}: {}", worker_id, e);
                            false
                        }
                    }
                } else {
                    false // Exited gracefully
                }
            } else {
                false
            };

            ShutdownResult {
                worker_id,
                graceful: graceful_success && !force_killed,
                force_killed,
            }
        });

        shutdown_tasks.push(task);
    }

    // Phase 3: Wait for all shutdowns with timeout
    let mut metrics = ShutdownMetrics {
        total_workers,
        ..Default::default()
    };

    let timeout_duration = Duration::from_secs(config.graceful_timeout_secs);

    for task in shutdown_tasks {
        let elapsed = shutdown_start.elapsed();
        let remaining = timeout_duration.saturating_sub(elapsed);

        if remaining.is_zero() {
            error!("Shutdown timeout ({}s) exceeded - aborting remaining workers", config.graceful_timeout_secs);
            metrics.timeout_exceeded += 1;
            task.abort();
            continue;
        }

        match tokio::time::timeout(remaining, task).await {
            Ok(Ok(result)) => {
                if result.graceful {
                    metrics.graceful_shutdown += 1;
                } else if result.force_killed {
                    metrics.force_killed += 1;
                } else {
                    metrics.timeout_exceeded += 1;
                }
                info!(
                    "Shutdown progress: {}/{} workers (graceful: {}, forced: {}, timeout: {})",
                    metrics.graceful_shutdown + metrics.force_killed + metrics.timeout_exceeded,
                    total_workers,
                    metrics.graceful_shutdown,
                    metrics.force_killed,
                    metrics.timeout_exceeded
                );
            }
            Ok(Err(e)) => {
                error!("Worker shutdown task failed: {}", e);
                metrics.timeout_exceeded += 1;
            }
            Err(_) => {
                error!("Worker shutdown task timed out");
                metrics.timeout_exceeded += 1;
            }
        }
    }

    // Phase 4: Clean up registry
    registry.clear().await;

    metrics.duration_secs = shutdown_start.elapsed().as_secs_f64();

    // Phase 5: Emit Prometheus metrics
    if let Err(e) = emit_shutdown_metrics(&metrics) {
        error!("Failed to emit shutdown metrics: {}", e);
    }

    // Phase 6: Log final metrics
    info!(
        "SHUTDOWN COMPLETE - Total: {}, Graceful: {}, Forced: {}, Timeout: {}, Duration: {:.2}s",
        metrics.total_workers,
        metrics.graceful_shutdown,
        metrics.force_killed,
        metrics.timeout_exceeded,
        metrics.duration_secs
    );

    if metrics.timeout_exceeded > 0 {
        warn!("{} workers exceeded shutdown timeout", metrics.timeout_exceeded);
    }

    metrics
}

/// Emit shutdown metrics to Prometheus
fn emit_shutdown_metrics(metrics: &ShutdownMetrics) -> Result<()> {
    use crate::metrics::{SHUTDOWN_DURATION_SECONDS, WORKERS_FORCE_KILLED_TOTAL, WORKERS_GRACEFUL_SHUTDOWN_TOTAL};

    SHUTDOWN_DURATION_SECONDS.observe(metrics.duration_secs);
    
    for _ in 0..metrics.graceful_shutdown {
        WORKERS_GRACEFUL_SHUTDOWN_TOTAL.inc();
    }
    
    for _ in 0..metrics.force_killed {
        WORKERS_FORCE_KILLED_TOTAL.inc();
    }

    Ok(())
}

/// Result of a single worker shutdown
#[derive(Debug)]
struct ShutdownResult {
    worker_id: String,
    graceful: bool,
    force_killed: bool,
}

/// Send HTTP shutdown request to worker
async fn shutdown_worker_http(worker_url: &str, timeout: Duration) -> Result<()> {
    let client = reqwest::Client::new();

    let response = client
        .post(format!("{}/v1/shutdown", worker_url))
        .timeout(timeout)
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("Worker returned non-success status: {}", response.status());
    }

    Ok(())
}

/// Wait for process to exit gracefully
async fn wait_for_process_exit(pid: u32, wait_duration: Duration) -> bool {
    tokio::time::sleep(wait_duration).await;
    !is_process_alive(pid)
}

/// Force-kill a worker process (SIGTERM → wait → SIGKILL)
async fn force_kill_worker(pid: u32, worker_id: &str) -> Result<bool> {
    use nix::sys::signal::{kill, Signal};
    use nix::unistd::Pid;

    info!("Force-killing worker {} (PID {})", worker_id, pid);

    let pid_obj = Pid::from_raw(pid as i32);

    // Try SIGTERM first
    if let Err(e) = kill(pid_obj, Signal::SIGTERM) {
        warn!("SIGTERM failed for PID {}: {} (process may be dead)", pid, e);
        return Ok(false);
    }

    info!("Sent SIGTERM to PID {}, waiting 10 seconds...", pid);

    // Wait 10 seconds
    tokio::time::sleep(Duration::from_secs(10)).await;

    // Check if still alive
    if is_process_alive(pid) {
        warn!("Process {} still alive after SIGTERM, sending SIGKILL", pid);
        if let Err(e) = kill(pid_obj, Signal::SIGKILL) {
            error!("SIGKILL failed for PID {}: {}", pid, e);
            return Err(anyhow::anyhow!("SIGKILL failed: {}", e));
        }
        info!("Sent SIGKILL to PID {}", pid);
        Ok(true)
    } else {
        info!("Process {} terminated gracefully after SIGTERM", pid);
        Ok(false)
    }
}

/// Check if a process is still alive
fn is_process_alive(pid: u32) -> bool {
    use nix::sys::signal::kill;
    use nix::unistd::Pid;

    // Signal 0 doesn't send a signal, just checks if process exists
    kill(Pid::from_raw(pid as i32), None).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{WorkerInfo, WorkerState};
    use std::time::SystemTime;

    #[tokio::test]
    async fn test_shutdown_config_default() {
        let config = ShutdownConfig::default();
        assert_eq!(config.graceful_timeout_secs, 30);
        assert_eq!(config.http_timeout_secs, 5);
        assert_eq!(config.force_kill_wait_secs, 10);
    }

    #[tokio::test]
    async fn test_shutdown_empty_registry() {
        let registry = Arc::new(WorkerRegistry::new());
        let config = ShutdownConfig::default();

        let metrics = shutdown_all_workers(registry, config).await;

        assert_eq!(metrics.total_workers, 0);
        assert_eq!(metrics.graceful_shutdown, 0);
        assert_eq!(metrics.force_killed, 0);
        assert_eq!(metrics.timeout_exceeded, 0);
    }

    #[tokio::test]
    async fn test_shutdown_metrics_default() {
        let metrics = ShutdownMetrics::default();
        assert_eq!(metrics.total_workers, 0);
        assert_eq!(metrics.graceful_shutdown, 0);
        assert_eq!(metrics.force_killed, 0);
        assert_eq!(metrics.timeout_exceeded, 0);
        assert_eq!(metrics.duration_secs, 0.0);
    }

    #[test]
    fn test_is_process_alive_self() {
        // Current process should be alive
        let pid = std::process::id();
        assert!(is_process_alive(pid));
    }

    #[test]
    fn test_is_process_alive_invalid() {
        // PID 999999 should not exist
        assert!(!is_process_alive(999999));
    }
}
