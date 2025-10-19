// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - PID tracking and force-kill implemented

//! Health monitoring loop
//!
//! Per test-001-mvp.md Phase 5: Monitor worker health every 30s
//!
//! Created by: TEAM-027

use crate::registry::WorkerRegistry;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tracing::{error, info, warn};

/// Health monitor loop - checks worker health every 30 seconds
///
/// # Arguments
/// * `registry` - Worker registry to monitor
///
/// # Behavior
/// - Polls each worker's health endpoint every 30s
/// - Logs health status
/// - TEAM-096: Fail-fast - removes workers after 3 failed health checks
/// - TEAM-101: Ready timeout - kills workers stuck in Loading > 30s
pub async fn health_monitor_loop(registry: Arc<WorkerRegistry>) {
    let mut interval = tokio::time::interval(Duration::from_secs(30));

    loop {
        interval.tick().await;

        let workers = registry.list().await;
        if workers.is_empty() {
            info!("🔍 Health monitor: No workers to check");
            continue;
        }

        info!("🔍 Health monitor: Checking {} workers", workers.len());

        for worker in workers {
            // TEAM-115: Check for stale workers (no heartbeat in 60 seconds)
            if let Some(last_heartbeat) = worker.last_heartbeat {
                let heartbeat_age = last_heartbeat.elapsed().unwrap_or(Duration::from_secs(0));
                if heartbeat_age > Duration::from_secs(60) {
                    warn!(
                        worker_id = %worker.id,
                        heartbeat_age_secs = heartbeat_age.as_secs(),
                        "⚠️  TEAM-115: Worker is stale (no heartbeat in 60s), removing from registry"
                    );
                    
                    // Force-kill the stale worker if we have PID
                    if let Some(pid) = worker.pid {
                        force_kill_worker(pid, &worker.id);
                    }
                    
                    // Remove from registry
                    registry.remove(&worker.id).await;
                    continue;
                }
            }

            // TEAM-101: Check for workers stuck in Loading state
            if worker.state == crate::registry::WorkerState::Loading {
                let loading_duration =
                    worker.last_activity.elapsed().unwrap_or(Duration::from_secs(0));
                if loading_duration > Duration::from_secs(30) {
                    error!(
                        worker_id = %worker.id,
                        duration_secs = loading_duration.as_secs(),
                        "TEAM-101: Worker stuck in Loading state, force-killing"
                    );

                    // Force-kill the worker
                    if let Some(pid) = worker.pid {
                        force_kill_worker(pid, &worker.id);
                    }

                    // Remove from registry
                    registry.remove(&worker.id).await;
                    continue;
                }
            }

            // TEAM-101: Process liveness check - verify process exists via PID
            if let Some(pid) = worker.pid {
                use sysinfo::{Pid, System};
                let mut sys = System::new();
                sys.refresh_processes();

                let pid_obj = Pid::from_u32(pid);
                if sys.process(pid_obj).is_none() {
                    error!(
                        worker_id = %worker.id,
                        pid = pid,
                        "TEAM-101: Worker process no longer exists (crashed), removing from registry"
                    );
                    registry.remove(&worker.id).await;
                    continue;
                }
            }

            // Check worker health: GET {worker.url}/v1/health
            let client = reqwest::Client::new();
            match client
                .get(format!("{}/v1/health", worker.url))
                .timeout(Duration::from_secs(5))
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {
                    info!(
                        worker_id = %worker.id,
                        url = %worker.url,
                        state = ?worker.state,
                        pid = ?worker.pid,
                        "✅ Worker healthy (HTTP + process liveness verified)"
                    );
                    // Reset counter on success
                    registry.update_state(&worker.id, worker.state).await;
                }
                Ok(response) => {
                    // TEAM-096: Increment fail counter and remove after 3 failures
                    let fail_count =
                        registry.increment_failed_health_checks(&worker.id).await.unwrap_or(0);
                    error!(
                        worker_id = %worker.id,
                        url = %worker.url,
                        status = %response.status(),
                        failed_checks = fail_count,
                        "❌ Worker unhealthy"
                    );

                    if fail_count >= 3 {
                        error!(
                            worker_id = %worker.id,
                            url = %worker.url,
                            "🚨 FAIL-FAST: Removing worker after 3 failed health checks"
                        );
                        registry.remove(&worker.id).await;
                    }
                }
                Err(e) => {
                    // TEAM-096: Increment fail counter and remove after 3 failures
                    let fail_count =
                        registry.increment_failed_health_checks(&worker.id).await.unwrap_or(0);
                    error!(
                        worker_id = %worker.id,
                        url = %worker.url,
                        error = %e,
                        failed_checks = fail_count,
                        "❌ Worker unreachable"
                    );

                    if fail_count >= 3 {
                        error!(
                            worker_id = %worker.id,
                            url = %worker.url,
                            "🚨 FAIL-FAST: Removing worker after 3 failed health checks"
                        );
                        registry.remove(&worker.id).await;
                    }
                }
            }
        }
    }
}

/// TEAM-101: Force-kill a worker process using its PID
fn force_kill_worker(pid: u32, worker_id: &str) {
    use sysinfo::{Pid, Signal, System};

    let mut sys = System::new();
    sys.refresh_processes();

    let pid_obj = Pid::from_u32(pid);
    if let Some(process) = sys.process(pid_obj) {
        info!(worker_id = worker_id, pid = pid, "TEAM-101: Force-killing worker process");

        if process.kill_with(Signal::Kill).is_some() {
            info!(
                worker_id = worker_id,
                pid = pid,
                signal = "SIGKILL",
                "Successfully force-killed worker"
            );
        } else {
            error!(worker_id = worker_id, pid = pid, "Failed to send SIGKILL to worker");
        }
    } else {
        info!(worker_id = worker_id, pid = pid, "Worker process already exited");
    }
}

/// TEAM-104: Determine if a worker should be restarted based on restart policy
///
/// # Restart Policy
/// - Maximum 3 restart attempts per worker
/// - Exponential backoff: 2^restart_count seconds (1s, 2s, 4s)
/// - Circuit breaker: stop restarting after max attempts
///
/// # Arguments
/// * `worker` - Worker info to check
///
/// # Returns
/// `true` if worker should be restarted, `false` otherwise
///
/// # Example
/// ```rust,no_run
/// use rbee_hive::registry::WorkerInfo;
///
/// if should_restart_worker(&worker) {
///     // Restart the worker
///     // 1. Increment restart_count
///     // 2. Set last_restart = SystemTime::now()
///     // 3. Spawn new worker with same config
/// }
/// ```
pub fn should_restart_worker(worker: &crate::registry::WorkerInfo) -> bool {
    const MAX_RESTARTS: u32 = 3;

    // Check restart count - circuit breaker
    if worker.restart_count >= MAX_RESTARTS {
        warn!(
            worker_id = %worker.id,
            restart_count = worker.restart_count,
            "TEAM-104: Worker exceeded max restart attempts (circuit breaker)"
        );
        return false;
    }

    // Check exponential backoff
    if let Some(last_restart) = worker.last_restart {
        let backoff_duration = Duration::from_secs(2u64.pow(worker.restart_count));
        let elapsed = SystemTime::now().duration_since(last_restart).unwrap_or(Duration::ZERO);

        if elapsed < backoff_duration {
            info!(
                worker_id = %worker.id,
                restart_count = worker.restart_count,
                backoff_remaining_secs = (backoff_duration - elapsed).as_secs(),
                "TEAM-104: Worker in backoff period"
            );
            return false;
        }
    }

    // Passed all checks - can restart
    info!(
        worker_id = %worker.id,
        restart_count = worker.restart_count,
        "TEAM-104: Worker eligible for restart"
    );
    true
}

// TEAM-031: Unit tests for monitor module
#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{WorkerInfo, WorkerState};
    use std::time::SystemTime;

    #[test]
    fn test_monitor_module_exists() {
        // Basic test to ensure module compiles
        // The actual health_monitor_loop is tested via integration tests
        assert!(true);
    }

    #[tokio::test]
    async fn test_health_monitor_with_empty_registry() {
        let registry = Arc::new(WorkerRegistry::new());
        let workers = registry.list().await;
        assert_eq!(workers.len(), 0);
    }

    #[tokio::test]
    async fn test_health_monitor_with_workers() {
        let registry = Arc::new(WorkerRegistry::new());

        let worker = WorkerInfo {
            id: "worker-1".to_string(),
            url: "http://localhost:8081".to_string(),
            model_ref: "hf:test/model".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: WorkerState::Idle,
            last_activity: SystemTime::now(),
            slots_total: 1,
            slots_available: 1,
            failed_health_checks: 0,
            pid: None,          // TEAM-101: Added pid field
            restart_count: 0,   // TEAM-104: Added restart_count field
            last_restart: None, // TEAM-104: Added last_restart field
            last_heartbeat: None, // TEAM-115: Added last_heartbeat field
        };

        registry.register(worker).await;
        let workers = registry.list().await;
        assert_eq!(workers.len(), 1);
    }
}
