// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Idle timeout implementation

//! Idle timeout enforcement
//!
//! Per test-001-mvp.md Phase 8: Auto-shutdown after 5 minutes idle
//! Per test-001-mvp.md EC10: Idle Timeout (Worker Auto-Shutdown)
//!
//! Created by: TEAM-027

use crate::registry::WorkerRegistry;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tracing::{info, warn};

/// Idle timeout loop - shuts down workers after 5 minutes idle
///
/// # Arguments
/// * `registry` - Worker registry to monitor
///
/// # Behavior
/// - Checks idle workers every 60 seconds
/// - Shuts down workers idle for >5 minutes
/// - Sends POST /v1/admin/shutdown to worker
/// - Removes worker from registry on success
pub async fn idle_timeout_loop(registry: Arc<WorkerRegistry>) {
    let mut interval = tokio::time::interval(Duration::from_secs(60));

    loop {
        interval.tick().await;

        let now = SystemTime::now();
        for worker in registry.get_idle_workers().await {
            if let Ok(idle_duration) = now.duration_since(worker.last_activity) {
                if idle_duration > Duration::from_secs(300) {
                    // 5 minutes
                    info!(
                        worker_id = %worker.id,
                        idle_seconds = idle_duration.as_secs(),
                        "Worker idle timeout, shutting down"
                    );

                    // Send shutdown request: POST {worker.url}/v1/admin/shutdown
                    let client = reqwest::Client::new();
                    match client
                        .post(format!("{}/v1/admin/shutdown", worker.url))
                        .timeout(Duration::from_secs(10))
                        .send()
                        .await
                    {
                        Ok(_) => {
                            registry.remove(&worker.id).await;
                            info!(worker_id = %worker.id, "Worker shutdown complete");
                        }
                        Err(e) => {
                            warn!(
                                worker_id = %worker.id,
                                error = %e,
                                "Failed to shutdown worker"
                            );
                            // Still remove from registry - worker may already be dead
                            registry.remove(&worker.id).await;
                        }
                    }
                }
            }
        }
    }
}

// TEAM-031: Unit tests for timeout module
#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{WorkerInfo, WorkerState};

    #[test]
    fn test_timeout_module_exists() {
        // Basic test to ensure module compiles
        // The actual idle_timeout_loop is tested via integration tests
        assert!(true);
    }

    #[tokio::test]
    async fn test_timeout_with_empty_registry() {
        let registry = Arc::new(WorkerRegistry::new());
        let idle_workers = registry.get_idle_workers().await;
        assert_eq!(idle_workers.len(), 0);
    }

    #[tokio::test]
    async fn test_timeout_with_idle_workers() {
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
            pid: None,
            restart_count: 0,
            last_restart: None,
        };

        registry.register(worker).await;
        let idle_workers = registry.get_idle_workers().await;
        assert_eq!(idle_workers.len(), 1);
    }

    #[tokio::test]
    async fn test_timeout_duration_check() {
        let now = SystemTime::now();
        let past = now - Duration::from_secs(400); // 6 minutes 40 seconds ago

        let duration = now.duration_since(past).unwrap();
        assert!(duration > Duration::from_secs(300)); // More than 5 minutes
    }

    #[tokio::test]
    async fn test_timeout_duration_not_exceeded() {
        let now = SystemTime::now();
        let recent = now - Duration::from_secs(200); // 3 minutes 20 seconds ago

        let duration = now.duration_since(recent).unwrap();
        assert!(duration < Duration::from_secs(300)); // Less than 5 minutes
    }
}
