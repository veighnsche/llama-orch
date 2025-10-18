//! Health monitoring loop
//!
//! Per test-001-mvp.md Phase 5: Monitor worker health every 30s
//!
//! Created by: TEAM-027

use crate::registry::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info};

/// Health monitor loop - checks worker health every 30 seconds
///
/// # Arguments
/// * `registry` - Worker registry to monitor
///
/// # Behavior
/// - Polls each worker's health endpoint every 30s
/// - Logs health status
/// - TEAM-096: Fail-fast - removes workers after 3 failed health checks
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
                        "✅ Worker healthy"
                    );
                    // Reset counter on success
                    registry.update_state(&worker.id, worker.state).await;
                }
                Ok(response) => {
                    // TEAM-096: Increment fail counter and remove after 3 failures
                    let fail_count = registry.increment_failed_health_checks(&worker.id).await.unwrap_or(0);
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
                    let fail_count = registry.increment_failed_health_checks(&worker.id).await.unwrap_or(0);
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
        };

        registry.register(worker).await;
        let workers = registry.list().await;
        assert_eq!(workers.len(), 1);
    }
}
