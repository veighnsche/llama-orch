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
                        .post(&format!("{}/v1/admin/shutdown", worker.url))
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
