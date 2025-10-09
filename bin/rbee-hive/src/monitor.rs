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
/// - TODO: Mark workers as unhealthy on failure
pub async fn health_monitor_loop(registry: Arc<WorkerRegistry>) {
    let mut interval = tokio::time::interval(Duration::from_secs(30));

    loop {
        interval.tick().await;

        for worker in registry.list().await {
            // Check worker health: GET {worker.url}/v1/health
            let client = reqwest::Client::new();
            match client
                .get(&format!("{}/v1/health", worker.url))
                .timeout(Duration::from_secs(5))
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => {
                    info!(worker_id = %worker.id, url = %worker.url, "Worker healthy");
                }
                Ok(response) => {
                    error!(
                        worker_id = %worker.id,
                        url = %worker.url,
                        status = %response.status(),
                        "Worker unhealthy"
                    );
                    // TODO: Mark worker as unhealthy in registry
                }
                Err(e) => {
                    error!(
                        worker_id = %worker.id,
                        url = %worker.url,
                        error = %e,
                        "Worker unreachable"
                    );
                    // TODO: Mark worker as unhealthy in registry
                }
            }
        }
    }
}
