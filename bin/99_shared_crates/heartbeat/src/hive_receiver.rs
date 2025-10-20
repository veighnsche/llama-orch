//! Hive heartbeat receiver - handles worker heartbeats
//!
//! Created by: TEAM-159
//! Consolidated from: rbee-hive/src/http/heartbeat.rs
//!
//! This module provides the logic for hives to receive heartbeats from workers.

use crate::traits::WorkerRegistry;
use crate::types::WorkerHeartbeatPayload;
use serde::Serialize;
use std::sync::Arc;
use tracing::{debug, warn};

/// Heartbeat response sent back to worker
#[derive(Debug, Serialize)]
pub struct HeartbeatResponse {
    /// Success message
    pub message: String,
}

/// Heartbeat error
#[derive(Debug, thiserror::Error)]
pub enum HeartbeatError {
    /// Worker not found in registry
    #[error("Worker not found: {0}")]
    WorkerNotFound(String),
    
    /// Other error
    #[error("{0}")]
    Other(String),
}

/// Handle worker heartbeat
///
/// Receives heartbeat from worker and updates registry.
///
/// # Arguments
/// * `registry` - Worker registry implementation
/// * `payload` - Heartbeat payload from worker
///
/// # Returns
/// * `Ok(HeartbeatResponse)` - Heartbeat acknowledged
/// * `Err(HeartbeatError::WorkerNotFound)` - Worker not in registry
///
/// # Example
/// ```no_run
/// use rbee_heartbeat::hive_receiver::handle_worker_heartbeat;
/// use rbee_heartbeat::types::WorkerHeartbeatPayload;
/// 
/// # async fn example(registry: std::sync::Arc<impl rbee_heartbeat::traits::WorkerRegistry>) {
/// let payload = WorkerHeartbeatPayload {
///     worker_id: "worker-123".to_string(),
///     timestamp: "2025-10-20T00:00:00Z".to_string(),
///     health_status: rbee_heartbeat::HealthStatus::Healthy,
/// };
///
/// let response = handle_worker_heartbeat(registry, payload).await.unwrap();
/// # }
/// ```
pub async fn handle_worker_heartbeat<R>(
    registry: Arc<R>,
    payload: WorkerHeartbeatPayload,
) -> Result<HeartbeatResponse, HeartbeatError>
where
    R: WorkerRegistry,
{
    debug!(
        worker_id = %payload.worker_id,
        timestamp = %payload.timestamp,
        "Received worker heartbeat"
    );

    // Update worker's last_heartbeat timestamp in registry
    let updated = registry.update_heartbeat(&payload.worker_id).await;

    if !updated {
        warn!(worker_id = %payload.worker_id, "Heartbeat from unknown worker");
        return Err(HeartbeatError::WorkerNotFound(payload.worker_id));
    }

    debug!(
        worker_id = %payload.worker_id,
        "Worker heartbeat processed - registry updated"
    );

    // NOTE: Relay to queen is handled by periodic hive heartbeat task
    // (see hive.rs: start_hive_heartbeat_task)
    // This collects ALL worker states and sends aggregated heartbeat to queen every 15s

    Ok(HeartbeatResponse {
        message: "Heartbeat received".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{HealthStatus, WorkerHeartbeatPayload};
    use async_trait::async_trait;
    use std::sync::Mutex;

    // Mock registry for testing
    struct MockRegistry {
        workers: Mutex<Vec<String>>,
    }

    impl MockRegistry {
        fn new() -> Self {
            Self {
                workers: Mutex::new(vec![]),
            }
        }

        fn add_worker(&self, worker_id: String) {
            self.workers.lock().unwrap().push(worker_id);
        }
    }

    #[async_trait]
    impl WorkerRegistry for MockRegistry {
        async fn update_heartbeat(&self, worker_id: &str) -> bool {
            self.workers.lock().unwrap().contains(&worker_id.to_string())
        }
    }

    #[tokio::test]
    async fn test_handle_worker_heartbeat_success() {
        let registry = Arc::new(MockRegistry::new());
        registry.add_worker("worker-123".to_string());

        let payload = WorkerHeartbeatPayload {
            worker_id: "worker-123".to_string(),
            timestamp: "2025-10-20T00:00:00Z".to_string(),
            health_status: HealthStatus::Healthy,
        };

        let result = handle_worker_heartbeat(registry, payload).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().message, "Heartbeat received");
    }

    #[tokio::test]
    async fn test_handle_worker_heartbeat_unknown_worker() {
        let registry = Arc::new(MockRegistry::new());
        // Don't add worker to registry

        let payload = WorkerHeartbeatPayload {
            worker_id: "unknown-worker".to_string(),
            timestamp: "2025-10-20T00:00:00Z".to_string(),
            health_status: HealthStatus::Healthy,
        };

        let result = handle_worker_heartbeat(registry, payload).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            HeartbeatError::WorkerNotFound(id) => assert_eq!(id, "unknown-worker"),
            _ => panic!("Expected WorkerNotFound error"),
        }
    }
}
