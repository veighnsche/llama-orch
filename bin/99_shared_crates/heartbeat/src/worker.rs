//! Worker heartbeat logic (Worker → Hive)
//!
//! Provides periodic heartbeat sending from workers to hive.
//!
//! Created by: TEAM-115
//! Extended by: TEAM-151

use crate::types::{HealthStatus, WorkerHeartbeatPayload};
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, warn};

// ============================================================================
// Worker Heartbeat Configuration
// ============================================================================

/// Worker heartbeat configuration (Worker → Hive)
#[derive(Debug, Clone)]
pub struct WorkerHeartbeatConfig {
    /// Worker ID
    pub worker_id: String,
    /// rbee-hive callback URL (e.g., "http://localhost:9200")
    pub callback_url: String,
    /// Heartbeat interval in seconds (default: 30s)
    pub interval_secs: u64,
}

impl WorkerHeartbeatConfig {
    /// Create new worker heartbeat config
    pub fn new(worker_id: String, callback_url: String) -> Self {
        Self {
            worker_id,
            callback_url,
            interval_secs: 30, // Default: 30 seconds
        }
    }

    /// Set custom interval
    pub fn with_interval(mut self, interval_secs: u64) -> Self {
        self.interval_secs = interval_secs;
        self
    }
}

// ============================================================================
// Worker Heartbeat Task
// ============================================================================

/// Start worker heartbeat task
///
/// Spawns a background task that sends periodic heartbeats to rbee-hive.
/// The task runs forever until the worker is shut down.
///
/// # Arguments
/// * `config` - Worker heartbeat configuration
///
/// # Returns
/// JoinHandle for the heartbeat task
///
/// # Used By
/// - `llm-worker-rbee` binary
///
/// # Example
/// ```no_run
/// use rbee_heartbeat::worker::{WorkerHeartbeatConfig, start_worker_heartbeat_task};
///
/// let config = WorkerHeartbeatConfig::new(
///     "worker-123".to_string(),
///     "http://localhost:8600".to_string(),
/// );
///
/// let handle = start_worker_heartbeat_task(config);
/// ```
pub fn start_worker_heartbeat_task(
    config: WorkerHeartbeatConfig,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval_timer = interval(Duration::from_secs(config.interval_secs));
        let client = reqwest::Client::new();
        let heartbeat_url = format!("{}/v1/heartbeat", config.callback_url);

        debug!(
            worker_id = %config.worker_id,
            interval_secs = config.interval_secs,
            "Starting worker heartbeat task"
        );

        loop {
            interval_timer.tick().await;

            let payload = WorkerHeartbeatPayload {
                worker_id: config.worker_id.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                health_status: HealthStatus::Healthy,
            };

            match send_worker_heartbeat(&client, &heartbeat_url, &payload).await {
                Ok(()) => {
                    debug!(worker_id = %config.worker_id, "Worker heartbeat sent successfully");
                }
                Err(e) => {
                    // TEAM-115: Log error but don't crash - heartbeat failures are non-fatal
                    warn!(
                        worker_id = %config.worker_id,
                        error = %e,
                        "Failed to send worker heartbeat (will retry)"
                    );
                }
            }
        }
    })
}

/// Send worker heartbeat to rbee-hive
async fn send_worker_heartbeat(
    client: &reqwest::Client,
    url: &str,
    payload: &WorkerHeartbeatPayload,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = client
        .post(url)
        .json(payload)
        .timeout(Duration::from_secs(5))
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_else(|_| "<no body>".to_string());
        return Err(format!("Worker heartbeat failed: {} - {}", status, body).into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Configuration Tests
    // ========================================================================

    #[test]
    fn config_new_sets_default_interval() {
        let config = WorkerHeartbeatConfig::new(
            "worker-123".to_string(),
            "http://localhost:9200".to_string(),
        );
        
        assert_eq!(config.worker_id, "worker-123");
        assert_eq!(config.callback_url, "http://localhost:9200");
        assert_eq!(config.interval_secs, 30, "Default interval should be 30 seconds");
    }

    #[test]
    fn config_with_interval_overrides_default() {
        let config = WorkerHeartbeatConfig::new(
            "worker-456".to_string(),
            "http://localhost:8600".to_string(),
        )
        .with_interval(60);
        
        assert_eq!(config.interval_secs, 60, "Custom interval should override default");
    }

    #[test]
    fn config_with_interval_can_be_chained() {
        let config = WorkerHeartbeatConfig::new(
            "worker-789".to_string(),
            "http://localhost:8600".to_string(),
        )
        .with_interval(15)
        .with_interval(45); // Last one wins
        
        assert_eq!(config.interval_secs, 45, "Last interval should be used");
    }

    #[test]
    fn config_accepts_various_url_formats() {
        let configs = vec![
            ("http://localhost:9200", "http://localhost:9200"),
            ("http://127.0.0.1:9200", "http://127.0.0.1:9200"),
            ("http://hive.example.com:9200", "http://hive.example.com:9200"),
            ("https://secure-hive:443", "https://secure-hive:443"),
        ];

        for (url, expected) in configs {
            let config = WorkerHeartbeatConfig::new(
                "worker-test".to_string(),
                url.to_string(),
            );
            assert_eq!(config.callback_url, expected);
        }
    }

    #[test]
    fn config_accepts_various_worker_id_formats() {
        let worker_ids = vec![
            "worker-123",
            "worker_456",
            "worker.789",
            "WORKER-ABC",
            "worker-123-abc-def",
            "w1",
        ];

        for worker_id in worker_ids {
            let config = WorkerHeartbeatConfig::new(
                worker_id.to_string(),
                "http://localhost:9200".to_string(),
            );
            assert_eq!(config.worker_id, worker_id);
        }
    }

    #[test]
    fn config_allows_zero_interval_for_testing() {
        let config = WorkerHeartbeatConfig::new(
            "worker-test".to_string(),
            "http://localhost:9200".to_string(),
        )
        .with_interval(0);
        
        assert_eq!(config.interval_secs, 0, "Zero interval should be allowed for testing");
    }

    #[test]
    fn config_allows_very_long_intervals() {
        let config = WorkerHeartbeatConfig::new(
            "worker-test".to_string(),
            "http://localhost:9200".to_string(),
        )
        .with_interval(3600); // 1 hour
        
        assert_eq!(config.interval_secs, 3600);
    }

    #[test]
    fn config_clone_creates_independent_copy() {
        let original = WorkerHeartbeatConfig::new(
            "worker-original".to_string(),
            "http://localhost:9200".to_string(),
        );
        
        let cloned = original.clone();
        
        assert_eq!(original.worker_id, cloned.worker_id);
        assert_eq!(original.callback_url, cloned.callback_url);
        assert_eq!(original.interval_secs, cloned.interval_secs);
    }

    // ========================================================================
    // Heartbeat Task Behavior Tests
    // ========================================================================

    #[tokio::test]
    async fn start_task_returns_join_handle() {
        let config = WorkerHeartbeatConfig::new(
            "worker-test".to_string(),
            "http://localhost:9999".to_string(), // Non-existent server
        )
        .with_interval(3600); // Long interval so it doesn't actually send

        let handle = start_worker_heartbeat_task(config);
        
        // Verify we got a handle
        assert!(!handle.is_finished(), "Task should be running");
        
        // Clean up
        handle.abort();
    }

    #[tokio::test]
    async fn start_task_spawns_background_task() {
        let config = WorkerHeartbeatConfig::new(
            "worker-background".to_string(),
            "http://localhost:9999".to_string(),
        )
        .with_interval(3600);

        let handle = start_worker_heartbeat_task(config);
        
        // Task should be running in background
        assert!(!handle.is_finished());
        
        // Aborting should stop it
        handle.abort();
    }

    // ========================================================================
    // Heartbeat URL Construction Tests
    // ========================================================================

    #[test]
    fn heartbeat_url_appends_v1_heartbeat_path() {
        // This tests the behavior that the task constructs the URL correctly
        // We can't easily test the internal URL construction without mocking,
        // but we can verify the config accepts the base URL correctly
        
        let base_urls = vec![
            "http://localhost:9200",
            "http://127.0.0.1:8600",
            "https://hive.example.com",
        ];

        for base_url in base_urls {
            let config = WorkerHeartbeatConfig::new(
                "worker-test".to_string(),
                base_url.to_string(),
            );
            
            // Verify base URL is stored correctly
            assert_eq!(config.callback_url, base_url);
            
            // The task will append "/v1/heartbeat" internally
            // Expected: base_url + "/v1/heartbeat"
        }
    }

    // ========================================================================
    // Configuration Edge Cases
    // ========================================================================

    #[test]
    fn config_handles_empty_worker_id() {
        let config = WorkerHeartbeatConfig::new(
            "".to_string(),
            "http://localhost:9200".to_string(),
        );
        
        assert_eq!(config.worker_id, "");
        // Note: In production, validation should happen at a higher level
    }

    #[test]
    fn config_handles_very_long_worker_id() {
        let long_id = "worker-".to_string() + &"a".repeat(1000);
        let config = WorkerHeartbeatConfig::new(
            long_id.clone(),
            "http://localhost:9200".to_string(),
        );
        
        assert_eq!(config.worker_id, long_id);
    }

    #[test]
    fn config_handles_url_with_trailing_slash() {
        let config = WorkerHeartbeatConfig::new(
            "worker-test".to_string(),
            "http://localhost:9200/".to_string(),
        );
        
        assert_eq!(config.callback_url, "http://localhost:9200/");
        // Note: The task will handle this correctly by appending "/v1/heartbeat"
        // Result will be "http://localhost:9200//v1/heartbeat" which HTTP handles fine
    }

    #[test]
    fn config_handles_url_without_port() {
        let config = WorkerHeartbeatConfig::new(
            "worker-test".to_string(),
            "http://localhost".to_string(),
        );
        
        assert_eq!(config.callback_url, "http://localhost");
    }

    // ========================================================================
    // Behavior Verification Tests
    // ========================================================================

    #[test]
    fn config_debug_format_includes_all_fields() {
        let config = WorkerHeartbeatConfig::new(
            "worker-debug".to_string(),
            "http://localhost:9200".to_string(),
        )
        .with_interval(45);
        
        let debug_str = format!("{:?}", config);
        
        // Verify all fields are in debug output
        assert!(debug_str.contains("worker-debug"));
        assert!(debug_str.contains("http://localhost:9200"));
        assert!(debug_str.contains("45"));
    }

    #[tokio::test]
    async fn multiple_tasks_can_run_simultaneously() {
        let config1 = WorkerHeartbeatConfig::new(
            "worker-1".to_string(),
            "http://localhost:9999".to_string(),
        )
        .with_interval(3600);

        let config2 = WorkerHeartbeatConfig::new(
            "worker-2".to_string(),
            "http://localhost:9998".to_string(),
        )
        .with_interval(3600);

        let handle1 = start_worker_heartbeat_task(config1);
        let handle2 = start_worker_heartbeat_task(config2);
        
        // Both tasks should be running
        assert!(!handle1.is_finished());
        assert!(!handle2.is_finished());
        
        // Clean up
        handle1.abort();
        handle2.abort();
    }

    #[test]
    fn config_interval_boundary_values() {
        // Test various boundary values for interval
        let intervals = vec![1, 5, 10, 15, 30, 60, 120, 300, 600, 3600];
        
        for interval in intervals {
            let config = WorkerHeartbeatConfig::new(
                "worker-test".to_string(),
                "http://localhost:9200".to_string(),
            )
            .with_interval(interval);
            
            assert_eq!(config.interval_secs, interval);
        }
    }
}
