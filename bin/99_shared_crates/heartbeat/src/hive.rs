//! Hive heartbeat logic (Hive → Queen)
//!
//! Provides periodic aggregated heartbeat sending from hive to queen.
//!
//! Created by: TEAM-151

use crate::types::{HiveHeartbeatPayload, WorkerState};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, warn};

// ============================================================================
// Hive Heartbeat Configuration
// ============================================================================

/// Hive heartbeat configuration (Hive → Queen)
///
/// The hive periodically sends aggregated worker states to queen
#[derive(Debug, Clone)]
pub struct HiveHeartbeatConfig {
    /// Hive ID (e.g., "localhost" or hostname)
    pub hive_id: String,
    /// Queen callback URL (e.g., "http://localhost:8500")
    pub queen_url: String,
    /// Heartbeat interval in seconds (default: 15s)
    pub interval_secs: u64,
    /// Authentication token for queen
    pub auth_token: String,
}

impl HiveHeartbeatConfig {
    /// Create new hive heartbeat config
    pub fn new(hive_id: String, queen_url: String, auth_token: String) -> Self {
        Self {
            hive_id,
            queen_url,
            auth_token,
            interval_secs: 15, // Default: 15 seconds (faster than worker heartbeats)
        }
    }

    /// Set custom interval
    pub fn with_interval(mut self, interval_secs: u64) -> Self {
        self.interval_secs = interval_secs;
        self
    }
}

// ============================================================================
// Worker State Provider Trait
// ============================================================================

/// Trait for getting worker states from hive registry
///
/// The hive must implement this to provide worker states for aggregation
pub trait WorkerStateProvider: Send + Sync {
    /// Get all worker states for heartbeat aggregation
    fn get_worker_states(&self) -> Vec<WorkerState>;
}

// ============================================================================
// Hive Heartbeat Task
// ============================================================================

/// Start hive heartbeat task
///
/// Spawns a background task that sends periodic aggregated heartbeats to queen-rbee.
/// The task collects worker states from the registry and sends them to queen.
///
/// # Arguments
/// * `config` - Hive heartbeat configuration
/// * `worker_provider` - Provider for getting worker states from registry
///
/// # Returns
/// JoinHandle for the heartbeat task
///
/// # Used By
/// - `rbee-hive` binary
///
/// # Example
/// ```no_run
/// use rbee_heartbeat::hive::{HiveHeartbeatConfig, start_hive_heartbeat_task, WorkerStateProvider};
/// use std::sync::Arc;
///
/// // Implement WorkerStateProvider for your registry
/// // impl WorkerStateProvider for MyRegistry { ... }
///
/// let config = HiveHeartbeatConfig::new(
///     "localhost".to_string(),
///     "http://localhost:8500".to_string(),
///     "auth-token".to_string(),
/// );
///
/// // let provider: Arc<dyn WorkerStateProvider> = Arc::new(my_registry);
/// // let handle = start_hive_heartbeat_task(config, provider);
/// ```
pub fn start_hive_heartbeat_task(
    config: HiveHeartbeatConfig,
    worker_provider: Arc<dyn WorkerStateProvider>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval_timer = interval(Duration::from_secs(config.interval_secs));
        let client = reqwest::Client::new();
        let heartbeat_url = format!("{}/v1/heartbeat", config.queen_url);

        debug!(
            hive_id = %config.hive_id,
            interval_secs = config.interval_secs,
            "Starting hive heartbeat task"
        );

        loop {
            interval_timer.tick().await;

            // Get current worker states from registry
            let workers = worker_provider.get_worker_states();

            let payload = HiveHeartbeatPayload {
                hive_id: config.hive_id.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                workers,
            };

            match send_hive_heartbeat(&client, &heartbeat_url, &config.auth_token, &payload).await
            {
                Ok(()) => {
                    debug!(
                        hive_id = %config.hive_id,
                        worker_count = payload.workers.len(),
                        "Hive heartbeat sent successfully"
                    );
                }
                Err(e) => {
                    // TEAM-151: Log error but don't crash - heartbeat failures are non-fatal
                    warn!(
                        hive_id = %config.hive_id,
                        error = %e,
                        "Failed to send hive heartbeat (will retry)"
                    );
                }
            }
        }
    })
}

/// Send hive heartbeat to queen-rbee
async fn send_hive_heartbeat(
    client: &reqwest::Client,
    url: &str,
    auth_token: &str,
    payload: &HiveHeartbeatPayload,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", auth_token))
        .json(payload)
        .timeout(Duration::from_secs(5))
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_else(|_| "<no body>".to_string());
        return Err(format!("Hive heartbeat failed: {} - {}", status, body).into());
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
        let config = HiveHeartbeatConfig::new(
            "localhost".to_string(),
            "http://localhost:8500".to_string(),
            "test-token".to_string(),
        );
        
        assert_eq!(config.hive_id, "localhost");
        assert_eq!(config.queen_url, "http://localhost:8500");
        assert_eq!(config.auth_token, "test-token");
        assert_eq!(config.interval_secs, 15, "Default interval should be 15 seconds");
    }

    #[test]
    fn config_with_interval_overrides_default() {
        let config = HiveHeartbeatConfig::new(
            "hive-prod".to_string(),
            "http://queen:8500".to_string(),
            "prod-token".to_string(),
        )
        .with_interval(30);
        
        assert_eq!(config.interval_secs, 30, "Custom interval should override default");
    }

    #[test]
    fn config_with_interval_can_be_chained() {
        let config = HiveHeartbeatConfig::new(
            "hive-test".to_string(),
            "http://localhost:8500".to_string(),
            "token".to_string(),
        )
        .with_interval(10)
        .with_interval(20); // Last one wins
        
        assert_eq!(config.interval_secs, 20, "Last interval should be used");
    }

    #[test]
    fn config_accepts_various_hive_id_formats() {
        let hive_ids = vec![
            "localhost",
            "hive-prod-1",
            "hive_test",
            "192.168.1.100",
            "hive.example.com",
            "h1",
        ];

        for hive_id in hive_ids {
            let config = HiveHeartbeatConfig::new(
                hive_id.to_string(),
                "http://localhost:8500".to_string(),
                "token".to_string(),
            );
            assert_eq!(config.hive_id, hive_id);
        }
    }

    #[test]
    fn config_accepts_various_queen_url_formats() {
        let urls = vec![
            ("http://localhost:8500", "http://localhost:8500"),
            ("http://127.0.0.1:8500", "http://127.0.0.1:8500"),
            ("http://queen.example.com:8500", "http://queen.example.com:8500"),
            ("https://secure-queen:443", "https://secure-queen:443"),
        ];

        for (url, expected) in urls {
            let config = HiveHeartbeatConfig::new(
                "hive-test".to_string(),
                url.to_string(),
                "token".to_string(),
            );
            assert_eq!(config.queen_url, expected);
        }
    }

    #[test]
    fn config_stores_auth_token_correctly() {
        let tokens = vec![
            "simple-token",
            "Bearer abc123",
            "very-long-token-with-many-characters-1234567890",
            "token-with-special-chars!@#$%",
        ];

        for token in tokens {
            let config = HiveHeartbeatConfig::new(
                "hive-test".to_string(),
                "http://localhost:8500".to_string(),
                token.to_string(),
            );
            assert_eq!(config.auth_token, token);
        }
    }

    #[test]
    fn config_allows_zero_interval_for_testing() {
        let config = HiveHeartbeatConfig::new(
            "hive-test".to_string(),
            "http://localhost:8500".to_string(),
            "token".to_string(),
        )
        .with_interval(0);
        
        assert_eq!(config.interval_secs, 0, "Zero interval should be allowed for testing");
    }

    #[test]
    fn config_allows_very_short_intervals() {
        let config = HiveHeartbeatConfig::new(
            "hive-test".to_string(),
            "http://localhost:8500".to_string(),
            "token".to_string(),
        )
        .with_interval(1); // 1 second
        
        assert_eq!(config.interval_secs, 1);
    }

    #[test]
    fn config_allows_very_long_intervals() {
        let config = HiveHeartbeatConfig::new(
            "hive-test".to_string(),
            "http://localhost:8500".to_string(),
            "token".to_string(),
        )
        .with_interval(3600); // 1 hour
        
        assert_eq!(config.interval_secs, 3600);
    }

    #[test]
    fn config_clone_creates_independent_copy() {
        let original = HiveHeartbeatConfig::new(
            "hive-original".to_string(),
            "http://localhost:8500".to_string(),
            "original-token".to_string(),
        );
        
        let cloned = original.clone();
        
        assert_eq!(original.hive_id, cloned.hive_id);
        assert_eq!(original.queen_url, cloned.queen_url);
        assert_eq!(original.auth_token, cloned.auth_token);
        assert_eq!(original.interval_secs, cloned.interval_secs);
    }

    // ========================================================================
    // WorkerStateProvider Trait Tests
    // ========================================================================

    struct MockWorkerStateProvider {
        states: Vec<WorkerState>,
    }

    impl WorkerStateProvider for MockWorkerStateProvider {
        fn get_worker_states(&self) -> Vec<WorkerState> {
            self.states.clone()
        }
    }

    #[test]
    fn worker_state_provider_returns_empty_list() {
        let provider = MockWorkerStateProvider {
            states: vec![],
        };
        
        let states = provider.get_worker_states();
        assert_eq!(states.len(), 0);
    }

    #[test]
    fn worker_state_provider_returns_single_worker() {
        let provider = MockWorkerStateProvider {
            states: vec![WorkerState {
                worker_id: "worker-1".to_string(),
                state: "Idle".to_string(),
                last_heartbeat: "2025-10-20T00:00:00Z".to_string(),
                health_status: "healthy".to_string(),
            }],
        };
        
        let states = provider.get_worker_states();
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].worker_id, "worker-1");
    }

    #[test]
    fn worker_state_provider_returns_multiple_workers() {
        let provider = MockWorkerStateProvider {
            states: vec![
                WorkerState {
                    worker_id: "worker-1".to_string(),
                    state: "Idle".to_string(),
                    last_heartbeat: "2025-10-20T00:00:00Z".to_string(),
                    health_status: "healthy".to_string(),
                },
                WorkerState {
                    worker_id: "worker-2".to_string(),
                    state: "Busy".to_string(),
                    last_heartbeat: "2025-10-20T00:00:05Z".to_string(),
                    health_status: "healthy".to_string(),
                },
                WorkerState {
                    worker_id: "worker-3".to_string(),
                    state: "Loading".to_string(),
                    last_heartbeat: "2025-10-20T00:00:10Z".to_string(),
                    health_status: "degraded".to_string(),
                },
            ],
        };
        
        let states = provider.get_worker_states();
        assert_eq!(states.len(), 3);
        assert_eq!(states[0].worker_id, "worker-1");
        assert_eq!(states[1].worker_id, "worker-2");
        assert_eq!(states[2].worker_id, "worker-3");
    }

    #[test]
    fn worker_state_provider_can_be_called_multiple_times() {
        let provider = MockWorkerStateProvider {
            states: vec![WorkerState {
                worker_id: "worker-test".to_string(),
                state: "Idle".to_string(),
                last_heartbeat: "2025-10-20T00:00:00Z".to_string(),
                health_status: "healthy".to_string(),
            }],
        };
        
        let states1 = provider.get_worker_states();
        let states2 = provider.get_worker_states();
        
        assert_eq!(states1.len(), states2.len());
        assert_eq!(states1[0].worker_id, states2[0].worker_id);
    }

    // ========================================================================
    // Heartbeat Task Behavior Tests
    // ========================================================================

    #[tokio::test]
    async fn start_task_returns_join_handle() {
        let config = HiveHeartbeatConfig::new(
            "hive-test".to_string(),
            "http://localhost:9999".to_string(), // Non-existent server
            "test-token".to_string(),
        )
        .with_interval(3600); // Long interval so it doesn't actually send

        let provider = Arc::new(MockWorkerStateProvider {
            states: vec![],
        });

        let handle = start_hive_heartbeat_task(config, provider);
        
        // Verify we got a handle
        assert!(!handle.is_finished(), "Task should be running");
        
        // Clean up
        handle.abort();
    }

    #[tokio::test]
    async fn start_task_spawns_background_task() {
        let config = HiveHeartbeatConfig::new(
            "hive-background".to_string(),
            "http://localhost:9999".to_string(),
            "token".to_string(),
        )
        .with_interval(3600);

        let provider = Arc::new(MockWorkerStateProvider {
            states: vec![],
        });

        let handle = start_hive_heartbeat_task(config, provider);
        
        // Task should be running in background
        assert!(!handle.is_finished());
        
        // Aborting should stop it
        handle.abort();
    }

    #[tokio::test]
    async fn start_task_accepts_provider_with_workers() {
        let config = HiveHeartbeatConfig::new(
            "hive-test".to_string(),
            "http://localhost:9999".to_string(),
            "token".to_string(),
        )
        .with_interval(3600);

        let provider = Arc::new(MockWorkerStateProvider {
            states: vec![
                WorkerState {
                    worker_id: "worker-1".to_string(),
                    state: "Idle".to_string(),
                    last_heartbeat: "2025-10-20T00:00:00Z".to_string(),
                    health_status: "healthy".to_string(),
                },
            ],
        });

        let handle = start_hive_heartbeat_task(config, provider);
        
        assert!(!handle.is_finished());
        handle.abort();
    }

    // ========================================================================
    // Configuration Edge Cases
    // ========================================================================

    #[test]
    fn config_handles_empty_hive_id() {
        let config = HiveHeartbeatConfig::new(
            "".to_string(),
            "http://localhost:8500".to_string(),
            "token".to_string(),
        );
        
        assert_eq!(config.hive_id, "");
        // Note: In production, validation should happen at a higher level
    }

    #[test]
    fn config_handles_empty_auth_token() {
        let config = HiveHeartbeatConfig::new(
            "hive-test".to_string(),
            "http://localhost:8500".to_string(),
            "".to_string(),
        );
        
        assert_eq!(config.auth_token, "");
        // Note: This might be intentional for local/dev mode
    }

    #[test]
    fn config_handles_very_long_hive_id() {
        let long_id = "hive-".to_string() + &"a".repeat(1000);
        let config = HiveHeartbeatConfig::new(
            long_id.clone(),
            "http://localhost:8500".to_string(),
            "token".to_string(),
        );
        
        assert_eq!(config.hive_id, long_id);
    }

    #[test]
    fn config_handles_url_with_trailing_slash() {
        let config = HiveHeartbeatConfig::new(
            "hive-test".to_string(),
            "http://localhost:8500/".to_string(),
            "token".to_string(),
        );
        
        assert_eq!(config.queen_url, "http://localhost:8500/");
    }

    #[test]
    fn config_handles_url_without_port() {
        let config = HiveHeartbeatConfig::new(
            "hive-test".to_string(),
            "http://localhost".to_string(),
            "token".to_string(),
        );
        
        assert_eq!(config.queen_url, "http://localhost");
    }

    // ========================================================================
    // Behavior Verification Tests
    // ========================================================================

    #[test]
    fn config_debug_format_includes_all_fields() {
        let config = HiveHeartbeatConfig::new(
            "hive-debug".to_string(),
            "http://localhost:8500".to_string(),
            "debug-token".to_string(),
        )
        .with_interval(25);
        
        let debug_str = format!("{:?}", config);
        
        // Verify all fields are in debug output
        assert!(debug_str.contains("hive-debug"));
        assert!(debug_str.contains("http://localhost:8500"));
        assert!(debug_str.contains("debug-token"));
        assert!(debug_str.contains("25"));
    }

    #[tokio::test]
    async fn multiple_hive_tasks_can_run_simultaneously() {
        let config1 = HiveHeartbeatConfig::new(
            "hive-1".to_string(),
            "http://localhost:9999".to_string(),
            "token1".to_string(),
        )
        .with_interval(3600);

        let config2 = HiveHeartbeatConfig::new(
            "hive-2".to_string(),
            "http://localhost:9998".to_string(),
            "token2".to_string(),
        )
        .with_interval(3600);

        let provider1 = Arc::new(MockWorkerStateProvider { states: vec![] });
        let provider2 = Arc::new(MockWorkerStateProvider { states: vec![] });

        let handle1 = start_hive_heartbeat_task(config1, provider1);
        let handle2 = start_hive_heartbeat_task(config2, provider2);
        
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
            let config = HiveHeartbeatConfig::new(
                "hive-test".to_string(),
                "http://localhost:8500".to_string(),
                "token".to_string(),
            )
            .with_interval(interval);
            
            assert_eq!(config.interval_secs, interval);
        }
    }

    #[test]
    fn hive_interval_is_faster_than_worker_default() {
        let hive_config = HiveHeartbeatConfig::new(
            "hive".to_string(),
            "http://localhost:8500".to_string(),
            "token".to_string(),
        );
        
        // Hive default is 15s, worker default is 30s
        assert_eq!(hive_config.interval_secs, 15);
        assert!(hive_config.interval_secs < 30, "Hive should send heartbeats faster than workers");
    }

    #[test]
    fn worker_state_provider_trait_is_send_sync() {
        // This test verifies the trait bounds are correct
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn WorkerStateProvider>>();
    }
}
