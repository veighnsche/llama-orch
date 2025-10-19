//! Heartbeat mechanism for worker health monitoring
//!
//! Workers send periodic heartbeats to rbee-hive to indicate they are alive and healthy.
//! If heartbeats stop, rbee-hive can detect the worker as stale and take action.
//!
//! Created by: TEAM-115

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, error, warn};

/// Heartbeat payload sent to rbee-hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatPayload {
    /// Worker ID
    pub worker_id: String,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
    /// Health status
    pub health_status: HealthStatus,
}

/// Worker health status
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Worker is healthy
    Healthy,
    /// Worker is degraded (e.g., high memory usage)
    Degraded,
}

/// Heartbeat configuration
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    /// Worker ID
    pub worker_id: String,
    /// rbee-hive callback URL (e.g., "http://localhost:9200")
    pub callback_url: String,
    /// Heartbeat interval in seconds (default: 30s)
    pub interval_secs: u64,
}

impl HeartbeatConfig {
    /// Create new heartbeat config
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

/// Start heartbeat task
///
/// Spawns a background task that sends periodic heartbeats to rbee-hive.
/// The task runs forever until the worker is shut down.
///
/// # Arguments
/// * `config` - Heartbeat configuration
///
/// # Returns
/// JoinHandle for the heartbeat task
pub fn start_heartbeat_task(config: HeartbeatConfig) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval_timer = interval(Duration::from_secs(config.interval_secs));
        let client = reqwest::Client::new();
        let heartbeat_url = format!("{}/v1/heartbeat", config.callback_url);

        debug!(
            worker_id = %config.worker_id,
            interval_secs = config.interval_secs,
            "Starting heartbeat task"
        );

        loop {
            interval_timer.tick().await;

            let payload = HeartbeatPayload {
                worker_id: config.worker_id.clone(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                health_status: HealthStatus::Healthy,
            };

            match send_heartbeat(&client, &heartbeat_url, &payload).await {
                Ok(()) => {
                    debug!(worker_id = %config.worker_id, "Heartbeat sent successfully");
                }
                Err(e) => {
                    // TEAM-115: Log error but don't crash - heartbeat failures are non-fatal
                    warn!(
                        worker_id = %config.worker_id,
                        error = %e,
                        "Failed to send heartbeat (will retry)"
                    );
                }
            }
        }
    })
}

/// Send heartbeat to rbee-hive
async fn send_heartbeat(
    client: &reqwest::Client,
    url: &str,
    payload: &HeartbeatPayload,
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
        return Err(format!("Heartbeat failed: {} - {}", status, body).into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heartbeat_config_new() {
        let config = HeartbeatConfig::new(
            "worker-123".to_string(),
            "http://localhost:9200".to_string(),
        );
        assert_eq!(config.worker_id, "worker-123");
        assert_eq!(config.callback_url, "http://localhost:9200");
        assert_eq!(config.interval_secs, 30);
    }

    #[test]
    fn test_heartbeat_config_with_interval() {
        let config = HeartbeatConfig::new(
            "worker-123".to_string(),
            "http://localhost:9200".to_string(),
        )
        .with_interval(60);
        assert_eq!(config.interval_secs, 60);
    }

    #[test]
    fn test_heartbeat_payload_serialization() {
        let payload = HeartbeatPayload {
            worker_id: "worker-123".to_string(),
            timestamp: "2025-10-19T00:00:00Z".to_string(),
            health_status: HealthStatus::Healthy,
        };

        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("worker-123"));
        assert!(json.contains("healthy"));
    }

    #[test]
    fn test_health_status_serialization() {
        let healthy = HealthStatus::Healthy;
        let degraded = HealthStatus::Degraded;

        let healthy_json = serde_json::to_string(&healthy).unwrap();
        let degraded_json = serde_json::to_string(&degraded).unwrap();

        assert_eq!(healthy_json, "\"healthy\"");
        assert_eq!(degraded_json, "\"degraded\"");
    }
}
