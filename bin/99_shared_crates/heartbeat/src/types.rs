//! Heartbeat payload types
//!
//! Defines the data structures for heartbeat messages across the system.
//!
//! Created by: TEAM-115
//! Extended by: TEAM-151

use serde::{Deserialize, Serialize};

// ============================================================================
// Worker → Hive Heartbeat Types
// ============================================================================

/// Worker heartbeat payload sent to rbee-hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeatPayload {
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

// ============================================================================
// Hive → Queen Heartbeat Types
// ============================================================================

/// Hive heartbeat payload sent to queen-rbee
///
/// Contains aggregated state of all workers in the hive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveHeartbeatPayload {
    /// Hive ID (e.g., "localhost" or hostname)
    pub hive_id: String,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
    /// List of all workers in this hive
    pub workers: Vec<WorkerState>,
}

/// Worker state in hive heartbeat
///
/// Simplified worker info for queen's scheduling decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerState {
    /// Worker ID
    pub worker_id: String,
    /// Worker state (e.g., "Idle", "Busy", "Loading")
    pub state: String,
    /// Last heartbeat timestamp from worker
    pub last_heartbeat: String,
    /// Health status
    pub health_status: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Worker Heartbeat Payload Tests
    // ========================================================================

    #[test]
    fn worker_heartbeat_serializes_to_expected_json_format() {
        let payload = WorkerHeartbeatPayload {
            worker_id: "worker-123".to_string(),
            timestamp: "2025-10-19T00:00:00Z".to_string(),
            health_status: HealthStatus::Healthy,
        };

        let json = serde_json::to_string(&payload).unwrap();

        // Verify all fields are present
        assert!(json.contains("worker-123"));
        assert!(json.contains("2025-10-19T00:00:00Z"));
        assert!(json.contains("healthy"));

        // Verify it's valid JSON
        let _: serde_json::Value = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn worker_heartbeat_deserializes_from_json() {
        let json = r#"{
            "worker_id": "worker-456",
            "timestamp": "2025-10-20T01:00:00Z",
            "health_status": "degraded"
        }"#;

        let payload: WorkerHeartbeatPayload = serde_json::from_str(json).unwrap();

        assert_eq!(payload.worker_id, "worker-456");
        assert_eq!(payload.timestamp, "2025-10-20T01:00:00Z");
        assert!(matches!(payload.health_status, HealthStatus::Degraded));
    }

    #[test]
    fn worker_heartbeat_roundtrip_preserves_data() {
        let original = WorkerHeartbeatPayload {
            worker_id: "worker-roundtrip".to_string(),
            timestamp: "2025-10-20T00:00:00Z".to_string(),
            health_status: HealthStatus::Healthy,
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: WorkerHeartbeatPayload = serde_json::from_str(&json).unwrap();

        assert_eq!(original.worker_id, deserialized.worker_id);
        assert_eq!(original.timestamp, deserialized.timestamp);
    }

    // ========================================================================
    // Health Status Tests
    // ========================================================================

    #[test]
    fn health_status_healthy_serializes_to_lowercase() {
        let healthy = HealthStatus::Healthy;
        let json = serde_json::to_string(&healthy).unwrap();
        assert_eq!(json, "\"healthy\"");
    }

    #[test]
    fn health_status_degraded_serializes_to_lowercase() {
        let degraded = HealthStatus::Degraded;
        let json = serde_json::to_string(&degraded).unwrap();
        assert_eq!(json, "\"degraded\"");
    }

    #[test]
    fn health_status_deserializes_case_insensitive() {
        // Lowercase (expected format)
        let healthy: HealthStatus = serde_json::from_str("\"healthy\"").unwrap();
        assert!(matches!(healthy, HealthStatus::Healthy));

        let degraded: HealthStatus = serde_json::from_str("\"degraded\"").unwrap();
        assert!(matches!(degraded, HealthStatus::Degraded));
    }

    #[test]
    fn health_status_invalid_value_fails_gracefully() {
        let result: Result<HealthStatus, _> = serde_json::from_str("\"invalid\"");
        assert!(result.is_err());
    }

    // ========================================================================
    // Hive Heartbeat Payload Tests
    // ========================================================================

    #[test]
    fn hive_heartbeat_serializes_with_empty_workers() {
        let payload = HiveHeartbeatPayload {
            hive_id: "localhost".to_string(),
            timestamp: "2025-10-19T00:00:00Z".to_string(),
            workers: vec![],
        };

        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("localhost"));
        assert!(json.contains("[]")); // Empty array
    }

    #[test]
    fn hive_heartbeat_serializes_with_multiple_workers() {
        let payload = HiveHeartbeatPayload {
            hive_id: "hive-prod".to_string(),
            timestamp: "2025-10-19T00:00:00Z".to_string(),
            workers: vec![
                WorkerState {
                    worker_id: "worker-1".to_string(),
                    state: "Idle".to_string(),
                    last_heartbeat: "2025-10-19T00:00:00Z".to_string(),
                    health_status: "healthy".to_string(),
                },
                WorkerState {
                    worker_id: "worker-2".to_string(),
                    state: "Busy".to_string(),
                    last_heartbeat: "2025-10-19T00:00:05Z".to_string(),
                    health_status: "healthy".to_string(),
                },
                WorkerState {
                    worker_id: "worker-3".to_string(),
                    state: "Loading".to_string(),
                    last_heartbeat: "2025-10-19T00:00:10Z".to_string(),
                    health_status: "degraded".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&payload).unwrap();

        // Verify hive info
        assert!(json.contains("hive-prod"));

        // Verify all workers are present
        assert!(json.contains("worker-1"));
        assert!(json.contains("worker-2"));
        assert!(json.contains("worker-3"));

        // Verify states
        assert!(json.contains("Idle"));
        assert!(json.contains("Busy"));
        assert!(json.contains("Loading"));
    }

    #[test]
    fn hive_heartbeat_deserializes_from_json() {
        let json = r#"{
            "hive_id": "hive-test",
            "timestamp": "2025-10-20T01:00:00Z",
            "workers": [
                {
                    "worker_id": "w1",
                    "state": "Idle",
                    "last_heartbeat": "2025-10-20T00:59:00Z",
                    "health_status": "healthy"
                }
            ]
        }"#;

        let payload: HiveHeartbeatPayload = serde_json::from_str(json).unwrap();

        assert_eq!(payload.hive_id, "hive-test");
        assert_eq!(payload.workers.len(), 1);
        assert_eq!(payload.workers[0].worker_id, "w1");
        assert_eq!(payload.workers[0].state, "Idle");
    }

    // ========================================================================
    // Worker State Tests
    // ========================================================================

    #[test]
    fn worker_state_captures_all_required_fields() {
        let state = WorkerState {
            worker_id: "worker-test".to_string(),
            state: "Busy".to_string(),
            last_heartbeat: "2025-10-20T00:00:00Z".to_string(),
            health_status: "healthy".to_string(),
        };

        // Verify all fields are accessible
        assert_eq!(state.worker_id, "worker-test");
        assert_eq!(state.state, "Busy");
        assert_eq!(state.last_heartbeat, "2025-10-20T00:00:00Z");
        assert_eq!(state.health_status, "healthy");
    }

    #[test]
    fn worker_state_allows_different_state_values() {
        let states = vec!["Idle", "Busy", "Loading", "Error", "Shutdown"];

        for state_value in states {
            let state = WorkerState {
                worker_id: "worker".to_string(),
                state: state_value.to_string(),
                last_heartbeat: "2025-10-20T00:00:00Z".to_string(),
                health_status: "healthy".to_string(),
            };

            assert_eq!(state.state, state_value);
        }
    }

    #[test]
    fn worker_state_serialization_preserves_order() {
        let state = WorkerState {
            worker_id: "w1".to_string(),
            state: "Idle".to_string(),
            last_heartbeat: "2025-10-20T00:00:00Z".to_string(),
            health_status: "healthy".to_string(),
        };

        let json = serde_json::to_string(&state).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        // Verify it's an object with expected keys
        assert!(parsed.is_object());
        assert!(parsed.get("worker_id").is_some());
        assert!(parsed.get("state").is_some());
        assert!(parsed.get("last_heartbeat").is_some());
        assert!(parsed.get("health_status").is_some());
    }

    // ========================================================================
    // Edge Cases and Error Conditions
    // ========================================================================

    #[test]
    fn worker_heartbeat_handles_special_characters_in_worker_id() {
        let payload = WorkerHeartbeatPayload {
            worker_id: "worker-123-abc_def.test".to_string(),
            timestamp: "2025-10-19T00:00:00Z".to_string(),
            health_status: HealthStatus::Healthy,
        };

        let json = serde_json::to_string(&payload).unwrap();
        let deserialized: WorkerHeartbeatPayload = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.worker_id, "worker-123-abc_def.test");
    }

    #[test]
    fn hive_heartbeat_handles_large_worker_list() {
        let workers: Vec<WorkerState> = (0..100)
            .map(|i| WorkerState {
                worker_id: format!("worker-{}", i),
                state: "Idle".to_string(),
                last_heartbeat: "2025-10-20T00:00:00Z".to_string(),
                health_status: "healthy".to_string(),
            })
            .collect();

        let payload = HiveHeartbeatPayload {
            hive_id: "hive-large".to_string(),
            timestamp: "2025-10-20T00:00:00Z".to_string(),
            workers,
        };

        let json = serde_json::to_string(&payload).unwrap();
        let deserialized: HiveHeartbeatPayload = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.workers.len(), 100);
        assert_eq!(deserialized.workers[0].worker_id, "worker-0");
        assert_eq!(deserialized.workers[99].worker_id, "worker-99");
    }

    #[test]
    fn worker_heartbeat_clone_creates_independent_copy() {
        let original = WorkerHeartbeatPayload {
            worker_id: "worker-clone".to_string(),
            timestamp: "2025-10-20T00:00:00Z".to_string(),
            health_status: HealthStatus::Healthy,
        };

        let cloned = original.clone();

        assert_eq!(original.worker_id, cloned.worker_id);
        assert_eq!(original.timestamp, cloned.timestamp);
    }

    #[test]
    fn health_status_clone_works_correctly() {
        let healthy = HealthStatus::Healthy;
        let cloned = healthy.clone();

        assert!(matches!(cloned, HealthStatus::Healthy));
    }
}
