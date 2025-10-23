//! Heartbeat payload types
//!
//! Defines the data structures for heartbeat messages across the system.
//!
//! Created by: TEAM-115
//! Extended by: TEAM-151
//! Cleaned by: TEAM-262 (removed hive heartbeat types after TEAM-261)

use serde::{Deserialize, Serialize};

// ============================================================================
// Worker → Queen Heartbeat Types (TEAM-261)
// ============================================================================

/// Worker heartbeat payload sent to queen-rbee
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
// TEAM-262: Removed HiveHeartbeatPayload and WorkerState
// ============================================================================
// After TEAM-261, workers send heartbeats directly to queen.
// Hive no longer aggregates worker state.

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
    // TEAM-262: Removed hive heartbeat and worker state tests
    // ========================================================================
    // After TEAM-261, these types are no longer used

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

    // TEAM-262: Removed large worker list test

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
