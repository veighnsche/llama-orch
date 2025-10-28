//! Hive heartbeat protocol
//!
//! TEAM-284: Heartbeat types for hive → queen communication

use crate::types::HiveInfo;
use serde::{Deserialize, Serialize};
use shared_contract::{HeartbeatPayload, HeartbeatTimestamp, HEARTBEAT_TIMEOUT_SECS};

/// Hive heartbeat message
///
/// Sent from hive to queen every 30 seconds to report status.
///
/// # Protocol
///
/// - **Endpoint:** `POST /v1/hive-heartbeat` on queen
/// - **Frequency:** Every 30 seconds
/// - **Timeout:** 90 seconds (3 missed heartbeats)
/// - **Action on timeout:** Queen marks hive as unavailable
///
/// # Flow
///
/// ```text
/// Hive (every 30s) → POST /v1/hive-heartbeat → Queen
///                                                 ↓
///                                        Update hive registry
///                                        Track last_heartbeat time
/// ```
///
/// # Example
///
/// ```
/// use hive_contract::{HiveInfo, HiveHeartbeat};
/// use shared_contract::{OperationalStatus, HealthStatus};
///
/// let hive = HiveInfo {
///     id: "localhost".to_string(),
///     hostname: "127.0.0.1".to_string(),
///     port: 9200,
///     operational_status: OperationalStatus::Ready,
///     health_status: HealthStatus::Healthy,
///     version: "0.1.0".to_string(),
/// };
///
/// let heartbeat = HiveHeartbeat::new(hive);
///
/// // Serialize to JSON
/// let json = serde_json::to_string(&heartbeat).unwrap();
///
/// // Send to queen: POST http://queen:8500/v1/hive-heartbeat
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveHeartbeat {
    /// Complete hive information
    pub hive: HiveInfo,

    /// Timestamp when heartbeat was sent
    pub timestamp: HeartbeatTimestamp,
}

impl HiveHeartbeat {
    /// Create a new heartbeat with current timestamp
    pub fn new(hive: HiveInfo) -> Self {
        Self { hive, timestamp: HeartbeatTimestamp::now() }
    }

    /// Check if heartbeat is recent (within timeout window)
    pub fn is_recent(&self) -> bool {
        self.timestamp.is_recent(HEARTBEAT_TIMEOUT_SECS)
    }
}

impl HeartbeatPayload for HiveHeartbeat {
    fn component_id(&self) -> &str {
        &self.hive.id
    }

    fn timestamp(&self) -> &HeartbeatTimestamp {
        &self.timestamp
    }
}

// TEAM-285: Implement HeartbeatItem for generic registry
impl heartbeat_registry::HeartbeatItem for HiveHeartbeat {
    type Info = HiveInfo;

    fn id(&self) -> &str {
        &self.hive.id
    }

    fn info(&self) -> Self::Info {
        self.hive.clone()
    }

    fn is_recent(&self) -> bool {
        self.timestamp.is_recent(HEARTBEAT_TIMEOUT_SECS)
    }

    fn is_available(&self) -> bool {
        self.hive.is_available()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use shared_contract::{HealthStatus, OperationalStatus};

    fn create_test_hive() -> HiveInfo {
        HiveInfo {
            id: "test-hive".to_string(),
            hostname: "localhost".to_string(),
            port: 9200,
            operational_status: OperationalStatus::Ready,
            health_status: HealthStatus::Healthy,
            version: "0.1.0".to_string(),
        }
    }

    #[test]
    fn heartbeat_new() {
        let hive = create_test_hive();
        let heartbeat = HiveHeartbeat::new(hive.clone());

        assert_eq!(heartbeat.hive.id, "test-hive");
        assert!(heartbeat.is_recent());
    }

    #[test]
    fn heartbeat_is_recent() {
        let hive = create_test_hive();

        // Recent heartbeat
        let heartbeat = HiveHeartbeat::new(hive.clone());
        assert!(heartbeat.is_recent());

        // Old heartbeat (91 seconds ago)
        let old_timestamp = HeartbeatTimestamp::from_datetime(Utc::now() - Duration::seconds(91));
        let old_heartbeat = HiveHeartbeat { hive, timestamp: old_timestamp };
        assert!(!old_heartbeat.is_recent());
    }

    #[test]
    fn heartbeat_payload_trait() {
        let hive = create_test_hive();
        let heartbeat = HiveHeartbeat::new(hive);

        // Test HeartbeatPayload trait methods
        assert_eq!(heartbeat.component_id(), "test-hive");
        assert!(heartbeat.is_recent(90));
    }

    #[test]
    fn heartbeat_serialization() {
        let hive = create_test_hive();
        let heartbeat = HiveHeartbeat::new(hive);

        let json = serde_json::to_string(&heartbeat).unwrap();

        // Verify all fields are present
        assert!(json.contains("\"id\":\"test-hive\""));
        assert!(json.contains("\"hostname\":\"localhost\""));
        assert!(json.contains("\"port\":9200"));

        // Verify it can be deserialized
        let deserialized: HiveHeartbeat = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.hive.id, "test-hive");
    }
}
