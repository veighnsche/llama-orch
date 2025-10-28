//! Hive contract types
//!
//! TEAM-284: Core types for hive contract

use serde::{Deserialize, Serialize};
use shared_contract::{HealthStatus, OperationalStatus};

/// Complete hive information
///
/// Contains all information about a hive that queen needs to track.
///
/// # Example
///
/// ```
/// use hive_contract::HiveInfo;
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
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HiveInfo {
    /// Hive ID (alias from config, e.g., "localhost", "gpu-server-1")
    pub id: String,

    /// Hostname or IP address
    pub hostname: String,

    /// HTTP port hive is listening on
    pub port: u16,

    /// Current operational status
    pub operational_status: OperationalStatus,

    /// Current health status
    pub health_status: HealthStatus,

    /// Hive version
    pub version: String,
    // TODO: TEAM-284: Add system stats
    // pub cpu_usage_percent: f32,
    // pub ram_used_gb: f32,
    // pub ram_total_gb: f32,
    // pub vram_per_device: Vec<VramInfo>,
    // pub temperature_celsius: Option<f32>,
}

impl HiveInfo {
    /// Check if hive is available (ready or busy, and healthy/degraded)
    pub fn is_available(&self) -> bool {
        self.operational_status.is_available() && self.health_status.is_operational()
    }

    /// Check if hive is ready to accept new workers
    pub fn is_ready(&self) -> bool {
        self.operational_status.is_ready() && self.health_status.is_healthy()
    }

    /// Get hive endpoint URL
    pub fn endpoint_url(&self) -> String {
        format!("http://{}:{}", self.hostname, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn hive_is_available() {
        let mut hive = create_test_hive();
        assert!(hive.is_available());

        hive.operational_status = OperationalStatus::Busy;
        assert!(hive.is_available());

        hive.operational_status = OperationalStatus::Stopped;
        assert!(!hive.is_available());
    }

    #[test]
    fn hive_is_ready() {
        let mut hive = create_test_hive();
        assert!(hive.is_ready());

        hive.operational_status = OperationalStatus::Busy;
        assert!(!hive.is_ready());

        hive.health_status = HealthStatus::Degraded { reason: "High load".to_string() };
        assert!(!hive.is_ready());
    }

    #[test]
    fn hive_endpoint_url() {
        let hive = create_test_hive();
        assert_eq!(hive.endpoint_url(), "http://localhost:9200");
    }

    #[test]
    fn hive_serialization() {
        let hive = create_test_hive();
        let json = serde_json::to_string(&hive).unwrap();

        assert!(json.contains("\"id\":\"test-hive\""));
        assert!(json.contains("\"hostname\":\"localhost\""));
        assert!(json.contains("\"port\":9200"));

        // Deserialize back
        let deserialized: HiveInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(hive, deserialized);
    }
}
