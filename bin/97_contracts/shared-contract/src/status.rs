//! Status types shared between workers and hives
//!
//! TEAM-284: Common status enums for health and operational state

use serde::{Deserialize, Serialize};

/// Health status of a component (worker or hive)
///
/// Indicates the current health state of the component.
/// This is separate from operational status.
///
/// # Examples
///
/// ```
/// use shared_contract::HealthStatus;
///
/// let healthy = HealthStatus::Healthy;
/// let degraded = HealthStatus::Degraded { reason: "High memory usage".to_string() };
/// let unhealthy = HealthStatus::Unhealthy { reason: "GPU error".to_string() };
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum HealthStatus {
    /// Component is healthy and operating normally
    Healthy,
    
    /// Component is degraded but still functional
    ///
    /// Examples: High memory usage, thermal throttling, slow responses
    Degraded {
        /// Reason for degradation
        reason: String,
    },
    
    /// Component is unhealthy and may not function correctly
    ///
    /// Examples: Hardware errors, critical resource exhaustion
    Unhealthy {
        /// Reason for unhealthy state
        reason: String,
    },
}

impl HealthStatus {
    /// Check if component is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }
    
    /// Check if component is degraded
    pub fn is_degraded(&self) -> bool {
        matches!(self, HealthStatus::Degraded { .. })
    }
    
    /// Check if component is unhealthy
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, HealthStatus::Unhealthy { .. })
    }
    
    /// Check if component is operational (healthy or degraded)
    pub fn is_operational(&self) -> bool {
        !self.is_unhealthy()
    }
}

/// Operational status of a component
///
/// Indicates what the component is currently doing.
/// This is separate from health status.
///
/// # Examples
///
/// ```
/// use shared_contract::OperationalStatus;
///
/// let starting = OperationalStatus::Starting;
/// let ready = OperationalStatus::Ready;
/// let busy = OperationalStatus::Busy;
/// let stopped = OperationalStatus::Stopped;
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OperationalStatus {
    /// Component is starting up
    Starting,
    
    /// Component is ready to accept work
    Ready,
    
    /// Component is busy processing work
    Busy,
    
    /// Component is shutting down
    Stopping,
    
    /// Component is stopped
    Stopped,
}

impl OperationalStatus {
    /// Check if component is ready to accept work
    pub fn is_ready(&self) -> bool {
        matches!(self, OperationalStatus::Ready)
    }
    
    /// Check if component is busy
    pub fn is_busy(&self) -> bool {
        matches!(self, OperationalStatus::Busy)
    }
    
    /// Check if component is available (ready or busy but not stopped)
    pub fn is_available(&self) -> bool {
        matches!(self, OperationalStatus::Ready | OperationalStatus::Busy)
    }
    
    /// Check if component is stopped or stopping
    pub fn is_stopped(&self) -> bool {
        matches!(self, OperationalStatus::Stopped | OperationalStatus::Stopping)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_status_healthy() {
        let status = HealthStatus::Healthy;
        assert!(status.is_healthy());
        assert!(!status.is_degraded());
        assert!(!status.is_unhealthy());
        assert!(status.is_operational());
    }

    #[test]
    fn health_status_degraded() {
        let status = HealthStatus::Degraded {
            reason: "High memory".to_string(),
        };
        assert!(!status.is_healthy());
        assert!(status.is_degraded());
        assert!(!status.is_unhealthy());
        assert!(status.is_operational());
    }

    #[test]
    fn health_status_unhealthy() {
        let status = HealthStatus::Unhealthy {
            reason: "GPU error".to_string(),
        };
        assert!(!status.is_healthy());
        assert!(!status.is_degraded());
        assert!(status.is_unhealthy());
        assert!(!status.is_operational());
    }

    #[test]
    fn operational_status_ready() {
        let status = OperationalStatus::Ready;
        assert!(status.is_ready());
        assert!(!status.is_busy());
        assert!(status.is_available());
        assert!(!status.is_stopped());
    }

    #[test]
    fn operational_status_busy() {
        let status = OperationalStatus::Busy;
        assert!(!status.is_ready());
        assert!(status.is_busy());
        assert!(status.is_available());
        assert!(!status.is_stopped());
    }

    #[test]
    fn operational_status_stopped() {
        let status = OperationalStatus::Stopped;
        assert!(!status.is_ready());
        assert!(!status.is_busy());
        assert!(!status.is_available());
        assert!(status.is_stopped());
    }

    #[test]
    fn health_status_serialization() {
        let healthy = HealthStatus::Healthy;
        let json = serde_json::to_string(&healthy).unwrap();
        assert!(json.contains("\"status\":\"healthy\""));

        let degraded = HealthStatus::Degraded {
            reason: "test".to_string(),
        };
        let json = serde_json::to_string(&degraded).unwrap();
        assert!(json.contains("\"status\":\"degraded\""));
        assert!(json.contains("\"reason\":\"test\""));
    }

    #[test]
    fn operational_status_serialization() {
        let ready = OperationalStatus::Ready;
        let json = serde_json::to_string(&ready).unwrap();
        assert_eq!(json, "\"ready\"");

        let busy = OperationalStatus::Busy;
        let json = serde_json::to_string(&busy).unwrap();
        assert_eq!(json, "\"busy\"");
    }
}
