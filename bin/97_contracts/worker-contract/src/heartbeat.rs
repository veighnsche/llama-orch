// TEAM-270: Heartbeat protocol
// TEAM-284: Updated to use shared-contract types

use crate::types::WorkerInfo;
use serde::{Deserialize, Serialize};

// TEAM-284: Re-export constants from shared-contract
pub use shared_contract::{HEARTBEAT_INTERVAL_SECS, HEARTBEAT_TIMEOUT_SECS};

/// Worker heartbeat message
///
/// Sent from worker to queen every 30 seconds to report status.
///
/// # Protocol
///
/// - **Endpoint:** `POST /v1/worker-heartbeat` on queen
/// - **Frequency:** Every 30 seconds
/// - **Timeout:** 90 seconds (3 missed heartbeats)
/// - **Action on timeout:** Queen marks worker as unavailable
///
/// # Flow
///
/// ```text
/// Worker (every 30s) → POST /v1/worker-heartbeat → Queen
///                                                     ↓
///                                            Update worker registry
///                                            Track last_heartbeat time
/// ```
///
/// # Example
///
/// ```
/// use worker_contract::{WorkerInfo, WorkerStatus, WorkerHeartbeat};
/// use chrono::Utc;
///
/// let worker = WorkerInfo {
///     id: "worker-abc123".to_string(),
///     model_id: "meta-llama/Llama-2-7b".to_string(),
///     device: "GPU-0".to_string(),
///     port: 9301,
///     status: WorkerStatus::Ready,
///     implementation: "llm-worker-rbee".to_string(),
///     version: "0.1.0".to_string(),
/// };
///
/// let heartbeat = WorkerHeartbeat {
///     worker,
///     timestamp: Utc::now(),
/// };
///
/// // Serialize to JSON
/// let json = serde_json::to_string(&heartbeat).unwrap();
///
/// // Send to queen: POST http://queen:8500/v1/worker-heartbeat
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHeartbeat {
    /// Complete worker information
    pub worker: WorkerInfo,

    /// Timestamp when heartbeat was sent
    /// TEAM-284: Using shared HeartbeatTimestamp
    pub timestamp: shared_contract::HeartbeatTimestamp,
}

/// Heartbeat acknowledgement from queen
///
/// Queen responds to heartbeat with acknowledgement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatAck {
    /// Status of acknowledgement
    pub status: String,

    /// Optional message from queen
    pub message: Option<String>,
}

impl WorkerHeartbeat {
    /// Create a new heartbeat with current timestamp
    pub fn new(worker: WorkerInfo) -> Self {
        Self {
            worker,
            timestamp: shared_contract::HeartbeatTimestamp::now(),
        }
    }

    /// Check if heartbeat is recent (within timeout window)
    pub fn is_recent(&self) -> bool {
        self.timestamp.is_recent(HEARTBEAT_TIMEOUT_SECS)
    }
}

// TEAM-284: Implement HeartbeatPayload trait
impl shared_contract::HeartbeatPayload for WorkerHeartbeat {
    fn component_id(&self) -> &str {
        &self.worker.id
    }

    fn timestamp(&self) -> &shared_contract::HeartbeatTimestamp {
        &self.timestamp
    }
}

// TEAM-285: Implement HeartbeatItem for generic registry
impl heartbeat_registry::HeartbeatItem for WorkerHeartbeat {
    type Info = WorkerInfo;

    fn id(&self) -> &str {
        &self.worker.id
    }

    fn info(&self) -> Self::Info {
        self.worker.clone()
    }

    fn is_recent(&self) -> bool {
        self.timestamp.is_recent(HEARTBEAT_TIMEOUT_SECS)
    }

    fn is_available(&self) -> bool {
        self.worker.is_available()
    }
}

impl HeartbeatAck {
    /// Create a success acknowledgement
    pub fn success(message: impl Into<String>) -> Self {
        Self { status: "ok".to_string(), message: Some(message.into()) }
    }

    /// Create an error acknowledgement
    pub fn error(message: impl Into<String>) -> Self {
        Self { status: "error".to_string(), message: Some(message.into()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::WorkerStatus;

    #[test]
    fn test_heartbeat_new() {
        let worker = WorkerInfo {
            id: "test".to_string(),
            model_id: "test-model".to_string(),
            device: "GPU-0".to_string(),
            port: 9301,
            status: WorkerStatus::Ready,
            implementation: "test".to_string(),
            version: "0.1.0".to_string(),
        };

        let heartbeat = WorkerHeartbeat::new(worker.clone());
        assert_eq!(heartbeat.worker.id, "test");
        assert!(heartbeat.is_recent());
    }

    #[test]
    fn test_heartbeat_is_recent() {
        let worker = WorkerInfo {
            id: "test".to_string(),
            model_id: "test-model".to_string(),
            device: "GPU-0".to_string(),
            port: 9301,
            status: WorkerStatus::Ready,
            implementation: "test".to_string(),
            version: "0.1.0".to_string(),
        };

        // Recent heartbeat
        let heartbeat = WorkerHeartbeat::new(worker.clone());
        assert!(heartbeat.is_recent());

        // Old heartbeat (91 seconds ago)
        let old_timestamp = chrono::Utc::now() - chrono::Duration::seconds(91);
        let old_heartbeat = WorkerHeartbeat { worker, timestamp: old_timestamp };
        assert!(!old_heartbeat.is_recent());
    }

    #[test]
    fn test_heartbeat_ack() {
        let ack = HeartbeatAck::success("Heartbeat received");
        assert_eq!(ack.status, "ok");
        assert_eq!(ack.message, Some("Heartbeat received".to_string()));

        let err = HeartbeatAck::error("Worker not found");
        assert_eq!(err.status, "error");
        assert_eq!(err.message, Some("Worker not found".to_string()));
    }

    #[test]
    fn test_heartbeat_serialization() {
        let worker = WorkerInfo {
            id: "test".to_string(),
            model_id: "test-model".to_string(),
            device: "GPU-0".to_string(),
            port: 9301,
            status: WorkerStatus::Ready,
            implementation: "test".to_string(),
            version: "0.1.0".to_string(),
        };

        let heartbeat = WorkerHeartbeat::new(worker);
        let json = serde_json::to_string(&heartbeat).unwrap();

        // Verify it can be deserialized
        let deserialized: WorkerHeartbeat = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.worker.id, "test");
    }
}
