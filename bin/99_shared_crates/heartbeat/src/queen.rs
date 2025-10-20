//! Queen heartbeat receiver logic (Hive → Queen)
//!
//! Provides endpoint handler for receiving heartbeats from hives.
//!
//! Created by: TEAM-158

use crate::types::HiveHeartbeatPayload;
use serde::Serialize;

// ============================================================================
// Queen Heartbeat Response
// ============================================================================

/// Response sent back to hive after receiving heartbeat
#[derive(Debug, Clone, Serialize)]
pub struct HeartbeatAcknowledgement {
    /// Whether the heartbeat was acknowledged
    pub acknowledged: bool,
}

impl HeartbeatAcknowledgement {
    /// Create a successful acknowledgement
    pub fn success() -> Self {
        Self { acknowledged: true }
    }
}

// ============================================================================
// Heartbeat Handler Trait
// ============================================================================

/// Trait for handling received heartbeats
///
/// The queen must implement this to process heartbeats from hives.
/// This allows the queen to update its catalog and registries.
pub trait HeartbeatHandler: Send + Sync {
    /// Handle a heartbeat from a hive
    ///
    /// # Arguments
    /// * `payload` - The heartbeat payload from the hive
    ///
    /// # Returns
    /// * `Ok(())` if heartbeat was processed successfully
    /// * `Err(String)` if there was an error processing the heartbeat
    fn handle_heartbeat(
        &self,
        payload: HiveHeartbeatPayload,
    ) -> impl std::future::Future<Output = Result<(), String>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acknowledgement_success_creates_acknowledged_response() {
        let ack = HeartbeatAcknowledgement::success();
        assert!(ack.acknowledged);
    }

    #[test]
    fn acknowledgement_serializes_correctly() {
        let ack = HeartbeatAcknowledgement::success();
        let json = serde_json::to_string(&ack).unwrap();
        assert!(json.contains("true"));
    }
}
