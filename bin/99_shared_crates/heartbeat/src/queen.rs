//! Queen heartbeat receiver logic (Worker → Queen)
//!
//! Provides types for receiving heartbeats from workers.
//!
//! Created by: TEAM-158
//! Simplified by: TEAM-261 (workers send directly to queen)
//! Cleaned by: TEAM-262 (removed hive heartbeat handling)

use serde::Serialize;

// ============================================================================
// Queen Heartbeat Response
// ============================================================================

/// Response sent back to worker after receiving heartbeat
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
// TEAM-262: Removed HeartbeatHandler trait - queen handles worker heartbeats
// directly via HTTP endpoint, no trait abstraction needed

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
