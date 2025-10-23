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
///
/// Simple acknowledgement indicating the heartbeat was received and processed.
/// Workers use this to confirm the queen is reachable.
///
/// # Example
///
/// ```rust
/// use rbee_heartbeat::HeartbeatAcknowledgement;
///
/// let ack = HeartbeatAcknowledgement::success();
/// assert!(ack.acknowledged);
/// ```
///
/// # TEAM-262
///
/// After TEAM-261 simplified heartbeat flow, this is now used for
/// Worker → Queen acknowledgements (not Hive → Queen).
#[derive(Debug, Clone, Serialize)]
pub struct HeartbeatAcknowledgement {
    /// Whether the heartbeat was acknowledged
    ///
    /// Always `true` for successful responses. Errors are returned as HTTP errors.
    pub acknowledged: bool,
}

impl HeartbeatAcknowledgement {
    /// Create a successful acknowledgement
    ///
    /// # Returns
    ///
    /// A new `HeartbeatAcknowledgement` with `acknowledged: true`
    ///
    /// # Example
    ///
    /// ```rust
    /// use rbee_heartbeat::HeartbeatAcknowledgement;
    ///
    /// let ack = HeartbeatAcknowledgement::success();
    /// assert!(ack.acknowledged);
    /// ```
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
