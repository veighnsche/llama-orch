//! Shared constants for heartbeat protocol
//!
//! TEAM-284: Common constants used by both workers and hives

/// Heartbeat interval in seconds
///
/// Components send heartbeats every 30 seconds.
pub const HEARTBEAT_INTERVAL_SECS: u64 = 30;

/// Heartbeat timeout in seconds
///
/// Queen marks component as unavailable after 90 seconds (3 missed heartbeats).
pub const HEARTBEAT_TIMEOUT_SECS: u64 = 90;

/// Maximum allowed heartbeat age for "recent" check
///
/// Same as timeout - heartbeats older than this are considered stale.
pub const MAX_HEARTBEAT_AGE_SECS: u64 = HEARTBEAT_TIMEOUT_SECS;

/// Recommended cleanup interval in seconds
///
/// How often registries should clean up stale entries.
pub const CLEANUP_INTERVAL_SECS: u64 = 60;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_reasonable() {
        assert_eq!(HEARTBEAT_INTERVAL_SECS, 30);
        assert_eq!(HEARTBEAT_TIMEOUT_SECS, 90);
        assert_eq!(MAX_HEARTBEAT_AGE_SECS, 90);
        assert_eq!(CLEANUP_INTERVAL_SECS, 60);

        // Timeout should be at least 2x interval
        assert!(HEARTBEAT_TIMEOUT_SECS >= HEARTBEAT_INTERVAL_SECS * 2);
    }
}
