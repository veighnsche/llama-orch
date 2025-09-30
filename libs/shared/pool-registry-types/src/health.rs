//! Health status types for pools

use serde::{Deserialize, Serialize};

/// Pool health state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthState {
    /// Pool is initializing
    Initializing,
    /// Pool is live and ready to serve
    Ready,
    /// Pool is draining (no new tasks)
    Draining,
    /// Pool has failed
    Failed,
    /// Pool is offline
    Offline,
}

/// Complete health status for a pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub state: HealthState,
    pub live: bool,
    pub ready: bool,
    pub draining: bool,
    pub last_error: Option<String>,
    pub last_heartbeat_ms: Option<u64>,
    pub engine: Option<String>,
    pub engine_version: Option<String>,
    pub device_mask: Option<String>,
    pub slots_total: Option<u32>,
    pub slots_free: Option<u32>,
    pub vram_free_bytes: Option<u64>,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            state: HealthState::Initializing,
            live: false,
            ready: false,
            draining: false,
            last_error: None,
            last_heartbeat_ms: None,
            engine: None,
            engine_version: None,
            device_mask: None,
            slots_total: None,
            slots_free: None,
            vram_free_bytes: None,
        }
    }
}

impl HealthStatus {
    pub fn is_available(&self) -> bool {
        self.ready && !self.draining && self.state == HealthState::Ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_not_available() {
        let health = HealthStatus::default();
        assert!(!health.is_available());
    }

    #[test]
    fn test_ready_is_available() {
        let health = HealthStatus {
            state: HealthState::Ready,
            live: true,
            ready: true,
            draining: false,
            ..Default::default()
        };
        assert!(health.is_available());
    }

    #[test]
    fn test_draining_not_available() {
        let health = HealthStatus {
            state: HealthState::Ready,
            live: true,
            ready: true,
            draining: true,
            ..Default::default()
        };
        assert!(!health.is_available());
    }
}
