//! Replica registry (planning-only).

use std::collections::HashMap;

use crate::health::HealthStatus;

#[derive(Debug, Clone)]
pub struct Replica;

#[derive(Debug, Clone, Default)]
pub struct Registry {
    pools: HashMap<String, PoolEntry>,
}

#[derive(Debug, Clone)]
pub struct PoolEntry {
    pub health: HealthStatus,
    pub last_heartbeat_ms: Option<i64>,
    pub version: Option<String>,
    pub last_error: Option<String>,
    pub active_leases: i32,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    pub fn set_health(&mut self, pool_id: impl Into<String>, health: HealthStatus) {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus {
                live: false,
                ready: false,
            },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
        });
        entry.health = health;
    }

    pub fn get_health(&self, pool_id: &str) -> Option<HealthStatus> {
        self.pools.get(pool_id).map(|e| e.health.clone())
    }

    pub fn set_last_error(&mut self, pool_id: impl Into<String>, err: impl Into<String>) {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus {
                live: false,
                ready: false,
            },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
        });
        entry.last_error = Some(err.into());
    }

    pub fn get_last_error(&self, pool_id: &str) -> Option<String> {
        self.pools
            .get(pool_id)
            .and_then(|e| e.last_error.as_ref().cloned())
    }

    pub fn set_heartbeat(&mut self, pool_id: impl Into<String>, ms: i64) {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus {
                live: false,
                ready: false,
            },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
        });
        entry.last_heartbeat_ms = Some(ms);
    }

    pub fn get_heartbeat(&self, pool_id: &str) -> Option<i64> {
        self.pools
            .get(pool_id)
            .and_then(|e| e.last_heartbeat_ms.as_ref().copied())
    }

    pub fn set_version(&mut self, pool_id: impl Into<String>, v: impl Into<String>) {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus {
                live: false,
                ready: false,
            },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
        });
        entry.version = Some(v.into());
    }

    pub fn get_version(&self, pool_id: &str) -> Option<String> {
        self.pools
            .get(pool_id)
            .and_then(|e| e.version.as_ref().cloned())
    }

    pub fn allocate_lease(&mut self, pool_id: impl Into<String>) -> i32 {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus { live: false, ready: false },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
        });
        entry.active_leases = entry.active_leases.saturating_add(1);
        entry.active_leases
    }

    pub fn release_lease(&mut self, pool_id: impl Into<String>) -> i32 {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus { live: false, ready: false },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
        });
        entry.active_leases = (entry.active_leases - 1).max(0);
        entry.active_leases
    }

    pub fn get_active_leases(&self, pool_id: &str) -> i32 {
        self.pools
            .get(pool_id)
            .map(|e| e.active_leases)
            .unwrap_or(0)
    }
}
