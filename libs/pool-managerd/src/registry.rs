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
    pub engine_version: Option<String>,
    pub engine_digest: Option<String>,
    pub engine_catalog_id: Option<String>,
    pub device_mask: Option<String>,
    pub slots_total: Option<i32>,
    pub slots_free: Option<i32>,
    pub perf_hints: Option<serde_json::Value>,
}

impl Registry {
    pub fn new() -> Self {
        Self { pools: HashMap::new() }
    }

    pub fn set_health(&mut self, pool_id: impl Into<String>, health: HealthStatus) {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus { live: false, ready: false },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
            engine_version: None,
            engine_digest: None,
            engine_catalog_id: None,
            device_mask: None,
            slots_total: None,
            slots_free: None,
            perf_hints: None,
        });
        entry.health = health;
    }

    pub fn get_health(&self, pool_id: &str) -> Option<HealthStatus> {
        self.pools.get(pool_id).map(|e| e.health.clone())
    }

    pub fn set_last_error(&mut self, pool_id: impl Into<String>, err: impl Into<String>) {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus { live: false, ready: false },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
            engine_version: None,
            engine_digest: None,
            engine_catalog_id: None,
            device_mask: None,
            slots_total: None,
            slots_free: None,
            perf_hints: None,
        });
        entry.last_error = Some(err.into());
    }

    pub fn get_last_error(&self, pool_id: &str) -> Option<String> {
        self.pools.get(pool_id).and_then(|e| e.last_error.as_ref().cloned())
    }

    pub fn set_heartbeat(&mut self, pool_id: impl Into<String>, ms: i64) {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus { live: false, ready: false },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
            engine_version: None,
            engine_digest: None,
            engine_catalog_id: None,
            device_mask: None,
            slots_total: None,
            slots_free: None,
            perf_hints: None,
        });
        entry.last_heartbeat_ms = Some(ms);
    }

    pub fn get_heartbeat(&self, pool_id: &str) -> Option<i64> {
        self.pools.get(pool_id).and_then(|e| e.last_heartbeat_ms.as_ref().copied())
    }

    pub fn set_version(&mut self, pool_id: impl Into<String>, v: impl Into<String>) {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus { live: false, ready: false },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
            engine_version: None,
            engine_digest: None,
            engine_catalog_id: None,
            device_mask: None,
            slots_total: None,
            slots_free: None,
            perf_hints: None,
        });
        entry.version = Some(v.into());
    }

    pub fn get_version(&self, pool_id: &str) -> Option<String> {
        self.pools.get(pool_id).and_then(|e| e.version.as_ref().cloned())
    }

    pub fn allocate_lease(&mut self, pool_id: impl Into<String>) -> i32 {
        let id = pool_id.into();
        let entry = self.pools.entry(id).or_insert(PoolEntry {
            health: HealthStatus { live: false, ready: false },
            last_heartbeat_ms: None,
            version: None,
            last_error: None,
            active_leases: 0,
            engine_version: None,
            engine_digest: None,
            engine_catalog_id: None,
            device_mask: None,
            slots_total: None,
            slots_free: None,
            perf_hints: None,
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
            engine_version: None,
            engine_digest: None,
            engine_catalog_id: None,
            device_mask: None,
            slots_total: None,
            slots_free: None,
            perf_hints: None,
        });
        entry.active_leases = (entry.active_leases - 1).max(0);
        entry.active_leases
    }

    pub fn get_active_leases(&self, pool_id: &str) -> i32 {
        self.pools.get(pool_id).map(|e| e.active_leases).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // OC-POOL-3001: registry stores and returns health readiness
    #[test]
    fn test_oc_pool_3001_health_and_meta() {
        let mut r = Registry::new();
        r.set_health("pool0", HealthStatus { live: true, ready: false });
        let h = r.get_health("pool0").expect("health");
        assert!(h.live);
        assert!(!h.ready);

        r.set_last_error("pool0", "preload failure");
        assert_eq!(r.get_last_error("pool0").as_deref(), Some("preload failure"));

        r.set_version("pool0", "v1");
        assert_eq!(r.get_version("pool0").as_deref(), Some("v1"));

        r.set_heartbeat("pool0", 1234);
        assert_eq!(r.get_heartbeat("pool0"), Some(1234));
    }

    // OC-POOL-3007: lease counters never go negative
    #[test]
    fn test_oc_pool_3007_leases_never_negative() {
        let mut r = Registry::new();
        assert_eq!(r.get_active_leases("pool0"), 0);
        assert_eq!(r.release_lease("pool0"), 0);
        assert_eq!(r.allocate_lease("pool0"), 1);
        assert_eq!(r.allocate_lease("pool0"), 2);
        assert_eq!(r.get_active_leases("pool0"), 2);
        assert_eq!(r.release_lease("pool0"), 1);
        assert_eq!(r.release_lease("pool0"), 0);
        assert_eq!(r.release_lease("pool0"), 0);
    }
}
