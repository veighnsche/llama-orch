//! Replica registry (planning-only).

use std::collections::HashMap;

use crate::health::HealthStatus;
use serde_json as json;

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

    /// Set engine version (from worker/engine), distinct from registry version.
    pub fn set_engine_version(&mut self, pool_id: impl Into<String>, v: impl Into<String>) {
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
        entry.engine_version = Some(v.into());
    }

    pub fn get_engine_version(&self, pool_id: &str) -> Option<String> {
        self.pools.get(pool_id).and_then(|e| e.engine_version.as_ref().cloned())
    }

    /// Optionally set engine metadata fields; only provided fields are updated.
    pub fn set_engine_meta<V, D, C>(
        &mut self,
        pool_id: impl Into<String>,
        version: Option<V>,
        digest: Option<D>,
        catalog_id: Option<C>,
    ) where
        V: Into<String>,
        D: Into<String>,
        C: Into<String>,
    {
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
        if let Some(v) = version { entry.engine_version = Some(v.into()); }
        if let Some(d) = digest { entry.engine_digest = Some(d.into()); }
        if let Some(c) = catalog_id { entry.engine_catalog_id = Some(c.into()); }
    }

    pub fn get_engine_digest(&self, pool_id: &str) -> Option<String> {
        self.pools.get(pool_id).and_then(|e| e.engine_digest.as_ref().cloned())
    }

    pub fn get_engine_catalog_id(&self, pool_id: &str) -> Option<String> {
        self.pools
            .get(pool_id)
            .and_then(|e| e.engine_catalog_id.as_ref().cloned())
    }

    pub fn set_device_mask(&mut self, pool_id: impl Into<String>, mask: impl Into<String>) {
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
        entry.device_mask = Some(mask.into());
    }

    pub fn get_device_mask(&self, pool_id: &str) -> Option<String> {
        self.pools.get(pool_id).and_then(|e| e.device_mask.as_ref().cloned())
    }

    pub fn set_slots(&mut self, pool_id: impl Into<String>, total: i32, free: i32) {
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
        entry.slots_total = Some(total);
        entry.slots_free = Some(free.max(0).min(total));
    }

    pub fn get_slots_total(&self, pool_id: &str) -> Option<i32> {
        self.pools.get(pool_id).and_then(|e| e.slots_total.as_ref().copied())
    }

    pub fn get_slots_free(&self, pool_id: &str) -> Option<i32> {
        self.pools.get(pool_id).and_then(|e| e.slots_free.as_ref().copied())
    }

    /// Convenience entrypoint for provisioners/orchestrator to register a Ready pool
    /// using a standard engine handoff JSON produced by engine-provisioner.
    /// Expected fields (best-effort): engine_version, device_mask, slots_total, slots_free.
    pub fn register_ready_from_handoff(
        &mut self,
        pool_id: impl Into<String>,
        handoff: &json::Value,
    ) {
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
        // Health
        entry.health = HealthStatus { live: true, ready: true };
        // Meta: only overwrite if present
        if let Some(v) = handoff.get("engine_version").and_then(|v| v.as_str()) {
            entry.engine_version = Some(v.to_string());
        }
        if let Some(dm) = handoff.get("device_mask").and_then(|v| v.as_str()) {
            entry.device_mask = Some(dm.to_string());
        }
        // Slots: prefer incoming values, otherwise preserve existing, default to 1/total if none
        let incoming_total = handoff.get("slots_total").and_then(|v| v.as_i64()).map(|x| x as i32);
        let incoming_free = handoff.get("slots_free").and_then(|v| v.as_i64()).map(|x| x as i32);
        let total = incoming_total
            .or(entry.slots_total)
            .unwrap_or(1);
        let mut free = incoming_free
            .or(entry.slots_free)
            .unwrap_or(total);
        free = free.max(0).min(total);
        entry.slots_total = Some(total);
        entry.slots_free = Some(free);
        // Clear last error on successful ready registration
        entry.last_error = None;
        // Heartbeat now (ms)
        let now_ms = {
            use std::time::{SystemTime, UNIX_EPOCH};
            let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
            (dur.as_secs() as i64) * 1000 + (dur.subsec_millis() as i64)
        };
        entry.last_heartbeat_ms = Some(now_ms);
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

    // OC-POOL-3101: registering from a valid handoff sets ready=true, fills meta/slots, clears last_error
    #[test]
    fn test_oc_pool_3101_register_ready_sets_ready_meta_and_clears_error() {
        let mut r = Registry::new();
        r.set_last_error("p1", "previous error");
        let handoff = json::json!({
            "engine_version": "llamacpp-source:v0-cpu",
            "device_mask": "GPU0",
            "slots_total": 2,
            "slots_free": 1
        });
        r.register_ready_from_handoff("p1", &handoff);
        let h = r.get_health("p1").expect("health");
        assert!(h.live && h.ready);
        assert_eq!(r.get_engine_version("p1").as_deref(), Some("llamacpp-source:v0-cpu"));
        assert_eq!(r.get_device_mask("p1").as_deref(), Some("GPU0"));
        assert_eq!(r.get_slots_total("p1"), Some(2));
        assert_eq!(r.get_slots_free("p1"), Some(1));
        assert!(r.get_heartbeat("p1").is_some());
        assert_eq!(r.get_last_error("p1"), None);
    }

    // OC-POOL-3102: repeated registrations update heartbeat/version; tolerate partial handoff fields
    #[test]
    fn test_oc_pool_3102_register_ready_idempotent_and_partial() {
        let mut r = Registry::new();
        let handoff1 = json::json!({ "engine_version": "v1", "slots_total": 4, "slots_free": 4 });
        r.register_ready_from_handoff("p2", &handoff1);
        let hb1 = r.get_heartbeat("p2").unwrap();
        // second handoff updates version, clamps free, and should not panic when fields are partial
        let handoff2 = json::json!({ "engine_version": "v2", "slots_free": 10 });
        r.register_ready_from_handoff("p2", &handoff2);
        let h = r.get_health("p2").unwrap();
        assert!(h.ready);
        assert_eq!(r.get_engine_version("p2").as_deref(), Some("v2"));
        assert_eq!(r.get_slots_total("p2"), Some(4));
        // free clamped to total
        assert_eq!(r.get_slots_free("p2"), Some(4));
        let hb2 = r.get_heartbeat("p2").unwrap();
        assert!(hb2 >= hb1);
    }
}
