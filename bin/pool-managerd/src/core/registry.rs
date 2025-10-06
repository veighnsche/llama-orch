//! Replica registry (planning-only).
//!
//! Status: PARTIALLY IMPLEMENTED (core registry + readiness API done; supervision/drain/reload pending).
//! Spec: ORCH-3002, ORCH-3010, ORCH-3027, ORCH-3028, ORCH-3038 (see README.md).
//! Completed (Owner E tasks from TODO_OWNERS_MVP_pt3.md):
//!   - Health/version/heartbeat/last_error getters/setters
//!   - Lease counters (allocate/release, never negative)
//!   - register_ready_from_handoff() API for provisioners
//!   - set_engine_meta() for optional engine metadata
//!   - Draining flag and lease refusal
//!   - Snapshots export for placement
//! TODO: Wire to supervision module (not yet implemented) for automatic health checks and backoff.
//! TODO: Wire to drain/reload module (drain.rs) for atomic model swaps.
//! TODO: Add VRAM tracking fields (vram_total_bytes, vram_free_bytes, compute_capability) per CHECKLIST.md.
//! TODO: Add perf hints (tokens_per_s, first_token_ms) for placement heuristics.
//! Integration: Used by orchestratord (state.rs:6), test-harness/bdd (pool_manager.rs:3).

use std::collections::HashMap;

use crate::health::HealthStatus;
use serde_json as json;
// Registry submodules
#[path = "registry/snapshot.rs"]
mod snapshot;
pub use snapshot::PoolSnapshot;
#[path = "registry/entry.rs"]
mod entry;
use entry::PoolEntry;
#[path = "registry/types.rs"]
mod types;
use types::UpdateFields;

#[derive(Debug, Clone)]
pub struct Replica;

#[derive(Debug, Clone, Default)]
pub struct Registry {
    pools: HashMap<String, PoolEntry>,
}

// PoolEntry type is defined in registry/entry.rs

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
            draining: false,
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
            draining: false,
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
            draining: false,
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
            draining: false,
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
            draining: false,
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
            draining: false,
        });
        if let Some(v) = version {
            entry.engine_version = Some(v.into());
        }
        if let Some(d) = digest {
            entry.engine_digest = Some(d.into());
        }
        if let Some(c) = catalog_id {
            entry.engine_catalog_id = Some(c.into());
        }
    }

    pub fn get_engine_digest(&self, pool_id: &str) -> Option<String> {
        self.pools.get(pool_id).and_then(|e| e.engine_digest.as_ref().cloned())
    }

    pub fn get_engine_catalog_id(&self, pool_id: &str) -> Option<String> {
        self.pools.get(pool_id).and_then(|e| e.engine_catalog_id.as_ref().cloned())
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
            draining: false,
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
            draining: false,
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
            draining: false,
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
        let total = incoming_total.or(entry.slots_total).unwrap_or(1);
        let mut free = incoming_free.or(entry.slots_free).unwrap_or(total);
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

    /// Mark pool as draining; when draining, new leases are refused.
    pub fn set_draining(&mut self, pool_id: impl Into<String>, draining: bool) {
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
            draining: false,
        });
        entry.draining = draining;
    }

    pub fn get_draining(&self, pool_id: &str) -> bool {
        self.pools.get(pool_id).map(|e| e.draining).unwrap_or(false)
    }

    /// Explicitly register a pool. Returns true if newly inserted.
    pub fn register(&mut self, pool_id: impl Into<String>) -> bool {
        let id = pool_id.into();
        if let std::collections::hash_map::Entry::Vacant(e) = self.pools.entry(id) {
            e.insert(PoolEntry {
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
                    draining: false,
                });
            true
        } else {
            false
        }
    }

    /// Explicitly deregister a pool. Returns true if an entry existed.
    pub fn deregister(&mut self, pool_id: &str) -> bool {
        self.pools.remove(pool_id).is_some()
    }

    /// Update merge for selected fields; immutable identity is preserved.
    pub fn update(&mut self, pool_id: impl Into<String>, fields: UpdateFields) {
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
            draining: false,
        });
        if let Some(v) = fields.engine_version {
            entry.engine_version = Some(v);
        }
        if let Some(dm) = fields.device_mask {
            entry.device_mask = Some(dm);
        }
        if let Some(t) = fields.slots_total {
            entry.slots_total = Some(t);
        }
        if let Some(f) = fields.slots_free {
            entry.slots_free = Some(f.max(0).min(entry.slots_total.unwrap_or(f)));
        }
        if let Some(ph) = fields.perf_hints {
            entry.perf_hints = Some(ph);
        }
    }

    /// Export typed, deterministically-ordered snapshots for consumers.
    pub fn snapshots(&self) -> Vec<PoolSnapshot> {
        let mut v: Vec<PoolSnapshot> = self
            .pools
            .iter()
            .map(|(pool_id, e)| PoolSnapshot {
                pool_id: pool_id.clone(),
                health: e.health.clone(),
                engine_version: e.engine_version.clone(),
                device_mask: e.device_mask.clone(),
                slots_total: e.slots_total,
                slots_free: e.slots_free,
                vram_total_bytes: None,
                vram_free_bytes: None,
                compute_capability: None,
                perf_hints: e.perf_hints.clone(),
                draining: e.draining,
            })
            .collect();
        v.sort_by(|a, b| a.pool_id.cmp(&b.pool_id));
        v
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
            draining: false,
        });
        if !entry.draining {
            entry.active_leases = entry.active_leases.saturating_add(1);
        }
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
            draining: false,
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
    use super::types::UpdateFields;
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

    // OC-POOL-3103: set_engine_meta updates only provided fields
    #[test]
    fn test_oc_pool_3103_set_engine_meta_partial_updates() {
        let mut r = Registry::new();
        // set version and catalog only
        r.set_engine_meta("p3", Some("eng-v1"), None::<&str>, Some("catalog-A"));
        assert_eq!(r.get_engine_version("p3").as_deref(), Some("eng-v1"));
        assert_eq!(r.get_engine_catalog_id("p3").as_deref(), Some("catalog-A"));
        assert_eq!(r.get_engine_digest("p3"), None);
        // update only digest; keep version/catalog
        r.set_engine_meta("p3", None::<&str>, Some("sha256:deadbeef"), None::<&str>);
        assert_eq!(r.get_engine_version("p3").as_deref(), Some("eng-v1"));
        assert_eq!(r.get_engine_catalog_id("p3").as_deref(), Some("catalog-A"));
        assert_eq!(r.get_engine_digest("p3").as_deref(), Some("sha256:deadbeef"));
    }

    // OC-POOL-3104: set_slots clamps free between 0 and total
    #[test]
    fn test_oc_pool_3104_set_slots_clamps_free() {
        let mut r = Registry::new();
        r.set_slots("p4", 3, 5);
        assert_eq!(r.get_slots_total("p4"), Some(3));
        assert_eq!(r.get_slots_free("p4"), Some(3));
        r.set_slots("p4", 3, -10);
        assert_eq!(r.get_slots_free("p4"), Some(0));
    }

    // OC-POOL-3105: set_engine_version sets the engine_version field directly
    #[test]
    fn test_oc_pool_3105_set_engine_version_direct() {
        let mut r = Registry::new();
        assert_eq!(r.get_engine_version("p5"), None);
        r.set_engine_version("p5", "engine-xyz");
        assert_eq!(r.get_engine_version("p5").as_deref(), Some("engine-xyz"));
    }

    // OC-POOL-3106: register/deregister idempotent and preserves identity
    #[test]
    fn test_oc_pool_3106_register_and_deregister() {
        let mut r = Registry::new();
        assert!(r.register("p6"));
        // idempotent register
        assert!(!r.register("p6"));
        assert!(r.get_health("p6").is_some());
        // deregister
        assert!(r.deregister("p6"));
        // idempotent deregister
        assert!(!r.deregister("p6"));
        assert!(r.get_health("p6").is_none());
    }

    // OC-POOL-3107: update merges fields without resetting identity
    #[test]
    fn test_oc_pool_3107_update_merges_fields() {
        let mut r = Registry::new();
        r.register("p7");
        r.update(
            "p7",
            UpdateFields {
                engine_version: Some("vA".into()),
                device_mask: None,
                slots_total: Some(8),
                slots_free: Some(3),
                perf_hints: None,
            },
        );
        assert_eq!(r.get_engine_version("p7").as_deref(), Some("vA"));
        assert_eq!(r.get_slots_total("p7"), Some(8));
        assert_eq!(r.get_slots_free("p7"), Some(3));
        // Partial update
        r.update(
            "p7",
            UpdateFields {
                engine_version: Some("vB".into()),
                device_mask: Some("GPU0".into()),
                slots_total: None,
                slots_free: Some(10),
                perf_hints: Some(json::json!({"tps": 1000})),
            },
        );
        assert_eq!(r.get_engine_version("p7").as_deref(), Some("vB"));
        assert_eq!(r.get_device_mask("p7").as_deref(), Some("GPU0"));
        // free clamped to total (8)
        assert_eq!(r.get_slots_free("p7"), Some(8));
    }

    // OC-POOL-3108: draining refuses new leases
    #[test]
    fn test_oc_pool_3108_draining_refuses_leases() {
        let mut r = Registry::new();
        r.set_slots("p8", 2, 2);
        assert_eq!(r.allocate_lease("p8"), 1);
        r.set_draining("p8", true);
        // allocation is refused; count unchanged
        assert_eq!(r.allocate_lease("p8"), 1);
        // releasing still decrements
        assert_eq!(r.release_lease("p8"), 0);
        assert!(r.get_draining("p8"));
    }

    // OC-POOL-3109: snapshots are deterministic and map fields correctly
    #[test]
    fn test_oc_pool_3109_snapshots_deterministic_and_mapped() {
        let mut r = Registry::new();
        r.register("b");
        r.register("a");
        r.set_health("a", HealthStatus { live: true, ready: true });
        r.set_engine_version("a", "v1");
        r.set_device_mask("a", "GPU0");
        r.set_slots("a", 4, 3);
        let snaps = r.snapshots();
        // sorted lexicographically: a then b
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[0].pool_id, "a");
        assert_eq!(snaps[1].pool_id, "b");
        // field mapping
        assert_eq!(snaps[0].health, HealthStatus { live: true, ready: true });
        assert_eq!(snaps[0].engine_version.as_deref(), Some("v1"));
        assert_eq!(snaps[0].device_mask.as_deref(), Some("GPU0"));
        assert_eq!(snaps[0].slots_total, Some(4));
        assert_eq!(snaps[0].slots_free, Some(3));
        assert_eq!(snaps[0].draining, false);
    }
}
