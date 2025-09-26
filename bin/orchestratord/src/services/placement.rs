//! Placement prefilter and decision cache (scaffold).

use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlacementKey {
    pub job_spec_hash: String,
    pub snapshot_version: String,
    pub policy: String,
}

#[derive(Clone, Debug)]
pub struct PlacementDecision {
    pub pool_id: String,
    pub replica_id: Option<String>,
}

#[derive(Default)]
pub struct PlacementCache {
    ttl: Duration,
    map: HashMap<PlacementKey, (PlacementDecision, Instant)>,
}

impl PlacementCache {
    pub fn with_ttl(ttl_ms: u64) -> Self {
        Self { ttl: Duration::from_millis(ttl_ms), map: HashMap::new() }
    }
    pub fn get(&mut self, k: &PlacementKey) -> Option<PlacementDecision> {
        if let Some((v, t0)) = self.map.get(k) {
            if t0.elapsed() <= self.ttl {
                return Some(v.clone());
            }
        }
        None
    }
    pub fn put(&mut self, k: PlacementKey, d: PlacementDecision) {
        self.map.insert(k, (d, Instant::now()));
    }
}

/// Very minimal feasibility prefilter (scaffold): always route to "default" pool.
pub fn prefilter_route_default() -> PlacementDecision {
    PlacementDecision { pool_id: "default".into(), replica_id: None }
}
