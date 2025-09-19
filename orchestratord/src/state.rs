use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use pool_managerd::registry::Registry as PoolRegistry;
use crate::ports::storage::ArtifactStore;
use adapter_host::AdapterHost;
use crate::services::placement::PlacementCache;

#[derive(Clone)]
pub struct AppState {
    pub logs: Arc<Mutex<Vec<String>>>,
    pub sessions: Arc<Mutex<HashMap<String, SessionInfo>>>,
    pub artifacts: Arc<Mutex<HashMap<String, serde_json::Value>>>,
    pub pool_manager: Arc<Mutex<PoolRegistry>>,
    pub draining_pools: Arc<Mutex<HashMap<String, bool>>>,
    pub artifact_store: Arc<dyn ArtifactStore>,
    pub cancellations: Arc<Mutex<HashSet<String>>>,
    pub adapter_host: Arc<AdapterHost>,
    pub capabilities_cache: Arc<Mutex<Option<serde_json::Value>>>,
    pub placement_cache: Arc<Mutex<PlacementCache>>,
}

#[derive(Debug, Clone, Default)]
pub struct SessionInfo {
    pub ttl_ms_remaining: i64,
    pub turns: i32,
    pub kv_bytes: i64,
    pub kv_warmth: bool,
    pub tokens_budget_remaining: i64,
    pub time_budget_remaining_ms: i64,
    pub cost_budget_remaining: f64,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            logs: Arc::new(Mutex::new(Vec::new())),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            artifacts: Arc::new(Mutex::new(HashMap::new())),
            pool_manager: Arc::new(Mutex::new(PoolRegistry::new())),
            draining_pools: Arc::new(Mutex::new(HashMap::new())),
            artifact_store: Arc::new(crate::infra::storage::inmem::InMemStore::default()),
            cancellations: Arc::new(Mutex::new(HashSet::new())),
            adapter_host: Arc::new(AdapterHost::new()),
            capabilities_cache: Arc::new(Mutex::new(None)),
            placement_cache: Arc::new(Mutex::new(PlacementCache::with_ttl(10_000))),
        }
    }
}

impl Default for AppState {
    fn default() -> Self { Self::new() }
}
