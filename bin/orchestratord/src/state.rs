use crate::ports::storage::ArtifactStore;
use crate::services::placement::PlacementCache;
use crate::admission::{MetricLabels, QueueWithMetrics};
use orchestrator_core::queue::Policy;
use adapter_host::AdapterHost;
use pool_managerd::registry::Registry as PoolRegistry;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

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
    // Admission state
    pub admission: Arc<Mutex<QueueWithMetrics>>, // single bounded FIFO with metrics
    pub admissions: Arc<Mutex<HashMap<String, AdmissionSnapshot>>>, // task_id -> admission snapshot
    // Autobind watcher state: set of pools already bound (to avoid rebinding)
    pub bound_pools: Arc<Mutex<HashSet<String>>>,
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
        // Configure admission queue capacity and policy via env for tests/dev
        let cap: usize = std::env::var("ORCHD_ADMISSION_CAPACITY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let policy = match std::env::var("ORCHD_ADMISSION_POLICY").ok().as_deref() {
            Some("drop-lru") => Policy::DropLru,
            _ => Policy::Reject,
        };
        let labels = MetricLabels {
            engine: "llamacpp".into(),
            engine_version: "v0".into(),
            pool_id: "default".into(),
            replica_id: "r0".into(),
        };
        let q = QueueWithMetrics::new(cap, policy, labels);
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
            admission: Arc::new(Mutex::new(q)),
            admissions: Arc::new(Mutex::new(HashMap::new())),
            bound_pools: Arc::new(Mutex::new(HashSet::new())),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AdmissionInfo {
    pub queue_position: i64,
    pub predicted_start_ms: i64,
}

#[derive(Debug, Clone)]
pub struct AdmissionSnapshot {
    pub info: AdmissionInfo,
    pub request: contracts_api_types::TaskRequest,
}
