use crate::admission::{MetricLabels, QueueWithMetrics};
use crate::clients::pool_manager::PoolManagerClient;
use crate::ports::storage::ArtifactStore;
use crate::services::placement::PlacementCache;
use crate::services::placement_v2::{PlacementService, PlacementStrategy};
// TODO: Remove adapter_host - migrating to direct worker communication
// use adapter_host::AdapterHost;
use orchestrator_core::queue::Policy;
use pool_registry::ServiceRegistry;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct AppState {
    pub logs: Arc<Mutex<Vec<String>>>,
    pub sessions: Arc<Mutex<HashMap<String, SessionInfo>>>,
    pub artifacts: Arc<Mutex<HashMap<String, serde_json::Value>>>,
    pub pool_manager: PoolManagerClient,
    pub draining_pools: Arc<Mutex<HashMap<String, bool>>>,
    pub artifact_store: Arc<dyn ArtifactStore>,
    pub cancellations: Arc<Mutex<HashSet<String>>>,
    // TODO: Remove adapter_host - migrating to direct worker communication
    // pub adapter_host: Arc<AdapterHost>,
    pub capabilities_cache: Arc<Mutex<Option<serde_json::Value>>>,
    pub placement_cache: Arc<Mutex<PlacementCache>>,
    // Admission state
    pub admission: Arc<Mutex<QueueWithMetrics>>, // single bounded FIFO with metrics
    pub admissions: Arc<Mutex<HashMap<String, AdmissionSnapshot>>>, // task_id -> admission snapshot
    // Autobind watcher state: set of pools already bound (to avoid rebinding)
    pub bound_pools: Arc<Mutex<HashSet<String>>>,
    // Cloud profile: Service registry for tracking GPU nodes
    pub service_registry: Option<ServiceRegistry>,
    pub cloud_profile: bool,
    // Multi-node placement service
    pub placement_service: PlacementService,
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
            pool_manager: PoolManagerClient::from_env(),
            draining_pools: Arc::new(Mutex::new(HashMap::new())),
            artifact_store: Arc::new(crate::infra::storage::inmem::InMemStore::default()),
            cancellations: Arc::new(Mutex::new(HashSet::new())),
            // TODO: Remove adapter_host - migrating to direct worker communication
            // adapter_host: Arc::new(AdapterHost::new()),
            capabilities_cache: Arc::new(Mutex::new(None)),
            placement_cache: Arc::new(Mutex::new(PlacementCache::with_ttl(10_000))),
            admission: Arc::new(Mutex::new(q)),
            admissions: Arc::new(Mutex::new(HashMap::new())),
            bound_pools: Arc::new(Mutex::new(HashSet::new())),
            // Cloud profile support
            cloud_profile: std::env::var("ORCHESTRATORD_CLOUD_PROFILE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(false),
            service_registry: if std::env::var("ORCHESTRATORD_CLOUD_PROFILE")
                .ok()
                .and_then(|v| v.parse::<bool>().ok())
                .unwrap_or(false)
            {
                let timeout_ms = std::env::var("ORCHESTRATORD_NODE_TIMEOUT_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(30_000);
                Some(ServiceRegistry::new(timeout_ms))
            } else {
                None
            },
            // Placement service (strategy from env)
            placement_service: {
                let strategy =
                    match std::env::var("ORCHESTRATORD_PLACEMENT_STRATEGY").ok().as_deref() {
                        Some("least-loaded") => PlacementStrategy::LeastLoaded,
                        Some("random") => PlacementStrategy::Random,
                        _ => PlacementStrategy::RoundRobin,
                    };
                PlacementService::new(strategy)
            },
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    /// Check if cloud profile is enabled
    pub fn cloud_profile_enabled(&self) -> bool {
        self.cloud_profile
    }

    /// Get service registry (panics if cloud profile disabled)
    pub fn service_registry(&self) -> &ServiceRegistry {
        self.service_registry
            .as_ref()
            .expect("Service registry not available. Enable ORCHESTRATORD_CLOUD_PROFILE=true")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_default_home_profile() {
        // Default should be HOME_PROFILE (cloud profile disabled)
        let state = AppState::new();
        assert!(!state.cloud_profile_enabled());
        assert!(state.service_registry.is_none());
    }

    #[test]
    fn test_app_state_cloud_profile_enabled() {
        std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");

        let state = AppState::new();
        assert!(state.cloud_profile_enabled());
        assert!(state.service_registry.is_some());

        // Should not panic
        let _registry = state.service_registry();

        std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
    }

    #[test]
    fn test_app_state_custom_node_timeout() {
        std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
        std::env::set_var("ORCHESTRATORD_NODE_TIMEOUT_MS", "60000");

        let state = AppState::new();
        assert!(state.service_registry.is_some());

        std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
        std::env::remove_var("ORCHESTRATORD_NODE_TIMEOUT_MS");
    }

    #[test]
    #[should_panic(expected = "Service registry not available")]
    fn test_service_registry_panics_when_disabled() {
        let state = AppState::new();
        let _ = state.service_registry();
    }
}
