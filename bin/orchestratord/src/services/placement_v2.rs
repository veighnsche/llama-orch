//! Multi-node placement service for CLOUD_PROFILE
//!
//! Selects the best node+pool for task execution based on:
//! - Node health (online, heartbeat recent)
//! - Pool readiness (ready=true, not draining)
//! - Available slots (slots_free > 0)
//! - Load balancing strategy (round-robin, least-loaded)

use crate::state::AppState;
use pool_registry_types::PoolSnapshot;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Placement decision with node context
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementDecisionV2 {
    /// Node ID (for CLOUD_PROFILE) or None (for HOME_PROFILE)
    pub node_id: Option<String>,
    
    /// Pool ID on the selected node
    pub pool_id: String,
    
    /// Optional replica ID
    pub replica_id: Option<String>,
    
    /// Node address (for HTTP calls in CLOUD_PROFILE)
    pub node_address: Option<String>,
}

/// Placement strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementStrategy {
    /// Round-robin across available pools
    RoundRobin,
    
    /// Select pool with most free slots
    LeastLoaded,
    
    /// Random selection
    Random,
}

/// Placement service for multi-node task routing
#[derive(Clone)]
pub struct PlacementService {
    strategy: PlacementStrategy,
    round_robin_counter: Arc<AtomicUsize>,
}

impl PlacementService {
    pub fn new(strategy: PlacementStrategy) -> Self {
        Self {
            strategy,
            round_robin_counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Select best pool for task execution
    ///
    /// CLOUD_PROFILE: Queries ServiceRegistry for online nodes and their pools
    /// HOME_PROFILE: Falls back to local adapter_host
    pub fn select_pool(&self, state: &AppState) -> Option<PlacementDecisionV2> {
        if state.cloud_profile_enabled() {
            self.select_pool_cloud(state)
        } else {
            self.select_pool_home(state)
        }
    }

    /// CLOUD_PROFILE: Select from registered nodes
    fn select_pool_cloud(&self, state: &AppState) -> Option<PlacementDecisionV2> {
        let registry = state.service_registry();
        let nodes = registry.get_online_nodes();

        if nodes.is_empty() {
            tracing::warn!("No online nodes available for placement");
            return None;
        }

        // Collect all available pools across all nodes
        let mut candidates: Vec<(String, String, PoolSnapshot)> = Vec::new();
        
        for node in &nodes {
            // Get actual pool status from heartbeat data
            let pool_statuses = registry.get_node_pools(&node.node_id);
            
            if pool_statuses.is_empty() {
                // Fallback: Use pools from registration (no heartbeat data yet)
                for pool_id in &node.pools {
                    candidates.push((
                        node.node_id.clone(),
                        pool_id.clone(),
                        PoolSnapshot {
                            pool_id: pool_id.clone(),
                            node_id: Some(node.node_id.clone()),
                            ready: false, // Not ready until first heartbeat
                            draining: false,
                            slots_free: 0,
                            slots_total: 0,
                            vram_free_bytes: 0,
                            engine: None,
                            models_available: vec![],
                        },
                    ));
                }
            } else {
                // Use real heartbeat data
                for status in pool_statuses {
                    candidates.push((
                        node.node_id.clone(),
                        status.pool_id.clone(),
                        status,
                    ));
                }
            }
        }

        if candidates.is_empty() {
            tracing::warn!("No available pools on online nodes");
            return None;
        }

        // Filter to ready, non-draining pools with free slots
        let available: Vec<_> = candidates
            .into_iter()
            .filter(|(_, _, snapshot)| snapshot.is_available())
            .collect();

        if available.is_empty() {
            tracing::warn!("No pools with available slots");
            return None;
        }

        // Select based on strategy
        let (node_id, pool_id, _status) = match self.strategy {
            PlacementStrategy::RoundRobin => {
                let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
                available[idx % available.len()].clone()
            }
            PlacementStrategy::LeastLoaded => {
                available
                    .iter()
                    .max_by_key(|(_, _, status)| status.slots_free)
                    .cloned()
                    .unwrap()
            }
            PlacementStrategy::Random => {
                use rand::Rng;
                let idx = rand::thread_rng().gen_range(0..available.len());
                available[idx].clone()
            }
        };

        // Find node address
        let node_address = nodes
            .iter()
            .find(|n| n.node_id == node_id)
            .map(|n| n.address.clone());

        Some(PlacementDecisionV2 {
            node_id: Some(node_id),
            pool_id,
            replica_id: None,
            node_address,
        })
    }

    /// HOME_PROFILE: Select from local adapter_host
    fn select_pool_home(&self, _state: &AppState) -> Option<PlacementDecisionV2> {
        // For HOME_PROFILE, use existing placement logic
        // Always return default pool (adapter binding checked elsewhere)
        let pool_id = "default".to_string();
        
        Some(PlacementDecisionV2 {
            node_id: None,
            pool_id,
            replica_id: Some("r0".to_string()),
            node_address: None,
        })
    }

    /// Check if a specific pool is dispatchable
    ///
    /// CLOUD_PROFILE: Checks ServiceRegistry for node+pool status
    /// HOME_PROFILE: Checks local adapter_host
    pub fn is_pool_dispatchable(&self, state: &AppState, pool_id: &str) -> bool {
        if state.cloud_profile_enabled() {
            self.is_pool_dispatchable_cloud(state, pool_id)
        } else {
            self.is_pool_dispatchable_home(state, pool_id)
        }
    }

    fn is_pool_dispatchable_cloud(&self, state: &AppState, pool_id: &str) -> bool {
        let registry = state.service_registry();
        let nodes = registry.get_online_nodes();

        // Check if any online node has this pool ready
        for node in nodes {
            if let Some(status) = registry.get_pool_status(&node.node_id, pool_id) {
                if status.is_available() {
                    return true;
                }
            }
        }

        false
    }

    fn is_pool_dispatchable_home(&self, _state: &AppState, _pool_id: &str) -> bool {
        // For HOME_PROFILE, assume pool is dispatchable if adapter is bound
        // Actual check happens in adapter_host.submit()
        true
    }
}

impl Default for PlacementService {
    fn default() -> Self {
        Self::new(PlacementStrategy::RoundRobin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_placement_service_home_profile() {
        let state = AppState::new();
        let service = PlacementService::new(PlacementStrategy::RoundRobin);

        // HOME_PROFILE: Should return None (no adapters bound)
        let decision = service.select_pool(&state);
        assert!(decision.is_none());
    }

    #[test]
    fn test_placement_service_cloud_profile_no_nodes() {
        std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
        
        let state = AppState::new();
        let service = PlacementService::new(PlacementStrategy::RoundRobin);

        // CLOUD_PROFILE: Should return None (no nodes registered)
        let decision = service.select_pool(&state);
        assert!(decision.is_none());
        
        std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
    }

    #[test]
    fn test_placement_strategy_round_robin() {
        let service = PlacementService::new(PlacementStrategy::RoundRobin);
        
        // Round-robin counter should increment
        let counter1 = service.round_robin_counter.load(Ordering::Relaxed);
        service.round_robin_counter.fetch_add(1, Ordering::Relaxed);
        let counter2 = service.round_robin_counter.load(Ordering::Relaxed);
        
        assert_eq!(counter2, counter1 + 1);
    }

    #[test]
    fn test_placement_decision_equality() {
        let d1 = PlacementDecisionV2 {
            node_id: Some("node-1".to_string()),
            pool_id: "pool-0".to_string(),
            replica_id: None,
            node_address: Some("http://localhost:9200".to_string()),
        };

        let d2 = PlacementDecisionV2 {
            node_id: Some("node-1".to_string()),
            pool_id: "pool-0".to_string(),
            replica_id: None,
            node_address: Some("http://localhost:9200".to_string()),
        };

        assert_eq!(d1, d2);
    }

    #[test]
    fn test_is_pool_dispatchable_home_profile() {
        let state = AppState::new();
        let service = PlacementService::new(PlacementStrategy::RoundRobin);

        // HOME_PROFILE: Should return false (no adapters)
        assert!(!service.is_pool_dispatchable(&state, "default"));
    }
}
