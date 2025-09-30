//! Catalog availability endpoint for CLOUD_PROFILE
//!
//! Provides visibility into which models are available on which nodes.
//! Useful for operators to verify model distribution before dispatching tasks.

use axum::{extract::State, response::IntoResponse, Json};
use http::StatusCode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::state::AppState;

/// Response for GET /v2/catalog/availability
#[derive(Debug, Serialize, Deserialize)]
pub struct CatalogAvailabilityResponse {
    /// Map of node_id -> list of available models
    pub nodes: HashMap<String, NodeCatalog>,

    /// Total unique models across all nodes
    pub total_models: usize,

    /// Models available on all nodes (fully replicated)
    pub replicated_models: Vec<String>,

    /// Models available on only one node (single point of failure)
    pub single_node_models: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeCatalog {
    /// Node ID
    pub node_id: String,

    /// Node address
    pub address: String,

    /// Online status
    pub online: bool,

    /// Models available on this node (across all pools)
    pub models: Vec<String>,

    /// Per-pool breakdown
    pub pools: Vec<PoolCatalog>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PoolCatalog {
    /// Pool ID
    pub pool_id: String,

    /// Pool ready status
    pub ready: bool,

    /// Models available in this pool
    pub models: Vec<String>,
}

/// GET /v2/catalog/availability
///
/// Returns catalog availability across all nodes in the cluster.
///
/// Example response:
/// ```json
/// {
///   "nodes": {
///     "gpu-node-1": {
///       "node_id": "gpu-node-1",
///       "address": "http://192.168.1.100:9200",
///       "online": true,
///       "models": ["llama-3.1-8b-instruct", "mistral-7b-instruct"],
///       "pools": [
///         {
///           "pool_id": "pool-0",
///           "ready": true,
///           "models": ["llama-3.1-8b-instruct"]
///         }
///       ]
///     }
///   },
///   "total_models": 2,
///   "replicated_models": [],
///   "single_node_models": ["llama-3.1-8b-instruct", "mistral-7b-instruct"]
/// }
/// ```
pub async fn get_catalog_availability(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, StatusCode> {
    // Only available in CLOUD_PROFILE
    if !state.cloud_profile_enabled() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    let registry = state.service_registry();
    let nodes = registry.get_online_nodes();

    let mut node_catalogs = HashMap::new();
    let mut all_models: HashMap<String, usize> = HashMap::new(); // model -> node count

    for node in &nodes {
        let pool_statuses = registry.get_node_pools(&node.node_id);

        let mut node_models: Vec<String> = Vec::new();
        let mut pool_catalogs: Vec<PoolCatalog> = Vec::new();

        for pool in &pool_statuses {
            // Collect models from this pool
            for model in &pool.models_available {
                if !node_models.contains(model) {
                    node_models.push(model.clone());
                }
                *all_models.entry(model.clone()).or_insert(0) += 1;
            }

            pool_catalogs.push(PoolCatalog {
                pool_id: pool.pool_id.clone(),
                ready: pool.ready,
                models: pool.models_available.clone(),
            });
        }

        node_models.sort();

        node_catalogs.insert(
            node.node_id.clone(),
            NodeCatalog {
                node_id: node.node_id.clone(),
                address: node.address.clone(),
                online: true,
                models: node_models,
                pools: pool_catalogs,
            },
        );
    }

    // Calculate replicated and single-node models
    let total_nodes = nodes.len();
    let mut replicated_models: Vec<String> = Vec::new();
    let mut single_node_models: Vec<String> = Vec::new();

    for (model, count) in &all_models {
        if *count == total_nodes && total_nodes > 1 {
            replicated_models.push(model.clone());
        } else if *count == 1 {
            single_node_models.push(model.clone());
        }
    }

    replicated_models.sort();
    single_node_models.sort();

    let response = CatalogAvailabilityResponse {
        nodes: node_catalogs,
        total_models: all_models.len(),
        replicated_models,
        single_node_models,
    };

    Ok((StatusCode::OK, Json(response)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_availability_response_serialization() {
        let response = CatalogAvailabilityResponse {
            nodes: HashMap::new(),
            total_models: 0,
            replicated_models: vec![],
            single_node_models: vec![],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("nodes"));
        assert!(json.contains("total_models"));
    }
}
