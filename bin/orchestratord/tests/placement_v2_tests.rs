//! Unit tests for PlacementService with model-aware filtering

use orchestratord::{
    services::placement_v2::{PlacementService, PlacementStrategy},
    state::AppState,
};
use pool_registry_types::{NodeInfo, PoolSnapshot};

#[test]
fn test_placement_filters_by_model_availability() {
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");

    let state = AppState::new();
    let placement = PlacementService::new(PlacementStrategy::RoundRobin);

    // Register two nodes with different models
    let node1 = NodeInfo::new(
        "gpu-node-1".to_string(),
        "machine-1".to_string(),
        "http://192.168.1.100:9200".to_string(),
        vec!["pool-0".to_string()],
        serde_json::json!({}),
    );

    let node2 = NodeInfo::new(
        "gpu-node-2".to_string(),
        "machine-2".to_string(),
        "http://192.168.1.101:9200".to_string(),
        vec!["pool-0".to_string()],
        serde_json::json!({}),
    );

    state.service_registry().register(node1).unwrap();
    state.service_registry().register(node2).unwrap();

    // Update pool status with different models
    let pool1 = PoolSnapshot {
        pool_id: "pool-0".to_string(),
        node_id: Some("gpu-node-1".to_string()),
        ready: true,
        draining: false,
        slots_free: 2,
        slots_total: 4,
        vram_free_bytes: 20_000_000_000,
        engine: Some("llamacpp".to_string()),
        models_available: vec!["llama-3.1-8b-instruct".to_string()],
    };

    let pool2 = PoolSnapshot {
        pool_id: "pool-0".to_string(),
        node_id: Some("gpu-node-2".to_string()),
        ready: true,
        draining: false,
        slots_free: 3,
        slots_total: 4,
        vram_free_bytes: 24_000_000_000,
        engine: Some("llamacpp".to_string()),
        models_available: vec!["mistral-7b-instruct".to_string()],
    };

    state.service_registry().update_pool_status("gpu-node-1", vec![pool1]);
    state.service_registry().update_pool_status("gpu-node-2", vec![pool2]);

    // Request placement with specific model
    let decision = placement.select_pool_with_model(&state, Some("llama-3.1-8b-instruct"));

    assert!(decision.is_some());
    let decision = decision.unwrap();
    assert_eq!(decision.node_id, Some("gpu-node-1".to_string()));
    assert_eq!(decision.pool_id, "pool-0");

    // Request placement with different model
    let decision = placement.select_pool_with_model(&state, Some("mistral-7b-instruct"));

    assert!(decision.is_some());
    let decision = decision.unwrap();
    assert_eq!(decision.node_id, Some("gpu-node-2".to_string()));

    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
}

#[test]
fn test_placement_returns_none_when_model_not_available() {
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");

    let state = AppState::new();
    let placement = PlacementService::new(PlacementStrategy::RoundRobin);

    // Register node with specific model
    let node = NodeInfo::new(
        "gpu-node-1".to_string(),
        "machine-1".to_string(),
        "http://192.168.1.100:9200".to_string(),
        vec!["pool-0".to_string()],
        serde_json::json!({}),
    );

    state.service_registry().register(node).unwrap();

    let pool = PoolSnapshot {
        pool_id: "pool-0".to_string(),
        node_id: Some("gpu-node-1".to_string()),
        ready: true,
        draining: false,
        slots_free: 2,
        slots_total: 4,
        vram_free_bytes: 20_000_000_000,
        engine: Some("llamacpp".to_string()),
        models_available: vec!["llama-3.1-8b-instruct".to_string()],
    };

    state.service_registry().update_pool_status("gpu-node-1", vec![pool]);

    // Request placement with unavailable model
    let decision = placement.select_pool_with_model(&state, Some("gpt-4"));

    assert!(decision.is_none());

    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
}

#[test]
fn test_placement_works_without_model_filter() {
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");

    let state = AppState::new();
    let placement = PlacementService::new(PlacementStrategy::RoundRobin);

    // Register node
    let node = NodeInfo::new(
        "gpu-node-1".to_string(),
        "machine-1".to_string(),
        "http://192.168.1.100:9200".to_string(),
        vec!["pool-0".to_string()],
        serde_json::json!({}),
    );

    state.service_registry().register(node).unwrap();

    let pool = PoolSnapshot {
        pool_id: "pool-0".to_string(),
        node_id: Some("gpu-node-1".to_string()),
        ready: true,
        draining: false,
        slots_free: 2,
        slots_total: 4,
        vram_free_bytes: 20_000_000_000,
        engine: Some("llamacpp".to_string()),
        models_available: vec!["llama-3.1-8b-instruct".to_string()],
    };

    state.service_registry().update_pool_status("gpu-node-1", vec![pool]);

    // Request placement without model filter
    let decision = placement.select_pool(&state);

    assert!(decision.is_some());
    let decision = decision.unwrap();
    assert_eq!(decision.node_id, Some("gpu-node-1".to_string()));

    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
}

#[test]
fn test_least_loaded_strategy_with_model_filter() {
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");

    let state = AppState::new();
    let placement = PlacementService::new(PlacementStrategy::LeastLoaded);

    // Register two nodes with same model but different loads
    for (node_id, slots_free) in [("gpu-node-1", 1), ("gpu-node-2", 3)] {
        let node = NodeInfo::new(
            node_id.to_string(),
            format!("machine-{}", node_id),
            format!("http://192.168.1.{}:9200", if node_id == "gpu-node-1" { 100 } else { 101 }),
            vec!["pool-0".to_string()],
            serde_json::json!({}),
        );

        state.service_registry().register(node).unwrap();

        let pool = PoolSnapshot {
            pool_id: "pool-0".to_string(),
            node_id: Some(node_id.to_string()),
            ready: true,
            draining: false,
            slots_free,
            slots_total: 4,
            vram_free_bytes: 20_000_000_000,
            engine: Some("llamacpp".to_string()),
            models_available: vec!["llama-3.1-8b-instruct".to_string()],
        };

        state.service_registry().update_pool_status(node_id, vec![pool]);
    }

    // Should select node with most free slots
    let decision = placement.select_pool_with_model(&state, Some("llama-3.1-8b-instruct"));

    assert!(decision.is_some());
    let decision = decision.unwrap();
    assert_eq!(decision.node_id, Some("gpu-node-2".to_string())); // Has 3 free slots

    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
}

#[test]
fn test_placement_skips_draining_pools() {
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");

    let state = AppState::new();
    let placement = PlacementService::new(PlacementStrategy::RoundRobin);

    // Register two nodes, one draining
    for (node_id, draining) in [("gpu-node-1", true), ("gpu-node-2", false)] {
        let node = NodeInfo::new(
            node_id.to_string(),
            format!("machine-{}", node_id),
            format!("http://192.168.1.{}:9200", if node_id == "gpu-node-1" { 100 } else { 101 }),
            vec!["pool-0".to_string()],
            serde_json::json!({}),
        );

        state.service_registry().register(node).unwrap();

        let pool = PoolSnapshot {
            pool_id: "pool-0".to_string(),
            node_id: Some(node_id.to_string()),
            ready: true,
            draining,
            slots_free: 2,
            slots_total: 4,
            vram_free_bytes: 20_000_000_000,
            engine: Some("llamacpp".to_string()),
            models_available: vec!["llama-3.1-8b-instruct".to_string()],
        };

        state.service_registry().update_pool_status(node_id, vec![pool]);
    }

    // Should skip draining node
    let decision = placement.select_pool_with_model(&state, Some("llama-3.1-8b-instruct"));

    assert!(decision.is_some());
    let decision = decision.unwrap();
    assert_eq!(decision.node_id, Some("gpu-node-2".to_string())); // Not draining

    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
}

#[test]
fn test_placement_skips_not_ready_pools() {
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");

    let state = AppState::new();
    let placement = PlacementService::new(PlacementStrategy::RoundRobin);

    let node = NodeInfo::new(
        "gpu-node-1".to_string(),
        "machine-1".to_string(),
        "http://192.168.1.100:9200".to_string(),
        vec!["pool-0".to_string()],
        serde_json::json!({}),
    );

    state.service_registry().register(node).unwrap();

    let pool = PoolSnapshot {
        pool_id: "pool-0".to_string(),
        node_id: Some("gpu-node-1".to_string()),
        ready: false, // Not ready
        draining: false,
        slots_free: 2,
        slots_total: 4,
        vram_free_bytes: 20_000_000_000,
        engine: Some("llamacpp".to_string()),
        models_available: vec!["llama-3.1-8b-instruct".to_string()],
    };

    state.service_registry().update_pool_status("gpu-node-1", vec![pool]);

    // Should return None (no ready pools)
    let decision = placement.select_pool_with_model(&state, Some("llama-3.1-8b-instruct"));

    assert!(decision.is_none());

    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
}
