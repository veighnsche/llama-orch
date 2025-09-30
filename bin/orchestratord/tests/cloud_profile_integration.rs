//! Integration tests for CLOUD_PROFILE features
//!
//! Tests node registration, heartbeat, catalog availability, and placement
//! with mocked multi-node scenarios.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use orchestratord::{app::router::build_router, state::AppState};
use serde_json::json;
use tower::ServiceExt;

/// Test node registration flow
#[tokio::test]
async fn test_node_registration_flow() {
    // Enable cloud profile
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
    std::env::set_var("LLORCH_API_TOKEN", "test-token-12345678901234567890");

    let state = AppState::new();
    let app = build_router(state.clone());

    // Register a node
    let register_body = json!({
        "node_id": "gpu-node-1",
        "machine_id": "machine-alpha",
        "address": "http://192.168.1.100:9200",
        "pools": ["pool-0", "pool-1"],
        "capabilities": {
            "gpus": [{"device": "GPU0", "vram_gb": 24}],
            "cpu_cores": 16
        },
        "version": "0.1.0"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v2/nodes/register")
                .header("Authorization", "Bearer test-token-12345678901234567890")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&register_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify node is registered
    let list_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v2/nodes")
                .header("Authorization", "Bearer test-token-12345678901234567890")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(list_response.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(list_response.into_body(), usize::MAX).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(body["count"], 1);
    assert_eq!(body["nodes"][0]["node_id"], "gpu-node-1");

    // Cleanup
    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
    std::env::remove_var("LLORCH_API_TOKEN");
}

/// Test heartbeat updates pool status
#[tokio::test]
async fn test_heartbeat_updates_pool_status() {
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
    std::env::set_var("LLORCH_API_TOKEN", "test-token-12345678901234567890");

    let state = AppState::new();
    let app = build_router(state.clone());

    // Register node first
    let register_body = json!({
        "node_id": "gpu-node-1",
        "machine_id": "machine-alpha",
        "address": "http://192.168.1.100:9200",
        "pools": ["pool-0"],
        "capabilities": {},
        "version": "0.1.0"
    });

    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v2/nodes/register")
                .header("Authorization", "Bearer test-token-12345678901234567890")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&register_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Send heartbeat with pool status
    let heartbeat_body = json!({
        "timestamp": "2025-10-01T00:00:00Z",
        "pools": [
            {
                "pool_id": "pool-0",
                "ready": true,
                "draining": false,
                "slots_free": 3,
                "slots_total": 4,
                "vram_free_bytes": 18000000000u64,
                "engine": "llamacpp"
            }
        ]
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v2/nodes/gpu-node-1/heartbeat")
                .header("Authorization", "Bearer test-token-12345678901234567890")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&heartbeat_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    // Verify pool status was updated
    let registry = state.service_registry();
    let pool_status = registry.get_pool_status("gpu-node-1", "pool-0");
    assert!(pool_status.is_some());

    let status = pool_status.unwrap();
    assert_eq!(status.ready, true);
    assert_eq!(status.slots_free, 3);
    assert_eq!(status.slots_total, 4);

    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
    std::env::remove_var("LLORCH_API_TOKEN");
}

/// Test catalog availability endpoint
#[tokio::test]
async fn test_catalog_availability_endpoint() {
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
    std::env::set_var("LLORCH_API_TOKEN", "test-token-12345678901234567890");

    let state = AppState::new();
    let app = build_router(state.clone());

    // Register two nodes
    for (node_id, models) in [
        ("gpu-node-1", vec!["llama-3.1-8b-instruct"]),
        ("gpu-node-2", vec!["mistral-7b-instruct"]),
    ] {
        let register_body = json!({
            "node_id": node_id,
            "machine_id": format!("machine-{}", node_id),
            "address": format!("http://192.168.1.{}:9200", if node_id == "gpu-node-1" { 100 } else { 101 }),
            "pools": ["pool-0"],
            "capabilities": {},
            "version": "0.1.0"
        });

        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v2/nodes/register")
                    .header("Authorization", "Bearer test-token-12345678901234567890")
                    .header("Content-Type", "application/json")
                    .body(Body::from(serde_json::to_string(&register_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Send heartbeat with models
        let heartbeat_body = json!({
            "timestamp": "2025-10-01T00:00:00Z",
            "pools": [
                {
                    "pool_id": "pool-0",
                    "ready": true,
                    "draining": false,
                    "slots_free": 2,
                    "slots_total": 4,
                    "vram_free_bytes": 20000000000u64,
                    "engine": "llamacpp",
                    "models_available": models
                }
            ]
        });

        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(&format!("/v2/nodes/{}/heartbeat", node_id))
                    .header("Authorization", "Bearer test-token-12345678901234567890")
                    .header("Content-Type", "application/json")
                    .body(Body::from(serde_json::to_string(&heartbeat_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
    }

    // Query catalog availability
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v2/catalog/availability")
                .header("Authorization", "Bearer test-token-12345678901234567890")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    // Verify response structure
    assert_eq!(body["total_models"], 2);
    assert_eq!(body["single_node_models"].as_array().unwrap().len(), 2);
    assert_eq!(body["replicated_models"].as_array().unwrap().len(), 0);

    // Verify node catalogs
    assert!(body["nodes"]["gpu-node-1"]["models"].as_array().unwrap().contains(&json!("llama-3.1-8b-instruct")));
    assert!(body["nodes"]["gpu-node-2"]["models"].as_array().unwrap().contains(&json!("mistral-7b-instruct")));

    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
    std::env::remove_var("LLORCH_API_TOKEN");
}

/// Test node deregistration
#[tokio::test]
async fn test_node_deregistration() {
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
    std::env::set_var("LLORCH_API_TOKEN", "test-token-12345678901234567890");

    let state = AppState::new();
    let app = build_router(state.clone());

    // Register node
    let register_body = json!({
        "node_id": "gpu-node-1",
        "machine_id": "machine-alpha",
        "address": "http://192.168.1.100:9200",
        "pools": ["pool-0"],
        "capabilities": {},
        "version": "0.1.0"
    });

    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v2/nodes/register")
                .header("Authorization", "Bearer test-token-12345678901234567890")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&register_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // Deregister node
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v2/nodes/gpu-node-1")
                .header("Authorization", "Bearer test-token-12345678901234567890")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    // Verify node is gone
    let list_response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v2/nodes")
                .header("Authorization", "Bearer test-token-12345678901234567890")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body_bytes = axum::body::to_bytes(list_response.into_body(), usize::MAX).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    assert_eq!(body["count"], 0);

    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
    std::env::remove_var("LLORCH_API_TOKEN");
}

/// Test authentication on node endpoints
#[tokio::test]
async fn test_node_endpoints_require_auth() {
    std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true");
    std::env::set_var("LLORCH_API_TOKEN", "test-token-12345678901234567890");

    let state = AppState::new();
    let app = build_router(state);

    // Try to register without token
    let register_body = json!({
        "node_id": "gpu-node-1",
        "machine_id": "machine-alpha",
        "address": "http://192.168.1.100:9200",
        "pools": ["pool-0"],
        "capabilities": {},
        "version": "0.1.0"
    });

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v2/nodes/register")
                // No Authorization header
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&register_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    // Try with wrong token
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v2/nodes/register")
                .header("Authorization", "Bearer wrong-token")
                .header("Content-Type", "application/json")
                .body(Body::from(serde_json::to_string(&register_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    std::env::remove_var("ORCHESTRATORD_CLOUD_PROFILE");
    std::env::remove_var("LLORCH_API_TOKEN");
}
