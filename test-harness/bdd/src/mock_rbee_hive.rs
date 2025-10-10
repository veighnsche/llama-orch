//! Mock rbee-hive server for BDD tests
//!
//! Created by: TEAM-054
//!
//! This module provides a mock rbee-hive server that runs on port 9200
//! (per the normative spec, NOT 8080 or 8090!) for testing purposes.

use axum::{
    routing::{get, post},
    Router, Json,
};
use std::net::SocketAddr;
use anyhow::Result;

/// Start the mock rbee-hive server on port 9200
pub async fn start_mock_rbee_hive() -> Result<()> {
    let app = Router::new()
        .route("/v1/health", get(handle_health))
        .route("/v1/workers/spawn", post(handle_spawn_worker))
        .route("/v1/workers/ready", post(handle_worker_ready))
        .route("/v1/workers/list", get(handle_list_workers));
    
    // CRITICAL: Port 9200, not 8080 or 8090!
    // See: test-harness/bdd/PORT_ALLOCATION.md for reference
    let addr: SocketAddr = "127.0.0.1:9200".parse()?;
    tracing::info!("🐝 Starting mock rbee-hive on {} (port 9200 per spec)", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "alive",
        "version": "0.1.0-mock"
    }))
}

async fn handle_spawn_worker(Json(req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    tracing::info!("Mock rbee-hive: spawning worker for request: {:?}", req);
    
    Json(serde_json::json!({
        "worker_id": "mock-worker-123",
        "url": "http://127.0.0.1:8001",
        "state": "loading"
    }))
}

async fn handle_worker_ready(Json(req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    tracing::info!("Mock rbee-hive: worker ready callback: {:?}", req);
    
    Json(serde_json::json!({
        "success": true
    }))
}

async fn handle_list_workers() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "workers": []
    }))
}
