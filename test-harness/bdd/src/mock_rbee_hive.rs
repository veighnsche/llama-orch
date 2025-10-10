//! Mock rbee-hive server for BDD tests
//!
//! Created by: TEAM-054
//! Modified by: TEAM-055 (added mock worker endpoint)
//! Modified by: TEAM-059 (real process spawning, not simulated)
//!
//! This module provides a mock rbee-hive server that runs on port 9200
//! (per the normative spec, NOT 8080 or 8090!) for testing purposes.
//! TEAM-059: Now spawns REAL worker processes instead of simulating them.

use axum::{
    routing::{get, post},
    Router, Json,
    body::Body,
    http::header,
    response::IntoResponse,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use anyhow::Result;

// TEAM-059: Shared state for tracking spawned worker processes
#[derive(Clone)]
struct RbeeHiveState {
    workers: Arc<Mutex<Vec<WorkerProcess>>>,
}

#[derive(Clone)]
struct WorkerProcess {
    worker_id: String,
    url: String,
    process: Arc<Mutex<Option<tokio::process::Child>>>,
}

/// Start the mock rbee-hive server on port 9200
/// TEAM-059: Now with real process management
pub async fn start_mock_rbee_hive() -> Result<()> {
    let state = RbeeHiveState {
        workers: Arc::new(Mutex::new(Vec::new())),
    };
    
    let app = Router::new()
        .route("/v1/health", get(handle_health))
        .route("/v1/workers/spawn", post(handle_spawn_worker))
        .route("/v1/workers/ready", post(handle_worker_ready))
        .route("/v1/workers/list", get(handle_list_workers))
        .with_state(state);
    
    // CRITICAL: Port 9200, not 8080 or 8090!
    // See: test-harness/bdd/PORT_ALLOCATION.md for reference
    let addr: SocketAddr = "127.0.0.1:9200".parse()?;
    tracing::info!("🐝 Starting mock rbee-hive on {} (port 9200 per spec)", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

// TEAM-059: Removed inline mock worker - now spawned as separate binary

async fn handle_health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "alive",
        "version": "0.1.0-mock"
    }))
}

// TEAM-059: Actually spawn a real worker process
async fn handle_spawn_worker(
    axum::extract::State(state): axum::extract::State<RbeeHiveState>,
    Json(req): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    tracing::info!("Mock rbee-hive: spawning REAL worker for request: {:?}", req);
    
    // Find available port (8001-8099)
    let port = 8001 + (state.workers.lock().await.len() as u16);
    let worker_id = format!("mock-worker-{}", port);
    let url = format!("http://127.0.0.1:{}", port);
    
    // TEAM-059: Spawn actual mock-worker binary
    let workspace_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| std::path::PathBuf::from(p).parent().unwrap().parent().unwrap().to_path_buf())
        .unwrap_or_else(|_| std::path::PathBuf::from("/home/vince/Projects/llama-orch"));
    
    let binary_path = workspace_dir.join("target/debug/mock-worker");
    
    match tokio::process::Command::new(&binary_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--worker-id")
        .arg(&worker_id)
        .arg("--queen-url")
        .arg("http://localhost:8080")
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
    {
        Ok(child) => {
            let worker = WorkerProcess {
                worker_id: worker_id.clone(),
                url: url.clone(),
                process: Arc::new(Mutex::new(Some(child))),
            };
            
            state.workers.lock().await.push(worker);
            
            tracing::info!("✅ Spawned real worker process: {} at {}", worker_id, url);
            
            Json(serde_json::json!({
                "worker_id": worker_id,
                "url": url,
                "state": "loading"
            }))
        }
        Err(e) => {
            tracing::error!("❌ Failed to spawn worker: {}", e);
            Json(serde_json::json!({
                "error": format!("Failed to spawn worker: {}", e)
            }))
        }
    }
}

async fn handle_worker_ready(Json(req): Json<serde_json::Value>) -> Json<serde_json::Value> {
    tracing::info!("Mock rbee-hive: worker ready callback: {:?}", req);
    
    Json(serde_json::json!({
        "success": true
    }))
}

// TEAM-059: Return actual spawned workers
async fn handle_list_workers(
    axum::extract::State(state): axum::extract::State<RbeeHiveState>,
) -> Json<serde_json::Value> {
    let workers = state.workers.lock().await;
    let worker_list: Vec<serde_json::Value> = workers
        .iter()
        .map(|w| serde_json::json!({
            "worker_id": w.worker_id,
            "url": w.url,
            "state": "idle"
        }))
        .collect();
    
    Json(serde_json::json!({
        "workers": worker_list
    }))
}

// TEAM-059: Worker endpoints moved to separate mock-worker binary
