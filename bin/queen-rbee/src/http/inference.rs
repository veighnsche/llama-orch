//! Inference task orchestration endpoint
//!
//! Endpoint:
//! - POST /v2/tasks - Create and execute an inference task
//!
//! This module handles the full orchestration flow:
//! 1. Query beehive registry for node details
//! 2. Establish SSH connection to rbee-hive
//! 3. Spawn worker on remote node
//! 4. Wait for worker to be ready
//! 5. Execute inference and stream results
//!
//! Created by: TEAM-046, TEAM-047
//! Refactored by: TEAM-052

use axum::{
    body::Body,
    extract::{Json, State},
    http::{header, StatusCode},
    response::IntoResponse,
};
use tracing::{error, info};

use crate::beehive_registry::BeehiveNode;
use crate::http::routes::AppState;
use crate::http::types::{InferenceRequest, InferenceTaskRequest, ReadyResponse, WorkerSpawnResponse};

/// Handle POST /v2/tasks
///
/// Full orchestration: registry lookup -> SSH -> spawn worker -> execute inference
pub async fn handle_create_inference_task(
    State(state): State<AppState>,
    Json(req): Json<InferenceTaskRequest>,
) -> impl IntoResponse {
    info!("Received inference task: node={}, model={}", req.node, req.model);
    
    // Step 1: Query rbee-hive registry for node SSH details
    let node = match state.beehive_registry.get_node(&req.node).await {
        Ok(Some(node)) => node,
        Ok(None) => {
            error!("Node not found in registry: {}", req.node);
            return (StatusCode::NOT_FOUND, format!("Node '{}' not registered", req.node)).into_response();
        }
        Err(e) => {
            error!("Failed to query registry: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Registry error: {}", e)).into_response();
        }
    };
    
    // Step 2: Determine rbee-hive URL (mock SSH for tests)
    // TEAM-053: Fixed port - rbee-hive uses 9200, not 8080 (queen-rbee's port)
    // Architecture: queen-rbee (8080) → rbee-hive (9200) → workers (8001+)
    let mock_ssh = std::env::var("MOCK_SSH").is_ok();
    let rbee_hive_url = if mock_ssh {
        // For tests, assume rbee-hive is already running on localhost:9200
        info!("🔌 Mock SSH: Using localhost rbee-hive at port 9200");
        "http://127.0.0.1:9200".to_string()
    } else {
        // Real SSH: start rbee-hive daemon on remote node
        info!("🔌 Establishing SSH connection to {}", node.ssh_host);
        match establish_rbee_hive_connection(&node).await {
            Ok(url) => url,
            Err(e) => {
                error!("Failed to connect to rbee-hive: {}", e);
                return (StatusCode::SERVICE_UNAVAILABLE, format!("SSH connection failed: {}", e)).into_response();
            }
        }
    };
    
    // Step 3: Spawn worker on rbee-hive
    info!("Spawning worker on rbee-hive at {}", rbee_hive_url);
    let client = reqwest::Client::new();
    let spawn_request = serde_json::json!({
        "model_ref": req.model,
        "backend": "cpu",
        "device": 0,
        "model_path": ""
    });
    
    let worker = match client
        .post(format!("{}/v1/workers/spawn", rbee_hive_url))
        .json(&spawn_request)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<WorkerSpawnResponse>().await {
                Ok(worker) => {
                    info!("Worker spawned: {} at {}", worker.worker_id, worker.url);
                    worker
                }
                Err(e) => {
                    error!("Failed to parse worker response: {}", e);
                    return (StatusCode::INTERNAL_SERVER_ERROR, format!("Worker spawn parse error: {}", e)).into_response();
                }
            }
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!("Worker spawn failed: HTTP {} - {}", status, body);
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Worker spawn failed: HTTP {}", status)).into_response();
        }
        Err(e) => {
            error!("Failed to spawn worker: {}", e);
            return (StatusCode::SERVICE_UNAVAILABLE, format!("Worker spawn request failed: {}", e)).into_response();
        }
    };
    
    // Step 4: Wait for worker ready
    info!("Waiting for worker {} to be ready", worker.worker_id);
    match wait_for_worker_ready(&worker.url).await {
        Ok(_) => info!("Worker ready: {}", worker.worker_id),
        Err(e) => {
            error!("Worker failed to become ready: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Worker ready timeout: {}", e)).into_response();
        }
    }
    
    // Step 5: Execute inference and stream results
    info!("Executing inference on worker {}", worker.worker_id);
    let inference_request = serde_json::json!({
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": true
    });
    
    let response = match client
        .post(format!("{}/v1/inference", worker.url))
        .json(&inference_request)
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp,
        Ok(resp) => {
            let status = resp.status();
            error!("Inference failed: HTTP {}", status);
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Inference failed: HTTP {}", status)).into_response();
        }
        Err(e) => {
            error!("Failed to execute inference: {}", e);
            return (StatusCode::SERVICE_UNAVAILABLE, format!("Inference request failed: {}", e)).into_response();
        }
    };
    
    // TEAM-048: Stream SSE response back to client (pass-through, don't re-wrap)
    // The worker already sends properly formatted SSE events, just proxy them
    let stream = response.bytes_stream();
    
    (
        [(header::CONTENT_TYPE, "text/event-stream")],
        Body::from_stream(stream)
    ).into_response()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Helper Functions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// TEAM-047: Establish rbee-hive connection via SSH
async fn establish_rbee_hive_connection(node: &BeehiveNode) -> anyhow::Result<String> {
    // Execute remote command to start rbee-hive daemon
    let start_command = format!(
        "{}/rbee-hive daemon --addr 0.0.0.0:8080 > /tmp/rbee-hive.log 2>&1 &",
        node.install_path
    );
    
    let (success, stdout, stderr) = crate::ssh::execute_remote_command(
        &node.ssh_host,
        node.ssh_port,
        &node.ssh_user,
        node.ssh_key_path.as_deref(),
        &start_command,
    )
    .await?;
    
    if !success {
        anyhow::bail!("Failed to start rbee-hive: {}", stderr);
    }
    
    info!("rbee-hive daemon started on {}: {}", node.ssh_host, stdout);
    
    // Wait for rbee-hive to be ready
    let rbee_hive_url = format!("http://{}:8080", node.ssh_host);
    wait_for_rbee_hive_ready(&rbee_hive_url).await?;
    
    Ok(rbee_hive_url)
}

/// TEAM-047: Wait for rbee-hive to be ready
/// TEAM-048: Enhanced with exponential backoff retry (EC1)
async fn wait_for_rbee_hive_ready(url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let mut backoff_ms = 100;
    let max_retries = 5;
    let timeout = std::time::Duration::from_secs(60);
    let start = std::time::Instant::now();
    
    for attempt in 0..max_retries {
        if start.elapsed() > timeout {
            anyhow::bail!("rbee-hive ready timeout after 60 seconds");
        }
        
        match client
            .get(format!("{}/health", url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                info!("rbee-hive is ready at {} (attempt {})", url, attempt + 1);
                return Ok(());
            }
            Ok(resp) => {
                info!("rbee-hive returned HTTP {}, retrying...", resp.status());
            }
            Err(e) if attempt < max_retries - 1 => {
                info!("Connection attempt {} failed: {}, retrying in {}ms", 
                      attempt + 1, e, backoff_ms);
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms *= 2; // Exponential backoff
            }
            Err(e) => {
                anyhow::bail!("Failed to connect to rbee-hive after {} attempts: {}", 
                             max_retries, e);
            }
        }
    }
    
    anyhow::bail!("rbee-hive ready timeout after {} retries", max_retries)
}

/// TEAM-047: Wait for worker to be ready
async fn wait_for_worker_ready(worker_url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(300); // 5 minutes
    
    loop {
        match client
            .get(format!("{}/v1/ready", worker_url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                if let Ok(ready) = response.json::<ReadyResponse>().await {
                    if ready.ready {
                        info!("Worker ready at {}", worker_url);
                        return Ok(());
                    }
                }
            }
            _ => {}
        }
        
        if start.elapsed() > timeout {
            anyhow::bail!("Worker ready timeout after 5 minutes");
        }
        
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
}

/// Handle POST /v1/inference
///
/// Simple inference endpoint that routes to an available worker
/// Created by: TEAM-084
pub async fn handle_inference_request(
    State(state): State<AppState>,
    Json(req): Json<InferenceRequest>,
) -> impl IntoResponse {
    info!("Received inference request: model={}", req.model.as_deref().unwrap_or("default"));
    
    // Find an idle worker
    let workers = state.worker_registry.list().await;
    let idle_worker = workers.iter().find(|w| w.state == crate::worker_registry::WorkerState::Idle);
    
    if let Some(worker) = idle_worker {
        info!("Routing request to worker: {} at {}", worker.id, worker.url);
        
        // Forward request to worker
        let client = reqwest::Client::new();
        match client
            .post(format!("{}/v1/inference", worker.url))
            .json(&req)
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await
        {
            Ok(response) => {
                info!("Worker responded with status: {}", response.status());
                
                // Stream the response back
                let status = response.status();
                let body = Body::from_stream(response.bytes_stream());
                
                (status, [(header::CONTENT_TYPE, "text/event-stream")], body).into_response()
            }
            Err(e) => {
                error!("Failed to forward request to worker: {}", e);
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Worker communication failed: {}", e),
                ).into_response()
            }
        }
    } else {
        error!("No idle workers available");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "No idle workers available",
        ).into_response()
    }
}
