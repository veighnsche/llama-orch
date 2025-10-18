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
//! TEAM-087: Fixed model_ref validation bug (HTTP 400 from rbee-hive)

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
// TEAM-113: Input validation for inference requests
use input_validation::{validate_model_ref, validate_identifier};

/// Handle POST /v2/tasks
///
/// Full orchestration: registry lookup -> SSH -> spawn worker -> execute inference
/// TEAM-087: Added model_ref validation
pub async fn handle_create_inference_task(
    State(state): State<AppState>,
    Json(req): Json<InferenceTaskRequest>,
) -> impl IntoResponse {
    info!("Received inference task: node={}, model={}", req.node, req.model);
    
    // TEAM-113: Validate inputs before processing
    if let Err(e) = validate_identifier(&req.node, 64) {
        error!("Invalid node name: {}", e);
        return (StatusCode::BAD_REQUEST, format!("Invalid node name: {}", e)).into_response();
    }
    
    // TEAM-087: Validate and normalize model reference
    // rbee-hive requires "provider:reference" format (e.g., "hf:model-name")
    let model_ref = if req.model.contains(':') {
        req.model.clone()
    } else {
        // Default to "hf:" prefix for convenience
        format!("hf:{}", req.model)
    };
    
    // TEAM-113: Validate model_ref format
    if let Err(e) = validate_model_ref(&model_ref) {
        error!("Invalid model reference: {}", e);
        return (StatusCode::BAD_REQUEST, format!("Invalid model_ref: {}", e)).into_response();
    }
    
    info!("Using model_ref: {}", model_ref);
    
    // TEAM-085: Handle localhost specially - no SSH needed!
    let rbee_hive_url = if req.node == "localhost" {
        info!("🏠 Localhost inference - starting rbee-hive locally");
        match ensure_local_rbee_hive_running().await {
            Ok(url) => url,
            Err(e) => {
                error!("Failed to start local rbee-hive: {}", e);
                return (StatusCode::SERVICE_UNAVAILABLE, format!("Failed to start rbee-hive: {}", e)).into_response();
            }
        }
    } else {
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
        if mock_ssh {
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
        }
    };
    
    // Step 3: Spawn worker on rbee-hive
    // TEAM-087: Enhanced spawn diagnostics
    info!("🚀 Spawning worker on rbee-hive at {}", rbee_hive_url);
    info!("   Model: {}", model_ref);
    info!("   Backend: cpu");
    
    let client = reqwest::Client::new();
    // TEAM-087: Use normalized model_ref
    let spawn_request = serde_json::json!({
        "model_ref": model_ref,
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
                    info!("✅ Worker spawned successfully:");
                    info!("   Worker ID: {}", worker.worker_id);
                    info!("   URL: {}", worker.url);
                    info!("   State: {}", worker.state);
                    worker
                }
                Err(e) => {
                    error!("❌ Failed to parse worker spawn response: {}", e);
                    return (StatusCode::INTERNAL_SERVER_ERROR, 
                            format!("Worker spawn parse error: {}. rbee-hive may have returned invalid JSON.", e)
                    ).into_response();
                }
            }
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!("❌ Worker spawn failed: HTTP {} - {}", status, body);
            return (StatusCode::INTERNAL_SERVER_ERROR, 
                    format!("Worker spawn failed: HTTP {} - {}. Check rbee-hive logs for details.", status, body)
            ).into_response();
        }
        Err(e) => {
            error!("❌ Failed to connect to rbee-hive: {}", e);
            return (StatusCode::SERVICE_UNAVAILABLE, 
                    format!("Worker spawn request failed: {}. Is rbee-hive running at {}?", e, rbee_hive_url)
            ).into_response();
        }
    };
    
    // Step 4: Wait for worker ready
    // TEAM-087: Enhanced timeout diagnostics
    info!("⏳ Waiting for worker {} to be ready at {}", worker.worker_id, worker.url);
    match wait_for_worker_ready(&worker.url).await {
        Ok(_) => {
            info!("✅ Worker {} is ready and accepting requests", worker.worker_id);
        }
        Err(e) => {
            error!("❌ Worker {} failed to become ready: {}", worker.worker_id, e);
            // TEAM-087: Return detailed error to help diagnose issues
            return (StatusCode::INTERNAL_SERVER_ERROR, 
                    format!("Worker ready timeout: {}. Check worker logs for details.", e)
            ).into_response();
        }
    }
    
    // Step 5: Execute inference and stream results
    info!("Executing inference on worker {}", worker.worker_id);
    // TEAM-093: Add required job_id field for worker's ExecuteRequest
    let job_id = format!("job-{}", uuid::Uuid::new_v4());
    let inference_request = serde_json::json!({
        "job_id": job_id,
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

/// TEAM-085: Start rbee-hive locally (no SSH!)
async fn ensure_local_rbee_hive_running() -> anyhow::Result<String> {
    // TEAM-085: Use port 9200 for rbee-hive (queen-rbee uses 8080)
    let rbee_hive_url = "http://127.0.0.1:9200";
    let client = reqwest::Client::new();
    
    // Check if rbee-hive is already running
    match client
        .get(format!("{}/v1/health", rbee_hive_url))
        .timeout(std::time::Duration::from_millis(500))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            info!("✓ rbee-hive already running locally");
            return Ok(rbee_hive_url.to_string());
        }
        _ => {
            info!("⚠️  rbee-hive not running, starting locally...");
        }
    }
    
    // Find rbee-hive binary
    let rbee_hive_binary = std::env::current_exe()?
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot find binary directory"))?
        .join("rbee-hive");
    
    if !rbee_hive_binary.exists() {
        anyhow::bail!(
            "rbee-hive binary not found at {:?}. Run: cargo build --bin rbee-hive",
            rbee_hive_binary
        );
    }
    
    // Start rbee-hive as local background process
    info!("🚀 Starting rbee-hive daemon locally...");
    
    // TEAM-088: CRITICAL FIX - Don't silence logs! We need to see what's happening!
    // Use RBEE_SILENT=1 to suppress logs if needed
    let (stdout_cfg, stderr_cfg) = if std::env::var("RBEE_SILENT").is_ok() {
        (std::process::Stdio::null(), std::process::Stdio::null())
    } else {
        (std::process::Stdio::inherit(), std::process::Stdio::inherit())
    };
    
    let mut child = tokio::process::Command::new(&rbee_hive_binary)
        .arg("daemon")
        .arg("--addr")
        .arg("127.0.0.1:9200")  // TEAM-085: Different port from queen-rbee (8080)
        .env("RBEE_WORKER_HOST", "127.0.0.1")
        .stdout(stdout_cfg)
        .stderr(stderr_cfg)
        .spawn()?;
    
    // Wait for rbee-hive to be ready (max 10 seconds)
    for attempt in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        match client
            .get(format!("{}/v1/health", rbee_hive_url))
            .timeout(std::time::Duration::from_millis(500))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                info!("✓ rbee-hive started successfully");
                
                // Detach the child process so it keeps running
                let _ = child.id();
                std::mem::forget(child);
                
                return Ok(rbee_hive_url.to_string());
            }
            _ => {
                // Check if process died
                if let Ok(Some(status)) = child.try_wait() {
                    anyhow::bail!("rbee-hive exited with status: {}", status);
                }
            }
        }
        
        if attempt % 10 == 0 && attempt > 0 {
            info!("  Waiting for rbee-hive... ({}/10s)", attempt / 10);
        }
    }
    
    anyhow::bail!("rbee-hive failed to start within 10 seconds")
}

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
/// TEAM-087: Enhanced timeout diagnostics
async fn wait_for_worker_ready(worker_url: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(300); // 5 minutes
    let mut attempt = 0;
    #[allow(unused_assignments)]
    let mut last_error: Option<String> = None;
    
    loop {
        attempt += 1;
        let elapsed = start.elapsed();
        
        // Log progress every 10 seconds
        if attempt % 5 == 0 {
            info!("Waiting for worker ready... attempt {} ({:.1}s elapsed)", attempt, elapsed.as_secs_f32());
        }
        
        match client
            .get(format!("{}/v1/ready", worker_url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                match response.json::<ReadyResponse>().await {
                    Ok(ready) => {
                        if ready.ready {
                            info!("✅ Worker ready at {} (took {:.1}s, {} attempts)", 
                                  worker_url, elapsed.as_secs_f32(), attempt);
                            return Ok(());
                        } else {
                            last_error = Some(format!("Worker not ready yet (state: {})", ready.state));
                        }
                    }
                    Err(e) => {
                        last_error = Some(format!("Failed to parse ready response: {}", e));
                        error!("Worker ready check parse error: {}", e);
                    }
                }
            }
            Ok(response) => {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                last_error = Some(format!("HTTP {} - {}", status, body));
                error!("Worker ready check failed: HTTP {} - {}", status, body);
            }
            Err(e) => {
                last_error = Some(format!("Connection error: {}", e));
                if attempt <= 3 || attempt % 10 == 0 {
                    error!("Worker connection error (attempt {}): {}", attempt, e);
                }
            }
        }
        
        if start.elapsed() > timeout {
            let diagnostic = format!(
                "Worker ready timeout after {:.1}s ({} attempts). Last error: {}. \
                 Worker URL: {}. Possible causes: (1) Model download failed, \
                 (2) Worker crashed during startup, (3) Worker binary missing dependencies, \
                 (4) Callback URL unreachable",
                elapsed.as_secs_f32(),
                attempt,
                last_error.unwrap_or_else(|| "No response".to_string()),
                worker_url
            );
            error!("{}", diagnostic);
            anyhow::bail!(diagnostic);
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
