//! Inference command - MVP cross-node inference
//!
//! Per test-001-mvp.md: 8-phase flow
//! - Phase 1: Worker Registry Check
//! - Phase 2: Pool Preflight
//! - Phase 3-5: Spawn Worker
//! - Phase 6: Worker Registration
//! - Phase 7: Worker Health Check
//! - Phase 8: Inference Execution
//!
//! Created by: TEAM-024
//! Modified by: TEAM-027

use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;
use serde::Deserialize;
use worker_registry::{WorkerInfo, WorkerRegistry};

use crate::pool_client::{PoolClient, SpawnWorkerRequest};

/// Handle infer command
///
/// Implements the 8-phase MVP flow from test-001-mvp.md
pub async fn handle(
    node: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> Result<()> {
    println!("{}", "=== MVP Cross-Node Inference ===".cyan().bold());
    println!("Node: {}", node.cyan());
    println!("Model: {}", model.cyan());
    println!("Prompt: {}", prompt);
    println!();

    // PHASE 1: Worker Registry Check
    println!("{}", "[Phase 1] Checking local worker registry...".yellow());
    let registry = WorkerRegistry::new(
        dirs::home_dir()
            .unwrap_or_default()
            .join(".rbee/workers.db")
            .to_string_lossy()
            .to_string(),
    );
    registry.init().await?;

    if let Some(worker) = registry.find_worker(&node, &model).await? {
        println!("{} Found existing worker: {}", "✓".green(), worker.url);
        println!();
        println!("{}", "[Phase 8] Executing inference...".yellow());
        execute_inference(&worker.url, prompt, max_tokens, temperature).await?;
        return Ok(());
    }
    println!("{} No existing worker found", "✗".red());
    println!();

    // PHASE 2: Pool Preflight
    println!("{}", "[Phase 2] Pool preflight check...".yellow());
    let pool_url = format!("http://{}.home.arpa:8080", node);
    let pool_client = PoolClient::new(pool_url.clone(), "api-key".to_string());

    let health = pool_client.health_check().await?;
    println!("{} Pool health: {} (version {})", "✓".green(), health.status, health.version);
    println!();

    // PHASE 3-5: Spawn Worker
    println!("{}", "[Phase 3-5] Spawning worker...".yellow());
    let spawn_request = SpawnWorkerRequest {
        model_ref: model.clone(),
        backend: "cpu".to_string(), // TODO: Detect backend from node capabilities
        device: 0,
        model_path: "/models/model.gguf".to_string(), // TODO: Get from catalog
    };

    let worker = pool_client.spawn_worker(spawn_request).await?;
    println!("{} Worker spawned: {} (state: {})", "✓".green(), worker.worker_id, worker.state);
    println!();

    // PHASE 6: Worker Registration
    println!("{}", "[Phase 6] Registering worker...".yellow());
    let worker_info = WorkerInfo {
        id: worker.worker_id.clone(),
        node: node.clone(),
        url: worker.url.clone(),
        model_ref: model,
        state: worker.state,
        last_health_check_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() as i64,
    };
    registry.register_worker(&worker_info).await?;
    println!("{} Worker registered in local registry", "✓".green());
    println!();

    // PHASE 7: Worker Health Check
    println!("{}", "[Phase 7] Waiting for worker ready...".yellow());
    wait_for_worker_ready(&worker.url).await?;
    println!("{} Worker ready!", "✓".green());
    println!();

    // PHASE 8: Inference Execution
    println!("{}", "[Phase 8] Executing inference...".yellow());
    execute_inference(&worker.url, prompt, max_tokens, temperature).await?;

    Ok(())
}

/// Wait for worker to be ready
///
/// Per test-001-mvp.md Phase 7: Worker Health Check
/// Polls GET /v1/ready until ready=true or timeout (5 minutes)
///
/// Modified by: TEAM-028
async fn wait_for_worker_ready(worker_url: &str) -> Result<()> {
    use std::io::Write;

    #[derive(Deserialize)]
    struct ReadyResponse {
        ready: bool,
        #[allow(dead_code)]
        state: String,
    }

    let client = reqwest::Client::new();
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(300); // 5 minutes

    print!("Waiting for worker ready");
    std::io::stdout().flush()?;

    loop {
        match client
            .get(&format!("{}/v1/ready", worker_url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => {
                if let Ok(ready) = response.json::<ReadyResponse>().await {
                    if ready.ready {
                        println!(" {}", "✓".green());
                        return Ok(());
                    }
                }
            }
            _ => {}
        }

        if start.elapsed() > timeout {
            println!(" {}", "✗".red());
            anyhow::bail!("Worker ready timeout after 5 minutes");
        }

        print!(".");
        std::io::stdout().flush()?;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
}

/// Execute inference with SSE streaming
///
/// Per test-001-mvp.md Phase 8: Inference Execution
/// Sends POST /v1/inference with stream=true and processes SSE events
///
/// Modified by: TEAM-028
async fn execute_inference(
    worker_url: &str,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> Result<()> {
    use std::io::Write;

    #[derive(Deserialize)]
    struct TokenEvent {
        #[serde(rename = "type")]
        event_type: String,
        #[serde(default)]
        token: String,
        #[serde(default)]
        done: bool,
        #[serde(default)]
        total_tokens: u32,
        #[serde(default)]
        duration_ms: u64,
    }

    let client = reqwest::Client::new();

    let request = serde_json::json!({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": true
    });

    let response = client
        .post(&format!("{}/v1/inference", worker_url))
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        anyhow::bail!("Inference request failed: HTTP {}", response.status());
    }

    println!("{}", "Tokens:".cyan());

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE events
        while let Some(pos) = buffer.find("\n\n") {
            let event = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();

            // Parse SSE format: "data: {...}"
            if let Some(json_str) = event.strip_prefix("data: ") {
                if json_str == "[DONE]" {
                    break;
                }

                if let Ok(token_event) = serde_json::from_str::<TokenEvent>(json_str) {
                    match token_event.event_type.as_str() {
                        "token" => {
                            print!("{}", token_event.token);
                            std::io::stdout().flush()?;
                        }
                        "end" => {
                            println!();
                            println!();
                            println!("{} Inference complete!", "✓".green().bold());
                            println!("Total tokens: {}", token_event.total_tokens.to_string().cyan());
                            println!("Duration: {} ms", token_event.duration_ms.to_string().cyan());

                            if token_event.duration_ms > 0 && token_event.total_tokens > 0 {
                                let tokens_per_sec = (token_event.total_tokens as f64
                                    / token_event.duration_ms as f64)
                                    * 1000.0;
                                println!("Speed: {:.2} tokens/sec", tokens_per_sec);
                            }
                            return Ok(());
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    Ok(())
}
