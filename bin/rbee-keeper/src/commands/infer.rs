//! Inference command - MVP cross-node inference
//!
//! Per test-001-mvp.md: 8-phase flow (simplified for ephemeral mode)
//! - Phase 1: Worker Registry Check (SKIPPED - ephemeral mode)
//! - Phase 2: Pool Preflight
//! - Phase 3-5: Spawn Worker
//! - Phase 6: Worker Registration (in pool manager)
//! - Phase 7: Worker Health Check
//! - Phase 8: Inference Execution
//!
//! TEAM-030: Removed local worker registry - ephemeral mode doesn't need persistence
//!
//! Created by: TEAM-024
//! Modified by: TEAM-027, TEAM-030

use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;
use serde::Deserialize;

use crate::pool_client::{PoolClient, SpawnWorkerRequest};

/// Handle infer command
///
/// Implements the 8-phase MVP flow from test-001-mvp.md (ephemeral mode)
///
/// TEAM-030: Ephemeral mode lifecycle
/// - Connects directly to rbee-hive (pool manager)
/// - No queen-rbee spawning (for now - M1+ feature)
/// - No local worker registry (pool manager handles it)
/// - Worker cleanup happens via pool manager shutdown
pub async fn handle(
    node: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
) -> Result<()> {
    println!("{}", "=== MVP Cross-Node Inference (Ephemeral Mode) ===".cyan().bold());
    println!("Node: {}", node.cyan());
    println!("Model: {}", model.cyan());
    println!("Prompt: {}", prompt);
    println!();

    // TEAM-030: Phase 1 skipped in ephemeral mode - always spawn fresh worker
    println!("{}", "[Phase 1] Skipped (ephemeral mode - no worker reuse)".yellow());

    // PHASE 2: Pool Preflight
    println!("{}", "[Phase 2] Pool preflight check...".yellow());
    // TEAM-029: Handle localhost specially (don't append .home.arpa)
    let pool_url = if node == "localhost" || node == "127.0.0.1" {
        format!("http://{}:8080", node)
    } else {
        format!("http://{}.home.arpa:8080", node)
    };
    let pool_client = PoolClient::new(pool_url.clone(), "api-key".to_string());

    let health = pool_client.health_check().await?;
    println!("{} Pool health: {} (version {})", "✓".green(), health.status, health.version);
    println!();

    // PHASE 3-5: Spawn Worker
    // TEAM-029: Pool will resolve model_path from catalog
    println!("{}", "[Phase 3-5] Spawning worker...".yellow());
    let spawn_request = SpawnWorkerRequest {
        model_ref: model.clone(),
        backend: "cpu".to_string(), // TODO: Detect backend from node capabilities
        device: 0,
        model_path: String::new(), // TEAM-029: Pool resolves this from catalog
    };

    let worker = pool_client.spawn_worker(spawn_request).await?;
    println!("{} Worker spawned: {} (state: {})", "✓".green(), worker.worker_id, worker.state);
    println!();

    // TEAM-030: Phase 6 - Worker registration happens in pool manager (in-memory)
    println!("{}", "[Phase 6] Worker registered in pool manager".yellow());

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
/// Modified by: TEAM-028, TEAM-029
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
    let mut consecutive_failures = 0;
    const MAX_CONSECUTIVE_FAILURES: u32 = 10; // Fail fast after 10 connection errors

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
                // Reset failure counter on successful connection
                consecutive_failures = 0;

                if let Ok(ready) = response.json::<ReadyResponse>().await {
                    if ready.ready {
                        println!(" {}", "✓".green());
                        return Ok(());
                    }
                    // Worker responded but not ready yet - keep waiting
                }
            }
            Ok(response) => {
                // TEAM-029: Worker responded with error status
                consecutive_failures += 1;
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    println!(" {}", "✗".red());
                    anyhow::bail!(
                        "Worker returned HTTP {} after {} attempts - worker may have failed to start",
                        response.status(),
                        consecutive_failures
                    );
                }
            }
            Err(e) => {
                // TEAM-029: Connection error - worker may not be running
                consecutive_failures += 1;
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    println!(" {}", "✗".red());
                    anyhow::bail!(
                        "Failed to connect to worker after {} attempts: {} - worker binary may not be running",
                        consecutive_failures,
                        e
                    );
                }
            }
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

    // TEAM-035: Match worker SSE event format from src/http/sse.rs
    #[derive(Deserialize)]
    struct TokenEvent {
        #[serde(rename = "type")]
        event_type: String,
        // Token event fields
        #[serde(default)]
        t: String, // token text
        #[serde(default)]
        i: u32, // token index
        // End event fields
        #[serde(default)]
        tokens_out: u32,
        #[serde(default)]
        decode_time_ms: u64,
    }

    let client = reqwest::Client::new();

    let request = serde_json::json!({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": true
    });

    let response =
        client.post(&format!("{}/v1/inference", worker_url)).json(&request).send().await?;

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

                // TEAM-035: Parse worker SSE events
                if let Ok(token_event) = serde_json::from_str::<TokenEvent>(json_str) {
                    match token_event.event_type.as_str() {
                        "token" => {
                            print!("{}", token_event.t);
                            std::io::stdout().flush()?;
                        }
                        "end" => {
                            println!();
                            println!();
                            println!("{} Inference complete!", "✓".green().bold());
                            println!("Total tokens: {}", token_event.tokens_out.to_string().cyan());
                            println!(
                                "Duration: {} ms",
                                token_event.decode_time_ms.to_string().cyan()
                            );

                            if token_event.decode_time_ms > 0 && token_event.tokens_out > 0 {
                                let tokens_per_sec = (token_event.tokens_out as f64
                                    / token_event.decode_time_ms as f64)
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
