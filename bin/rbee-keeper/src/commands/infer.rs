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
//! TEAM-048: Refactored to use queen-rbee orchestration endpoint
//! TEAM-050: Fixed stream error handling to prevent exit code 1
//! TEAM-085: CRITICAL FIX - Auto-start queen-rbee if not running (ONE COMMAND INFERENCE)
//!
//! Created by: TEAM-024
//! Modified by: TEAM-027, TEAM-030, TEAM-048, TEAM-050, TEAM-085

use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;
use std::io::Write;
use crate::queen_lifecycle::ensure_queen_rbee_running;

/// Handle infer command
///
/// TEAM-048: Refactored to use queen-rbee's /v2/tasks endpoint
/// TEAM-055: Added backend and device parameters per test-001 spec
/// - All orchestration logic moved to queen-rbee
/// - rbee-keeper is now a thin client
/// - Simplifies CLI and centralizes orchestration
pub async fn handle(
    node: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    backend: Option<String>,
    device: Option<u32>,
) -> Result<()> {
    println!("{}", "=== Inference via queen-rbee Orchestration ===".cyan().bold());
    println!("Node: {}", node.cyan());
    println!("Model: {}", model.cyan());
    println!("Prompt: {}", prompt);
    println!();

    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8080";
    // TEAM-085: CRITICAL FIX - Ensure queen-rbee is running
    ensure_queen_rbee_running(&client, queen_url).await?;

    println!("{}", "[queen-rbee] Submitting inference task...".yellow());

    // Submit inference task to queen-rbee
    let task_request = serde_json::json!({
        "node": node,
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "backend": backend,
        "device": device,
    });

    // TEAM-055: Retry logic with exponential backoff
    // TEAM-085: Add narration so user sees what's happening during retries
    let mut last_error = None;
    let mut response = None;
    for attempt in 0..3 {
        if attempt > 0 {
            println!("{}", format!("  ⏳ Retry attempt {}/3...", attempt + 1).dimmed());
        }
        
        println!("{}", format!("  🔌 Connecting to queen-rbee at {}...", queen_url).dimmed());
        
        match client
            .post(format!("{}/v2/tasks", queen_url))
            .json(&task_request)
            .timeout(Duration::from_secs(30))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                response = Some(resp);
                break;
            }
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("Inference request failed: HTTP {} - {}", status, body);
            }
            Err(e) if attempt < 2 => {
                tracing::warn!("⚠️ Attempt {} failed: {}, retrying...", attempt + 1, e);
                let error_msg = format!("HTTP {}", response.status());
                println!("{}", format!("  ❌ Error: {}", error_msg).red());
                
                // Try to get error body for debugging
                if let Ok(body) = response.text().await {
                    if !body.is_empty() {
                        println!("{}", format!("  📄 Response body: {}", body).dimmed());
                    }
                }
                
                last_error = Some(error_msg);
                
                if attempt < 2 {
                    let backoff = 1000 * (attempt + 1) as u64;
                    println!("{}", format!("  ⏱️  Backing off for {}ms...", backoff).dimmed());
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                }
            }
            Err(e) => {
                println!("{}", format!("  ❌ Connection error: {}", e).red());
                println!("{}", format!("  💡 Is queen-rbee actually running? Check: curl http://localhost:8080/health").dimmed());
                
                last_error = Some(e.to_string());
                
                if attempt < 2 {
                    let backoff = 1000 * (attempt + 1) as u64;
                    println!("{}", format!("  ⏱️  Backing off for {}ms...", backoff).dimmed());
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                }
            }

    let response = match response {
        Some(r) => r,
        None => {
            if let Some(e) = last_error {
                anyhow::bail!("Failed to submit inference task after 3 attempts: {}", e);
            } else {
                anyhow::bail!("Failed to submit inference task");
            }
        }
    };

    println!("{}", "Tokens:".cyan());

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut done = false;  // TEAM-049: Track when [DONE] is received

    while let Some(chunk) = stream.next().await {
        if done {
            // TEAM-049: Stop processing after [DONE] to avoid race condition
            break;
        }

        // TEAM-050: Handle stream errors gracefully - don't propagate them as function errors
        // Stream errors (including normal closure) should not cause exit code 1
        let chunk = match chunk {
            Ok(bytes) => bytes,
            Err(_e) => {
                // If we've already seen [DONE], stream closure is expected
                if done {
                    break;
                }
                // Otherwise continue - the stream might recover or be done
                // (Stream errors after [DONE] are normal and should not cause exit code 1)
                continue;
            }
        };
        
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        // Process complete SSE events
        while let Some(pos) = buffer.find("\n\n") {
            let event = buffer[..pos].to_string();
            buffer = buffer[pos + 2..].to_string();

            // Parse SSE format: "data: {...}"
            if let Some(json_str) = event.strip_prefix("data: ") {
                if json_str == "[DONE]" {
                    // TEAM-049: Set flag and break from event loop
                    // The outer loop will exit on next iteration
                    done = true;
                    break;
                }

                // Parse and display token events
                if let Ok(token_event) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if let Some(token) = token_event.get("t").and_then(|t| t.as_str()) {
                        print!("{}", token);
                        std::io::stdout().flush()?;
                    }
                }
            }
        }
    }

    println!("\n");
    Ok(())
}

// TEAM-085: Moved to shared module commands/queen_lifecycle.rs
// All commands that talk to queen-rbee use the same lifecycle management

// TEAM-048: Removed wait_for_worker_ready and execute_inference - now handled by queen-rbee
