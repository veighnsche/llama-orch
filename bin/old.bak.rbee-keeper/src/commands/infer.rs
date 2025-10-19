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
//! TEAM-086: Added detailed diagnostic output between submission and error messages
//! TEAM-088: Added RBEE_NO_RETRY env var to disable retries for faster dev feedback
//!
//! Created by: TEAM-024
//! Modified by: TEAM-027, TEAM-030, TEAM-048, TEAM-050, TEAM-085, TEAM-086, TEAM-088

use crate::queen_lifecycle::ensure_queen_rbee_running;
use anyhow::Result;
use colored::Colorize;
use futures::StreamExt;
use std::io::Write;
use std::time::Duration;
// TEAM-113: Input validation for CLI arguments
use input_validation::{validate_identifier, validate_model_ref};

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
    // TEAM-113: Validate inputs before sending to queen-rbee
    validate_model_ref(&model)
        .map_err(|e| anyhow::anyhow!("Invalid model reference format: {}", e))?;

    validate_identifier(&node, 64).map_err(|e| anyhow::anyhow!("Invalid node name: {}", e))?;

    // Validate backend if provided
    if let Some(ref backend_name) = backend {
        validate_identifier(backend_name, 64)
            .map_err(|e| anyhow::anyhow!("Invalid backend name: {}", e))?;
    }

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
    // TEAM-086: Added more diagnostic output between submission and error
    // TEAM-088: RBEE_NO_RETRY=1 to disable retries for faster dev feedback
    let max_attempts = if std::env::var("RBEE_NO_RETRY").is_ok() {
        println!("{}", "  🚫 RBEE_NO_RETRY set - will fail fast without retries".yellow());
        1
    } else {
        3
    };

    let mut last_error = None;
    let mut response = None;
    for attempt in 0..max_attempts {
        if attempt > 0 {
            println!(
                "{}",
                format!("  ⏳ Retry attempt {}/{}...", attempt + 1, max_attempts).dimmed()
            );
        }

        println!("{}", format!("  🔌 Connecting to queen-rbee at {}...", queen_url).dimmed());
        println!("{}", format!("  📤 Sending POST request to {}/v2/tasks", queen_url).dimmed());
        println!("{}", format!("  📋 Request payload: node={}, model={}", node, model).dimmed());

        match client
            .post(format!("{}/v2/tasks", queen_url))
            .json(&task_request)
            .timeout(Duration::from_secs(30))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                println!(
                    "{}",
                    format!("  ✅ Request accepted by queen-rbee (HTTP {})", resp.status()).green()
                );
                response = Some(resp);
                break;
            }
            Ok(resp) => {
                // TEAM-086: Non-success HTTP status
                let status = resp.status();
                println!("{}", format!("  ❌ HTTP error: {}", status).red());

                // Try to get error body for debugging
                match resp.text().await {
                    Ok(body) if !body.is_empty() => {
                        println!("{}", format!("  📄 Response body: {}", body).dimmed());
                        last_error = Some(format!("HTTP {} - {}", status, body));
                    }
                    Ok(_) => {
                        last_error = Some(format!("HTTP {} (no body)", status));
                    }
                    Err(e) => {
                        println!(
                            "{}",
                            format!("  ⚠️  Could not read response body: {}", e).dimmed()
                        );
                        last_error = Some(format!("HTTP {}", status));
                    }
                }

                if attempt < max_attempts - 1 {
                    let backoff = 1000 * (attempt + 1) as u64;
                    println!(
                        "{}",
                        format!("  ⏱️  Backing off for {}ms before retry...", backoff).dimmed()
                    );
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                }
            }
            Err(e) => {
                // TEAM-086: Connection/network error
                println!("{}", format!("  ❌ Connection error: {}", e).red());

                // TEAM-086: More specific diagnostics based on error type
                let error_str = e.to_string();
                if error_str.contains("Connection refused") {
                    println!(
                        "{}",
                        "  💡 queen-rbee is not responding on port 8080".to_string().yellow()
                    );
                    println!(
                        "{}",
                        "  💡 Verify: curl http://localhost:8080/health".to_string().dimmed()
                    );
                } else if error_str.contains("timeout") {
                    println!("{}", "  💡 Request timed out after 30 seconds".to_string().yellow());
                    println!("{}", "  💡 queen-rbee may be overloaded or stuck".to_string().dimmed());
                } else if error_str.contains("dns") || error_str.contains("resolve") {
                    println!("{}", "  💡 DNS resolution failed for localhost".to_string().yellow());
                } else {
                    println!("{}", "  💡 Network error occurred".to_string().yellow());
                }

                last_error = Some(e.to_string());

                if attempt < max_attempts - 1 {
                    let backoff = 1000 * (attempt + 1) as u64;
                    println!(
                        "{}",
                        format!("  ⏱️  Backing off for {}ms before retry...", backoff).dimmed()
                    );
                    tokio::time::sleep(Duration::from_millis(backoff)).await;
                }
            }
        }
    }

    let response = match response {
        Some(r) => r,
        None => {
            println!();
            println!("{}", "❌ All retry attempts exhausted".red().bold());
            if let Some(e) = last_error {
                anyhow::bail!("Failed to submit inference task after 3 attempts: {}", e);
            } else {
                anyhow::bail!("Failed to submit inference task after 3 attempts");
            }
        }
    };

    println!("{}", "Tokens:".cyan());

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut done = false; // TEAM-049: Track when [DONE] is received

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
                // TEAM-094: Filter by event type to only show actual tokens
                if let Ok(token_event) = serde_json::from_str::<serde_json::Value>(json_str) {
                    // Check if it's a token event
                    if token_event.get("type").and_then(|t| t.as_str()) == Some("token") {
                        if let Some(token) = token_event.get("t").and_then(|t| t.as_str()) {
                            print!("{}", token);
                            std::io::stdout().flush()?;
                        }
                    }
                    // Also handle other event types silently (narration, started, end)
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
