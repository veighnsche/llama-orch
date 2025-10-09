//! Inference test command
//!
//! Created by: TEAM-024
//!
//! This command allows testing inference directly on a worker
//! via HTTP POST to /execute endpoint.
//!
//! NOTE: This is for TESTING only. In production (M2+), inference
//! should go through orchestratord, not directly to workers.

use anyhow::Result;
use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

#[derive(Serialize)]
struct InferenceRequest {
    job_id: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Deserialize)]
struct TokenEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    t: String,
    #[serde(default)]
    i: u32,
    #[serde(default)]
    tokens_out: u32,
    #[serde(default)]
    decode_time_ms: u64,
    #[serde(default)]
    stop_reason: String,
}

pub fn handle(worker: String, prompt: String, max_tokens: u32, temperature: f32) -> Result<()> {
    println!("{}", "Testing inference on worker...".cyan().bold());
    println!("Worker: {}", worker);
    println!("Prompt: {}", prompt);
    println!("Max tokens: {}", max_tokens);
    println!("Temperature: {}", temperature);
    println!();

    // Generate job ID
    let job_id = format!("llorch-test-{}", chrono::Utc::now().timestamp());

    // Build request
    let request = InferenceRequest {
        job_id: job_id.clone(),
        prompt: prompt.clone(),
        max_tokens,
        temperature,
    };

    let request_json = serde_json::to_string(&request)?;

    // Build curl command
    let url = format!("http://{}/execute", worker);
    
    println!("{}", "Sending request...".cyan());
    println!();

    // Execute curl with streaming
    let mut child = Command::new("curl")
        .args(&[
            "-N",
            "-X", "POST",
            &url,
            "-H", "Content-Type: application/json",
            "-d", &request_json,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("Failed to capture stdout"))?;
    let reader = BufReader::new(stdout);

    println!("{}", "Response:".green().bold());
    println!("{}", "─".repeat(80));

    let mut token_count = 0;
    let mut started = false;

    for line in reader.lines() {
        let line = line?;
        
        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Parse SSE format (data: {...})
        if let Some(json_str) = line.strip_prefix("data: ") {
            if let Ok(event) = serde_json::from_str::<TokenEvent>(json_str) {
                match event.event_type.as_str() {
                    "started" => {
                        if !started {
                            println!("{} Inference started (job: {})", "✓".green(), job_id);
                            print!("{}", "Tokens: ".cyan());
                            started = true;
                        }
                    }
                    "token" => {
                        print!("{}", event.t);
                        token_count += 1;
                    }
                    "end" => {
                        println!();
                        println!();
                        println!("{}", "─".repeat(80));
                        println!("{} Inference complete!", "✓".green().bold());
                        println!("Tokens generated: {}", event.tokens_out.to_string().cyan());
                        println!("Time: {} ms", event.decode_time_ms.to_string().cyan());
                        println!("Stop reason: {}", event.stop_reason.cyan());
                        
                        if event.decode_time_ms > 0 && event.tokens_out > 0 {
                            let tokens_per_sec = (event.tokens_out as f64 / event.decode_time_ms as f64) * 1000.0;
                            println!("Speed: {:.2} tokens/sec", tokens_per_sec);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    child.wait()?;

    if token_count == 0 {
        println!("{}", "⚠ No tokens received. Check worker status.".yellow());
    }

    Ok(())
}
