//! Job submission client for queen-rbee
//!
//! This is the ONLY way rbee-keeper talks to queen-rbee:
//! 1. Ensure queen is running
//! 2. POST /v1/jobs with operation payload
//! 3. GET /jobs/{job_id}/stream and stream narration to stdout
//! 4. Cleanup queen handle

use anyhow::Result;
use futures::StreamExt;
use observability_narration_core::Narration;

use crate::operations::{ACTOR_RBEE_KEEPER, ACTION_JOB_SUBMIT, ACTION_JOB_STREAM, ACTION_JOB_COMPLETE};

/// Submit a job to queen-rbee and stream its narration output.
///
/// This handles the complete lifecycle:
/// - Ensures queen is running
/// - Submits the job
/// - Streams SSE narration events
/// - Cleans up queen handle
pub async fn submit_and_stream_job(
    client: &reqwest::Client,
    queen_url: &str,
    job_payload: serde_json::Value,
) -> Result<()> {
    // Ensure queen is running
    let queen_handle = rbee_keeper_queen_lifecycle::ensure_queen_running(queen_url).await?;
    
    // Submit job to queen
    let res = client.post(format!("{}/v1/jobs", queen_url)).json(&job_payload).send().await?;
    let json: serde_json::Value = res.json().await?;
    
    // Extract job_id and sse_url
    let job_id = json["job_id"].as_str().ok_or_else(|| anyhow::anyhow!("No job_id in response"))?;
    let sse_url = json["sse_url"].as_str().ok_or_else(|| anyhow::anyhow!("No sse_url in response"))?;
    
    // Extract operation name for narration
    let operation = job_payload["operation"].as_str().unwrap_or("unknown");
    let hive_id = job_payload["hive_id"].as_str();
    
    let mut narration = Narration::new(ACTOR_RBEE_KEEPER, ACTION_JOB_SUBMIT, job_id)
        .operation(operation)
        .human(format!("📋 Job {} submitted", job_id));
    
    if let Some(hid) = hive_id {
        narration = narration.hive_id(hid);
    }
    narration.emit();
    
    // Stream narration from job's SSE endpoint
    let sse_full_url = format!("{}{}", queen_url, sse_url);
    let response = client.get(&sse_full_url).send().await?;
    
    if !response.status().is_success() {
        let error = response.text().await?;
        std::mem::forget(queen_handle);
        anyhow::bail!("Failed to connect to SSE stream: {}", error);
    }
    
    let mut narration = Narration::new(ACTOR_RBEE_KEEPER, ACTION_JOB_STREAM, job_id)
        .operation(operation)
        .human("📡 Streaming results...");
    
    if let Some(hid) = hive_id {
        narration = narration.hive_id(hid);
    }
    narration.emit();
    
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        
        // Print each SSE event to stdout
        for line in text.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..]; // Remove "data: " prefix
                // Just print the data directly without wrapping in narration
                println!("{}", data);
                
                // Check for [DONE] marker
                if data.contains("[DONE]") {
                    let mut narration = Narration::new(ACTOR_RBEE_KEEPER, ACTION_JOB_COMPLETE, job_id)
                        .operation(operation)
                        .human("✅ Complete");
                    
                    if let Some(hid) = hive_id {
                        narration = narration.hive_id(hid);
                    }
                    narration.emit();
                    std::mem::forget(queen_handle);
                    return Ok(());
                }
            }
        }
    }
    
    // Cleanup
    std::mem::forget(queen_handle);
    Ok(())
}
