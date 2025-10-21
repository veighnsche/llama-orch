//! Job submission client for queen-rbee
//!
//! This is the ONLY way rbee-keeper talks to queen-rbee:
//! 1. Ensure queen is running
//! 2. POST /v1/jobs with operation payload
//! 3. GET /jobs/{job_id}/stream and stream narration to stdout
//! 4. Cleanup queen handle
//!
//! TEAM-185: Updated narration to use operation field instead of embedding in human message
//! TEAM-185: Added hive_id to narration context
//! TEAM-185: Replaced hardcoded action strings with constants from operations module
//! TEAM-186: Accept Operation directly, serialize internally (DRY)

use anyhow::Result;
use futures::StreamExt;
use observability_narration_core::Narration;
use rbee_operations::Operation;

use crate::narration::{ACTOR_RBEE_KEEPER, ACTION_JOB_SUBMIT, ACTION_JOB_STREAM, ACTION_JOB_COMPLETE};
use crate::queen_lifecycle::ensure_queen_running;

/// Submit a job to queen-rbee and stream its narration output.
///
/// TEAM-186: Now accepts Operation directly instead of pre-serialized JSON
///
/// This handles the complete lifecycle:
/// - Serializes operation to JSON (DRY - no more repeated serialization!)
/// - Ensures queen is running
/// - Submits the job
/// - Streams SSE narration events
/// - Cleans up queen handle
pub async fn submit_and_stream_job(
    client: &reqwest::Client,
    queen_url: &str,
    operation: Operation,
) -> Result<()> {
    // TEAM-186: Serialize operation here (DRY - single place!)
    let job_payload = serde_json::to_value(&operation)
        .expect("Failed to serialize operation");
    
    // Ensure queen is running
    let queen_handle = ensure_queen_running(queen_url).await?;
    
    // Submit job to queen
    let res = client.post(format!("{}/v1/jobs", queen_url)).json(&job_payload).send().await?;
    let json: serde_json::Value = res.json().await?;
    
    // Extract job_id and sse_url
    let job_id = json["job_id"].as_str().ok_or_else(|| anyhow::anyhow!("No job_id in response"))?;
    let sse_url = json["sse_url"].as_str().ok_or_else(|| anyhow::anyhow!("No sse_url in response"))?;
    
    // TEAM-186: Use Operation methods instead of JSON parsing
    let operation_name = operation.name();
    let hive_id = operation.hive_id();
    
    let mut narration = Narration::new(ACTOR_RBEE_KEEPER, ACTION_JOB_SUBMIT, job_id)
        .operation(operation_name)
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
        .operation(operation_name)
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
            // TEAM-187: Use strip_prefix() instead of manual slicing (Clippy fix)
            if let Some(data) = line.strip_prefix("data: ") {
                // Just print the data directly without wrapping in narration
                println!("{}", data);
                
                // Check for [DONE] marker
                if data.contains("[DONE]") {
                    let mut narration = Narration::new(ACTOR_RBEE_KEEPER, ACTION_JOB_COMPLETE, job_id)
                        .operation(operation_name)
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
