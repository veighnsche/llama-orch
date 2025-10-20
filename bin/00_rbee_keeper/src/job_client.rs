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

use crate::actions::{ACTOR_RBEE_KEEPER, ACTION_STREAM};

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
    
    Narration::new(ACTOR_RBEE_KEEPER, "job_submitted", job_id)
        .human(format!("📋 Job {} submitted", job_id))
        .emit();
    
    // Stream narration from job's SSE endpoint
    let sse_full_url = format!("{}{}", queen_url, sse_url);
    let response = client.get(&sse_full_url).send().await?;
    
    if !response.status().is_success() {
        let error = response.text().await?;
        std::mem::forget(queen_handle);
        anyhow::bail!("Failed to connect to SSE stream: {}", error);
    }
    
    Narration::new(ACTOR_RBEE_KEEPER, ACTION_STREAM, &sse_full_url)
        .human("📡 Streaming events from queen-rbee...")
        .emit();
    
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        
        // Print each SSE event to stdout
        for line in text.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..]; // Remove "data: " prefix
                Narration::new(ACTOR_RBEE_KEEPER, ACTION_STREAM, "token").human(data).emit();
                
                // Check for [DONE] marker
                if data.contains("[DONE]") {
                    Narration::new(ACTOR_RBEE_KEEPER, ACTION_STREAM, "complete")
                        .human("✅ Stream complete")
                        .emit();
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
