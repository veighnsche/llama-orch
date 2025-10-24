//! Inference command handler
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-186: Use typed Operation enum instead of JSON strings
//! TEAM-187: Eliminated all clones by moving owned values directly

use anyhow::Result;
use operations_contract::{InferRequest, Operation}; // TEAM-284: Renamed from rbee_operations

use crate::job_client::submit_and_stream_job;

#[allow(clippy::too_many_arguments)]
pub async fn handle_infer(
    hive_id: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<u32>,
    device: Option<String>,
    worker_id: Option<String>,
    stream: bool,
    queen_url: &str,
) -> Result<()> {
    // TEAM-284: Use typed InferRequest
    let operation = Operation::Infer(InferRequest {
        hive_id,
        model,
        prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        device,
        worker_id,
        stream,
    });
    submit_and_stream_job(queen_url, operation).await
}
