// TEAM-353: Job router for worker - mirrors Hive/Queen pattern
//
// This module routes Operation enums from operations-contract to appropriate handlers.
// Pattern: Same as Hive (bin/20_rbee_hive/src/job_router.rs)

use anyhow::{anyhow, Result};
use job_server::JobRegistry;
use observability_narration_core::sse_sink;
use operations_contract::Operation;
use serde::Serialize;
use std::sync::Arc;

use crate::backend::request_queue::{GenerationRequest, RequestQueue, TokenResponse};
use crate::common::SamplingConfig;

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<TokenResponse>>,
    pub queue: Arc<RequestQueue>,
}

/// Response from job creation
#[derive(Debug, Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Create a new job and store its payload
///
/// TEAM-353: Mirrors Hive pattern
/// Called by HTTP layer to create jobs.
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    // Parse operation from JSON
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow!("Failed to parse operation: {}", e))?;

    // Route to appropriate handler
    match operation {
        Operation::Infer(req) => execute_infer(state, req).await,
        _ => Err(anyhow!("Unsupported operation for worker: {:?}", operation)),
    }
}

/// Execute inference operation
///
/// TEAM-353: Handles Operation::Infer
async fn execute_infer(
    state: JobState,
    req: operations_contract::InferRequest,
) -> Result<JobResponse> {
    // Create job in registry
    let job_id = state.registry.create_job();

    // Create SSE channel for narration (job status updates)
    sse_sink::create_job_channel(job_id.clone(), 1000);

    // Create token channel for inference output
    let (response_tx, response_rx) = tokio::sync::mpsc::unbounded_channel();
    state.registry.set_token_receiver(&job_id, response_rx);

    // Convert to sampling config
    let config = SamplingConfig {
        temperature: req.temperature,
        top_p: req.top_p.unwrap_or(0.9),
        top_k: req.top_k.unwrap_or(50),
        repetition_penalty: 1.1, // Not in InferRequest, use default
        min_p: None,             // Not in InferRequest
        stop_sequences: vec![],
        stop_strings: vec![],    // Not in InferRequest
        seed: 42,                // Not in InferRequest, use default
        max_tokens: req.max_tokens,
    };

    // Create generation request
    let generation_request = GenerationRequest {
        request_id: job_id.clone(),
        prompt: req.prompt,
        config,
        response_tx,
    };

    // Add to queue
    state
        .queue
        .add_request(generation_request)
        .map_err(|e| anyhow!("Failed to queue request: {}", e))?;

    // Return job response
    Ok(JobResponse {
        job_id: job_id.clone(),
        sse_url: format!("/v1/jobs/{}/stream", job_id),
    })
}
