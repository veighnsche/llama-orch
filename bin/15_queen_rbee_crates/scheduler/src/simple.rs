// TEAM-275: Simple scheduler implementation (first available worker)
// TEAM-380: Migrated to n!() macro (manual context for trait methods)
use crate::types::{
    JobRequest, ScheduleResult, SchedulerError, WorkerInferenceRequest, WorkerJobResponse,
};
use crate::JobScheduler;
use observability_narration_core::{n, with_narration_context, NarrationContext}; // TEAM-380: Migrated to n!() macro
use queen_rbee_telemetry_registry::TelemetryRegistry; // TEAM-374
use std::sync::Arc;

/// Simple scheduler: Pick first available worker for model
///
/// # Algorithm
///
/// 1. Find workers serving requested model
/// 2. Filter by online (recent heartbeat) + available (Ready status)
/// 3. Return **first match** (no load balancing)
///
/// # Future Improvements (M2+)
///
/// This will be replaced/supplemented by RhaiScheduler which supports:
/// - Custom routing logic via Rhai scripts
/// - Load balancing (round-robin, least-loaded)
/// - Cost optimization
/// - Latency optimization
/// - Multi-modal routing
/// - Geo-filtering
/// - Custom policies
///
/// See: `.business/stakeholders/RHAI_PROGRAMMABLE_SCHEDULER.md`
pub struct SimpleScheduler {
    worker_registry: Arc<TelemetryRegistry>, // TEAM-374
}

impl SimpleScheduler {
    /// Create a new simple scheduler
    pub fn new(worker_registry: Arc<TelemetryRegistry>) -> Self { // TEAM-374
        Self { worker_registry }
    }

    /// Execute job on selected worker and stream results
    ///
    /// This handles the full job execution flow:
    /// 1. POST to worker's /v1/inference endpoint
    /// 2. Connect to worker's SSE stream
    /// 3. Stream tokens back via line_handler
    ///
    /// # Arguments
    ///
    /// * `result` - Schedule result with worker info
    /// * `request` - Original job request
    /// * `line_handler` - Callback for each SSE line
    ///
    /// TEAM-380: Migrated to n!() macro (manual context - can't use macro on methods)
    pub async fn execute_job<F>(
        &self,
        result: ScheduleResult,
        request: JobRequest,
        mut line_handler: F,
    ) -> Result<(), SchedulerError>
    where
        F: FnMut(&str) -> Result<(), SchedulerError>,
    {
        // TEAM-380: Manual context setup for trait method
        let job_id = &request.job_id;
        let ctx = NarrationContext::new().with_job_id(job_id);
        
        with_narration_context(ctx, async move {
        let worker_url = &result.worker_url;

        // Step 1: POST to worker's /v1/inference endpoint
        // TEAM-380: Migrated to n!() macro
        n!("infer_post_start", "📤 Sending inference request to worker at {}", worker_url);

        let client = reqwest::Client::new();
        let worker_request = WorkerInferenceRequest {
            prompt: request.prompt.clone(),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
        };

        let response = client
            .post(format!("{}/v1/inference", worker_url))
            .json(&worker_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());

            // TEAM-380: Migrated to n!() macro
            n!("infer_post_err", "❌ Worker returned error {}: {}", status, error_text);

            return Err(SchedulerError::WorkerError { status, message: error_text });
        }

        let worker_job: WorkerJobResponse = response.json().await.map_err(|e| {
            // TEAM-380: Migrated to n!() macro
            n!("infer_parse_fail", "❌ Failed to parse worker response: {}", e);
            SchedulerError::ParseError(e.to_string())
        })?;

        // TEAM-380: Migrated to n!() macro
        n!("infer_job_created", "✅ Worker job created: {}", worker_job.job_id);

        // Step 2: Connect to worker's SSE stream
        let stream_url = format!("{}{}", worker_url, worker_job.sse_url);

        // TEAM-380: Migrated to n!() macro
        n!("infer_stream_conn", "🔗 Connecting to worker SSE stream: {}", stream_url);

        let stream_response = client.get(&stream_url).send().await.map_err(|e| {
            // TEAM-380: Migrated to n!() macro
            n!("infer_stream_fail", "❌ Failed to connect to worker stream: {}", e);
            SchedulerError::StreamConnectionFailed(e.to_string())
        })?;

        // Step 3: Stream tokens back to client
        use futures::StreamExt;

        // TEAM-380: Migrated to n!() macro
        n!("infer_streaming", "📡 Streaming tokens from worker...");

        let mut stream = stream_response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                // TEAM-380: Migrated to n!() macro
                n!("infer_chunk_err", "❌ Error reading stream chunk: {}", e);
                SchedulerError::StreamReadError(e.to_string())
            })?;

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if !line.is_empty() {
                    // Strip SSE prefix if present
                    let clean_line = if line.starts_with("data: ") { &line[6..] } else { &line };

                    // Forward to client
                    line_handler(clean_line)?;

                    // Check for [DONE] marker
                    if clean_line == "[DONE]" {
                        // TEAM-380: Migrated to n!() macro
                        n!("infer_complete", "✅ Inference complete");
                        return Ok(());
                    }
                }
            }
        }

        // Process remaining buffer
        if !buffer.is_empty() {
            let clean_line = if buffer.starts_with("data: ") { &buffer[6..] } else { &buffer };
            line_handler(clean_line.trim())?;
        }

        // TEAM-380: Migrated to n!() macro
        n!("infer_done", "✅ Inference streaming complete");

        Ok(())
        }).await // TEAM-380: Close with_narration_context
    }
}

#[async_trait::async_trait]
impl JobScheduler for SimpleScheduler {
    /// TEAM-380: Migrated to n!() macro (manual context for trait method)
    async fn schedule(&self, request: JobRequest) -> Result<ScheduleResult, SchedulerError> {
        // TEAM-380: Manual context setup for trait method
        let job_id = &request.job_id;
        let model = &request.model;
        let ctx = NarrationContext::new().with_job_id(job_id);
        
        with_narration_context(ctx, async move {

        // TEAM-380: Migrated to n!() macro
        n!("infer_schedule", "🔍 Finding worker for model '{}'", model);

        // Find best worker for model
        let worker = self.worker_registry.find_best_worker_for_model(model).ok_or_else(|| {
            // TEAM-380: Migrated to n!() macro
            n!("infer_no_worker", "❌ No available worker found for model '{}'", model);
            SchedulerError::NoWorkersAvailable { model: model.clone() }
        })?;

        // TEAM-374: ProcessStats uses different fields than WorkerInfo
        let worker_id = format!("{}-{}", worker.group, worker.instance);
        let worker_port: u16 = worker.instance.parse().unwrap_or(8080);
        let model_name = worker.model.clone().unwrap_or_else(|| model.to_string());
        
        // TEAM-380: Migrated to n!() macro
        n!("infer_worker_sel", "✅ Selected worker '{}' for model '{}' at localhost:{}", worker_id, model, worker_port);

        Ok(ScheduleResult {
            worker_id,
            worker_url: format!("http://localhost:{}", worker_port),
            worker_port,
            model: model_name,
            device: worker.group.clone(), // TEAM-374: Use group as device
        })
        }).await // TEAM-380: Close with_narration_context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_request_serialization() {
        let req = WorkerInferenceRequest {
            prompt: "Hello".to_string(),
            max_tokens: 20,
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"prompt\":\"Hello\""));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"top_p\":0.9"));
        assert!(!json.contains("\"top_k\"")); // Should be omitted
    }
}
