// TEAM-275: Simple scheduler implementation (first available worker)
use crate::types::{
    JobRequest, ScheduleResult, SchedulerError, WorkerInferenceRequest, WorkerJobResponse,
};
use crate::JobScheduler;
use observability_narration_core::NarrationFactory;
use queen_rbee_telemetry_registry::TelemetryRegistry; // TEAM-374
use std::sync::Arc;

const NARRATE: NarrationFactory = NarrationFactory::new("scheduler");

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
    pub async fn execute_job<F>(
        &self,
        result: ScheduleResult,
        request: JobRequest,
        mut line_handler: F,
    ) -> Result<(), SchedulerError>
    where
        F: FnMut(&str) -> Result<(), SchedulerError>,
    {
        let job_id = &request.job_id;
        let worker_url = &result.worker_url;

        // Step 1: POST to worker's /v1/inference endpoint
        NARRATE
            .action("infer_post_start")
            .job_id(job_id)
            .context(worker_url)
            .human("📤 Sending inference request to worker at {}")
            .emit();

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

            NARRATE
                .action("infer_post_err")
                .job_id(job_id)
                .context(&status.to_string())
                .context(&error_text)
                .human("❌ Worker returned error {}: {}")
                .error_kind("worker_error")
                .emit();

            return Err(SchedulerError::WorkerError { status, message: error_text });
        }

        let worker_job: WorkerJobResponse = response.json().await.map_err(|e| {
            NARRATE
                .action("infer_parse_fail")
                .job_id(job_id)
                .context(&e.to_string())
                .human("❌ Failed to parse worker response: {}")
                .error_kind("parse_failed")
                .emit();

            SchedulerError::ParseError(e.to_string())
        })?;

        NARRATE
            .action("infer_job_created")
            .job_id(job_id)
            .context(&worker_job.job_id)
            .human("✅ Worker job created: {}")
            .emit();

        // Step 2: Connect to worker's SSE stream
        let stream_url = format!("{}{}", worker_url, worker_job.sse_url);

        NARRATE
            .action("infer_stream_conn")
            .job_id(job_id)
            .context(&stream_url)
            .human("🔗 Connecting to worker SSE stream: {}")
            .emit();

        let stream_response = client.get(&stream_url).send().await.map_err(|e| {
            NARRATE
                .action("infer_stream_fail")
                .job_id(job_id)
                .context(&e.to_string())
                .human("❌ Failed to connect to worker stream: {}")
                .error_kind("stream_connection_failed")
                .emit();

            SchedulerError::StreamConnectionFailed(e.to_string())
        })?;

        // Step 3: Stream tokens back to client
        use futures::StreamExt;

        NARRATE
            .action("infer_streaming")
            .job_id(job_id)
            .human("📡 Streaming tokens from worker...")
            .emit();

        let mut stream = stream_response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                NARRATE
                    .action("infer_chunk_err")
                    .job_id(job_id)
                    .context(&e.to_string())
                    .human("❌ Error reading stream chunk: {}")
                    .error_kind("stream_read_error")
                    .emit();

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
                        NARRATE
                            .action("infer_complete")
                            .job_id(job_id)
                            .human("✅ Inference complete")
                            .emit();
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

        NARRATE.action("infer_done").job_id(job_id).human("✅ Inference streaming complete").emit();

        Ok(())
    }
}

#[async_trait::async_trait]
impl JobScheduler for SimpleScheduler {
    async fn schedule(&self, request: JobRequest) -> Result<ScheduleResult, SchedulerError> {
        let job_id = &request.job_id;
        let model = &request.model;

        NARRATE
            .action("infer_schedule")
            .job_id(job_id)
            .context(model)
            .human("🔍 Finding worker for model '{}'")
            .emit();

        // Find best worker for model
        let worker = self.worker_registry.find_best_worker_for_model(model).ok_or_else(|| {
            NARRATE
                .action("infer_no_worker")
                .job_id(job_id)
                .context(model)
                .human("❌ No available worker found for model '{}'")
                .error_kind("no_worker")
                .emit();

            SchedulerError::NoWorkersAvailable { model: model.clone() }
        })?;

        // TEAM-374: ProcessStats uses different fields than WorkerInfo
        let worker_id = format!("{}-{}", worker.group, worker.instance);
        let worker_port: u16 = worker.instance.parse().unwrap_or(8080);
        let model_name = worker.model.clone().unwrap_or_else(|| model.to_string());
        
        NARRATE
            .action("infer_worker_sel")
            .job_id(job_id)
            .context(&worker_id)
            .context(model)
            .context(&format!("localhost:{}", worker_port))
            .human("✅ Selected worker '{}' for model '{}' at {}")
            .emit();

        Ok(ScheduleResult {
            worker_id,
            worker_url: format!("http://localhost:{}", worker_port),
            worker_port,
            model: model_name,
            device: worker.group.clone(), // TEAM-374: Use group as device
        })
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
