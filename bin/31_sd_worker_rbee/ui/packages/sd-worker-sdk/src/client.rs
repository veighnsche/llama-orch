// TEAM-391: SD Worker Client - WASM wrapper around job-client
// Pattern: Same as llm-worker-sdk client

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// SD Worker Client for submitting image generation jobs
///
/// # Example
///
/// ```javascript
/// const client = new SDWorkerClient('http://localhost:8600', 'sd-worker-1');
/// const result = await client.generateImage({
///     prompt: 'a photo of a cat',
///     steps: 20,
///     width: 512,
///     height: 512
/// });
/// ```
#[wasm_bindgen]
pub struct SDWorkerClient {
    base_url: String,
    worker_id: String,
}

#[wasm_bindgen]
impl SDWorkerClient {
    /// Create a new SD Worker client
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the worker (e.g., "http://localhost:8600")
    /// * `worker_id` - Worker identifier
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String, worker_id: String) -> Self {
        web_sys::console::log_1(
            &format!("🎨 [SD Worker SDK] Creating client for {}", base_url).into(),
        );
        Self { base_url, worker_id }
    }

    /// Submit a text-to-image generation job
    ///
    /// # Arguments
    ///
    /// * `request` - Generation request (prompt, steps, dimensions, etc.)
    ///
    /// # Returns
    ///
    /// Promise that resolves to the job ID
    #[wasm_bindgen(js_name = generateImage)]
    pub async fn generate_image(&self, request: JsValue) -> Result<String, JsValue> {
        // TODO: TEAM-392+ will implement this using job-client
        // For now, return a stub
        web_sys::console::log_1(&"🎨 [SD Worker SDK] generateImage called (stub)".into());
        Ok("job_stub_123".to_string())
    }

    /// Get the status of a generation job
    ///
    /// # Arguments
    ///
    /// * `job_id` - Job identifier
    ///
    /// # Returns
    ///
    /// Promise that resolves to job status
    #[wasm_bindgen(js_name = getJobStatus)]
    pub async fn get_job_status(&self, job_id: String) -> Result<JsValue, JsValue> {
        // TODO: TEAM-392+ will implement this
        web_sys::console::log_1(
            &format!("🎨 [SD Worker SDK] getJobStatus({}) called (stub)", job_id).into(),
        );
        Ok(JsValue::NULL)
    }

    /// Stream generation progress via SSE
    ///
    /// # Arguments
    ///
    /// * `job_id` - Job identifier
    /// * `callback` - Callback function for progress events
    #[wasm_bindgen(js_name = streamProgress)]
    pub fn stream_progress(
        &self,
        job_id: String,
        callback: js_sys::Function,
    ) -> Result<(), JsValue> {
        // TODO: TEAM-392+ will implement SSE streaming
        web_sys::console::log_1(
            &format!("🎨 [SD Worker SDK] streamProgress({}) called (stub)", job_id).into(),
        );
        Ok(())
    }
}
