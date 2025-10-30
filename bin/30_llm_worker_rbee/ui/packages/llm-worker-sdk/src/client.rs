// TEAM-353: WorkerClient - thin wrapper around job-client for Worker UI
// Pattern: Same as HiveClient

use wasm_bindgen::prelude::*;
use job_client::JobClient;
use operations_contract::Operation;
use crate::conversions::{js_to_operation, error_to_js};

/// Main client for Worker UI operations
///
/// TEAM-353: This is a thin wrapper around the existing JobClient
/// from the job-client shared crate. We just add WASM bindings.
#[wasm_bindgen]
pub struct WorkerClient {
    /// TEAM-353: Reuse existing job-client!
    inner: JobClient,
    /// TEAM-353: Store worker_id (identifier for this worker)
    worker_id: String,
}

#[wasm_bindgen]
impl WorkerClient {
    /// Create a new WorkerClient
    ///
    /// # Arguments
    /// * `base_url` - Base URL of worker (e.g., "http://192.168.1.101:7840")
    /// * `worker_id` - Worker identifier (e.g., "worker-1" or hostname)
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String, worker_id: String) -> Self {
        // TEAM-353: Wrap JobClient and store worker_id
        Self {
            inner: JobClient::new(base_url),
            worker_id,
        }
    }

    /// Get the base URL
    #[wasm_bindgen(getter)]
    pub fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }

    /// Get the worker ID
    #[wasm_bindgen(getter, js_name = workerId)]
    pub fn worker_id(&self) -> String {
        self.worker_id.clone()
    }

    /// Submit a job and stream results
    ///
    /// TEAM-353: This wraps JobClient::submit_and_stream() from job-client crate
    ///
    /// # Arguments
    /// * `operation` - JavaScript object representing an operation
    /// * `on_line` - JavaScript callback function called for each line
    ///
    /// # Returns
    /// Promise that resolves to job_id
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const jobId = await client.submitAndStream(
    ///   { operation: 'infer', model: 'llama-3.2-1b', prompt: 'Hello' },
    ///   (line) => console.log(line)
    /// );
    /// ```
    #[wasm_bindgen(js_name = submitAndStream)]
    pub async fn submit_and_stream(
        &self,
        operation: JsValue,
        on_line: js_sys::Function,
    ) -> Result<String, JsValue> {
        // TEAM-353: Convert JS operation to Rust
        let op: Operation = js_to_operation(operation)?;

        // TEAM-353: Clone callback for use in closure
        let callback = on_line.clone();

        // TEAM-353: Use existing job-client!
        let job_id = self.inner
            .submit_and_stream(op, move |line| {
                // Call JavaScript callback
                let this = JsValue::null();
                let line_js = JsValue::from_str(line);
                
                // Ignore callback errors (non-critical)
                let _ = callback.call1(&this, &line_js);
                
                Ok(())
            })
            .await
            .map_err(error_to_js)?;

        Ok(job_id)
    }

    /// Submit a job without streaming
    ///
    /// TEAM-353: Wraps JobClient::submit()
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const jobId = await client.submit({ operation: 'infer', model: 'llama-3.2-1b', prompt: 'Hello' });
    /// ```
    #[wasm_bindgen]
    pub async fn submit(&self, operation: JsValue) -> Result<String, JsValue> {
        let op: Operation = js_to_operation(operation)?;
        
        self.inner
            .submit(op)
            .await
            .map_err(error_to_js)
    }
}
