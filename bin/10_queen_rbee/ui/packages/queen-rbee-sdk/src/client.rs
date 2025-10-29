// TEAM-286: QueenClient - thin wrapper around job-client for Queen UI

use wasm_bindgen::prelude::*;
use job_client::JobClient;
use operations_contract::Operation;
use crate::conversions::{js_to_operation, error_to_js};

/// Main client for Queen UI operations
///
/// TEAM-286: This is a thin wrapper around the existing JobClient
/// from the job-client shared crate. We just add WASM bindings.
#[wasm_bindgen]
pub struct QueenClient {
    /// TEAM-286: Reuse existing job-client!
    inner: JobClient,
}

#[wasm_bindgen]
impl QueenClient {
    /// Create a new QueenClient
    ///
    /// # Arguments
    /// * `base_url` - Base URL of queen-rbee (e.g., "http://localhost:7833")
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        // TEAM-286: Just wrap the existing JobClient
        Self {
            inner: JobClient::new(base_url),
        }
    }

    /// Get the base URL
    #[wasm_bindgen(getter)]
    pub fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }

    /// Submit a job and stream results
    ///
    /// TEAM-286: This wraps JobClient::submit_and_stream() from job-client crate
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
    ///   { operation: 'status' },
    ///   (line) => console.log(line)
    /// );
    /// ```
    #[wasm_bindgen(js_name = submitAndStream)]
    pub async fn submit_and_stream(
        &self,
        operation: JsValue,
        on_line: js_sys::Function,
    ) -> Result<String, JsValue> {
        // TEAM-286: Convert JS operation to Rust
        let op: Operation = js_to_operation(operation)?;

        // TEAM-286: Clone callback for use in closure
        let callback = on_line.clone();

        // TEAM-286: Use existing job-client!
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
    /// TEAM-286: Wraps JobClient::submit()
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const jobId = await client.submit({ operation: 'status' });
    /// ```
    #[wasm_bindgen]
    pub async fn submit(&self, operation: JsValue) -> Result<String, JsValue> {
        let op: Operation = js_to_operation(operation)?;
        
        self.inner
            .submit(op)
            .await
            .map_err(error_to_js)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CONVENIENCE METHODS
    // ═══════════════════════════════════════════════════════════════════════
    // Queen UI only needs Status for heartbeat monitoring.
    // All worker/model/infer operations belong to Hive UI.

    /// Convenience: Status (streaming)
    ///
    /// # JavaScript Example
    /// ```javascript
    /// await client.status((line) => console.log(line));
    /// ```
    #[wasm_bindgen]
    pub async fn status(
        &self,
        callback: js_sys::Function,
    ) -> Result<String, JsValue> {
        use crate::operations::OperationBuilder;
        let op = OperationBuilder::status();
        self.submit_and_stream(op, callback).await
    }
}

