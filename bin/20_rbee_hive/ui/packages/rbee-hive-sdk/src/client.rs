// TEAM-353: HiveClient - thin wrapper around job-client for Hive UI
// Pattern: Same as QueenClient (TEAM-286)

use wasm_bindgen::prelude::*;
use job_client::JobClient;
use operations_contract::Operation;
use crate::conversions::{js_to_operation, error_to_js};

/// Main client for Hive UI operations
///
/// TEAM-353: This is a thin wrapper around the existing JobClient
/// from the job-client shared crate. We just add WASM bindings.
#[wasm_bindgen]
pub struct HiveClient {
    /// TEAM-353: Reuse existing job-client!
    inner: JobClient,
    /// TEAM-353: Store hive_id (hostname/IP of this Hive)
    hive_id: String,
}

#[wasm_bindgen]
impl HiveClient {
    /// Create a new HiveClient
    ///
    /// # Arguments
    /// * `base_url` - Base URL of rbee-hive (e.g., "http://192.168.1.100:7835")
    /// * `hive_id` - Hive identifier (hostname or IP, e.g., "192.168.1.100")
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String, hive_id: String) -> Self {
        // TEAM-353: Wrap JobClient and store hive_id
        Self {
            inner: JobClient::new(base_url),
            hive_id,
        }
    }

    /// Get the base URL
    #[wasm_bindgen(getter)]
    pub fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }

    /// Get the hive ID
    #[wasm_bindgen(getter, js_name = hiveId)]
    pub fn hive_id(&self) -> String {
        self.hive_id.clone()
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
    ///   { operation: 'worker_process_list', hive_id: 'localhost' },
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
    /// const jobId = await client.submit({ operation: 'worker_process_list', hive_id: 'localhost' });
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
