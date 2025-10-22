//! Shared HTTP client for job submission and SSE streaming
//!
//! TEAM-259: Consolidate job submission patterns
//!
//! This crate provides a reusable pattern for:
//! - Submitting operations to /v1/jobs endpoints
//! - Streaming SSE responses from /v1/jobs/{job_id}/stream
//! - Processing narration events
//!
//! # Usage
//!
//! ```rust,no_run
//! use job_client::JobClient;
//! use rbee_operations::Operation;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let client = JobClient::new("http://localhost:8500");
//!
//! let operation = Operation::HiveList;
//!
//! client.submit_and_stream(operation, |line| {
//!     println!("{}", line);
//!     Ok(())
//! }).await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use futures::stream::StreamExt;
use rbee_operations::Operation;

/// HTTP client for job submission and SSE streaming
///
/// TEAM-259: Shared pattern used by:
/// - rbee-keeper → queen-rbee
/// - queen-rbee → rbee-hive
#[derive(Debug, Clone)]
pub struct JobClient {
    base_url: String,
    client: reqwest::Client,
}

impl JobClient {
    /// Create a new job client for the given base URL
    ///
    /// # Example
    /// ```
    /// use job_client::JobClient;
    ///
    /// let client = JobClient::new("http://localhost:8500");
    /// ```
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Create a new job client with a custom reqwest client
    ///
    /// Useful for setting timeouts, custom headers, etc.
    pub fn with_client(base_url: impl Into<String>, client: reqwest::Client) -> Self {
        Self {
            base_url: base_url.into(),
            client,
        }
    }

    /// Submit a job and stream its SSE responses
    ///
    /// TEAM-259: Core pattern shared across rbee-keeper and queen-rbee
    ///
    /// # Arguments
    /// * `operation` - The operation to submit
    /// * `line_handler` - Callback for each SSE line (without "data: " prefix)
    ///
    /// # Returns
    /// * `Ok(job_id)` - The job ID returned by the server
    /// * `Err(_)` - If submission or streaming fails
    pub async fn submit_and_stream<F>(
        &self,
        operation: Operation,
        mut line_handler: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Result<()>,
    {
        // 1. Serialize operation to JSON
        let payload = serde_json::to_value(&operation)
            .map_err(|e| anyhow::anyhow!("Failed to serialize operation: {}", e))?;

        // 2. POST to /v1/jobs endpoint
        let job_response: serde_json::Value = self
            .client
            .post(format!("{}/v1/jobs", self.base_url))
            .json(&payload)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to submit job: {}", e))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse job response: {}", e))?;

        // 3. Extract job_id from response
        let job_id = job_response
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Server did not return job_id"))?
            .to_string();

        // 4. Connect to SSE stream
        let stream_url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
        let response = self
            .client
            .get(&stream_url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to connect to SSE stream: {}", e))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("SSE stream returned error: {}", error));
        }

        // 5. Stream bytes and process lines
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| anyhow::anyhow!("Stream error: {}", e))?;
            let text = String::from_utf8(chunk.to_vec())
                .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in stream: {}", e))?;

            // Process each line
            for line in text.lines() {
                // Strip "data: " prefix if present (SSE format)
                let data = line.strip_prefix("data: ").unwrap_or(line);
                
                // Skip empty lines
                if data.is_empty() {
                    continue;
                }

                // Call handler for each line
                line_handler(data)?;

                // Check for [DONE] marker
                if data.contains("[DONE]") {
                    return Ok(job_id);
                }
            }
        }

        // Stream ended without [DONE] marker
        Ok(job_id)
    }

    /// Submit a job without streaming (fire and forget)
    ///
    /// Returns the job_id immediately without waiting for completion.
    pub async fn submit(&self, operation: Operation) -> Result<String> {
        let payload = serde_json::to_value(&operation)
            .map_err(|e| anyhow::anyhow!("Failed to serialize operation: {}", e))?;

        let job_response: serde_json::Value = self
            .client
            .post(format!("{}/v1/jobs", self.base_url))
            .json(&payload)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to submit job: {}", e))?
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse job response: {}", e))?;

        let job_id = job_response
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Server did not return job_id"))?
            .to_string();

        Ok(job_id)
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = JobClient::new("http://localhost:8500");
        assert_eq!(client.base_url(), "http://localhost:8500");
    }

    #[test]
    fn test_strip_data_prefix() {
        let line = "data: Hello world";
        let stripped = line.strip_prefix("data: ").unwrap_or(line);
        assert_eq!(stripped, "Hello world");
    }
}
