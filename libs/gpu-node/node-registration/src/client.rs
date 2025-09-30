//! HTTP client for registration

use anyhow::{Context, Result};
use service_registry::{
    RegisterRequest, RegisterResponse, 
    HeartbeatRequest, HeartbeatResponse
};
use tracing::debug;

/// Registration client for communicating with orchestratord
pub struct RegistrationClient {
    orchestratord_url: String,
    client: reqwest::Client,
    api_token: Option<String>,
}

impl RegistrationClient {
    /// Create a new registration client
    pub fn new(orchestratord_url: String, api_token: Option<String>) -> Self {
        Self {
    }

    impl RegistrationClient {
        /// Create a new registration client
        pub fn new(base_url: String, api_token: Option<String>) -> Self {
            Self {
                base_url,
            .context("Failed to send registration request")?;

        let status = response.status();
        let body = response.json::<RegisterResponse>().await
            .context("Failed to parse registration response")?;

        if !status.is_success() {
            anyhow::bail!("Registration failed: {}", body.message);

        Ok(body)

    pub async fn heartbeat(&self, node_id: &str, request: HeartbeatRequest) -> Result<()> {
        let url = format!("{}/v2/nodes/{}/heartbeat", self.base_url, node_id);
        
        let mut req = self.client.post(&url).json(&request);

        let mut req = self.client.delete(&url);

        // Add Bearer token if configured
        if let Some(token) = &self.api_token {
            req = req.bearer_auth(token);
        }

        let response = req
            .send()
            .await

        let status = response.status();
        let body = response.json::<HeartbeatResponse>().await

        if !status.is_success() {
            .context("Failed to send deregistration request")?;

        if !response.status().is_success() {
            anyhow::bail!("Deregistration failed");
        }

        Ok(())
    }
}
