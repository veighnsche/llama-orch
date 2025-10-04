//! HTTP client for registration

use anyhow::{Context, Result};
use pool_registry::{HeartbeatRequest, HeartbeatResponse, RegisterRequest, RegisterResponse};
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
        Self { orchestratord_url, client: reqwest::Client::new(), api_token }
    }

    /// Register node
    pub async fn register(&self, request: RegisterRequest) -> Result<RegisterResponse> {
        let url = format!("{}/v2/nodes/register", self.orchestratord_url);

        debug!(url = %url, "Sending registration request");

        let mut req = self.client.post(&url).json(&request);

        if let Some(token) = &self.api_token {
            req = req.bearer_auth(token);
        }

        let response = req.send().await.context("Failed to send registration request")?;

        let status = response.status();
        let body = response
            .json::<RegisterResponse>()
            .await
            .context("Failed to parse registration response")?;

        if !status.is_success() {
            anyhow::bail!("Registration failed: {}", body.message);
        }

        Ok(body)
    }

    /// Send heartbeat
    pub async fn heartbeat(
        &self,
        node_id: &str,
        request: HeartbeatRequest,
    ) -> Result<HeartbeatResponse> {
        let url = format!("{}/v2/nodes/{}/heartbeat", self.orchestratord_url, node_id);

        let mut req = self.client.post(&url).json(&request);

        if let Some(token) = &self.api_token {
            req = req.bearer_auth(token);
        }

        let response = req.send().await.context("Failed to send heartbeat")?;

        let status = response.status();
        let body = response
            .json::<HeartbeatResponse>()
            .await
            .context("Failed to parse heartbeat response")?;

        if !status.is_success() {
            anyhow::bail!("Heartbeat failed");
        }

        Ok(body)
    }

    /// Deregister node
    pub async fn deregister(&self, node_id: &str) -> Result<()> {
        let url = format!("{}/v2/nodes/{}", self.orchestratord_url, node_id);

        let mut req = self.client.delete(&url);

        if let Some(token) = &self.api_token {
            req = req.bearer_auth(token);
        }

        let response = req.send().await.context("Failed to send deregistration request")?;

        if !response.status().is_success() {
            anyhow::bail!("Deregistration failed");
        }

        Ok(())
    }
}
