//! Device detector implementation for queen-rbee
//!
//! Created by: TEAM-159
//!
//! Implements the DeviceDetector trait from rbee-heartbeat to make HTTP
//! requests to hives for device detection.

use async_trait::async_trait;
use rbee_heartbeat::traits::{DetectionError, DeviceDetector, DeviceResponse};

/// HTTP-based device detector
///
/// Makes HTTP GET requests to hive's /v1/devices endpoint
pub struct HttpDeviceDetector {
    client: reqwest::Client,
}

impl HttpDeviceDetector {
    /// Create a new HTTP device detector
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for HttpDeviceDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DeviceDetector for HttpDeviceDetector {
    async fn detect_devices(&self, hive_url: &str) -> Result<DeviceResponse, DetectionError> {
        let url = format!("{}/v1/devices", hive_url);
        
        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| DetectionError::Http(e.to_string()))?
            .json()
            .await
            .map_err(|e| DetectionError::Parse(e.to_string()))
    }
}
