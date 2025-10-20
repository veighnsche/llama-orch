//! Mock rbee-hive /v1/devices endpoint for testing device detection
//!
//! Created by: TEAM-159
//!
//! This module provides a mock HTTP server that simulates rbee-hive's
//! /v1/devices endpoint. Used in BDD tests where we can't spin up a real
//! rbee-hive daemon.
//!
//! What it mocks: rbee-hive's GET /v1/devices endpoint
//! What it returns: JSON with CPU, GPU, models, workers info

use async_trait::async_trait;
use rbee_heartbeat::traits::{DetectionError, DeviceDetector, DeviceResponse};
use serde_json::json;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

/// Mock device response builder for rbee-hive's /v1/devices endpoint
pub struct MockDeviceResponse {
    pub cpu_cores: u32,
    pub cpu_ram_gb: u32,
    pub gpus: Vec<MockGpu>,
    pub models: usize,
    pub workers: usize,
}

/// Mock GPU data from rbee-hive
pub struct MockGpu {
    pub id: String,
    pub name: String,
    pub vram_gb: u32,
}

impl MockDeviceResponse {
    /// Create a default mock response with 2 GPUs
    pub fn default_response() -> Self {
        Self {
            cpu_cores: 8,
            cpu_ram_gb: 32,
            gpus: vec![
                MockGpu {
                    id: "gpu0".to_string(),
                    name: "RTX 3060".to_string(),
                    vram_gb: 12,
                },
                MockGpu {
                    id: "gpu1".to_string(),
                    name: "RTX 3090".to_string(),
                    vram_gb: 24,
                },
            ],
            models: 0,
            workers: 0,
        }
    }

    /// Create a CPU-only mock response
    pub fn cpu_only() -> Self {
        Self {
            cpu_cores: 4,
            cpu_ram_gb: 16,
            gpus: vec![],
            models: 0,
            workers: 0,
        }
    }

    /// Create a custom mock response
    pub fn new(cpu_cores: u32, cpu_ram_gb: u32) -> Self {
        Self {
            cpu_cores,
            cpu_ram_gb,
            gpus: vec![],
            models: 0,
            workers: 0,
        }
    }

    /// Add a GPU
    pub fn with_gpu(mut self, id: String, name: String, vram_gb: u32) -> Self {
        self.gpus.push(MockGpu { id, name, vram_gb });
        self
    }

    /// Convert to JSON
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "cpu": {
                "cores": self.cpu_cores,
                "ram_gb": self.cpu_ram_gb
            },
            "gpus": self.gpus.iter().map(|gpu| json!({
                "id": gpu.id,
                "name": gpu.name,
                "vram_gb": gpu.vram_gb
            })).collect::<Vec<_>>(),
            "models": self.models,
            "workers": self.workers
        })
    }
}

/// Start a mock rbee-hive server that responds to /v1/devices
pub async fn start_mock_hive_device_endpoint(response: MockDeviceResponse) -> MockServer {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/v1/devices"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response.to_json()))
        .mount(&mock_server)
        .await;

    mock_server
}

/// Mock device detector that calls the mock rbee-hive /v1/devices endpoint
///
/// This replaces the real HttpDeviceDetector which would make HTTP calls
/// to a real rbee-hive instance.
pub struct MockHiveDeviceDetector {
    base_url: String,
    client: reqwest::Client,
}

impl MockHiveDeviceDetector {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl DeviceDetector for MockHiveDeviceDetector {
    async fn detect_devices(&self, _hive_url: &str) -> Result<DeviceResponse, DetectionError> {
        // Use the mock rbee-hive URL instead of the provided hive_url
        // In real code, this would call http://real-hive:8600/v1/devices
        let url = format!("{}/v1/devices", self.base_url);
        
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
