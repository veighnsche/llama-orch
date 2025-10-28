//! Request types for operations
//!
//! TEAM-284: Typed request structures for all operations
//!
//! These types provide compile-time guarantees that requests are well-formed.

use serde::{Deserialize, Serialize};

// ============================================================================
// Worker Operation Requests
// ============================================================================

/// Request to spawn a worker process
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerSpawnRequest {
    /// Hive ID where worker should be spawned
    pub hive_id: String,
    /// Model to load
    pub model: String,
    /// Worker type (e.g., "cpu", "cuda", "metal")
    pub worker: String,
    /// Device index
    pub device: u32,
}

/// Request to list worker processes on a hive
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessListRequest {
    /// Hive ID to query
    pub hive_id: String,
}

/// Request to get worker process details
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessGetRequest {
    /// Hive ID where worker is running
    pub hive_id: String,
    /// Process ID
    pub pid: u32,
}

/// Request to delete (kill) a worker process
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessDeleteRequest {
    /// Hive ID where worker is running
    pub hive_id: String,
    /// Process ID to kill
    pub pid: u32,
}

// ============================================================================
// Model Operation Requests
// ============================================================================

/// Request to download a model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelDownloadRequest {
    /// Hive ID where model should be downloaded
    pub hive_id: String,
    /// Model identifier (e.g., "meta-llama/Llama-2-7b")
    pub model: String,
}

/// Request to list models on a hive
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelListRequest {
    /// Hive ID to query
    pub hive_id: String,
}

/// Request to get model details
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelGetRequest {
    /// Hive ID where model is stored
    pub hive_id: String,
    /// Model ID
    pub id: String,
}

/// Request to delete a model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelDeleteRequest {
    /// Hive ID where model is stored
    pub hive_id: String,
    /// Model ID to delete
    pub id: String,
}

// ============================================================================
// Inference Request
// ============================================================================

/// Request to perform inference
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferRequest {
    /// Hive ID (for routing)
    pub hive_id: String,
    /// Model to use
    pub model: String,
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature (0.0 - 2.0)
    pub temperature: f32,
    /// Top-p sampling (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Device to use (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Specific worker ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
    /// Stream response
    #[serde(default = "default_stream")]
    pub stream: bool,
}

fn default_stream() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_spawn_request_serialization() {
        let request = WorkerSpawnRequest {
            hive_id: "localhost".to_string(),
            model: "test-model".to_string(),
            worker: "cpu".to_string(),
            device: 0,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: WorkerSpawnRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(request, deserialized);
    }

    #[test]
    fn test_infer_request_optional_fields() {
        let request = InferRequest {
            hive_id: "localhost".to_string(),
            model: "test-model".to_string(),
            prompt: "hello".to_string(),
            max_tokens: 20,
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: None,
            device: None,
            worker_id: None,
            stream: true,
        };

        let json = serde_json::to_string(&request).unwrap();

        // top_p should be present
        assert!(json.contains("\"top_p\":0.9"));

        // top_k should be omitted
        assert!(!json.contains("\"top_k\""));
    }
}
