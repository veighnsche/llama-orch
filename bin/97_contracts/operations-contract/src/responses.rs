//! Response types for operations
//!
//! TEAM-284: Typed response structures for all operations
//! TEAM-380: Added documentation about narration-based output
//!
//! These types provide compile-time guarantees that responses are well-formed.
//!
//! # Important Note on Hive Implementation
//!
//! **rbee-hive currently returns narration events via SSE, not structured JSON responses.**
//!
//! These response types are defined for:
//! 1. **Type safety** - Documenting expected response structure
//! 2. **Future use** - When structured responses are needed
//! 3. **API clients** - Programmatic access (future)
//!
//! Current behavior:
//! - Hive operations return human-readable narration events
//! - Events are streamed via SSE as text lines
//! - Clients parse narration text (e.g., "✅ Worker 'worker-cpu-9301' spawned (PID: 12345, port: 9301)")
//!
//! To use structured responses in the future, hive would need to:
//! - Serialize these types to JSON
//! - Send as SSE data events
//! - Maintain backward compatibility with narration

use serde::{Deserialize, Serialize};

// ============================================================================
// Worker Operation Responses
// ============================================================================

/// Response from spawning a worker
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerSpawnResponse {
    /// Assigned worker ID
    pub worker_id: String,
    /// Port worker is listening on
    pub port: u16,
    /// Process ID
    pub pid: u32,
    /// Status message
    pub status: String,
}

/// Worker process information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessInfo {
    /// Process ID
    pub pid: u32,
    /// Worker ID
    pub worker_id: String,
    /// Model being served
    pub model: String,
    /// Port
    pub port: u16,
    /// Status (e.g., "running", "starting")
    pub status: String,
}

/// Response from listing worker processes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessListResponse {
    /// List of worker processes
    pub workers: Vec<WorkerProcessInfo>,
}

/// Response from getting worker process details
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessGetResponse {
    /// Worker process information
    pub worker: WorkerProcessInfo,
}

/// Response from deleting a worker process
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessDeleteResponse {
    /// Success message
    pub message: String,
    /// Process ID that was killed
    pub pid: u32,
}

// ============================================================================
// Model Operation Responses
// ============================================================================

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelInfo {
    /// Model ID
    pub id: String,
    /// Model name
    pub name: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Download status
    pub status: String,
}

/// Response from downloading a model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelDownloadResponse {
    /// Model ID
    pub model_id: String,
    /// Status message
    pub status: String,
}

/// Response from listing models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelListResponse {
    /// List of models
    pub models: Vec<ModelInfo>,
}

/// Response from getting model details
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelGetResponse {
    /// Model information
    pub model: ModelInfo,
}

/// Response from deleting a model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelDeleteResponse {
    /// Success message
    pub message: String,
    /// Model ID that was deleted
    pub model_id: String,
}

// ============================================================================
// Inference Response
// ============================================================================

/// Response from inference (streamed)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferResponse {
    /// Generated text
    pub text: String,
    /// Tokens generated
    pub tokens: u32,
    /// Whether generation is complete
    pub done: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_spawn_response_serialization() {
        let response = WorkerSpawnResponse {
            worker_id: "worker-123".to_string(),
            port: 9301,
            pid: 12345,
            status: "running".to_string(),
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: WorkerSpawnResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(response, deserialized);
    }

    #[test]
    fn test_model_list_response() {
        let response = ModelListResponse {
            models: vec![
                ModelInfo {
                    id: "model-1".to_string(),
                    name: "Test Model 1".to_string(),
                    size_bytes: 1024,
                    status: "ready".to_string(),
                },
                ModelInfo {
                    id: "model-2".to_string(),
                    name: "Test Model 2".to_string(),
                    size_bytes: 2048,
                    status: "downloading".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: ModelListResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(response, deserialized);
        assert_eq!(deserialized.models.len(), 2);
    }
}
