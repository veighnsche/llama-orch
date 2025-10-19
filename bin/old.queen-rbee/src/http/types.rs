//! Request and Response types for queen-rbee HTTP API
//!
//! Created by: TEAM-043
//! Refactored by: TEAM-052

use crate::beehive_registry::BeehiveNode;
use serde::{Deserialize, Serialize};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Beehive Registry Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Deserialize)]
pub struct AddNodeRequest {
    pub node_name: String,
    pub ssh_host: String,
    #[serde(default = "default_ssh_port")]
    pub ssh_port: u16,
    pub ssh_user: String,
    pub ssh_key_path: Option<String>,
    pub git_repo_url: String,
    pub git_branch: String,
    pub install_path: String,
    // TEAM-052: Backend capabilities
    pub backends: Option<String>, // JSON array: ["cuda", "metal", "cpu"]
    pub devices: Option<String>,  // JSON object: {"cuda": 2, "metal": 1, "cpu": 1}
}

fn default_ssh_port() -> u16 {
    22
}

#[derive(Debug, Serialize)]
pub struct AddNodeResponse {
    pub success: bool,
    pub message: String,
    pub node_name: String,
}

#[derive(Debug, Serialize)]
pub struct ListNodesResponse {
    pub nodes: Vec<BeehiveNode>,
}

#[derive(Debug, Serialize)]
pub struct RemoveNodeResponse {
    pub success: bool,
    pub message: String,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Health Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Worker Management Types (TEAM-046)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// TEAM-084: Worker registration request from rbee-hive
#[derive(Debug, Deserialize)]
pub struct RegisterWorkerRequest {
    pub worker_id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub node_name: String,
    pub slots_total: Option<u32>,
    pub vram_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct RegisterWorkerResponse {
    pub success: bool,
    pub message: String,
    pub worker_id: String,
}

// TEAM-124: Worker ready notification from rbee-hive
#[derive(Debug, Deserialize)]
pub struct WorkerReadyNotification {
    pub worker_id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
}

#[derive(Debug, Serialize)]
pub struct WorkerReadyResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub node: String,
    pub state: String,
    pub model_ref: Option<String>,
    pub url: String,
}

#[derive(Debug, Serialize)]
pub struct WorkersListResponse {
    pub workers: Vec<WorkerInfo>,
}

#[derive(Debug, Serialize)]
pub struct WorkerHealthInfo {
    pub worker_id: String,
    pub state: String,
    pub ready: bool,
}

#[derive(Debug, Serialize)]
pub struct WorkersHealthResponse {
    pub status: String,
    pub workers: Vec<WorkerHealthInfo>,
}

#[derive(Debug, Deserialize)]
pub struct ShutdownWorkerRequest {
    pub worker_id: String,
}

#[derive(Debug, Serialize)]
pub struct ShutdownWorkerResponse {
    pub success: bool,
    pub message: String,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Inference Task Types (TEAM-046)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Deserialize)]
pub struct InferenceTaskRequest {
    pub node: String,
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

// TEAM-084: Simple inference request (OpenAI-compatible)
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_tokens() -> u32 {
    100
}

fn default_temperature() -> f32 {
    0.7
}

// TEAM-047: Internal types for worker communication
#[derive(Debug, Deserialize)]
pub(crate) struct WorkerSpawnResponse {
    pub worker_id: String,
    pub url: String,
    #[allow(dead_code)]
    pub state: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ReadyResponse {
    pub ready: bool,
    #[allow(dead_code)]
    pub state: String,
}
