// World state for BDD tests
// Created by: TEAM-040

use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Default, cucumber::World)]
pub struct World {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Topology & Configuration
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Node topology: node_name -> (hostname, components, capabilities)
    pub topology: HashMap<String, NodeInfo>,

    /// Current node we're operating from
    pub current_node: Option<String>,

    /// queen-rbee URL
    pub queen_rbee_url: Option<String>,

    /// Model catalog path
    pub model_catalog_path: Option<PathBuf>,

    /// rbee-hive registry database path
    pub registry_db_path: Option<String>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // rbee-hive Registry State (TEAM-041)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Registered rbee-hive nodes: node_name -> BeehiveNode
    pub beehive_nodes: HashMap<String, BeehiveNode>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Worker Registry State
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Registered workers: worker_id -> WorkerInfo
    pub workers: HashMap<String, WorkerInfo>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Model Catalog State
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Model catalog entries: model_ref -> ModelCatalogEntry
    pub model_catalog: HashMap<String, ModelCatalogEntry>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Node Resources
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Available RAM per node: node_name -> MB
    pub node_ram: HashMap<String, usize>,

    /// Available backends per node: node_name -> Vec<backend>
    pub node_backends: HashMap<String, Vec<String>>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Command Execution State
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Last command executed
    pub last_command: Option<String>,

    /// Last command exit code
    pub last_exit_code: Option<i32>,

    /// Last command stdout
    pub last_stdout: String,

    /// Last command stderr
    pub last_stderr: String,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // HTTP Request/Response State
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Last HTTP request sent
    pub last_http_request: Option<HttpRequest>,

    /// Last HTTP response received
    pub last_http_response: Option<HttpResponse>,

    /// SSE events received
    pub sse_events: Vec<SseEvent>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Inference State
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Tokens generated during inference
    pub tokens_generated: Vec<String>,

    /// Narration messages received
    pub narration_messages: Vec<NarrationMessage>,

    /// Inference metrics
    pub inference_metrics: Option<InferenceMetrics>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Error State
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Last error received
    pub last_error: Option<ErrorResponse>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Temporary Resources
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Temporary directory for test artifacts
    pub temp_dir: Option<tempfile::TempDir>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Process Management (TEAM-043)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Running queen-rbee process
    pub queen_rbee_process: Option<tokio::process::Child>,

    /// Running rbee-hive processes
    pub rbee_hive_processes: Vec<tokio::process::Child>,

    /// Running worker processes
    pub worker_processes: Vec<tokio::process::Child>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Supporting Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub hostname: String,
    pub components: Vec<String>,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BeehiveNode {
    pub node_name: String,
    pub ssh_host: String,
    pub ssh_port: u16,
    pub ssh_user: String,
    pub ssh_key_path: Option<String>,
    pub git_repo_url: String,
    pub git_branch: String,
    pub install_path: String,
    pub last_connected_unix: Option<i64>,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub state: String,
    pub backend: String,
    pub device: u32,
    pub slots_total: u32,
    pub slots_available: u32,
}

#[derive(Debug, Clone)]
pub struct ModelCatalogEntry {
    pub provider: String,
    pub reference: String,
    pub local_path: PathBuf,
    pub size_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
}

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
}

#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event_type: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct NarrationMessage {
    pub actor: String,
    pub action: String,
    pub human: String,
    pub cute: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    pub tokens_out: u32,
    pub decode_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

impl World {
    /// Clear all state for a fresh scenario
    pub fn clear(&mut self) {
        self.topology.clear();
        self.current_node = None;
        self.queen_rbee_url = None;
        self.model_catalog_path = None;
        self.registry_db_path = None;
        self.beehive_nodes.clear();
        self.workers.clear();
        self.model_catalog.clear();
        self.node_ram.clear();
        self.node_backends.clear();
        self.last_command = None;
        self.last_exit_code = None;
        self.last_stdout.clear();
        self.last_stderr.clear();
        self.last_http_request = None;
        self.last_http_response = None;
        self.sse_events.clear();
        self.tokens_generated.clear();
        self.narration_messages.clear();
        self.inference_metrics = None;
        self.last_error = None;
        self.temp_dir = None;
        self.queen_rbee_process = None;
        self.rbee_hive_processes.clear();
        self.worker_processes.clear();
    }
}

impl Drop for World {
    fn drop(&mut self) {
        // TEAM-043: Kill all running processes
        if let Some(mut proc) = self.queen_rbee_process.take() {
            let _ = proc.start_kill();
            tracing::debug!("Killed queen-rbee process");
        }

        for mut proc in self.rbee_hive_processes.drain(..) {
            let _ = proc.start_kill();
        }

        for mut proc in self.worker_processes.drain(..) {
            let _ = proc.start_kill();
        }

        tracing::debug!("World dropped, cleaning up resources");
    }
}
