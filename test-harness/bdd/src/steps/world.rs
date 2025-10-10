// World state for BDD tests
// Created by: TEAM-040
// Modified by: TEAM-061 (added HTTP client factory with timeouts)
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: MUST import and test REAL product code from /bin/
// ⚠️ DO NOT use mock servers - wire up actual rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-064 (added explicit warning preservation notice)

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use rbee_hive::registry::WorkerRegistry;

// TEAM-064: Wrapper for WorkerRegistry to implement Debug
pub struct DebugWorkerRegistry(WorkerRegistry);

impl std::fmt::Debug for DebugWorkerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerRegistry").finish_non_exhaustive()
    }
}

impl DebugWorkerRegistry {
    pub fn new() -> Self {
        Self(WorkerRegistry::new())
    }
    
    pub fn inner_mut(&mut self) -> &mut WorkerRegistry {
        &mut self.0
    }
}

#[derive(Debug, cucumber::World)]
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

    /// Last HTTP response received (body as String)
    /// TEAM-058: Changed from HttpResponse to Option<String> for simpler access
    pub last_http_response: Option<String>,
    
    /// Last HTTP status code
    /// TEAM-058: Added for status code tracking
    pub last_http_status: Option<u16>,

    /// SSE events received
    pub sse_events: Vec<SseEvent>,
    
    /// Start time for latency tracking
    /// TEAM-058: Added for latency verification
    pub start_time: Option<std::time::Instant>,

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

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Product Integration (TEAM-063)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// rbee-hive worker registry (actual product code)
    pub hive_registry: Option<DebugWorkerRegistry>,
    
    /// Next available port for workers
    pub next_worker_port: u16,
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

impl Default for World {
    fn default() -> Self {
        Self {
            topology: HashMap::new(),
            current_node: None,
            queen_rbee_url: None,
            model_catalog_path: None,
            registry_db_path: None,
            beehive_nodes: HashMap::new(),
            workers: HashMap::new(),
            model_catalog: HashMap::new(),
            node_ram: HashMap::new(),
            node_backends: HashMap::new(),
            last_command: None,
            last_exit_code: None,
            last_stdout: String::new(),
            last_stderr: String::new(),
            last_http_request: None,
            last_http_response: None,
            last_http_status: None,
            sse_events: Vec::new(),
            start_time: None,
            tokens_generated: Vec::new(),
            narration_messages: Vec::new(),
            inference_metrics: None,
            last_error: None,
            temp_dir: None,
            queen_rbee_process: None,
            rbee_hive_processes: Vec::new(),
            worker_processes: Vec::new(),
            hive_registry: Some(DebugWorkerRegistry::new()),
            next_worker_port: 8001,
        }
    }
}

impl World {
    /// Get or create the hive registry
    pub fn hive_registry(&mut self) -> &mut WorkerRegistry {
        if self.hive_registry.is_none() {
            self.hive_registry = Some(DebugWorkerRegistry::new());
        }
        self.hive_registry.as_mut().unwrap().inner_mut()
    }

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
        self.last_http_status = None;
        self.sse_events.clear();
        self.start_time = None;
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
        // TEAM-051: DON'T kill queen-rbee - it's a shared global instance
        // Only kill scenario-specific processes (rbee-hive, workers)
        // TEAM-061: Enhanced cleanup with timeout to prevent hangs
        
        if let Some(_proc) = self.queen_rbee_process.take() {
            // Just drop the reference - the global instance will be cleaned up at the end
            tracing::debug!("Released reference to global queen-rbee");
        }

        for mut proc in self.rbee_hive_processes.drain(..) {
            let _ = proc.start_kill();
        }

        for mut proc in self.worker_processes.drain(..) {
            let _ = proc.start_kill();
        }

        // TEAM-061: Give processes time to die, but don't hang forever
        std::thread::sleep(Duration::from_millis(500));

        tracing::debug!("World dropped, cleaning up scenario-specific resources");
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HTTP Client Factory (TEAM-061)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// TEAM-061: Create HTTP client with aggressive timeouts to prevent hangs
/// 
/// All HTTP requests in BDD tests MUST use this client factory.
/// - Total request timeout: 10 seconds
/// - Connection timeout: 5 seconds
/// 
/// This prevents tests from hanging indefinitely when servers don't respond.
pub fn create_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(10))        // Total request timeout
        .connect_timeout(Duration::from_secs(5)) // Connection timeout
        .build()
        .expect("Failed to create HTTP client with timeouts")
}
