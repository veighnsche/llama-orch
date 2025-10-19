// World state for BDD tests
// Created by: TEAM-040
// Modified by: TEAM-061 (added HTTP client factory with timeouts)
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-064 (added explicit warning preservation notice)
// Modified by: TEAM-099 (added chrono import for deadline propagation tests)

use rbee_hive::registry::WorkerRegistry;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

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

// TEAM-080: Wrapper for queen_rbee::WorkerRegistry to implement Debug
pub struct DebugQueenRegistry(queen_rbee::WorkerRegistry);

impl std::fmt::Debug for DebugQueenRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueenWorkerRegistry").finish_non_exhaustive()
    }
}

impl DebugQueenRegistry {
    pub fn new() -> Self {
        Self(queen_rbee::WorkerRegistry::new())
    }

    pub fn inner(&self) -> &queen_rbee::WorkerRegistry {
        &self.0
    }
}

#[derive(cucumber::World)]
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

    /// rbee-hive URL (TEAM-085: Added for resource management tests)
    pub rbee_hive_url: Option<String>,

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

    /// Last worker ID (TEAM-098: For PID tracking tests)
    pub last_worker_id: Option<String>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Test Action Tracking (TEAM-078)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Last action performed (for step tracking)
    pub last_action: Option<String>,

    // TEAM-085: Narration capture
    pub narration_enabled: bool,
    pub last_narration: Option<Vec<observability_narration_core::CapturedNarration>>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Concurrency Testing (TEAM-080)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// queen-rbee worker registry for concurrency tests
    #[allow(dead_code)]
    pub queen_registry: Option<DebugQueenRegistry>,

    /// Concurrent operation results
    pub concurrent_results: Vec<Result<String, String>>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Concurrency Testing Extensions (TEAM-081)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Concurrent task handles for async operations
    pub concurrent_handles: Vec<tokio::task::JoinHandle<bool>>,

    /// Active request ID for tracking
    pub active_request_id: Option<String>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // P0 Security Testing (TEAM-097)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Authentication enabled flag
    pub auth_enabled: bool,

    /// Expected API token for auth tests
    pub expected_token: Option<String>,

    /// Queen-rbee URL for auth tests
    pub queen_url: Option<String>,

    /// rbee-hive URL for validation tests
    pub hive_url: Option<String>,

    /// llm-worker-rbee URL for auth tests
    pub worker_url: Option<String>,

    /// Last HTTP status code
    pub last_status_code: Option<u16>,

    /// Last HTTP response body
    pub last_response_body: Option<String>,

    /// Last HTTP response headers
    pub last_response_headers: Option<reqwest::header::HeaderMap>,

    /// Last error message
    pub last_error_message: Option<String>,

    /// Last request body
    pub last_request_body: Option<String>,

    /// Timing measurements for auth tests
    pub timing_measurements: Option<Vec<Duration>>,

    /// Timing measurements for invalid tokens
    pub timing_measurements_invalid: Option<Vec<Duration>>,

    /// Bind address for queen-rbee
    pub bind_address: Option<String>,

    /// Process started flag
    pub process_started: bool,

    /// Queen token for multi-component tests
    pub queen_token: Option<String>,

    /// Hive token for multi-component tests
    pub hive_token: Option<String>,

    /// Secret file path for secrets tests
    pub secret_file_path: Option<String>,

    /// File permissions for secrets tests
    pub file_permissions: Option<String>,

    /// Last config for secrets tests
    pub last_config: Option<String>,

    /// Systemd credential path
    pub systemd_credential_path: Option<String>,

    /// Systemd credential name
    pub systemd_credential_name: Option<String>,

    /// Hive token file path
    pub hive_token_file: Option<String>,

    /// Worker token file path
    pub worker_token_file: Option<String>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // P1 Audit Logging (TEAM-099)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Audit logging enabled flag
    pub audit_enabled: bool,

    /// Audit log entries (JSON objects)
    pub audit_log_entries: Vec<serde_json::Value>,

    /// Last audit entry hash
    pub audit_last_hash: Option<String>,

    /// Audit log file path
    pub audit_log_path: Option<PathBuf>,

    /// Tampered audit entry number
    pub audit_tampered_entry: Option<usize>,

    /// Audit log rotation size in MB
    pub audit_rotation_size_mb: Option<usize>,

    /// Current audit log size in MB
    pub audit_current_size_mb: Option<usize>,

    /// Audit log rotated flag
    pub audit_rotated: bool,

    /// Audit directory free space in MB
    pub audit_free_space_mb: Option<usize>,

    /// Audit logs consumed space in MB
    pub audit_consumed_mb: Option<usize>,

    /// Last warning message
    pub last_warning: Option<String>,

    /// Process restarted flag
    pub process_restarted: bool,

    /// Last model reference
    pub last_model_ref: Option<String>,

    /// Last node name
    pub last_node: Option<String>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // P1 Deadline Propagation (TEAM-099)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Request timeout in seconds
    pub request_timeout_secs: Option<u64>,

    /// Request deadline timestamp
    pub request_deadline: Option<chrono::DateTime<chrono::Utc>>,

    /// Queen-rbee received request flag
    pub queen_received_request: bool,

    /// Queen-rbee calculated deadline
    pub queen_deadline: Option<chrono::DateTime<chrono::Utc>>,

    /// Hive received deadline
    pub hive_received_deadline: Option<chrono::DateTime<chrono::Utc>>,

    /// Worker received deadline
    pub worker_received_deadline: Option<chrono::DateTime<chrono::Utc>>,

    /// Worker processing duration
    pub worker_processing_duration: Option<Duration>,

    /// Deadline exceeded flag
    pub deadline_exceeded: bool,

    /// Deadline exceeded at duration
    pub deadline_exceeded_at: Option<Duration>,

    /// Queen cancelled request flag
    pub queen_cancelled_request: bool,

    /// Hive received cancellation flag
    pub hive_received_cancellation: bool,

    /// Worker received cancellation flag
    pub worker_received_cancellation: bool,

    /// Worker stopped flag
    pub worker_stopped: bool,

    /// Worker spawned flag
    pub worker_spawned: bool,

    /// Hive received request flag
    pub hive_received_request: bool,

    /// Worker received request flag
    pub worker_received_request: bool,

    /// Last request headers
    pub last_request_headers: HashMap<String, String>,

    /// Last response content type
    pub last_response_content_type: Option<String>,

    /// Last error code
    pub last_error_code: Option<String>,

    /// Worker processing flag
    pub worker_processing: bool,

    /// Worker stopped token generation flag
    pub worker_stopped_tokens: bool,

    /// Worker released GPU flag
    pub worker_released_gpu: bool,

    /// Worker slot available flag
    pub worker_slot_available: bool,

    /// Last worker event
    pub last_worker_event: Option<String>,

    /// Malicious deadline attempt
    pub malicious_deadline_attempt: Option<chrono::DateTime<chrono::Utc>>,

    /// Deadline extension rejected flag
    pub deadline_extension_rejected: bool,

    /// Expected timeout in seconds
    pub expected_timeout_secs: Option<u64>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // P2 Metrics & Observability (TEAM-100 - THE CENTENNIAL TEAM! 💯🎉)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Narration capture adapter    // TEAM-100: Narration verification (observability-narration-core)
    pub narration_adapter: Option<observability_narration_core::CaptureAdapter>,

    // TEAM-106: Integration testing fields
    pub integration_env_ready: bool,
    pub request_start_time: Option<std::time::Instant>,
    pub active_requests: Vec<String>,
    pub registered_workers: Vec<String>,
    pub auth_required: bool,
    pub auth_token: Option<String>,

    /// Pool-managerd URL
    pub pool_managerd_url: Option<String>,

    /// Metrics enabled flag
    pub metrics_enabled: bool,

    /// Last response status
    pub last_response_status: Option<u16>,

    /// Request count for metrics
    pub request_count: usize,

    /// Correlation ID for request tracking
    pub correlation_id: Option<String>,

    /// Environment variable overrides
    pub env_overrides: HashMap<String, String>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // P2 Configuration Management (TEAM-100)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Config file path
    pub config_file_path: Option<String>,

    /// Config file content
    pub config_content: Option<String>,

    /// Config valid flag
    pub config_valid: bool,

    /// Config loaded flag
    pub config_loaded: bool,

    /// Config validation passed flag
    pub config_validation_passed: bool,

    /// Config validation error
    pub config_validation_error: Option<String>,

    /// Config values
    pub config_values: HashMap<String, String>,

    /// Config updated flag
    pub config_updated: bool,

    /// Config reloaded flag
    pub config_reloaded: bool,

    /// Pool-managerd running flag
    pub pool_managerd_running: bool,

    /// Startup failed flag
    pub startup_failed: bool,

    /// Exit code
    pub exit_code: Option<i32>,

    /// Example config path
    pub example_config_path: Option<String>,

    /// Example config validated flag
    pub example_config_validated: bool,

    /// Config has secrets flag
    pub config_has_secrets: bool,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEAM-105: Cascading Shutdown Testing (LIFE-007, LIFE-008, LIFE-009)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Worker count for shutdown tests
    pub worker_count: Option<u32>,

    /// Shutdown start time for performance measurement
    pub shutdown_start_time: Option<std::time::Instant>,

    /// Number of responsive workers (respond to shutdown within timeout)
    pub responsive_workers: Option<u32>,

    /// Number of unresponsive workers (require force-kill)
    pub unresponsive_workers: Option<u32>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEAM-101: Worker PID Tracking & Force-Kill Testing (LIFE-001 to LIFE-015)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Last worker PID for tracking
    pub last_worker_pid: Option<u32>,

    /// Worker timeout in seconds
    pub worker_timeout: Option<u64>,

    /// Force-killed worker PID
    pub force_killed_pid: Option<u32>,

    /// Worker hung flag
    pub worker_hung: bool,

    /// Health check performed flag
    pub health_check_performed: bool,

    /// Ready timeout in seconds
    pub ready_timeout: Option<u64>,

    /// Worker crashed flag
    pub worker_crashed: bool,

    /// Worker zombie flag
    pub worker_zombie: bool,

    /// Worker responded flag
    pub worker_responded: bool,

    /// Worker response time in seconds
    pub worker_response_time: Option<u64>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEAM-118: Missing Step Fields (Batch 1)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// SSH connection timeout in seconds
    pub ssh_timeout: Option<u64>,

    /// Worker PIDs: worker_id -> PID
    pub worker_pids: HashMap<String, u32>,

    /// Worker slots configuration
    pub worker_slots: Option<usize>,

    /// Validation passed flag
    pub validation_passed: bool,

    /// Target node for requests
    pub target_node: Option<String>,

    /// Worker heartbeat T0 timestamp
    pub worker_heartbeat_t0: Option<std::time::SystemTime>,

    /// Catalog queried flag
    pub catalog_queried: bool,

    /// Worker busy flag
    pub worker_busy: bool,

    /// Worker accepting requests flag
    pub worker_accepting_requests: bool,

    /// Backup path
    pub backup_path: Option<String>,

    /// rbee-keeper configuration
    pub keeper_config: Option<String>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEAM-119: Missing Step Fields (Batch 2)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// GPU VRAM free per device: device_id -> bytes
    pub gpu_vram_free: HashMap<u32, u64>,

    /// System RAM available in bytes
    pub system_ram_available: Option<u64>,

    /// GPU temperature in Celsius
    pub gpu_temperature: Option<i32>,

    /// CPU cores count
    pub cpu_cores: Option<usize>,

    /// GPU total VRAM in bytes
    pub gpu_vram_total: Option<u64>,

    /// Bandwidth limit in bytes per second
    pub bandwidth_limit: Option<u64>,

    /// Disk I/O capacity percentage
    pub disk_io_percent: Option<u8>,

    /// Base URL for requests
    pub base_url: Option<String>,

    /// API token for authentication
    pub api_token: Option<String>,

    /// File readable by world flag
    pub file_readable_by_world: bool,

    /// File readable by group flag
    pub file_readable_by_group: bool,

    /// Queen started with config flag
    pub queen_started_with_config: bool,

    /// Queen request count
    pub queen_request_count: Option<usize>,

    /// Log messages
    pub log_messages: Vec<String>,

    /// File content checked flag
    pub file_content_checked: bool,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEAM-121: Missing Step Fields (Batch 4)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Model being downloaded by provisioner
    pub downloading_model: Option<String>,

    /// Health check interval in seconds
    pub health_check_interval: Option<u64>,

    /// Workers have different models flag
    pub workers_have_different_models: bool,

    /// pool-managerd narration enabled
    pub pool_managerd_narration: bool,

    /// pool-managerd cute mode enabled
    pub pool_managerd_cute_mode: bool,

    /// queen-rbee requested metrics flag
    pub queen_requested_metrics: bool,

    /// Narration has source_location field
    pub narration_has_source_location: bool,

    /// Narration redaction string
    pub narration_redaction: Option<String>,

    /// Worker state (idle, busy, etc.)
    pub worker_state: Option<String>,

    /// Registry database available flag
    pub registry_available: bool,

    /// Crash detected flag
    pub crash_detected: bool,

    /// Concurrent registrations count
    pub concurrent_registrations: Option<usize>,

    /// Concurrent requests count
    pub concurrent_requests: Option<usize>,

    /// queen-rbee restarted flag
    pub queen_restarted: bool,

    /// rbee-hive restarted flag
    pub hive_restarted: bool,

    /// Inference duration in minutes
    pub inference_duration: Option<u64>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEAM-120: Missing Step Fields (Batch 3)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Queen-rbee started flag
    pub queen_started: bool,

    /// Unwrap search performed flag
    pub unwrap_search_performed: bool,

    /// rbee-hive crashed flag
    pub hive_crashed: bool,

    /// Log has correlation_id flag
    pub log_has_correlation_id: bool,

    /// Audit has token fingerprint flag
    pub audit_has_token_fingerprint: bool,

    /// Hash chain valid flag
    pub hash_chain_valid: bool,

    /// Audit fields list
    pub audit_fields: Vec<String>,

    /// Warning messages list
    pub warning_messages: Vec<String>,

    /// pool-managerd has GPU workers flag
    pub pool_managerd_has_gpu: bool,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEAM-129: Error Handling Fields
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Code analyzed flag (for unwrap detection)
    pub code_analyzed: bool,

    /// Number of unwrap() calls found in production code
    pub unwrap_calls_found: usize,

    /// Code scan completed flag
    pub code_scan_completed: bool,

    /// Error occurred flag
    pub error_occurred: bool,

    /// Worker processing inference flag
    pub worker_processing_inference: bool,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEAM-127: CLI Commands Testing
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// Config files created during tests
    pub config_files: Vec<String>,

    /// Environment variables set during tests
    pub env_vars: HashMap<String, String>,

    /// Remote node for SSH commands
    pub remote_node: Option<String>,

    /// Remote command executed flag
    pub remote_command_executed: bool,

    /// SSH connections: node -> connected
    pub ssh_connections: HashMap<String, bool>,

    /// Installation path for binaries
    pub install_path: Option<String>,

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEAM-129: Lifecycle Management Fields
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    /// rbee-hive daemon running flag
    pub hive_daemon_running: bool,

    /// rbee-hive port
    pub hive_port: Option<u16>,

    /// rbee-hive persistent daemon flag
    pub hive_persistent: bool,

    /// rbee-keeper spawn configured flag
    pub keeper_spawn_configured: bool,

    /// rbee-hive already running flag
    pub hive_already_running: bool,

    /// rbee-hive manually started flag
    pub hive_manually_started: bool,

    /// Spawned by rbee-keeper flag
    pub spawned_by_keeper: bool,

    /// Worker ready flag
    pub worker_ready: bool,

    /// Elapsed seconds
    pub elapsed_seconds: Option<u64>,

    /// Idle minutes
    pub idle_minutes: Option<u64>,

    /// No requests received flag
    pub no_requests_received: bool,

    /// SIGTERM sent flag
    pub sigterm_sent: bool,

    /// Inference completed flag
    pub inference_completed: bool,

    /// Inference started flag
    pub inference_started: bool,

    /// rbee-hive exited flag
    pub hive_exited: bool,

    /// Idle timeout minutes
    pub idle_timeout_minutes: Option<u64>,

    /// Worker timeout reached flag
    pub worker_timeout_reached: bool,

    /// rbee-hive accepting requests flag
    pub hive_accepting_requests: bool,

    /// Worker running flag
    pub worker_running: bool,
}

// TEAM-101: Manual Debug implementation to handle CaptureAdapter which doesn't implement Debug
impl std::fmt::Debug for World {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("World")
            .field("topology", &self.topology)
            .field("current_node", &self.current_node)
            .field(
                "narration_adapter",
                &self.narration_adapter.as_ref().map(|_| "<CaptureAdapter>"),
            )
            .finish_non_exhaustive()
    }
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
    // TEAM-118: Added for capabilities tracking
    pub capabilities: Vec<String>,
}

impl Default for WorkerInfo {
    fn default() -> Self {
        Self {
            id: String::new(),
            url: String::new(),
            model_ref: String::new(),
            state: "idle".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            slots_total: 1,
            slots_available: 1,
            capabilities: Vec::new(),
        }
    }
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
            rbee_hive_url: None, // TEAM-085: Added for resource management tests
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
            last_worker_id: None, // TEAM-098: PID tracking tests
            last_action: None,

            // TEAM-085: Narration
            narration_enabled: false,
            last_narration: None,           // TEAM-078: Action tracking
            queen_registry: None,           // TEAM-080: Concurrency testing
            concurrent_results: Vec::new(), // TEAM-080: Concurrent operation results
            concurrent_handles: Vec::new(), // TEAM-081: Concurrent task handles
            active_request_id: None,        // TEAM-081: Active request tracking

            // TEAM-097: P0 Security Testing
            auth_enabled: false,
            expected_token: None,
            queen_url: None,
            hive_url: None,
            worker_url: None,
            last_status_code: None,
            last_response_body: None,
            last_response_headers: None,
            last_error_message: None,
            last_request_body: None,
            timing_measurements: None,
            timing_measurements_invalid: None,
            bind_address: None,
            process_started: false,
            queen_token: None,
            hive_token: None,
            secret_file_path: None,
            file_permissions: None,
            last_config: None,
            systemd_credential_path: None,
            systemd_credential_name: None,
            hive_token_file: None,
            worker_token_file: None,

            // TEAM-099: P1 Audit Logging
            audit_enabled: false,
            audit_log_entries: Vec::new(),
            audit_last_hash: None,
            audit_log_path: None,
            audit_tampered_entry: None,
            audit_rotation_size_mb: None,
            audit_current_size_mb: None,
            audit_rotated: false,
            audit_free_space_mb: None,
            audit_consumed_mb: None,
            last_warning: None,
            process_restarted: false,
            last_model_ref: None,
            last_node: None,

            // TEAM-099: P1 Deadline Propagation
            request_timeout_secs: None,
            request_deadline: None,
            queen_received_request: false,
            queen_deadline: None,
            hive_received_deadline: None,
            worker_received_deadline: None,
            worker_processing_duration: None,
            deadline_exceeded: false,
            deadline_exceeded_at: None,
            queen_cancelled_request: false,
            hive_received_cancellation: false,
            worker_received_cancellation: false,
            worker_stopped: false,
            worker_spawned: false,
            hive_received_request: false,
            worker_received_request: false,
            last_request_headers: HashMap::new(),
            last_response_content_type: None,
            last_error_code: None,
            worker_processing: false,
            worker_stopped_tokens: false,
            worker_released_gpu: false,
            worker_slot_available: false,
            last_worker_event: None,
            malicious_deadline_attempt: None,
            deadline_extension_rejected: false,
            expected_timeout_secs: None,

            // TEAM-100: P2 Metrics & Observability
            narration_adapter: None,
            pool_managerd_url: None,
            metrics_enabled: false,
            last_response_status: None,
            request_count: 0,
            correlation_id: None,
            env_overrides: HashMap::new(),

            // TEAM-100: P2 Configuration Management
            config_file_path: None,
            config_content: None,
            config_valid: false,
            config_loaded: false,
            config_validation_passed: false,
            config_validation_error: None,
            config_values: HashMap::new(),
            config_updated: false,
            config_reloaded: false,
            pool_managerd_running: false,
            startup_failed: false,
            exit_code: None,
            example_config_path: None,
            example_config_validated: false,
            config_has_secrets: false,

            // TEAM-105: Cascading Shutdown Testing
            worker_count: None,
            shutdown_start_time: None,
            responsive_workers: None,
            unresponsive_workers: None,

            // TEAM-101: Worker PID Tracking & Force-Kill Testing
            last_worker_pid: None,
            worker_timeout: None,
            force_killed_pid: None,
            worker_hung: false,
            health_check_performed: false,
            ready_timeout: None,
            worker_crashed: false,
            worker_zombie: false,
            worker_responded: false,
            worker_response_time: None,

            // TEAM-106: Integration testing
            integration_env_ready: false,
            request_start_time: None,
            active_requests: Vec::new(),
            registered_workers: Vec::new(),
            auth_required: false,
            auth_token: None,

            // TEAM-118: Missing Step Fields (Batch 1)
            ssh_timeout: None,
            worker_pids: HashMap::new(),
            worker_slots: None,
            validation_passed: false,
            target_node: None,
            worker_heartbeat_t0: None,
            catalog_queried: false,
            worker_busy: false,
            worker_accepting_requests: true,
            backup_path: None,
            keeper_config: None,

            // TEAM-119: Missing Step Fields (Batch 2)
            gpu_vram_free: HashMap::new(),
            system_ram_available: None,
            gpu_temperature: None,
            cpu_cores: None,
            gpu_vram_total: None,
            bandwidth_limit: None,
            disk_io_percent: None,
            base_url: None,
            api_token: None,
            file_readable_by_world: false,
            file_readable_by_group: false,
            queen_started_with_config: false,
            queen_request_count: None,
            log_messages: Vec::new(),
            file_content_checked: false,

            // TEAM-121: Missing Step Fields (Batch 4)
            downloading_model: None,
            health_check_interval: None,
            workers_have_different_models: false,
            pool_managerd_narration: false,
            pool_managerd_cute_mode: false,
            queen_requested_metrics: false,
            narration_has_source_location: false,
            narration_redaction: None,
            worker_state: None,
            registry_available: true,
            crash_detected: false,
            concurrent_registrations: None,
            concurrent_requests: None,
            queen_restarted: false,
            hive_restarted: false,
            inference_duration: None,

            // TEAM-120: Missing Step Fields (Batch 3)
            queen_started: false,
            unwrap_search_performed: false,
            hive_crashed: false,
            log_has_correlation_id: false,
            audit_has_token_fingerprint: false,
            hash_chain_valid: false,
            audit_fields: Vec::new(),
            warning_messages: Vec::new(),
            pool_managerd_has_gpu: false,
            worker_processing_inference: false,

            // TEAM-129: Error Handling Fields
            code_analyzed: false,
            unwrap_calls_found: 0,
            code_scan_completed: false,
            error_occurred: false,

            // TEAM-127: CLI Commands Testing
            config_files: Vec::new(),
            env_vars: HashMap::new(),
            remote_node: None,
            remote_command_executed: false,
            ssh_connections: HashMap::new(),
            install_path: None,

            // TEAM-129: Lifecycle Management Fields
            hive_daemon_running: false,
            hive_port: None,
            hive_persistent: false,
            keeper_spawn_configured: false,
            hive_already_running: false,
            hive_manually_started: false,
            spawned_by_keeper: false,
            worker_ready: false,
            elapsed_seconds: None,
            idle_minutes: None,
            no_requests_received: false,
            sigterm_sent: false,
            inference_completed: false,
            inference_started: false,
            hive_exited: false,
            idle_timeout_minutes: None,
            worker_timeout_reached: false,
            hive_accepting_requests: false,
            worker_running: false,
        }
    }
}

impl World {
    /// TEAM-121: Check if a service is available
    pub async fn check_service_available(&self, url: &str) -> bool {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();
        
        match client.get(url).send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }
    
    /// TEAM-121: Skip test if services not available
    pub async fn require_services(&self) -> Result<(), String> {
        let queen_available = self.check_service_available("http://localhost:8080/health").await;
        let hive_available = self.check_service_available("http://localhost:8081/health").await;
        
        if !queen_available || !hive_available {
            return Err("Services not available - skipping integration test".to_string());
        }
        
        Ok(())
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

    /// TEAM-082: Reset state for a fresh scenario (comprehensive cleanup)
    pub fn reset_for_scenario(&mut self) {
        self.concurrent_handles.clear();
        self.concurrent_results.clear();
        self.active_request_id = None;
        self.sse_events.clear();
        self.tokens_generated.clear();
        self.last_http_response = None;
        self.last_http_status = None;
        self.last_error = None;
        self.start_time = None;

        tracing::debug!("TEAM-082: World state reset for new scenario");
    }

    /// Clear all state for a fresh scenario
    pub fn clear(&mut self) {
        self.topology.clear();
        self.current_node = None;
        self.queen_rbee_url = None;
        self.rbee_hive_url = None; // TEAM-085: Clear rbee-hive URL
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
        self.last_action = None; // TEAM-078: Clear action tracking
    }

    /// TEAM-100: Get or create correlation ID for request tracking
    pub fn get_or_create_correlation_id(&mut self) -> String {
        if let Some(id) = &self.correlation_id {
            id.clone()
        } else {
            let id = format!("req-test-{}", uuid::Uuid::new_v4());
            self.correlation_id = Some(id.clone());
            id
        }
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
        .timeout(Duration::from_secs(10)) // Total request timeout
        .connect_timeout(Duration::from_secs(5)) // Connection timeout
        .build()
        .expect("Failed to create HTTP client with timeouts")
}
