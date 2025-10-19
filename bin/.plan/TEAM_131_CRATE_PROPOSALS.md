# TEAM-131: Detailed Crate Proposals

**Binary:** rbee-hive  
**Date:** 2025-10-19  
**Status:** Investigation Phase

---

## CRATE 1: rbee-hive-registry

**LOC:** 644  
**Risk:** Low  
**Priority:** High (foundation crate)

### Purpose
Thread-safe in-memory worker state management with health tracking, PID management, and force-kill capabilities.

### Source Files
- `src/registry.rs` → `src/lib.rs`

### Public API
```rust
// Core types
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}

pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub state: WorkerState,
    pub last_activity: SystemTime,
    pub slots_total: u32,
    pub slots_available: u32,
    pub failed_health_checks: u32,
    pub pid: Option<u32>,
    pub restart_count: u32,
    pub last_restart: Option<SystemTime>,
    pub last_heartbeat: Option<SystemTime>,
}

pub enum WorkerState {
    Loading,
    Idle,
    Busy,
}

// Core operations
impl WorkerRegistry {
    pub fn new() -> Self;
    pub async fn register(&self, worker: WorkerInfo);
    pub async fn update_state(&self, worker_id: &str, state: WorkerState);
    pub async fn get(&self, worker_id: &str) -> Option<WorkerInfo>;
    pub async fn list(&self) -> Vec<WorkerInfo>;
    pub async fn remove(&self, worker_id: &str) -> Option<WorkerInfo>;
    pub async fn find_idle_worker(&self, model_ref: &str) -> Option<WorkerInfo>;
    pub async fn get_idle_workers(&self) -> Vec<WorkerInfo>;
    pub async fn find_by_node_and_model(&self, node: &str, model_ref: &str) -> Option<WorkerInfo>;
    pub async fn clear(&self);
    
    // Health tracking
    pub async fn increment_failed_health_checks(&self, worker_id: &str) -> Option<u32>;
    pub async fn update_heartbeat(&self, worker_id: &str) -> bool;
    
    // Force-kill
    pub async fn force_kill_worker(&self, worker_id: &str) -> Result<bool, String>;
}
```

### Dependencies
```toml
[dependencies]
serde = { workspace = true, features = ["derive"] }
tokio = { workspace = true }
chrono = { version = "0.4", features = ["serde"] }
nix = { version = "0.27", features = ["signal", "process"] }
sysinfo = { workspace = true }
tracing = { workspace = true }
```

### Test Coverage
Current: ~60% (18 unit tests)  
Target: 85%+

**Add:**
- Property-based tests for concurrent operations
- Stress tests for high-churn scenarios
- Edge cases for force-kill logic

### Migration Notes
- No breaking changes to external APIs
- Already well-tested
- Used by: http, monitor, shutdown, timeout, cli

---

## CRATE 2: rbee-hive-http-server

**LOC:** 576  
**Risk:** Medium  
**Priority:** High

### Purpose
HTTP server and endpoints for worker/model management, health checks, metrics, and heartbeat.

### Source Files
```
src/http/workers.rs → src/workers.rs
src/http/models.rs → src/models.rs
src/http/health.rs → src/health.rs
src/http/heartbeat.rs → src/heartbeat.rs
src/http/metrics.rs → src/metrics_endpoint.rs
src/http/routes.rs → src/routes.rs
src/http/server.rs → src/server.rs
```

### Public API
```rust
// Server lifecycle
pub struct HttpServer {
    addr: SocketAddr,
    router: Router,
}

impl HttpServer {
    pub fn new(addr: SocketAddr, state: AppState) -> Self;
    pub async fn start(self) -> Result<()>;
    pub async fn shutdown(self) -> Result<()>;
}

// Route factory
pub fn create_router(state: AppState) -> Router;

// Shared state
pub struct AppState {
    pub registry: Arc<WorkerRegistry>,
    pub model_catalog: Arc<ModelCatalog>,
    pub provisioner: Arc<ModelProvisioner>,
    pub download_tracker: Arc<DownloadTracker>,
}

// Request/Response types
pub struct SpawnWorkerRequest {
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub model_path: String,
}

pub struct SpawnWorkerResponse {
    pub worker_id: String,
    pub url: String,
    pub state: String,
}

pub struct WorkerReadyRequest {
    pub worker_id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
}

pub struct ListWorkersResponse {
    pub workers: Vec<WorkerInfo>,
}
```

### Dependencies
```toml
[dependencies]
axum = { workspace = true }
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "cors"] }
tokio = { workspace = true, features = ["full"] }
serde = { workspace = true, features = ["derive"] }
serde_json = "1.0"
tracing = { workspace = true }
uuid = { workspace = true, features = ["v4"] }
reqwest = { workspace = true, features = ["json"] }
futures = { workspace = true }
async-stream = "0.3"

# Local crates
rbee-hive-registry = { path = "../rbee-hive-registry" }
rbee-hive-provisioner = { path = "../rbee-hive-provisioner" }
rbee-hive-metrics = { path = "../rbee-hive-metrics" }
rbee-hive-http-middleware = { path = "../rbee-hive-http-middleware" }

# Shared crates
input-validation = { path = "../../shared-crates/input-validation" }
model-catalog = { path = "../../shared-crates/model-catalog" }
```

### Coupling Analysis
**Tight coupling with:**
- `rbee-hive-registry` - Worker state (acceptable)
- `rbee-hive-provisioner` - Model provisioning (acceptable)
- `model-catalog` - Model metadata (shared crate, acceptable)

**Mitigation:** Keep coupling at interface level only

### Test Coverage
Current: ~40%  
Target: 75%+

**Add:**
- Integration tests for each endpoint
- Mock provisioner for testing
- SSE streaming tests

---

## CRATE 3: rbee-hive-http-middleware

**LOC:** 177  
**Risk:** Low  
**Priority:** Medium

### Purpose
HTTP middleware for authentication, CORS, and tracing.

### Source Files
```
src/http/middleware/auth.rs → src/auth.rs
```

### Public API
```rust
// Middleware builders
pub fn auth_middleware() -> Middleware;
pub fn cors_middleware() -> CorsLayer;
pub fn tracing_middleware() -> TraceLayer;

// Configuration
pub struct AuthConfig {
    pub enabled: bool,
    pub jwt_secret: Option<String>,
}

pub struct CorsConfig {
    pub allowed_origins: Vec<String>,
    pub allowed_methods: Vec<String>,
}
```

### Dependencies
```toml
[dependencies]
axum = { workspace = true }
tower = "0.5"
tower-http = { version = "0.6", features = ["trace", "cors"] }
tracing = { workspace = true }

# Shared crates
auth-min = { path = "../../shared-crates/auth-min" }
```

### Reusability
Can be used by:
- queen-rbee (future)
- rbee-keeper (future)
- Any HTTP server in the monorepo

---

## CRATE 4: rbee-hive-provisioner

**LOC:** 478  
**Risk:** Low  
**Priority:** High

### Purpose
Model download, provisioning, and progress tracking.

### Source Files
```
src/provisioner/operations.rs → src/operations.rs
src/provisioner/catalog.rs → src/catalog.rs
src/provisioner/download.rs → src/download.rs
src/provisioner/types.rs → src/types.rs
src/download_tracker.rs → src/tracker.rs
```

### Public API
```rust
pub struct ModelProvisioner {
    catalog: Arc<ModelCatalog>,
    download_tracker: Arc<DownloadTracker>,
}

impl ModelProvisioner {
    pub fn new(catalog: Arc<ModelCatalog>, tracker: Arc<DownloadTracker>) -> Self;
    pub async fn download_model(&self, reference: &str, provider: &str) -> Result<PathBuf>;
    pub async fn provision_model(&self, model_ref: &str) -> Result<ModelInfo>;
    pub fn get_model_size(&self, path: &Path) -> Result<u64>;
}

// Download tracking
pub struct DownloadTracker {
    downloads: Arc<RwLock<HashMap<String, broadcast::Sender<DownloadEvent>>>>,
}

impl DownloadTracker {
    pub fn new() -> Self;
    pub async fn start_download(&self) -> String;
    pub async fn send_progress(&self, download_id: &str, event: DownloadEvent) -> Result<()>;
    pub async fn subscribe(&self, download_id: &str) -> Option<broadcast::Receiver<DownloadEvent>>;
}

pub enum DownloadEvent {
    Downloading { bytes_downloaded: u64, bytes_total: u64, speed_mbps: f64 },
    Complete { local_path: String },
    Error { message: String },
}
```

### Dependencies
```toml
[dependencies]
tokio = { workspace = true }
reqwest = { workspace = true, features = ["stream"] }
serde = { workspace = true, features = ["derive"] }
uuid = { workspace = true, features = ["v4"] }
anyhow = "1.0"
tracing = { workspace = true }

# Shared crates
model-catalog = { path = "../../shared-crates/model-catalog" }
```

---

## CRATE 5: rbee-hive-monitor

**LOC:** 301  
**Risk:** Low  
**Priority:** Medium

### Purpose
Background health monitoring loop with fail-fast protocol and process liveness checks.

### Source Files
```
src/monitor.rs → src/lib.rs
```

### Public API
```rust
// Monitor loop (never returns)
pub async fn health_monitor_loop(registry: Arc<WorkerRegistry>) -> !;

// Configuration
pub struct MonitorConfig {
    pub check_interval_secs: u64,      // default: 30
    pub max_failed_checks: u32,         // default: 3
    pub ready_timeout_secs: u64,        // default: 30
    pub heartbeat_timeout_secs: u64,    // default: 60
}

// Helpers
pub fn should_restart_worker(worker: &WorkerInfo) -> bool;
pub fn force_kill_worker(pid: u32, worker_id: &str);
```

### Dependencies
```toml
[dependencies]
tokio = { workspace = true }
reqwest = { workspace = true, features = ["json"] }
sysinfo = { workspace = true }
tracing = { workspace = true }

# Local crates
rbee-hive-registry = { path = "../rbee-hive-registry" }
rbee-hive-restart = { path = "../rbee-hive-restart" }
```

---

## CRATE 6: rbee-hive-resources

**LOC:** 390  
**Risk:** Low  
**Priority:** Medium

### Purpose
System resource monitoring (memory, VRAM, disk) and limit enforcement.

### Source Files
```
src/resources.rs → src/lib.rs
```

### Public API
```rust
// Resource info
pub struct ResourceInfo {
    pub memory_total_bytes: u64,
    pub memory_available_bytes: u64,
    pub disk_total_bytes: u64,
    pub disk_available_bytes: u64,
}

pub fn get_resource_info() -> Result<ResourceInfo>;

// Memory limits
pub struct MemoryLimits {
    pub max_worker_memory_bytes: u64,  // default: 8GB
    pub min_free_memory_bytes: u64,    // default: 2GB
}

pub fn check_memory_available(required: u64, limits: &MemoryLimits) -> Result<()>;
pub fn check_worker_memory_usage(pid: u32, limit: u64) -> Result<bool>;

// VRAM limits
pub struct VramLimits {
    pub min_free_vram_bytes: u64,  // default: 1GB
}

pub fn check_vram_available(device: u32, required: u64, limits: &VramLimits) -> Result<()>;
pub fn estimate_model_vram_bytes(model_size: u64) -> u64;

// Disk limits
pub struct DiskLimits {
    pub min_free_disk_bytes: u64,  // default: 10GB
}

pub fn check_disk_space_available(required: u64, limits: &DiskLimits) -> Result<()>;
```

### Dependencies
```toml
[dependencies]
sysinfo = { workspace = true }
anyhow = "1.0"
serde = { workspace = true, features = ["derive"] }
tracing = { workspace = true }

# Shared crates
gpu-info = { path = "../../shared-crates/gpu-info" }
```

### Reusability
Pure functions, no state. Can be used by:
- queen-rbee (resource allocation)
- llm-worker-rbee (self-monitoring)
- rbee-keeper (cluster-wide resource tracking)

---

## CRATE 7: rbee-hive-shutdown

**LOC:** 349  
**Risk:** Medium  
**Priority:** High

### Purpose
Graceful shutdown orchestration with force-kill fallback and metrics.

### Source Files
```
src/shutdown.rs → src/lib.rs
```

### Public API
```rust
pub async fn shutdown_all_workers(
    registry: Arc<WorkerRegistry>,
    config: ShutdownConfig,
) -> ShutdownMetrics;

pub struct ShutdownConfig {
    pub graceful_timeout_secs: u64,     // default: 30
    pub http_timeout_secs: u64,         // default: 5
    pub force_kill_wait_secs: u64,      // default: 10
}

pub struct ShutdownMetrics {
    pub total_workers: usize,
    pub graceful_shutdown: usize,
    pub force_killed: usize,
    pub timeout_exceeded: usize,
    pub duration_secs: f64,
}
```

### Dependencies
```toml
[dependencies]
tokio = { workspace = true }
reqwest = { workspace = true, features = ["json"] }
nix = { version = "0.27", features = ["signal", "process"] }
anyhow = "1.0"
tracing = { workspace = true }

# Local crates
rbee-hive-registry = { path = "../rbee-hive-registry" }
rbee-hive-metrics = { path = "../rbee-hive-metrics" }
```

---

## CRATE 8: rbee-hive-metrics

**LOC:** 332  
**Risk:** Low  
**Priority:** Medium

### Purpose
Prometheus metrics collection and export.

### Source Files
```
src/metrics.rs → src/lib.rs
```

### Public API
```rust
// Metrics (lazy_static)
pub static WORKERS_BY_STATE: GaugeVec;
pub static WORKERS_FAILED_HEALTH: IntGauge;
pub static WORKERS_RESTART_COUNT: IntGauge;
pub static MODELS_DOWNLOADED_TOTAL: IntCounter;
pub static DOWNLOADS_ACTIVE: IntGauge;
pub static SHUTDOWN_DURATION_SECONDS: Histogram;
pub static WORKERS_GRACEFUL_SHUTDOWN_TOTAL: IntCounter;
pub static WORKERS_FORCE_KILLED_TOTAL: IntCounter;
pub static WORKER_RESTART_FAILURES_TOTAL: IntCounter;
pub static CIRCUIT_BREAKER_ACTIVATIONS_TOTAL: IntCounter;
pub static MEMORY_AVAILABLE_BYTES: IntGauge;
pub static DISK_AVAILABLE_BYTES: IntGauge;

// Update functions
pub async fn update_worker_metrics(registry: Arc<WorkerRegistry>);
pub async fn update_resource_metrics();
pub fn render_metrics() -> Result<String>;
```

### Dependencies
```toml
[dependencies]
prometheus = "0.13"
lazy_static = "1.4"
anyhow = "1.0"

# Local crates (for metric updates only)
rbee-hive-registry = { path = "../rbee-hive-registry", optional = true }
```

---

## CRATE 9: rbee-hive-restart

**LOC:** 280  
**Risk:** Low  
**Priority:** Low

### Purpose
Worker restart policy with exponential backoff, circuit breaker, and idle timeout.

### Source Files
```
src/restart.rs → src/lib.rs
src/timeout.rs → src/timeout.rs
```

### Public API
```rust
// Restart policy
pub struct RestartPolicy {
    pub max_attempts: u32,                      // default: 3
    pub base_backoff_secs: u64,                 // default: 1
    pub max_backoff_secs: u64,                  // default: 60
    pub circuit_breaker_threshold: u32,         // default: 5
    pub circuit_breaker_window_secs: u64,       // default: 300
    pub jitter_enabled: bool,                   // default: true
}

impl RestartPolicy {
    pub fn calculate_backoff(&self, attempt: u32) -> Duration;
    pub fn check_restart_allowed(
        &self,
        restart_count: u32,
        last_restart: Option<SystemTime>,
    ) -> Result<Duration>;
}

pub enum RestartError {
    MaxAttemptsExceeded(u32),
    CircuitBreakerOpen(u32, u64),
    BackoffNotElapsed(u64),
}

// Idle timeout
pub struct IdleTimeoutConfig {
    pub enabled: bool,
    pub timeout_secs: u64,  // default: 300
}

pub async fn idle_timeout_loop(
    registry: Arc<WorkerRegistry>,
    config: IdleTimeoutConfig,
);
```

### Dependencies
```toml
[dependencies]
tokio = { workspace = true }
thiserror = { workspace = true }
rand = "0.8"
```

### Reusability
Pure algorithm, no state. Can be reused by any worker manager.

---

## CRATE 10: rbee-hive-cli

**LOC:** 593  
**Risk:** Medium  
**Priority:** High

### Purpose
CLI commands and user interface layer.

### Source Files
```
src/cli.rs → src/cli.rs
src/commands/daemon.rs → src/commands/daemon.rs
src/commands/worker.rs → src/commands/worker.rs
src/commands/models.rs → src/commands/models.rs
src/commands/status.rs → src/commands/status.rs
src/commands/detect.rs → src/commands/detect.rs
src/worker_provisioner.rs → src/worker_provisioner.rs
```

### Public API
```rust
pub struct Cli {
    pub command: Commands,
}

pub enum Commands {
    Models { action: ModelsAction },
    Worker { action: WorkerAction },
    Status,
    Daemon { addr: String },
    Detect,
}

pub async fn handle_command(cli: Cli) -> Result<()>;

// Command handlers
pub mod commands {
    pub mod models;
    pub mod worker;
    pub mod daemon;
    pub mod status;
    pub mod detect;
}

// Worker spawning
pub async fn spawn_worker_process(
    backend: &str,
    model_path: &Path,
    device: u32,
    worker_url: &str,
    manager_url: &str,
    worker_id: &str,
    model_ref: &str,
) -> Result<u32>;  // Returns PID
```

### Dependencies
```toml
[dependencies]
clap = { version = "4.5", features = ["derive"] }
anyhow = "1.0"
colored = "2.0"
indicatif = { workspace = true }
tokio = { workspace = true, features = ["full"] }
serde_json = "1.0"
hostname = "0.4"

# All local crates (orchestration layer)
rbee-hive-registry = { path = "../rbee-hive-registry" }
rbee-hive-http-server = { path = "../rbee-hive-http-server" }
rbee-hive-provisioner = { path = "../rbee-hive-provisioner" }
rbee-hive-monitor = { path = "../rbee-hive-monitor" }
rbee-hive-resources = { path = "../rbee-hive-resources" }
rbee-hive-shutdown = { path = "../rbee-hive-shutdown" }
rbee-hive-metrics = { path = "../rbee-hive-metrics" }
rbee-hive-restart = { path = "../rbee-hive-restart" }

# Shared crates
model-catalog = { path = "../../shared-crates/model-catalog" }
gpu-info = { path = "../../shared-crates/gpu-info" }
```

---

## SUMMARY TABLE

| Crate | LOC | Dependencies (Local) | Risk | Test Target |
|-------|-----|---------------------|------|-------------|
| registry | 644 | None | Low | 85% |
| http-server | 576 | registry, provisioner, metrics, middleware | Medium | 75% |
| http-middleware | 177 | None | Low | 80% |
| provisioner | 478 | None | Low | 80% |
| monitor | 301 | registry, restart | Low | 75% |
| resources | 390 | None | Low | 85% |
| shutdown | 349 | registry, metrics | Medium | 80% |
| metrics | 332 | registry (optional) | Low | 70% |
| restart | 280 | None | Low | 90% |
| cli | 593 | ALL | Medium | 65% |

**Total:** 4,120 LOC in libraries + ~100 LOC thin binary wrapper
