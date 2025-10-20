# Missing Crates Analysis

**Date:** 2025-10-19  
**Source:** Comparison of happy flow requirements vs. current implementation  
**Reference Documents:**
- `/home/vince/Projects/llama-orch/bin/a_human_wrote_this.md` (authoritative)
- `/home/vince/Projects/llama-orch/bin/a_chatGPT_5_refined_this.md` (refined)
- `/home/vince/Projects/llama-orch/bin/TECHNICAL_SUMMARY.md` (current state)

---

## 🎯 Executive Summary

Based on the happy flow requirements, **7 critical crates are missing** from the current implementation:

| # | Missing Crate | Owner Binary | Priority | LOC Estimate |
|---|---------------|--------------|----------|--------------|
| 1 | **polling** | rbee-keeper | 🔴 CRITICAL | ~80 |
| 2 | **health** | queen-rbee | 🔴 CRITICAL | ~60 |
| 3 | **hive-catalog** | queen-rbee | 🔴 CRITICAL | ~200 |
| 4 | **scheduler** | queen-rbee | 🔴 CRITICAL | ~150 |
| 5 | **vram-checker** | rbee-hive | 🔴 CRITICAL | ~100 |
| 6 | **worker-catalog** | rbee-hive | 🔴 CRITICAL | ~180 |
| 7 | **sse-relay** | All binaries | 🟡 HIGH | ~120 |

**Total Missing:** ~890 LOC across 7 crates

---

## 📋 Detailed Analysis by Binary

### 1. rbee-keeper Missing Crates

#### 1.1 polling (CRITICAL)

**Location:** `bin/05_rbee_keeper_crates/polling/`  
**Package Name:** `rbee-keeper-polling`

**Required By Happy Flow:**
- Line 15: "the bee keeper polls the queen until she gives a healthy sign"
- Line 17: "when the bee keeper successfully polls a pong to its ping"

**Purpose:**
- Poll queen-rbee health endpoint until ready
- Configurable retry logic (interval, max attempts, timeout)
- Exponential backoff support

**API:**
```rust
pub struct HealthPoller {
    target_url: String,
    config: PollingConfig,
}

pub struct PollingConfig {
    pub interval_ms: u64,
    pub max_attempts: u32,
    pub timeout_ms: u64,
    pub exponential_backoff: bool,
}

impl HealthPoller {
    pub async fn poll_until_healthy(&self) -> Result<()>;
    pub async fn check_once(&self) -> Result<bool>;
}
```

**Dependencies:**
- `tokio` (async runtime)
- `reqwest` (HTTP client)
- `rbee-http-client` (shared)

**Estimated LOC:** ~80

---

### 2. queen-rbee Missing Crates

#### 2.1 health (CRITICAL)

**Location:** `bin/15_queen_rbee_crates/health/`  
**Package Name:** `queen-rbee-health`

**Required By Happy Flow:**
- Line 15: "queen bee health crate is missing"
- Line 16: "The queen bee wakes up and immediately starts the http server"
- Line 17: "when the bee keeper successfully polls a pong to its ping"

**Purpose:**
- Health check endpoint (`/health`)
- Readiness check (HTTP server started)
- Liveness check (system operational)

**API:**
```rust
pub struct HealthCheck {
    state: Arc<RwLock<HealthState>>,
}

pub enum HealthState {
    Starting,
    Ready,
    Degraded,
    Unhealthy,
}

impl HealthCheck {
    pub fn new() -> Self;
    pub async fn mark_ready(&self);
    pub async fn check(&self) -> HealthStatus;
}

pub struct HealthStatus {
    pub state: HealthState,
    pub uptime_seconds: u64,
    pub http_server_ready: bool,
}
```

**Dependencies:**
- `tokio` (async runtime)
- `axum` (HTTP framework)

**Estimated LOC:** ~60

---

#### 2.2 hive-catalog (CRITICAL)

**Location:** `bin/15_queen_rbee_crates/hive-catalog/`  
**Package Name:** `queen-rbee-hive-catalog`

**Required By Happy Flow:**
- Line 25: "The queen bee looks at the hive catalog (missing crate hive catalog is sqlite)"
- Line 30: "The queen bee adds the local pc to the hive catalog"
- Line 38: "the queen bee will check the hive catalog for their devices"
- Line 45: "the queen bee updates the hive catalog with the devices"

**Purpose:**
- SQLite-based persistent storage of hives
- Track hive capabilities (CPU, GPUs, VRAM)
- Track hive status (online, offline, degraded)
- Query hives by capability

**Schema:**
```sql
CREATE TABLE hives (
    id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL,
    port INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    last_seen_at INTEGER
);

CREATE TABLE hive_devices (
    hive_id TEXT NOT NULL,
    device_type TEXT NOT NULL,  -- 'cpu', 'gpu'
    device_index INTEGER,
    device_name TEXT,
    vram_bytes INTEGER,
    FOREIGN KEY (hive_id) REFERENCES hives(id)
);

CREATE TABLE hive_capabilities (
    hive_id TEXT NOT NULL,
    capability_key TEXT NOT NULL,
    capability_value TEXT,
    FOREIGN KEY (hive_id) REFERENCES hives(id)
);
```

**API:**
```rust
pub struct HiveCatalog {
    db: SqliteConnection,
}

pub struct HiveEntry {
    pub id: String,
    pub hostname: String,
    pub port: u16,
    pub status: HiveStatus,
    pub devices: Vec<DeviceInfo>,
    pub capabilities: HashMap<String, String>,
}

pub struct DeviceInfo {
    pub device_type: DeviceType,  // CPU, GPU
    pub device_index: Option<u32>,
    pub device_name: String,
    pub vram_bytes: Option<u64>,
}

impl HiveCatalog {
    pub async fn new(db_path: &Path) -> Result<Self>;
    pub async fn add_hive(&self, entry: HiveEntry) -> Result<()>;
    pub async fn get_hive(&self, id: &str) -> Result<Option<HiveEntry>>;
    pub async fn list_hives(&self) -> Result<Vec<HiveEntry>>;
    pub async fn update_devices(&self, hive_id: &str, devices: Vec<DeviceInfo>) -> Result<()>;
    pub async fn update_capabilities(&self, hive_id: &str, caps: HashMap<String, String>) -> Result<()>;
    pub async fn remove_hive(&self, id: &str) -> Result<()>;
}
```

**Dependencies:**
- `sqlx` or `rusqlite` (SQLite)
- `tokio` (async runtime)
- `serde` (serialization)
- `rbee-types` (shared types)

**Estimated LOC:** ~200

**Note:** This is DIFFERENT from `hive-registry` (which is in-memory RAM). Catalog = persistent SQLite, Registry = runtime RAM.

---

#### 2.3 scheduler (CRITICAL)

**Location:** `bin/15_queen_rbee_crates/scheduler/`  
**Package Name:** `queen-rbee-scheduler`

**Required By Happy Flow:**
- Line 51-53: "The queen bee scheduler will pick the strongest device. in this case gpu1."
- Narration: "the basic scheduler has picked GPU1 for our inference job"

**Purpose:**
- Device selection based on capabilities
- Basic scheduler: pick strongest GPU
- Advanced scheduler: load balancing, VRAM availability, etc.
- Pluggable scheduler interface

**API:**
```rust
pub trait Scheduler: Send + Sync {
    async fn schedule(&self, request: ScheduleRequest) -> Result<ScheduleDecision>;
}

pub struct ScheduleRequest {
    pub model_ref: String,
    pub model_size_bytes: u64,
    pub available_hives: Vec<HiveInfo>,
}

pub struct HiveInfo {
    pub hive_id: String,
    pub devices: Vec<DeviceInfo>,
    pub available_vram: HashMap<String, u64>,  // device_id -> vram_bytes
}

pub struct ScheduleDecision {
    pub hive_id: String,
    pub device_id: String,
    pub device_type: DeviceType,
}

// Basic implementation
pub struct BasicScheduler;

impl Scheduler for BasicScheduler {
    async fn schedule(&self, request: ScheduleRequest) -> Result<ScheduleDecision> {
        // Pick strongest GPU (highest VRAM)
    }
}

// Advanced implementation (future)
pub struct LoadBalancingScheduler {
    // Consider current load, VRAM usage, etc.
}
```

**Dependencies:**
- `tokio` (async runtime)
- `rbee-types` (shared types)

**Estimated LOC:** ~150

---

### 3. rbee-hive Missing Crates

#### 3.1 vram-checker (CRITICAL)

**Location:** `bin/25_rbee_hive_crates/vram-checker/`  
**Package Name:** `rbee-hive-vram-checker`

**Required By Happy Flow:**
- Line 59: "The bee hive checks 'the VRAM CHECKER' (missing crate?)"
- Line 60: "There is enough room in VRAM"
- Line 61: "The bee hive responds with a 204"

**Purpose:**
- Check available VRAM on specific GPU
- Account for already-loaded models
- Admission control for new models
- Support for CUDA, Metal, CPU fallback

**API:**
```rust
pub struct VramChecker {
    device_monitor: DeviceMonitor,
}

pub struct VramCheckRequest {
    pub device_id: String,
    pub required_bytes: u64,
}

pub struct VramCheckResult {
    pub has_room: bool,
    pub available_bytes: u64,
    pub total_bytes: u64,
    pub used_bytes: u64,
}

impl VramChecker {
    pub async fn new() -> Result<Self>;
    pub async fn check(&self, request: VramCheckRequest) -> Result<VramCheckResult>;
    pub async fn get_device_info(&self, device_id: &str) -> Result<DeviceVramInfo>;
}

pub struct DeviceVramInfo {
    pub device_id: String,
    pub device_type: DeviceType,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub used_bytes: u64,
}
```

**Dependencies:**
- `sysinfo` (system monitoring)
- CUDA libraries (for NVIDIA GPUs)
- Metal libraries (for Apple GPUs)
- `rbee-hive-device-detection` (existing)

**Estimated LOC:** ~100

---

#### 3.2 worker-catalog (CRITICAL)

**Location:** `bin/25_rbee_hive_crates/worker-catalog/`  
**Package Name:** `rbee-hive-worker-catalog`

**Required By Happy Flow:**
- Line 44: "its worker catalog (which is empty, and we're missing a crate for worker catalog)"
- Line 89: "the bee hive adds the worker to the worker catalog"
- Line 98: "The bee hive looks in the worker catalog for the location of the worker on disk"

**Purpose:**
- SQLite-based persistent storage of worker binaries
- Track worker versions and capabilities
- Track worker binary paths on disk
- Support for multiple worker types (CPU, CUDA, Metal)

**Schema:**
```sql
CREATE TABLE workers (
    id TEXT PRIMARY KEY,
    worker_type TEXT NOT NULL,  -- 'cpu', 'cuda', 'metal'
    version TEXT NOT NULL,
    binary_path TEXT NOT NULL,
    checksum TEXT,
    capabilities TEXT,  -- JSON
    created_at INTEGER NOT NULL,
    last_used_at INTEGER
);
```

**API:**
```rust
pub struct WorkerCatalog {
    db: SqliteConnection,
}

pub struct WorkerEntry {
    pub id: String,
    pub worker_type: WorkerType,
    pub version: String,
    pub binary_path: PathBuf,
    pub checksum: Option<String>,
    pub capabilities: HashMap<String, String>,
}

pub enum WorkerType {
    Cpu,
    Cuda,
    Metal,
}

impl WorkerCatalog {
    pub async fn new(db_path: &Path) -> Result<Self>;
    pub async fn add_worker(&self, entry: WorkerEntry) -> Result<()>;
    pub async fn get_worker(&self, id: &str) -> Result<Option<WorkerEntry>>;
    pub async fn get_worker_by_type(&self, worker_type: WorkerType) -> Result<Option<WorkerEntry>>;
    pub async fn list_workers(&self) -> Result<Vec<WorkerEntry>>;
    pub async fn remove_worker(&self, id: &str) -> Result<()>;
    pub async fn update_last_used(&self, id: &str) -> Result<()>;
}
```

**Dependencies:**
- `sqlx` or `rusqlite` (SQLite)
- `tokio` (async runtime)
- `serde` (serialization)

**Estimated LOC:** ~180

**Note:** This is DIFFERENT from `worker-registry` (which is in-memory RAM). Catalog = persistent SQLite (available workers), Registry = runtime RAM (running workers).

---

### 4. Cross-Binary Missing Crates

#### 4.1 sse-relay (HIGH PRIORITY)

**Location:** `bin/99_shared_crates/sse-relay/`  
**Package Name:** `sse-relay`

**Required By Happy Flow:**
- Line 23: "The bee keeper makes a SSE connection with the queen bee"
- Line 67: "the bee hive responds with a GET link that is the SSE connection"
- Line 72: "narration (bee hive -> sse -> queen bee -> sse -> beekeeper -> stdout)"
- Line 116: "the worker bee responds with a get link for the SSE connection"

**Purpose:**
- SSE (Server-Sent Events) client and server utilities
- Multi-hop SSE relay (hive → queen → keeper → stdout)
- Narration message formatting
- Stream multiplexing

**API:**
```rust
// Server side
pub struct SseServer {
    tx: broadcast::Sender<SseMessage>,
}

impl SseServer {
    pub fn new() -> Self;
    pub async fn send(&self, message: SseMessage) -> Result<()>;
    pub fn subscribe(&self) -> broadcast::Receiver<SseMessage>;
}

// Client side
pub struct SseClient {
    url: String,
}

impl SseClient {
    pub async fn connect(&self) -> Result<SseStream>;
}

pub struct SseStream {
    stream: EventSource,
}

impl SseStream {
    pub async fn next(&mut self) -> Option<Result<SseMessage>>;
}

// Relay (middleware)
pub struct SseRelay {
    upstream: SseClient,
    downstream: SseServer,
}

impl SseRelay {
    pub async fn relay(&self) -> Result<()>;
    pub async fn relay_with_transform<F>(&self, transform: F) -> Result<()>
    where
        F: Fn(SseMessage) -> SseMessage;
}

// Message types
pub struct SseMessage {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
}

pub struct NarrationMessage {
    pub source: String,  // "bee keeper", "queen bee", "bee hive", "worker bee"
    pub message: String,
    pub timestamp: SystemTime,
}
```

**Dependencies:**
- `axum` (HTTP framework)
- `tokio` (async runtime)
- `tokio-stream` (stream utilities)
- `futures` (stream combinators)

**Estimated LOC:** ~120

**Usage Pattern:**
```rust
// In bee-keeper
let sse_client = SseClient::new(queen_sse_url);
let mut stream = sse_client.connect().await?;
while let Some(msg) = stream.next().await {
    println!("{}", msg.data);  // stdout
}

// In queen-rbee (relay)
let upstream = SseClient::new(hive_sse_url);
let downstream = SseServer::new();
let relay = SseRelay::new(upstream, downstream);
relay.relay_with_transform(|msg| {
    // Add "queen bee → " prefix
    NarrationMessage::transform(msg, "queen bee")
}).await?;

// In rbee-hive
let sse_server = SseServer::new();
sse_server.send(SseMessage {
    event: Some("narration".to_string()),
    data: "total size of the model is 500 MB, starting download".to_string(),
    id: None,
}).await?;
```

---

## 📊 Summary by Priority

### 🔴 CRITICAL (Must Have for Happy Flow)

1. **rbee-keeper-polling** - Poll queen health until ready
2. **queen-rbee-health** - Health check endpoint
3. **queen-rbee-hive-catalog** - Persistent hive storage (SQLite)
4. **queen-rbee-scheduler** - Device selection logic
5. **rbee-hive-vram-checker** - VRAM admission control
6. **rbee-hive-worker-catalog** - Persistent worker storage (SQLite)

**Total:** 6 crates, ~770 LOC

### 🟡 HIGH (Needed for Production)

7. **sse-relay** - SSE streaming and relay

**Total:** 1 crate, ~120 LOC

---

## 🔍 Clarifications Needed

### 1. Catalog vs. Registry Distinction

The happy flow mentions both "catalog" and "registry" for hives and workers:

**Catalog (SQLite - Persistent):**
- `queen-rbee-hive-catalog` - Available hives and their capabilities
- `rbee-hive-model-catalog` - Available models (already exists ✅)
- `rbee-hive-worker-catalog` - Available worker binaries (MISSING ❌)

**Registry (RAM - Runtime):**
- `queen-rbee-hive-registry` - Running hives (already exists ✅)
- `queen-rbee-worker-registry` - All workers across hives (already exists ✅)
- `rbee-hive-worker-registry` - Running workers on THIS hive (already exists ✅)

**Confirmed:** This distinction is correct and intentional.

---

### 2. Health Check Implementation

The happy flow mentions:
- Line 15: "queen bee health crate is missing"
- Line 16: "The queen bee wakes up and immediately starts the http server"

**Question:** Should health be:
1. A separate crate (`queen-rbee-health`)? ✅ RECOMMENDED
2. Part of `queen-rbee-http-server`? ❌ Less modular

**Recommendation:** Separate crate for reusability and testing.

---

### 3. SSE Relay Architecture

The happy flow shows multi-hop SSE:
```
bee hive → sse → queen bee → sse → bee keeper → stdout
```

**Question:** Should this be:
1. A shared crate (`sse-relay`)? ✅ RECOMMENDED
2. Implemented separately in each binary? ❌ Code duplication

**Recommendation:** Shared crate with relay utilities.

---

## 📝 Implementation Order

### Phase 1: Core Infrastructure (Week 1)

1. **queen-rbee-health** (60 LOC)
   - Simple health check endpoint
   - Blocks: rbee-keeper-polling

2. **rbee-keeper-polling** (80 LOC)
   - Poll queen until healthy
   - Depends on: queen-rbee-health

3. **queen-rbee-hive-catalog** (200 LOC)
   - SQLite schema + CRUD operations
   - Critical for hive management

### Phase 2: Scheduling & Admission (Week 2)

4. **queen-rbee-scheduler** (150 LOC)
   - Basic scheduler implementation
   - Depends on: queen-rbee-hive-catalog

5. **rbee-hive-vram-checker** (100 LOC)
   - VRAM admission control
   - Depends on: rbee-hive-device-detection (exists)

6. **rbee-hive-worker-catalog** (180 LOC)
   - SQLite schema + CRUD operations
   - Critical for worker management

### Phase 3: Streaming (Week 3)

7. **sse-relay** (120 LOC)
   - SSE client, server, relay utilities
   - Needed for narration streaming

---

## ✅ Validation Checklist

After implementing missing crates, verify:

- [ ] rbee-keeper can poll queen-rbee health
- [ ] queen-rbee exposes `/health` endpoint
- [ ] queen-rbee can persist hives to SQLite (hive-catalog)
- [ ] queen-rbee can query hive capabilities from catalog
- [ ] queen-rbee scheduler can pick strongest device
- [ ] rbee-hive can check VRAM availability
- [ ] rbee-hive can persist worker binaries to SQLite (worker-catalog)
- [ ] rbee-hive can resolve worker binary paths from catalog
- [ ] SSE streaming works: hive → queen → keeper → stdout
- [ ] Narration messages flow correctly through SSE relay

---

## 📚 References

- **Happy Flow (Authoritative):** `/home/vince/Projects/llama-orch/bin/a_human_wrote_this.md`
- **Happy Flow (Refined):** `/home/vince/Projects/llama-orch/bin/a_chatGPT_5_refined_this.md`
- **Current Implementation:** `/home/vince/Projects/llama-orch/bin/TECHNICAL_SUMMARY.md`
- **Root Cargo.toml:** `/home/vince/Projects/llama-orch/Cargo.toml`

---

**Status:** ✅ ANALYSIS COMPLETE  
**Next Step:** Implement Phase 1 crates (health, polling, hive-catalog)  
**Total Missing:** 7 crates, ~890 LOC

---

**END OF MISSING CRATES ANALYSIS**
