# rbee Architecture Overview - Part 3: Shared Infrastructure

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Living Document

---

## Job Client/Server Pattern

### Purpose

**Unified pattern for job submission and SSE streaming across all components.**

### Problem Solved

Before consolidation (TEAM-259), the same pattern was duplicated:
- `rbee-keeper/src/job_client.rs` (~120 LOC)
- `queen-rbee/src/hive_forwarder.rs` (~120 LOC)

**Same logic:**
1. Serialize operation to JSON
2. POST to `/v1/jobs`
3. Extract `job_id` from response
4. Connect to SSE stream
5. Process streaming lines

### Solution: Shared Crates

#### 1. job-server (Server-Side)

**Location:** `bin/99_shared_crates/job-server/`

**Purpose:** Track active jobs and provide SSE channels.

**API:**
```rust
pub struct JobRegistry<T> {
    jobs: Arc<RwLock<HashMap<String, Job<T>>>>,
}

impl<T> JobRegistry<T> {
    pub fn new() -> Self
    
    pub fn create_job(&self, job_id: String) -> mpsc::Receiver<T>
    
    pub fn send_event(&self, job_id: &str, event: T) -> Result<()>
    
    pub fn complete_job(&self, job_id: &str) -> Result<()>
    
    pub fn get_receiver(&self, job_id: &str) -> Option<mpsc::Receiver<T>>
}
```

**Usage (Server):**
```rust
// 1. Create job
let job_id = Uuid::new_v4().to_string();
let receiver = registry.create_job(job_id.clone());

// 2. Spawn task to process operation
tokio::spawn(async move {
    // Do work...
    registry.send_event(&job_id, "Starting...").await?;
    registry.send_event(&job_id, "Progress...").await?;
    registry.complete_job(&job_id).await?;
});

// 3. Return job_id to client
Ok(Json(CreateJobResponse { job_id }))
```

**SSE Streaming:**
```rust
// GET /v1/jobs/:job_id/stream
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>> {
    let receiver = state.registry.get_receiver(&job_id)
        .ok_or_else(|| anyhow!("Job not found"))?;
    
    let stream = ReceiverStream::new(receiver).map(|line| {
        Ok(Event::default().data(line))
    });
    
    Ok(Sse::new(stream))
}
```

#### 2. job-client (Client-Side)

**Location:** `bin/99_shared_crates/job-client/`

**Purpose:** Submit jobs and handle SSE streaming.

**API:**
```rust
pub struct JobClient {
    base_url: String,
    client: reqwest::Client,
}

impl JobClient {
    pub fn new(base_url: impl Into<String>) -> Self
    
    pub async fn submit_and_stream<F>(
        &self,
        operation: Operation,
        line_handler: F,
    ) -> Result<String>
    where
        F: Fn(String) -> Result<()>
}
```

#### 3. rbee-operations (Shared Types)

**Location:** `bin/99_shared_crates/rbee-operations/`

**Purpose:** Shared Operation enum.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Operation {
    HiveInstall { alias: String, binary_path: Option<String> },
    HiveStart { alias: String },
    WorkerSpawn { hive_id: String, model: String, device: String },
    Infer { prompt: String, model: String },
    // ... more operations
}

impl Operation {
    pub fn should_forward_to_hive(&self) -> bool {
        matches!(self, Operation::WorkerSpawn { .. } | Operation::ModelDownload { .. })
    }
}
```

---

## Observability: Narration System

### Purpose

**Human-readable event streaming with SSE routing.**

### Core API

```rust
pub const NARRATE: NarrationFactory = NarrationFactory::new("actor");

NARRATE
    .action("worker_spawn")
    .job_id(&job_id)  // ← CRITICAL for SSE routing!
    .context(&worker_id)
    .human("🚀 Spawning worker {}")
    .emit();
```

**Key Insight:** SSE sink requires `job_id` for routing!

---

## Security Crates (5 Total)

### 1. auth-min - Authentication

**Purpose:** Timing-safe authentication primitives.

**Features:**
- Timing-safe token comparison
- Token fingerprinting
- Bind policy enforcement

### 2. audit-logging - Compliance

**Purpose:** Immutable audit trail with tamper detection.

**Features:**
- 32 GDPR event types
- 7-year retention
- Blockchain-style hash chains
- Tamper detection

### 3. input-validation - Injection Prevention

**Purpose:** Prevent injection attacks.

**Features:**
- Path traversal prevention
- SQL injection prevention
- Log injection prevention
- Command injection prevention

### 4. secrets-management - Credential Handling

**Purpose:** Secure credential storage.

**Features:**
- File-based secrets (not env vars)
- Zeroization on drop
- Timing-safe verification
- Rotation support

### 5. deadline-propagation - Resource Protection

**Purpose:** Timeout enforcement.

**Features:**
- Hard deadline enforcement
- Visual countdown
- Cascading timeouts
- Resource exhaustion prevention

---

## Configuration Management

### rbee-config Crate

**Location:** `bin/99_shared_crates/rbee-config/`

**Purpose:** Unified configuration for queen.

### File Structure

```
~/.config/rbee/
├── config.toml           # Queen settings
├── hives.conf            # Hive definitions (SSH config style)
└── capabilities.yaml     # Auto-generated device info
```

### config.toml

```toml
[queen]
port = 8500
bind_addr = "127.0.0.1"

[hives.localhost]
host = "localhost"
port = 9000
ssh_user = "vince"

[hives.remote-gpu]
host = "gpu.home.arpa"
port = 9000
ssh_user = "vince"
ssh_key = "~/.ssh/id_ed25519"
```

### RbeeConfig API

```rust
pub struct RbeeConfig {
    pub queen: QueenConfig,
    pub hives: HashMap<String, HiveConfig>,
}

impl RbeeConfig {
    pub fn load() -> Result<Self>
    
    pub fn get_hive(&self, alias: &str) -> Option<&HiveConfig>
    
    pub fn save(&self) -> Result<()>
}
```

---

## Shared Utilities

### timeout-enforcer

**Purpose:** Hard timeout with visual feedback.

```rust
use timeout_enforcer::TimeoutEnforcer;

// Client-side (no job_id)
TimeoutEnforcer::new(Duration::from_secs(30))
    .enforce(slow_operation())
    .await?;

// Server-side (with job_id for SSE)
TimeoutEnforcer::new(Duration::from_secs(30))
    .with_job_id(&job_id)
    .enforce(slow_operation())
    .await?;
```

**TEAM-661 Fix:** Added `.with_job_id()` for SSE routing.

### daemon-lifecycle

**Purpose:** Daemon management utilities.

**Features:**
- Process daemonization
- PID file management
- Signal handling
- Graceful shutdown

### heartbeat

**Purpose:** Health monitoring protocol.

**Types:**
```rust
pub struct WorkerHeartbeatPayload {
    pub worker_id: String,
    pub timestamp: String,
    pub health_status: HealthStatus,
}

pub enum HealthStatus {
    Healthy,
    Degraded,
}
```

**TEAM-261:** Workers send heartbeats directly to queen (not through hive).

---

## Crate Organization

### Directory Structure

```
bin/99_shared_crates/
├── audit-logging/          # GDPR compliance
├── auth-min/               # Authentication
├── deadline-propagation/   # Timeouts
├── input-validation/       # Injection prevention
├── secrets-management/     # Credential storage
├── narration-core/         # Observability
├── narration-macros/       # Macros
├── job-server/             # Server-side job tracking
├── job-client/             # Client-side job submission
├── rbee-operations/        # Shared Operation enum
├── rbee-config/            # Configuration
├── timeout-enforcer/       # Hard timeouts
├── daemon-lifecycle/       # Daemon utilities
├── heartbeat/              # Health monitoring
└── auto-update/            # Binary updates
```

### Dependency Graph

```
rbee-keeper, queen-rbee, rbee-hive, llm-worker-rbee
    ↓
job-client, job-server, rbee-operations
    ↓
narration-core, timeout-enforcer
    ↓
auth-min, audit-logging, input-validation, secrets-management
```

---

## Key Principles

### 1. Shared Infrastructure

**Principle:** Don't duplicate infrastructure code.

**Implementation:**
- job-client/job-server (TEAM-259: 200 LOC saved)
- rbee-operations (single source of truth)
- narration-core (consistent observability)

### 2. SSE Routing

**Principle:** All server-side narration must include job_id.

**Implementation:**
```rust
// ✅ CORRECT
NARRATE.action("x").job_id(&job_id).emit();

// ❌ WRONG (events go to stdout, not SSE)
NARRATE.action("x").emit();
```

### 3. Security First

**Principle:** 5 security crates for defense-in-depth.

**Implementation:**
- auth-min (timing-safe)
- audit-logging (immutable)
- input-validation (injection prevention)
- secrets-management (zeroization)
- deadline-propagation (resource protection)

### 4. Type Safety

**Principle:** Share types to prevent mismatches.

**Implementation:**
- rbee-operations (Operation enum)
- heartbeat (WorkerHeartbeatPayload)
- rbee-config (RbeeConfig)

---

## Next: Part 4 - Data Flow & Protocols

The next document covers data flow and communication protocols:
- Request Flow (End-to-End)
- SSE Streaming
- Heartbeat Architecture
- Operation Routing

**See:** `.arch/03_DATA_FLOW_PART_4.md`
