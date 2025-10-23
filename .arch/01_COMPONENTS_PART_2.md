# rbee Architecture Overview - Part 2: Component Deep Dive

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Living Document

---

## User Interfaces

rbee provides **two complementary interfaces** for managing infrastructure:

1. **rbee-keeper (CLI)** - Human operators, interactive workflows
2. **rbee-sdk (Library)** - Programmatic access, application integration

Both are thin HTTP clients to queen-rbee. All business logic lives in queen.

---

## rbee-keeper (CLI - User Interface)

### Purpose

**rbee-keeper is the PRIMARY user interface for human operators managing rbee infrastructure.**

NOT a testing tool - it's how operators interact with the entire system.

### Responsibilities

1. **Queen Lifecycle Management** (NEW)
   - Build queen with different configurations
   - Start/stop/status queen daemon
   - Rebuild queen with local-hive feature
   - Query queen build info

2. **Hive Lifecycle Management**
   - Install hive binaries on machines
   - Start/stop/status hive daemons
   - Uninstall hives
   - Refresh hive capabilities
   - Smart prompts for localhost optimization

3. **Worker Management** (via queen)
   - Spawn workers on specific hives
   - List workers across all hives
   - Get worker details
   - Stop workers

4. **Model Management** (via queen → hive)
   - Download models from Hugging Face
   - List available models
   - Get model metadata
   - Delete models

5. **Inference Testing**
   - Submit inference requests
   - Stream tokens in real-time
   - Test model availability

### Architecture

```
bin/00_rbee_keeper/
├── src/
│   ├── main.rs                 # CLI entry point, command routing
│   ├── job_client.rs           # HTTP client, SSE streaming
│   └── lib.rs                  # Re-exports
├── bdd/                        # BDD integration tests
└── Cargo.toml
```

### Key Patterns

#### 1. Job Submission Pattern

```rust
// Submit operation to queen
let operation = Operation::HiveStart {
    alias: "localhost".to_string(),
};

let client = JobClient::new("http://localhost:8500");
client.submit_and_stream(operation, |line| {
    println!("{}", line);  // Print SSE events
    Ok(())
}).await?;
```

#### 2. Command Structure

```rust
#[derive(Parser)]
enum Commands {
    // Queen commands (NEW)
    Queen {
        #[command(subcommand)]
        command: QueenCommands,
    },
    
    // Hive commands
    Hive {
        #[command(subcommand)]
        command: HiveCommands,
    },
    
    // Worker commands
    Worker {
        #[command(subcommand)]
        command: WorkerCommands,
    },
    
    // Model commands
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },
    
    // Inference command
    Infer {
        prompt: String,
        model: String,
        // ... other args
    },
}

// NEW: Queen lifecycle commands
enum QueenCommands {
    Start,
    Stop,
    Status,
    Rebuild {
        #[arg(long)]
        with_local_hive: bool,
    },
    Info,  // Show build configuration
}
```

### User Experience

#### Real-Time Feedback

All operations provide real-time SSE streaming:

```bash
$ rbee-keeper hive start localhost
🔍 Checking hive health...
✅ Hive is healthy
🚀 Starting hive...
✅ Hive started successfully
[DONE]
```

#### Smart Prompts (NEW)

Intelligent recommendations for optimal configuration:

```bash
$ rbee-keeper hive install localhost
⚠️  Performance Notice:
   You're installing a hive on localhost, but your queen-rbee
   was built without the 'local-hive' feature.
   
   📊 Performance comparison:
      • Current setup:  ~5-10ms overhead (HTTP)
      • Integrated:     ~0.1ms overhead (direct calls)
   
   💡 Recommendation:
      Rebuild queen-rbee with integrated hive for 50-100x faster
      localhost operations:
      
      $ rbee-keeper queen rebuild --with-local-hive
   
   ℹ️  Or continue with distributed setup if you have specific needs.
   
   Continue with distributed setup? [y/N]: 
```

#### Error Handling

Clear error messages with context:

```bash
$ rbee-keeper worker spawn --model llama-3-8b --device GPU-0
❌ Error: GPU-0 not found
💡 Available devices:
  - GPU-0 (NVIDIA GeForce RTX 3090, 24GB)
  - CPU-0 (16 cores, 64GB RAM)
```

### Dependencies

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }
job-client = { path = "../99_shared_crates/job-client" }
rbee-operations = { path = "../99_shared_crates/rbee-operations" }
anyhow = "1.0"
```

### Status

✅ **M0 Complete**
- All hive commands working
- Basic inference working
- SSE streaming operational

---

## rbee-sdk (Library - Programmatic Interface)

### Purpose

**rbee-sdk provides programmatic access to rbee infrastructure for application integration.**

Enables developers to embed rbee capabilities into Rust and TypeScript/JavaScript applications.

### Key Features

1. **Single-Source Design**
   - Rust core library
   - Compiles to native (Rust apps)
   - Compiles to WASM (TypeScript/JavaScript apps)
   - TypeScript bindings auto-generated

2. **Type-Safe API**
   - Same Operation types as rbee-keeper
   - Compile-time safety
   - Auto-completion in IDEs

3. **Async/Streaming**
   - Built on tokio/reqwest (Rust)
   - Async/await in TypeScript
   - SSE streaming support

4. **Cross-Platform**
   - Native: Linux, macOS, Windows
   - WASM: Node.js, browsers

### Architecture

```
consumers/rbee-sdk/
├── src/
│   ├── lib.rs                 # Public API exports
│   ├── client.rs              # HTTP client (mimics keeper)
│   ├── types.rs               # Re-exports from rbee-operations
│   ├── stream.rs              # SSE stream wrapper
│   └── error.rs               # Error types
├── ts/
│   ├── index.ts               # TypeScript wrapper
│   └── package.json           # npm package
└── Cargo.toml                 # Features: native, wasm
```

### Rust API Example

```rust
use rbee_sdk::RbeeClient;

#[tokio::main]
async fn main() -> Result<()> {
    let client = RbeeClient::new("http://localhost:8500");
    
    // Spawn worker
    let mut stream = client.worker_spawn("llama-3-8b", "GPU-0").await?;
    while let Some(event) = stream.next().await {
        println!("{}", event);
    }
    
    // Run inference
    let mut stream = client.infer("Hello, world!", "llama-3-8b").await?;
    while let Some(token) = stream.next().await {
        print!("{}", token);
    }
    
    Ok(())
}
```

### TypeScript API Example

```typescript
import { RbeeClient } from '@rbee/sdk';

const client = new RbeeClient('http://localhost:8500');

// Spawn worker
const stream = await client.workerSpawn('llama-3-8b', 'GPU-0');
for await (const event of stream) {
  console.log(event);
}

// Run inference
const inferStream = await client.infer('Hello, world!', 'llama-3-8b');
for await (const token of inferStream) {
  process.stdout.write(token);
}
```

### SDK vs CLI Comparison

| Feature | rbee-keeper (CLI) | rbee-sdk (Library) |
|---------|-------------------|-------------------|
| **Interface** | Command-line | Rust/TypeScript API |
| **Output** | Pretty-printed | Structured data |
| **Auto-start** | Yes (queen lifecycle) | No (expects queen running) |
| **Prompts** | Interactive | No prompts |
| **Use case** | Operators, DevOps | Applications, integrations |

**Both use the same HTTP API and Operation types.**

### Use Cases

1. **Rust Applications** - Batch processing, custom tooling, integration tests
2. **Node.js Services** - Web servers with LLM, API gateways, background workers
3. **Browser Applications** - Interactive UIs, chat interfaces, real-time streaming
4. **Automation** - CI/CD pipelines, testing frameworks, monitoring tools

### Dependencies

```toml
[dependencies]
rbee-operations = { path = "../../bin/99_shared_crates/rbee-operations" }
rbee-job-client = { path = "../../bin/99_shared_crates/rbee-job-client" }
reqwest = { version = "0.11", features = ["json", "stream"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }

# WASM support
wasm-bindgen = { version = "0.2", optional = true }
tsify = { version = "0.4", optional = true }

[features]
default = ["native"]
native = []
wasm = ["dep:tsify", "dep:wasm-bindgen"]
```

### Status

- **Version:** 0.0.0 (design phase)
- **Implementation:** Stubs only (client methods unimplemented)
- **Effort:** 22-32 hours to complete

**See:** `.arch/SDK_ARCHITECTURE.md` for complete design and implementation plan.

---

## queen-rbee (HTTP Daemon - The Brain)

### Purpose

**queen-rbee is THE BRAIN - makes ALL intelligent decisions.**

### Build Configurations

Queen supports two build modes optimized for different deployments:

#### Distributed Queen (Default)

```bash
cargo build --bin queen-rbee
```

- All operations forwarded via HTTP (job-client)
- Manages remote hives
- ~5-10ms overhead per operation
- Requires separate rbee-hive binary for localhost

#### Integrated Queen (local-hive feature)

```bash
cargo build --bin queen-rbee --features local-hive
```

- Direct Rust calls for localhost operations (~0.1ms)
- HTTP forwarding for remote hives (still available!)
- 50-100x faster localhost operations
- No separate rbee-hive binary needed for localhost

**Key Insight:** Integrated queen provides best of both worlds - fast local + distributed capability.

### Responsibilities

1. **Operation Routing**
   - Hive operations → Execute directly
   - Worker/Model operations → Forward to hive (HTTP or direct)
   - Infer operations → Schedule to worker (TODO)

2. **Job Management**
   - Track all operations via job_id
   - Provide SSE streams per job
   - Handle job cancellation

3. **Registry Management**
   - Hive registry (track available hives)
   - Worker registry (track available workers) [TODO]
   - Model availability (which models on which hives)

4. **Scheduling** (Future M2)
   - Load balancing across workers
   - Model-aware placement
   - Queue management

### Architecture

```
bin/10_queen_rbee/
├── src/
│   ├── main.rs                 # HTTP server, router
│   ├── job_router.rs           # Operation routing logic
│   ├── hive_forwarder.rs       # Forward operations to hive
│   ├── http/
│   │   ├── mod.rs              # HTTP module exports
│   │   ├── jobs.rs             # Job endpoints
│   │   ├── heartbeat.rs        # Heartbeat endpoints
│   │   └── health.rs           # Health check
│   ├── narration.rs            # Observability constants
│   └── lib.rs                  # Re-exports
├── Cargo.toml
└── README.md
```

### Key Components

#### 1. Job Router

**Purpose:** Route operations to appropriate handler.

**Logic:**
```rust
pub async fn route_operation(
    operation: Operation,
    job_id: String,
    state: SchedulerState,
) -> Result<()> {
    match operation {
        // Hive lifecycle (handled in queen)
        Operation::HiveInstall { .. } => {
            execute_hive_install(...).await
        }
        Operation::HiveStart { .. } => {
            execute_hive_start(...).await
        }
        // ... more hive operations
        
        // Inference (queen schedules, routes to worker)
        Operation::Infer { .. } => {
            // TODO: IMPLEMENT INFERENCE SCHEDULING
            // 1. Query hive for available workers
            // 2. Select worker (load balancing)
            // 3. Direct HTTP to worker
            // 4. Stream tokens back to client
            Err(anyhow!("Not yet implemented"))
        }
        
        // Worker/Model operations (forward to hive)
        op if op.should_forward_to_hive() => {
            hive_forwarder::forward_to_hive(&job_id, op, state.config).await
        }
        
        // Unknown operations
        _ => Err(anyhow!("Unsupported operation"))
    }
}
```

**Key Insight:** Different operations take different paths!

#### 2. Hive Forwarder (Dual-Mode)

**Purpose:** Forward worker/model operations to appropriate hive (HTTP or direct).

**Implementation (Distributed Queen):**
```rust
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
    config: Arc<RbeeConfig>,
) -> Result<()> {
    let hive_id = operation.get_hive_id()?;
    let hive = config.get_hive(hive_id)?;
    let hive_url = format!("http://{}:{}", hive.host, hive.port);
    
    // Always use HTTP (no local hive embedded)
    let client = JobClient::new(hive_url);
    client.submit_and_stream(operation, |line| {
        NARRATE.action("hive_forward")
            .job_id(job_id)
            .context(&line)
            .emit();
        Ok(())
    }).await
}
```

**Implementation (Integrated Queen with local-hive):**
```rust
pub async fn forward_to_hive(
    job_id: &str,
    operation: Operation,
    config: Arc<RbeeConfig>,
) -> Result<()> {
    let hive_id = operation.get_hive_id()?;
    
    #[cfg(feature = "local-hive")]
    {
        // Check if this is localhost
        if is_localhost_hive(hive_id, &config) {
            // Direct Rust calls (FAST!)
            return forward_via_local_hive(job_id, operation).await;
        }
    }
    
    // Remote hive: Use HTTP (always available)
    let hive = config.get_hive(hive_id)?;
    let hive_url = format!("http://{}:{}", hive.host, hive.port);
    let client = JobClient::new(hive_url);
    client.submit_and_stream(operation, |line| {
        NARRATE.action("hive_forward")
            .job_id(job_id)
            .context(&line)
            .emit();
        Ok(())
    }).await
}

#[cfg(feature = "local-hive")]
async fn forward_via_local_hive(
    job_id: &str,
    operation: Operation,
) -> Result<()> {
    // Direct calls to embedded hive crates
    match operation {
        Operation::WorkerSpawn { model, device, .. } => {
            rbee_hive_worker_lifecycle::spawn_worker(
                job_id, &model, &device
            ).await?;
        }
        
        Operation::ModelDownload { model_id, .. } => {
            rbee_hive_model_provisioner::download_model(
                job_id, &model_id
            ).await?;
        }
        
        // ... other operations
    }
    
    Ok(())
}
```

**Key Insight:** Same API, different implementation based on build configuration!

#### 3. Heartbeat Handler

**Purpose:** Accept worker heartbeats for scheduling.

**Implementation:**
```rust
pub async fn handle_worker_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<WorkerHeartbeatPayload>,
) -> Result<Json<HttpHeartbeatAcknowledgement>> {
    eprintln!(
        "💓 Worker heartbeat: worker_id={}, timestamp={}, health_status={:?}",
        payload.worker_id, payload.timestamp, payload.health_status
    );
    
    // TODO: Update worker registry
    // state.worker_registry.update_worker_state(&payload.worker_id, payload);
    
    Ok(Json(HttpHeartbeatAcknowledgement {
        status: "ok".to_string(),
        message: format!("Heartbeat received from worker {}", payload.worker_id),
    }))
}
```

#### 4. HTTP Routes

```rust
fn create_router(
    job_server: Arc<JobRegistry<String>>,
    config: Arc<RbeeConfig>,
    hive_registry: Arc<HiveRegistry>,
) -> Router {
    Router::new()
        .route("/health", get(handle_health))
        .route("/v1/shutdown", post(handle_shutdown))
        .route("/v1/heartbeat", post(handle_heartbeat))           // Hive heartbeat (deprecated)
        .route("/v1/worker-heartbeat", post(handle_worker_heartbeat)) // Worker heartbeat (TEAM-261)
        .route("/v1/jobs", post(handle_create_job))
        .route("/v1/jobs/:job_id/stream", get(handle_stream_job))
        .with_state(...)
}
```

### State Management

#### Job Registry

```rust
pub struct JobRegistry<T> {
    jobs: Arc<RwLock<HashMap<String, Job<T>>>>,
}

pub struct Job<T> {
    pub id: String,
    pub status: JobStatus,
    pub channel: mpsc::Sender<T>,
}
```

**Purpose:** Track active jobs and their SSE channels.

#### Hive Registry

```rust
pub struct HiveRegistry {
    hives: Arc<RwLock<HashMap<String, HiveState>>>,
}

pub struct HiveState {
    pub hive_id: String,
    pub last_heartbeat: String,
    pub workers: Vec<WorkerState>,
}
```

**Purpose:** Track available hives and their capabilities.

### Configuration

```toml
# ~/.config/rbee/config.toml
[queen]
port = 8500
bind_addr = "127.0.0.1"

[hives.localhost]
host = "localhost"
port = 9000
ssh_user = "vince"
```

### Dependencies

```toml
[features]
default = []
local-hive = [
    "rbee-hive-device-detection",
    "rbee-hive-worker-lifecycle",
    "rbee-hive-model-provisioner",
]

[dependencies]
# Core (always available)
axum = { workspace = true }
tokio = { workspace = true }
job-server = { path = "../99_shared_crates/job-server" }
job-client = { path = "../99_shared_crates/job-client" }  # Always available for remote hives
rbee-operations = { path = "../99_shared_crates/rbee-operations" }
rbee-config = { path = "../99_shared_crates/rbee-config" }
hive-lifecycle = { path = "../15_queen_rbee_crates/hive-lifecycle" }
hive-registry = { path = "../15_queen_rbee_crates/hive-registry" }

# Conditional (only with local-hive feature)
rbee-hive-device-detection = { path = "../25_rbee_hive_crates/device-detection", optional = true }
rbee-hive-worker-lifecycle = { path = "../25_rbee_hive_crates/worker-lifecycle", optional = true }
rbee-hive-model-provisioner = { path = "../25_rbee_hive_crates/model-provisioner", optional = true }
```

### Status

🚧 **In Progress**
- ✅ HTTP server operational
- ✅ Job registry working
- ✅ Hive operations working
- ✅ Worker/model forwarding working (HTTP)
- 🚧 local-hive feature (planned)
- ❌ Inference scheduling (TODO)
- ❌ Worker registry (TODO)
- ❌ Load balancing (TODO)

---

### Rhai Programmable Scheduler (M2 - Future)

**Status:** ⚠️ **OUT OF SCOPE** for current milestone (M0/M1)  
**Planned:** M2 milestone  
**Spec:** `bin/.specs/00_llama-orch.md` [SYS-6.1.5]

#### Purpose

**Rhai** is an embedded Rust scripting language that will power queen-rbee's intelligent scheduling decisions.

**Think of it as:**
- The "policy execution engine" of queen-rbee
- User-programmable orchestration logic
- Dynamic, scriptable decision-making without recompilation

#### Architecture

```
Inference Request → Queen → Rhai Scheduler Script → Worker Selection
                     ↓
              [scheduler.rhai]
                     ↓
        evaluate(job, pools, workers)
                     ↓
              Selected Worker
```

#### Two Deployment Modes

**1. Platform Mode (Multi-Tenant Marketplace)**

**Purpose:** Secure, fair, multi-tenant GPU marketplace

**Scheduler:** `platform-scheduler.rhai` (built-in, immutable)

**Characteristics:**
- ✅ **Immutable** - Cannot be modified by users
- ✅ **Multi-tenant fairness** - Fair resource allocation
- ✅ **SLA compliance** - Guarantees service levels
- ✅ **Security-first** - Sandboxed execution
- ✅ **Quota enforcement** - Per-tenant limits
- ✅ **Capacity management** - Rejects with 429 when full

**Example Logic:**
```rhai
// platform-scheduler.rhai (immutable)
fn schedule_job(job, pools, workers) {
    // 1. Check tenant quota
    if tenant_over_quota(job.tenant_id) {
        return reject("quota_exceeded");
    }
    
    // 2. Find worker with capacity
    let available = workers.filter(|w| w.status == "ready" && w.vram_free > job.vram_required);
    
    if available.is_empty() {
        return reject_429("no_capacity");  // Queue full
    }
    
    // 3. Fair scheduling (round-robin by tenant)
    let worker = fair_select(available, job.tenant_id);
    
    return accept(worker.id);
}
```

**2. Home/Lab Mode (Personal Infrastructure)**

**Purpose:** Maximize hardware utilization, user-customizable

**Scheduler:** `~/.config/rbee/scheduler.rhai` (user-editable)

**Characteristics:**
- ✅ **User-editable** - Full customization
- ✅ **No quotas** - Use all available resources
- ✅ **Best-effort** - Accept jobs if any worker available
- ✅ **Power user features** - Custom policies (device affinity, priority, etc.)
- ✅ **Learning-friendly** - Examples provided

**Example Logic:**
```rhai
// ~/.config/rbee/scheduler.rhai (user-editable)
fn schedule_job(job, pools, workers) {
    // 1. Find workers with matching model
    let candidates = workers.filter(|w| w.model == job.model && w.status == "ready");
    
    if candidates.is_empty() {
        // No worker available, spawn one!
        return spawn_worker(job.model, "cuda:0");
    }
    
    // 2. Prefer GPU with most free VRAM
    let worker = candidates.max_by(|w| w.vram_free);
    
    return accept(worker.id);
}
```

**Advanced Example (Device Affinity):**
```rhai
// Custom policy: Prefer specific GPU for certain models
fn schedule_job(job, pools, workers) {
    // Large models → GPU-0 (24GB)
    if job.model.contains("70b") || job.model.contains("405b") {
        let gpu0_workers = workers.filter(|w| w.device == "cuda:0");
        if !gpu0_workers.is_empty() {
            return accept(gpu0_workers[0].id);
        }
    }
    
    // Small models → GPU-1 (smaller, but available)
    if job.model.contains("7b") || job.model.contains("8b") {
        let gpu1_workers = workers.filter(|w| w.device == "cuda:1");
        if !gpu1_workers.is_empty() {
            return accept(gpu1_workers[0].id);
        }
    }
    
    // Fallback: Any available worker
    let available = workers.filter(|w| w.status == "ready");
    if !available.is_empty() {
        return accept(available[0].id);
    }
    
    return reject("no_workers");
}
```

#### Scheduler API

**Input (provided to script):**
```rust
struct SchedulerInput {
    job: JobRequest,
    pools: Vec<WorkerPool>,
    workers: Vec<WorkerState>,
}

struct JobRequest {
    job_id: String,
    tenant_id: Option<String>,  // Platform mode only
    model: String,
    prompt: String,
    vram_required: u64,
    priority: u8,
}

struct WorkerState {
    id: String,
    status: String,  // "ready", "busy", "error"
    model: String,
    device: String,
    vram_used: u64,
    vram_free: u64,
    vram_total: u64,
    requests_total: u64,
}
```

**Output (returned from script):**
```rust
enum SchedulerDecision {
    Accept { worker_id: String },
    Reject { reason: String },
    Reject429 { reason: String },  // Queue full (platform mode)
    SpawnWorker { model: String, device: String },  // Home/lab mode
}
```

#### Rhai Integration

```rust
// bin/10_queen_rbee/src/scheduler.rs (M2)
use rhai::{Engine, AST};

pub struct RhaiScheduler {
    engine: Engine,
    script: AST,
}

impl RhaiScheduler {
    pub fn new(script_path: &Path) -> Result<Self> {
        let mut engine = Engine::new();
        
        // Register custom functions
        engine.register_fn("tenant_over_quota", check_tenant_quota);
        engine.register_fn("fair_select", fair_select_worker);
        engine.register_fn("reject", |reason: &str| SchedulerDecision::Reject(reason));
        engine.register_fn("reject_429", |reason: &str| SchedulerDecision::Reject429(reason));
        engine.register_fn("accept", |worker_id: &str| SchedulerDecision::Accept(worker_id));
        engine.register_fn("spawn_worker", |model: &str, device: &str| {
            SchedulerDecision::SpawnWorker(model, device)
        });
        
        // Load and compile script
        let script = std::fs::read_to_string(script_path)?;
        let ast = engine.compile(&script)?;
        
        Ok(Self { engine, script: ast })
    }
    
    pub fn schedule(&self, input: SchedulerInput) -> Result<SchedulerDecision> {
        // Call Rhai function
        let decision: SchedulerDecision = self.engine.call_fn(
            &mut rhai::Scope::new(),
            &self.script,
            "schedule_job",
            (input.job, input.pools, input.workers),
        )?;
        
        Ok(decision)
    }
}
```

**Usage in job_router.rs:**
```rust
// M2: Inference scheduling with Rhai
Operation::Infer { prompt, model, .. } => {
    // 1. Load scheduler
    let scheduler = state.scheduler.read().unwrap();
    
    // 2. Prepare input
    let input = SchedulerInput {
        job: JobRequest {
            job_id: job_id.clone(),
            tenant_id: None,  // Home mode
            model: model.clone(),
            prompt: prompt.clone(),
            vram_required: estimate_vram(&model),
            priority: 0,
        },
        pools: state.worker_registry.get_pools(),
        workers: state.worker_registry.get_workers(),
    };
    
    // 3. Run scheduler
    let decision = scheduler.schedule(input)?;
    
    // 4. Execute decision
    match decision {
        SchedulerDecision::Accept { worker_id } => {
            // Forward to selected worker
            forward_to_worker(&job_id, &worker_id, prompt).await?;
        }
        SchedulerDecision::SpawnWorker { model, device } => {
            // Spawn worker first, then forward
            spawn_worker_and_infer(&job_id, &model, &device, prompt).await?;
        }
        SchedulerDecision::Reject { reason } => {
            return Err(anyhow!("Scheduling failed: {}", reason));
        }
        SchedulerDecision::Reject429 { reason } => {
            return Err(anyhow!("Queue full: {}", reason));
        }
    }
    
    Ok(())
}
```

#### Benefits

**For Platform Operators:**
- ✅ Immutable, auditable scheduling policy
- ✅ Fair resource allocation
- ✅ Quota enforcement
- ✅ SLA compliance
- ✅ Predictable behavior

**For Home/Lab Users:**
- ✅ Full customization
- ✅ No artificial limits
- ✅ Experiment with policies
- ✅ Device affinity
- ✅ Priority scheduling

**For Developers:**
- ✅ No recompilation needed
- ✅ Hot-reload scheduler scripts
- ✅ Easy to test policies
- ✅ Rhai is safe (sandboxed)

#### Implementation Plan (M2)

**Phase 1: Basic Rhai Integration (8-12 hours)**
- Add rhai dependency
- Create RhaiScheduler struct
- Load and compile scripts
- Basic function registration

**Phase 2: Platform Scheduler (12-16 hours)**
- Implement platform-scheduler.rhai
- Quota enforcement
- Fair scheduling (round-robin)
- 429 rejection when full
- SLA tracking

**Phase 3: Home/Lab Scheduler (8-12 hours)**
- User-editable scheduler.rhai
- Example scripts
- SpawnWorker support
- Device affinity examples

**Phase 4: Advanced Features (12-16 hours)**
- Hot-reload support
- Scheduler metrics
- A/B testing (multiple schedulers)
- Scheduler validation

**Total Effort:** 40-56 hours

#### Example Scheduler Scripts

**Location:** `bin/10_queen_rbee/schedulers/`
- `platform-scheduler.rhai` - Platform mode (immutable)
- `home-basic.rhai` - Simple best-effort
- `home-device-affinity.rhai` - GPU selection by model size
- `home-priority.rhai` - Priority-based scheduling
- `home-power-save.rhai` - Prefer idle workers

#### Security Considerations

**Platform Mode:**
- Scheduler script is read-only (embedded in binary)
- No file I/O from script
- No network access from script
- Limited CPU budget per scheduling decision

**Home/Lab Mode:**
- User has full control (their hardware)
- Still sandboxed (no arbitrary code execution)
- CPU limits to prevent infinite loops

#### Future Enhancements

1. **Scheduler Marketplace** - Users share scheduler scripts
2. **ML-based Scheduling** - Rhai calls into ML model for predictions
3. **Multi-objective Optimization** - Balance cost, latency, throughput
4. **A/B Testing** - Run multiple schedulers, compare results

---

### Build Info Endpoint (NEW)

Queen exposes build configuration for rbee-keeper to query:

```rust
// GET /v1/build-info
{
    "version": "0.1.0",
    "features": ["local-hive"],  // or [] for distributed
    "build_timestamp": "2025-10-23T10:00:00Z"
}
```

---

## rbee-hive (HTTP Daemon - Pool Manager)

### Purpose

**rbee-hive manages worker lifecycle on a SINGLE machine.**

### Responsibilities

1. **Worker Lifecycle**
   - Spawn workers (detect GPU, start process)
   - List local workers
   - Stop workers
   - Track worker status

2. **Model Catalog**
   - Track available models
   - Model download coordination
   - Model metadata storage

3. **Device Detection**
   - GPU enumeration (nvidia-smi)
   - CPU detection
   - VRAM/RAM reporting

4. **Capabilities Reporting**
   - Report available devices to queen
   - Update capabilities on change

### Architecture

```
bin/20_rbee_hive/
├── src/
│   ├── main.rs                 # HTTP server, capabilities
│   ├── job_router.rs           # Operation routing
│   ├── http/
│   │   ├── mod.rs              # HTTP exports
│   │   └── jobs.rs             # Job endpoints
│   ├── narration.rs            # Observability constants
│   └── lib.rs                  # Re-exports
├── bdd/                        # BDD tests
└── Cargo.toml
```

### Key Components

#### 1. Capabilities Detection

**Purpose:** Enumerate GPUs and CPUs for queen.

```rust
async fn get_capabilities() -> Json<CapabilitiesResponse> {
    NARRATE.action("caps_request")
        .human("📡 Received capabilities request from queen")
        .emit();
    
    NARRATE.action("caps_gpu_check")
        .human("🔍 Detecting GPUs via nvidia-smi...")
        .emit();
    
    // Detect GPUs
    let gpu_info = rbee_hive_device_detection::detect_gpus();
    
    NARRATE.action("caps_gpu_found")
        .context(gpu_info.count.to_string())
        .human(if gpu_info.count > 0 {
            "✅ Found {} GPU(s)"
        } else {
            "ℹ️  No GPUs detected, using CPU only"
        })
        .emit();
    
    // Build device list
    let mut devices = vec![];
    
    // Add GPUs
    for gpu in gpu_info.devices {
        devices.push(HiveDevice {
            id: format!("GPU-{}", gpu.index),
            name: gpu.name,
            device_type: "gpu",
            vram_gb: Some(gpu.vram_total_gb() as u32),
            compute_capability: Some(format!("{}.{}", gpu.compute_capability.0, gpu.compute_capability.1)),
        });
    }
    
    // Add CPU
    let cpu_cores = rbee_hive_device_detection::get_cpu_cores();
    let system_ram_gb = rbee_hive_device_detection::get_system_ram_gb();
    
    devices.push(HiveDevice {
        id: "CPU-0",
        name: format!("CPU ({} cores)", cpu_cores),
        device_type: "cpu",
        vram_gb: Some(system_ram_gb),
        compute_capability: None,
    });
    
    Json(CapabilitiesResponse { devices })
}
```

#### 2. Job Router

**Purpose:** Route worker/model operations to appropriate handler.

```rust
pub async fn route_operation(
    operation: Operation,
    job_id: String,
) -> Result<()> {
    match operation {
        // Worker operations
        Operation::WorkerSpawn { .. } => {
            // TODO: Implement worker spawning
            Err(anyhow!("Not yet implemented"))
        }
        Operation::WorkerList { .. } => {
            // TODO: Implement worker listing
            Err(anyhow!("Not yet implemented"))
        }
        
        // Model operations
        Operation::ModelDownload { .. } => {
            // TODO: Implement model download
            Err(anyhow!("Not yet implemented"))
        }
        Operation::ModelList { .. } => {
            // TODO: Implement model listing
            Err(anyhow!("Not yet implemented"))
        }
        
        // Inference (REJECTED - should not be here!)
        Operation::Infer { .. } => {
            Err(anyhow!(
                "Infer operation should NOT be routed to hive! \
                 Queen should route inference directly to workers. \
                 This indicates a routing bug in queen-rbee."
            ))
        }
        
        // Unsupported
        _ => Err(anyhow!("Operation not supported by hive"))
    }
}
```

#### 3. HTTP Server

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    NARRATE.action("startup")
        .context(&args.port.to_string())
        .human("🐝 Starting on port {}")
        .emit();
    
    // Initialize job registry
    let job_registry = Arc::new(JobRegistry::new());
    
    // Create router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/capabilities", get(get_capabilities))
        .route("/v1/jobs", post(handle_create_job))
        .route("/v1/jobs/:job_id/stream", get(handle_stream_job))
        .with_state(HiveState { registry: job_registry });
    
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    
    NARRATE.action("listen")
        .context(&format!("http://{}", addr))
        .human("✅ Listening on {}")
        .emit();
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

### TEAM-261 Simplification

**Removed:**
- ❌ Hive heartbeat task (no longer sends heartbeat to queen)
- ❌ HiveWorkerProvider (no worker aggregation)
- ❌ CLI args: `--hive-id`, `--queen-url`

**Why?**
- Simpler architecture
- Workers send heartbeats directly to queen
- Hive is purely lifecycle management

**Impact:**
- ~110 LOC removed
- Clearer responsibilities
- Single source of truth (queen)

### Dependencies

```toml
[dependencies]
axum = { workspace = true }
tokio = { workspace = true }
job-server = { path = "../99_shared_crates/job-server" }
rbee-operations = { path = "../99_shared_crates/rbee-operations" }
rbee-hive-device-detection = { path = "../25_rbee_hive_crates/device-detection" }
```

### Status

✅ **M0 Complete (HTTP Server)**
- ✅ HTTP server running
- ✅ Job registry working
- ✅ Capabilities detection working
- ✅ SSE streaming working
- ❌ Worker lifecycle (TODO)
- ❌ Model catalog (TODO)

---

## llm-worker-rbee (HTTP Daemon - Executor)

### Purpose

**llm-worker-rbee loads ONE model and executes inference.**

### Responsibilities

1. **Model Loading**
   - Load model into VRAM/RAM
   - Initialize backend (CUDA/Metal/CPU)
   - Report readiness

2. **Inference Execution**
   - Accept inference requests
   - Generate tokens
   - Stream tokens via SSE

3. **Health Reporting**
   - Send heartbeat to queen (TEAM-261)
   - Report VRAM usage
   - Report health status

### Architecture

```
bin/30_llm_worker_rbee/
├── src/
│   ├── main.rs                 # HTTP server
│   ├── heartbeat.rs            # Heartbeat to queen
│   ├── http/
│   │   └── routes.rs           # HTTP routes
│   └── lib.rs                  # Re-exports
├── bdd/                        # BDD tests
└── Cargo.toml
```

### Key Components

#### 1. Heartbeat (TEAM-261)

**Purpose:** Send heartbeat directly to queen (not hive).

```rust
pub async fn send_heartbeat_to_queen(
    worker_id: &str,
    queen_url: &str,
    health_status: HealthStatus,
) -> Result<()> {
    Narration::new("worker-heartbeat", "send_heartbeat", worker_id)
        .human(format!("Sending heartbeat to queen at {}", queen_url))
        .emit();
    
    let payload = WorkerHeartbeatPayload {
        worker_id: worker_id.to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        health_status,
    };
    
    // TODO: Implement HTTP POST to queen
    // POST {queen_url}/v1/worker-heartbeat with payload
    
    Ok(())
}

pub fn start_heartbeat_task(
    worker_id: String,
    queen_url: String,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            if let Err(e) = send_heartbeat_to_queen(
                &worker_id,
                &queen_url,
                HealthStatus::Healthy,
            ).await {
                eprintln!("Failed to send heartbeat: {}", e);
            }
        }
    })
}
```

#### 2. Inference Endpoint

```rust
// POST /v1/inference
pub async fn handle_inference(
    State(state): State<WorkerState>,
    Json(request): Json<InferenceRequest>,
) -> Result<Response> {
    // 1. Generate tokens
    let tokens = state.model.generate(
        &request.prompt,
        request.max_tokens,
        request.temperature,
    )?;
    
    // 2. Stream via SSE
    let stream = tokens.map(|token| {
        Event::default().data(token)
    });
    
    Ok(Sse::new(stream).into_response())
}
```

#### 3. Worker State

```rust
pub struct WorkerState {
    worker_id: String,
    model: Arc<Model>,
    backend: Backend,
    device_id: u32,
}
```

### Process Lifecycle

1. **Spawn:** Hive spawns worker process
2. **Load:** Worker loads model into VRAM/RAM
3. **Ready:** Worker starts accepting requests
4. **Heartbeat:** Worker sends heartbeat to queen every 30s
5. **Execute:** Worker processes inference requests
6. **Shutdown:** Worker receives SIGTERM, gracefully shuts down

### Dependencies

```toml
[dependencies]
axum = { workspace = true }
tokio = { workspace = true }
rbee-heartbeat = { path = "../99_shared_crates/heartbeat" }
# Model loading (varies by backend)
```

### Status

✅ **M0 Complete**
- ✅ HTTP server running
- ✅ Model loading working
- ✅ Inference working
- ✅ SSE streaming working
- 🚧 Heartbeat (TODO: implement HTTP POST)

---

## Inter-Component Communication

### Request Flow Example: Worker Spawn

```
1. User runs command:
   $ rbee-keeper worker spawn --model llama-3-8b --device GPU-0

2. rbee-keeper submits to queen:
   POST http://localhost:8500/v1/jobs
   Body: {
     "operation": {
       "WorkerSpawn": {
         "hive_id": "localhost",
         "model": "llama-3-8b",
         "device": "GPU-0"
       }
     }
   }
   Response: { "job_id": "uuid-123" }

3. rbee-keeper connects to SSE:
   GET http://localhost:8500/v1/jobs/uuid-123/stream

4. queen routes operation:
   - Checks: operation.should_forward_to_hive() → true
   - Forwards to hive via job-client

5. queen forwards to hive:
   POST http://localhost:9000/v1/jobs
   Body: { "operation": { "WorkerSpawn": { ... } } }

6. rbee-hive receives operation:
   - Routes to worker lifecycle handler
   - Spawns worker process
   - Emits SSE events

7. SSE events flow back:
   hive → queen → rbee-keeper
   data: {"event": "worker_spawn", "message": "Spawning worker..."}
   data: {"event": "worker_spawn", "message": "✅ Worker started"}
   data: [DONE]

8. rbee-keeper displays output:
   Spawning worker...
   ✅ Worker started
   [DONE]
```

### Key Insights

1. **SSE Chaining:** Events flow through multiple hops
2. **Job Isolation:** Each operation has unique job_id
3. **Dual-Call Pattern:** POST for submission, GET for streaming
4. **Real-Time Feedback:** User sees progress immediately

---

## Next: Part 3 - Shared Infrastructure

The next document covers shared infrastructure that all components use:
- Job Client/Server Pattern
- Observability (Narration)
- Security Crates
- Configuration Management

**See:** `.arch/02_SHARED_INFRASTRUCTURE_PART_3.md`
