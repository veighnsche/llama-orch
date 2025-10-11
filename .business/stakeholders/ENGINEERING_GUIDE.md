# 🐝 rbee: Engineering Guide

> **A practical guide for engineers building production-grade AI orchestration** 🍯

**Pronunciation:** rbee (pronounced "are-bee")  
**Version:** 0.1.0  
**Date:** 2025-10-11  
**Audience:** Engineers, Contributors, Technical Leads  
**Website:** https://rbee.dev

**🎯 FOR ENGINEERS:** This guide explains how we actually build rbee—from architecture decisions to error handling patterns to the BDD workflow we use every day.

---

## Quick Start

```bash
# Clone & build
git clone https://github.com/yourorg/llama-orch
cd llama-orch
cargo build --release

# Run BDD tests
cd test-harness/bdd
cargo test --bin bdd-runner

# Check specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/error_handling.feature cargo test --bin bdd-runner
```

**Current Status (October 2025):**
- ✅ **42/62 BDD scenarios passing** (68% complete)
- ✅ **41 error handling functions** implemented
- ✅ Backend detection operational (CUDA, Metal, CPU)
- ✅ Dual registry system working
- 🚧 Lifecycle management in progress

---

## Critical Design Principles

### 1. FAIL FAST - NO FALLBACK ⚡

**Policy:** GPU errors FAIL IMMEDIATELY with exit code 1

```rust
// ❌ NEVER - No automatic fallback
if cuda_failed {
    warn!("Falling back to CPU");
    use_cpu();  // BANNED!
}

// ✅ ALWAYS - Fail fast
if cuda_failed {
    error!("CUDA device failed");
    return Err(Error::CudaDeviceFailed);
    // Exit code 1, NO fallback
}
```

**Why:** System integrity over convenience. User must explicitly choose backend.

### 2. Smart/Dumb Separation 🧠

**ALL intelligence lives in queen-rbee. Everything else executes commands.**

```
queen-rbee  → Makes decisions (scheduling, routing)
rbee-hive   → Executes commands (spawn worker, report state)
worker-rbee → Generates tokens (no routing logic)
```

### 3. Cascading Shutdown 🔄

**When queen-rbee dies, EVERYTHING dies gracefully.**

```
SIGTERM → queen-rbee
  ↓ SSH SIGTERM to all rbee-hives
  ↓ POST /shutdown to all workers
  ↓ Workers unload models from VRAM
  ↓ Clean exit, no orphans, no leaks
```

### 4. Process Isolation 📦

Each worker runs in separate process with its own memory context (CUDA/Metal/CPU).

---

## Error Handling Philosophy

### Industry Research (TEAM-075)

Studied production systems: **llama.cpp**, **candle-vllm**, **mistral.rs**, **ollama**

**Key Findings:**
- ✅ Structured error types with context
- ✅ Exponential backoff for transient errors
- ✅ Circuit breaker for repeated failures
- ✅ Resource cleanup guarantees (RAII)
- ✅ Actionable error messages

### Error Response Structure

```rust
ErrorResponse {
    code: "GPU_VRAM_EXHAUSTED",  // Machine-readable (UPPER_SNAKE_CASE)
    message: "GPU out of memory: 8GB required, 6GB available",
    details: Some(json!({
        "required_vram_gb": 8,
        "available_vram_gb": 6,
        "suggested_action": "Use smaller model or explicitly select CPU backend",
        "alternative_models": ["llama-3.1-3b", "phi-3-mini"]
    })),
}
```

### Exit Codes & HTTP Status

```rust
// Exit codes
0   = Success
1   = Generic error / FAIL FAST
124 = Timeout (standard Unix)
137 = SIGKILL (128 + 9)
255 = SSH/network failure

// HTTP status codes
400 = Bad Request (validation error)
429 = Too Many Requests (rate limited)
500 = Internal Server Error
503 = Service Unavailable (capacity/overload)
504 = Gateway Timeout
```

---

## System Architecture Deep Dive

### The 4-Binary System Explained

#### 👑🐝 queen-rbee (The Orchestrator)

**Location:** `bin/queen-rbee/`  
**Type:** HTTP Daemon  
**Port:** 8080  
**Language:** Rust + Axum + Rhai

**Core Responsibilities:**
1. **Admission Control** - Accept/reject inference requests based on capacity
2. **Queue Management** - FIFO queue with priority support
3. **Scheduling** - Select which worker executes which task
4. **Worker Registry** - Track all active workers (ephemeral, in-memory)
5. **Hive Registry** - Track all rbee-hive nodes (persistent, SQLite)
6. **SSE Relay** - Stream tokens from workers to clients
7. **Rhai Scripting** - User-defined routing logic
8. **Marketplace Federation** - Route to external GPU providers

**Database Schema (SQLite):**
```sql
-- Beehive Registry (Persistent)
CREATE TABLE beehives (
    node_name TEXT PRIMARY KEY,
    ssh_host TEXT NOT NULL,
    ssh_port INTEGER DEFAULT 22,
    ssh_user TEXT NOT NULL,
    ssh_key_path TEXT NOT NULL,
    git_repo_url TEXT,
    git_branch TEXT DEFAULT 'main',
    added_at INTEGER NOT NULL
);

-- Worker Registry (Ephemeral - In-Memory HashMap)
// WorkerId → WorkerInfo
struct WorkerInfo {
    id: String,
    url: String,           // http://hostname:port
    model_ref: String,
    backend: String,       // cuda, metal, cpu
    device: Option<u8>,    // GPU device number
    state: WorkerState,    // loading, idle, busy
    slots_total: u32,
    slots_available: u32,
    last_heartbeat: Instant,
}
```

**API Endpoints:**
```rust
// Task submission
POST   /v2/tasks
Body: { "prompt": "...", "model": "...", "max_tokens": 100 }
Response: { "job_id": "uuid", "status": "queued" }

// SSE streaming
GET    /v2/tasks/{job_id}/events
Response: text/event-stream
  data: {"token": "Once", "index": 0}
  data: {"token": " upon", "index": 1}
  data: {"done": true}

// Registry management
POST   /v2/registry/beehives/add
GET    /v2/registry/beehives/list
POST   /v2/registry/beehives/remove
GET    /v2/workers/list
GET    /v2/workers/health
POST   /v2/workers/shutdown
```

#### 🍯🏠 rbee-hive (Pool Manager)

**Location:** `bin/rbee-hive/` (currently CLI, will become HTTP daemon in M1)  
**Type:** CLI (M0) → HTTP Daemon (M1)  
**Port:** 9200 (M1)  
**Language:** Rust

**Core Responsibilities:**
1. **Model Catalog** - Track downloaded models (SQLite)
2. **Model Download** - Fetch from Hugging Face with progress
3. **Backend Detection** - Detect CUDA, Metal, CPU availability
4. **Worker Spawning** - Launch worker processes
5. **Worker Lifecycle** - Monitor health, enforce idle timeout
6. **Preflight Checks** - Verify VRAM, disk space before spawn
7. **Orphan Cleanup** - Detect and kill dead workers

**Database Schema (SQLite):**
```sql
CREATE TABLE models (
    id TEXT PRIMARY KEY,              -- sha256 hash
    provider TEXT NOT NULL,           -- 'hf' or 'file'
    reference TEXT NOT NULL,          -- 'meta-llama/Llama-3.1-8B'
    local_path TEXT NOT NULL,         -- '/models/llama-3.1-8b.gguf'
    size_bytes INTEGER,
    downloaded_at INTEGER,
    last_used_at INTEGER
);

CREATE INDEX idx_models_reference ON models(reference);
```

**Commands (M0):**
```bash
# Backend detection
rbee-hive detect

# Model management
rbee-hive model download --ref meta-llama/Llama-3.1-8B
rbee-hive model list

# Worker management
rbee-hive worker spawn cuda --model tinyllama.gguf --gpu 0 --port 8001
rbee-hive worker list
rbee-hive worker kill <pid>
```

**API Endpoints (M1):**
```rust
// Model operations
POST   /v1/models/download
GET    /v1/models/download/progress  // SSE
GET    /v1/models/list

// Worker operations
POST   /v1/workers/spawn
GET    /v1/workers/list
POST   /v1/workers/ready             // Callback from worker
POST   /v1/workers/shutdown
```

#### 🐝💪 llm-worker-rbee (Worker)

**Location:** `bin/llm-worker-rbee/`  
**Type:** HTTP Daemon  
**Ports:** 8001+  
**Language:** Rust + Candle (ML framework)

**Core Responsibilities:**
1. **Model Loading** - Load GGUF into VRAM/memory
2. **Inference Execution** - Generate tokens via Candle
3. **SSE Streaming** - Stream tokens token-by-token
4. **State Management** - Track loading/idle/busy state
5. **Ready Callback** - Notify rbee-hive when ready
6. **Graceful Shutdown** - Unload model, release VRAM

**State Machine:**
```
┌─────────┐
│ loading │ ──────────────┐
└────┬────┘               │
     │                    │
     ↓                    ↓
┌─────────┐          ┌──────────┐
│  idle   │ ←──────→ │   busy   │
└────┬────┘          └─────┬────┘
     │                     │
     │                     │
     └─────────┬───────────┘
               ↓
          ┌──────────┐
          │ stopping │
          └────┬─────┘
               ↓
          ┌──────────┐
          │ stopped  │
          └──────────┘
```

**API Endpoints:**
```rust
// Inference
POST   /v1/execute
Body: { "prompt": "...", "max_tokens": 100, "stream": true }
Response: text/event-stream (SSE)

// Health & readiness
GET    /v1/health
GET    /v1/ready
GET    /v1/loading/progress  // SSE during model load

// Lifecycle
POST   /v1/cancel            // Cancel active inference
POST   /v1/admin/shutdown    // Graceful shutdown
```

**Worker Startup Flow:**
```
1. Parse CLI args (--model, --backend, --port, --gpu)
2. Initialize HTTP server on port
3. Send POST /v1/workers/ready to rbee-hive
4. Start model loading (async)
5. Stream loading progress via SSE
6. Transition to idle state
7. Accept inference requests
```

#### 🧑‍🌾🐝 rbee-keeper (User Interface)

**Location:** `bin/rbee-keeper/`  
**Type:** CLI + Web UI  
**Language:** Rust (CLI) + Vue.js (Web UI)

**Core Responsibilities:**
1. **queen-rbee Lifecycle** - Start/stop/status daemon
2. **SSH Configuration** - Manage remote node credentials
3. **Inference Interface** - Submit jobs, display results
4. **Visual Management** - Web UI for monitoring (future)

**CLI Commands:**
```bash
# Setup
rbee-keeper setup add-node --name mac --ssh-host mac.local
rbee-keeper setup install --node mac
rbee-keeper setup list-nodes
rbee-keeper setup remove-node --name mac

# Daemon management (M0 - in progress)
rbee-keeper daemon start
rbee-keeper daemon stop
rbee-keeper daemon status

# Inference
rbee-keeper infer --node mac --model tinyllama --prompt "hello"

# Hive management (M1)
rbee-keeper hive start --node mac
rbee-keeper hive stop --node mac
rbee-keeper hive status --node mac

# Worker management (M1)
rbee-keeper worker start --node mac --model llama-7b
rbee-keeper worker stop --node mac --worker-id <id>
rbee-keeper worker list --node mac
```

---

## Complete Orchestration Flow (40+ Steps)

### Cold Start Inference (No Workers Available)

```
═══════════════════════════════════════════════════════════
PHASE 0: Setup (One-Time Configuration)
═══════════════════════════════════════════════════════════

Step 1:  User runs: rbee-keeper setup add-node --name workstation
Step 2:  rbee-keeper → queen-rbee: POST /v2/registry/beehives/add
Step 3:  queen-rbee tests SSH connection to workstation
Step 4:  queen-rbee saves to beehives table (SQLite)
Step 5:  rbee-keeper → queen-rbee: POST /v2/registry/beehives/install
Step 6:  queen-rbee runs via SSH: git clone + cargo build
Step 7:  Binaries installed on remote node

═══════════════════════════════════════════════════════════
PHASE 1: Job Submission
═══════════════════════════════════════════════════════════

Step 8:  User runs: rbee-keeper infer --node workstation --model tinyllama
Step 9:  rbee-keeper spawns queen-rbee as child process (port 8080)
Step 10: rbee-keeper → queen-rbee: POST /v2/tasks
         Body: { "prompt": "hello", "model": "tinyllama", "node": "workstation" }
Step 11: queen-rbee queries beehives table for SSH details
Step 12: queen-rbee creates job_id, adds to queue

═══════════════════════════════════════════════════════════
PHASE 2: Hive Startup
═══════════════════════════════════════════════════════════

Step 13: queen-rbee → SSH: start rbee-hive daemon on workstation
Step 14: rbee-hive starts HTTP daemon on port 9200
Step 15: rbee-hive responds: { "status": "ready", "port": 9200 }

═══════════════════════════════════════════════════════════
PHASE 3: Worker Registry Check
═══════════════════════════════════════════════════════════

Step 16: queen-rbee → rbee-hive: GET /v1/workers/list
Step 17: rbee-hive responds: { "workers": [] }  (empty)
Step 18: queen-rbee decides: need to spawn worker

═══════════════════════════════════════════════════════════
PHASE 4: Model Provisioning
═══════════════════════════════════════════════════════════

Step 19: queen-rbee → rbee-hive: POST /v1/models/download
         Body: { "reference": "tinyllama", "provider": "hf" }
Step 20: rbee-hive queries model catalog (SQLite)
Step 21: rbee-hive: Model not found, initiating download
Step 22: rbee-hive downloads from Hugging Face
Step 23: rbee-hive streams progress via SSE:
         data: {"bytes_downloaded": 1048576, "total_bytes": 10485760}
Step 24: rbee-keeper displays progress bar to user
Step 25: rbee-hive inserts model into catalog (SQLite)

═══════════════════════════════════════════════════════════
PHASE 5: Worker Preflight
═══════════════════════════════════════════════════════════

Step 26: rbee-hive checks VRAM availability (NVML query)
Step 27: rbee-hive checks backend availability (CUDA detected)
Step 28: rbee-hive checks disk space (sufficient)
Step 29: rbee-hive selects port: 8001 (first available)

═══════════════════════════════════════════════════════════
PHASE 6: Worker Startup
═══════════════════════════════════════════════════════════

Step 30: rbee-hive spawns: llm-cuda-worker-rbee --model tinyllama.gguf --port 8001 --gpu 0
Step 31: Worker process starts, initializes HTTP server
Step 32: Worker → rbee-hive: POST /v1/workers/ready
         Body: { "worker_id": "uuid", "port": 8001, "model": "tinyllama" }
Step 33: rbee-hive → queen-rbee: POST /v2/registry/workers/ready
Step 34: queen-rbee adds worker to in-memory registry

═══════════════════════════════════════════════════════════
PHASE 7: Model Loading
═══════════════════════════════════════════════════════════

Step 35: Worker loads model into VRAM (asynchronous)
Step 36: Worker streams loading progress via SSE:
         data: {"stage": "loading_weights", "progress": 0.5}
Step 37: rbee-keeper polls: GET /v1/ready (returns 503 while loading)
Step 38: Worker completes loading → state = "idle"
Step 39: Worker responds: GET /v1/ready → 200 OK

═══════════════════════════════════════════════════════════
PHASE 8: Inference Execution
═══════════════════════════════════════════════════════════

Step 40: queen-rbee routes job to worker (direct HTTP)
Step 41: queen-rbee → worker: POST http://workstation:8001/v1/execute
Step 42: Worker state: idle → busy
Step 43: Worker generates tokens via Candle
Step 44: Worker streams tokens via SSE:
         data: {"token": "Hello", "index": 0}
         data: {"token": " world", "index": 1}
         data: {"done": true, "total_tokens": 2}
Step 45: queen-rbee relays SSE stream to rbee-keeper
Step 46: rbee-keeper displays tokens to stdout
Step 47: Worker state: busy → idle

═══════════════════════════════════════════════════════════
PHASE 9: Idle Timeout (5 minutes later)
═══════════════════════════════════════════════════════════

Step 48: rbee-hive monitors worker health every 30s
Step 49: Worker idle for 5 minutes (no requests)
Step 50: rbee-hive → worker: POST /v1/admin/shutdown
Step 51: Worker unloads model from VRAM
Step 52: Worker exits cleanly (exit code 0)
Step 53: VRAM freed for other applications
Step 54: queen-rbee removes worker from registry
```

---

## Production Error Patterns

### Pattern 1: GPU Errors - FAIL FAST

```rust
#[when(expr = "CUDA device {int} fails")]
pub async fn when_cuda_device_fails(world: &mut World, device: u8) {
    world.last_exit_code = Some(1);
    world.last_error = Some(ErrorResponse {
        code: "CUDA_DEVICE_FAILED".to_string(),
        message: format!("CUDA device {} initialization failed", device),
        details: Some(json!({
            "device": device,
            "suggested_action": "Check GPU drivers or explicitly select CPU backend"
        })),
    });
    tracing::error!("❌ CUDA FAILED - exiting immediately (NO FALLBACK)");
}
```

### Pattern 2: Exponential Backoff

```rust
// Retry schedule: 1s, 2s, 4s, 8s, 16s (max 5 attempts)
for attempt in 0..max_attempts {
    match operation() {
        Ok(result) => return Ok(result),
        Err(e) if is_transient(&e) => {
            let delay = Duration::from_secs(1) * 2_u32.pow(attempt);
            let jitter = rand::random::<u64>() % 1000;
            tokio::time::sleep(delay + Duration::from_millis(jitter)).await;
        }
        Err(e) => return Err(e),
    }
}
```

### Pattern 3: Circuit Breaker

```rust
// Open after 5 failures, cooldown 60s
match self.state {
    CircuitState::Open => {
        if self.last_failure.elapsed() > Duration::from_secs(60) {
            self.state = CircuitState::HalfOpen;
        } else {
            return Err(Error::CircuitBreakerOpen);
        }
    }
    _ => {}
}
```

### Pattern 4: Concurrent Limits

```rust
// Return 503 when at capacity
if current_requests >= max_concurrent {
    return Ok(Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header("Retry-After", "30")
        .body(json!({
            "error": {
                "code": "SERVICE_UNAVAILABLE",
                "details": { "retry_after_seconds": 30 }
            }
        })).unwrap());
}
```

### Pattern 5: Timeout Cascades

```rust
// Timeout hierarchy: outer > inner (leave cleanup time)
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const INFERENCE_TIMEOUT: Duration = Duration::from_secs(25);
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

match tokio::time::timeout(INFERENCE_TIMEOUT, run_inference()).await {
    Ok(Ok(response)) => Ok(response),
    Err(_) => {
        // Graceful cancellation with cleanup timeout
        let _ = tokio::time::timeout(CLEANUP_TIMEOUT, cancel_inference()).await;
        Err(Error::InferenceTimeout)
    }
}
```

### Pattern 6: Resource Cleanup

```rust
// CRITICAL: NEVER block in Drop!

// ❌ BAD - Blocks during drop
impl Drop for Resource {
    fn drop(&mut self) {
        std::thread::sleep(Duration::from_secs(5)); // HANGS!
    }
}

// ✅ GOOD - Non-blocking
impl Drop for Resource {
    fn drop(&mut self) {
        // Let OS handle cleanup
    }
}

// ✅ BEST - Explicit cleanup method
impl Resource {
    pub async fn shutdown(&mut self) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(5), self.cleanup()).await?;
        Ok(())
    }
}
```

---

## Development Workflow (TEAM-XXX Pattern)

### How We Build rbee

```
1. Write Gherkin Feature (.feature file)
   ↓
2. Implement Step Definitions (Rust)
   ↓
3. Run BDD Tests (bdd-runner)
   ↓
4. Iterate Until Green
   ↓
5. Handoff to Next Team
```

### Example: TEAM-075 Error Handling

**Phase 1: Research (2h)** → Industry error patterns from llama.cpp, candle-vllm  
**Phase 2: Documentation (1h)** → 3 markdown files created  
**Phase 3: Implementation (1h)** → 15 MVP edge case functions  
**Phase 4: Verification (<1h)** → `cargo check` passes  

**Result:** 15 functions, 3 docs, zero test fraud, FAIL FAST enforced

---

## Testing Strategy

### BDD Tests

```bash
cd test-harness/bdd
cargo test --bin bdd-runner  # Run all tests

# Target specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/error_handling.feature cargo test
```

**Current:** 42/62 scenarios passing (68%)

### Proof Bundles

```bash
LLORCH_RUN_ID=test-001 cargo test
```

**Output:**
```
.proof_bundle/integration/test-001/
├── seeds.json         # RNG seeds
├── transcript.ndjson  # SSE events
├── metadata.json      # Model, device, versions
└── result.txt         # Final output
```

**Benefits:** Deterministic testing, debugging aid, regression detection

---

## Code Architecture

```
llama-orch/
├── bin/
│   ├── queen-rbee/          # Orchestrator (smart)
│   ├── rbee-keeper/         # CLI + Web UI
│   ├── llm-worker-rbee/     # LLM worker (dumb)
│   └── shared-crates/       # 11 built libraries
├── test-harness/bdd/
│   ├── src/steps/           # Step definitions
│   │   ├── error_handling.rs    # 41 functions
│   │   ├── error_helpers.rs
│   │   └── world.rs
│   ├── ERROR_*.md           # TEAM-075 docs
│   └── tests/features/      # Gherkin features
└── reference/               # Industry code
    ├── llama.cpp/
    ├── candle-vllm/
    └── mistral.rs/
```

### Shared Crates (Use These!)

```rust
use audit_logging::AuditLog;        // GDPR compliance
use auth_min::AuthToken;            // Minimal auth
use gpu_info::detect_gpus;          // Backend detection
use narration_core::Narrator;       // User messages
use deadline_propagation::Deadline; // Timeout handling
use proof_bundle::ProofBundle;      // Test artifacts
```

---

## Contributing

### 1. Find/Create Feature

```gherkin
Scenario: GPU VRAM exhaustion fails immediately
  When GPU VRAM is exhausted
  Then rbee-hive fails immediately
  And exit code is 1
```

### 2. Implement Step

```rust
// TEAM-XXX: Your signature
#[when(expr = "GPU VRAM is exhausted")]
pub async fn when_gpu_vram_exhausted(world: &mut World) {
    world.last_exit_code = Some(1);
    world.last_error = Some(ErrorResponse {
        code: "GPU_VRAM_EXHAUSTED".to_string(),
        message: "GPU out of memory".to_string(),
        details: Some(json!({ "suggested_action": "Use smaller model" })),
    });
}
```

### 3. Test & Document

```bash
cargo check --bin bdd-runner
cargo test --bin bdd-runner

# Create TEAM_XXX_SUMMARY.md
```

---

## Common Pitfalls

### ❌ Blocking in Drop
```rust
// NEVER do this - causes hangs!
impl Drop for Resource {
    fn drop(&mut self) { std::thread::sleep(Duration::from_secs(5)); }
}
```

### ❌ Automatic Fallback
```rust
// NO GPU fallback to CPU!
if cuda_failed { use_cpu(); }  // BANNED!
```

### ❌ Vague Errors
```rust
// Bad: "Something went wrong"
// Good: "GPU VRAM exhausted: 8GB required, 6GB available"
```

### ❌ Missing Error Details
```rust
// Always include suggested_action in details
```

---

## Multi-Modal Protocol Handling

### Protocol Matrix

| Capability  | Protocol      | Content-Type           | Workers              | Response Format |
|-------------|---------------|------------------------|----------------------|-----------------|
| text-gen    | SSE           | text/event-stream      | llm-*-worker-rbee    | Streaming       |
| image-gen   | JSON          | application/json       | sd-*-worker-rbee     | Base64/Binary   |
| audio-gen   | Binary        | audio/mpeg             | tts-*-worker-rbee    | Binary stream   |
| embedding   | JSON          | application/json       | embed-*-worker-rbee  | Vector array    |

### Text Generation (SSE Streaming)

```rust
// Request
POST /v1/execute
Content-Type: application/json
{
  "prompt": "write a story",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": true
}

// Response
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

data: {"token": "Once", "index": 0, "logprob": -0.5}
data: {"token": " upon", "index": 1, "logprob": -0.3}
data: {"token": " a", "index": 2, "logprob": -0.2}
data: {"done": true, "total_tokens": 100, "finish_reason": "length"}
```

### Image Generation (JSON Response)

```rust
// Request
POST /v1/execute
Content-Type: application/json
{
  "prompt": "a cat on a couch",
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "guidance_scale": 7.5
}

// Response
HTTP/1.1 200 OK
Content-Type: application/json
{
  "images": [{
    "format": "png",
    "data": "iVBORw0KGgoAAAANSUhEUgAA...",  // base64
    "width": 1024,
    "height": 1024,
    "seed": 42
  }],
  "generation_time_ms": 2500
}
```

### Audio Generation (Binary Stream)

```rust
// Request
POST /v1/execute
Content-Type: application/json
{
  "text": "Hello, world!",
  "voice": "en-US-female",
  "format": "mp3"
}

// Response
HTTP/1.1 200 OK
Content-Type: audio/mpeg
Content-Length: 45678

<binary MP3 data>
```

### Embeddings (JSON Response)

```rust
// Request
POST /v1/execute
Content-Type: application/json
{
  "input": "The quick brown fox",
  "model": "text-embedding-ada-002"
}

// Response
HTTP/1.1 200 OK
Content-Type: application/json
{
  "embeddings": [
    [0.123, -0.456, 0.789, ...],  // 1536 dimensions
  ],
  "model": "text-embedding-ada-002",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

---

## Advanced Patterns & Optimizations

### Pattern 7: Model Corruption Detection

```rust
// TEAM-075: SHA256 verification after download
pub async fn verify_model_checksum(
    file_path: &Path,
    expected_sha256: &str,
) -> Result<()> {
    let mut hasher = Sha256::new();
    let mut file = File::open(file_path).await?;
    let mut buffer = vec![0; 8192];
    
    loop {
        let n = file.read(&mut buffer).await?;
        if n == 0 { break; }
        hasher.update(&buffer[..n]);
    }
    
    let actual = format!("{:x}", hasher.finalize());
    
    if actual != expected_sha256 {
        // Delete corrupted file
        tokio::fs::remove_file(file_path).await?;
        
        return Err(Error::ModelCorrupted {
            expected: expected_sha256.to_string(),
            actual,
            action: "deleted_and_retrying",
        });
    }
    
    Ok(())
}
```

### Pattern 8: Worker Health Monitoring

```rust
// rbee-hive monitors worker health every 30s
pub struct HealthMonitor {
    workers: Arc<RwLock<HashMap<WorkerId, WorkerInfo>>>,
    check_interval: Duration,
}

impl HealthMonitor {
    pub async fn run(&self) {
        let mut interval = tokio::time::interval(self.check_interval);
        
        loop {
            interval.tick().await;
            
            let workers = self.workers.read().await;
            for (id, info) in workers.iter() {
                // Check heartbeat
                if info.last_heartbeat.elapsed() > Duration::from_secs(60) {
                    tracing::warn!("Worker {} heartbeat timeout", id);
                    // Mark as unhealthy
                }
                
                // Check idle timeout (5 minutes)
                if info.state == WorkerState::Idle 
                    && info.last_request.elapsed() > Duration::from_secs(300) 
                {
                    tracing::info!("Worker {} idle timeout, shutting down", id);
                    let _ = self.shutdown_worker(id).await;
                }
            }
        }
    }
}
```

### Pattern 9: Request Deduplication

```rust
// Prevent duplicate requests for same prompt
pub struct RequestDeduplicator {
    pending: Arc<RwLock<HashMap<String, Arc<Notify>>>>,
}

impl RequestDeduplicator {
    pub async fn execute<F, T>(&self, key: String, f: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
        T: Clone,
    {
        // Check if request already pending
        let notify = {
            let mut pending = self.pending.write().await;
            if let Some(notify) = pending.get(&key) {
                // Wait for existing request
                notify.clone()
            } else {
                // Start new request
                let notify = Arc::new(Notify::new());
                pending.insert(key.clone(), notify.clone());
                notify
            }
        };
        
        // Execute or wait
        let result = f.await;
        
        // Notify waiters
        notify.notify_waiters();
        self.pending.write().await.remove(&key);
        
        result
    }
}
```

### Pattern 10: Adaptive Batching

```rust
// Batch multiple small requests for efficiency
pub struct AdaptiveBatcher {
    batch_size: usize,
    batch_timeout: Duration,
    pending: Vec<Request>,
}

impl AdaptiveBatcher {
    pub async fn add_request(&mut self, req: Request) -> Result<Response> {
        self.pending.push(req);
        
        // Flush if batch full or timeout
        if self.pending.len() >= self.batch_size {
            return self.flush_batch().await;
        }
        
        // Wait for more requests or timeout
        tokio::select! {
            _ = tokio::time::sleep(self.batch_timeout) => {
                self.flush_batch().await
            }
            req = self.next_request() => {
                self.pending.push(req);
                if self.pending.len() >= self.batch_size {
                    self.flush_batch().await
                } else {
                    // Continue batching
                    Ok(Response::Pending)
                }
            }
        }
    }
    
    async fn flush_batch(&mut self) -> Result<Response> {
        let batch = std::mem::take(&mut self.pending);
        // Send batch to worker
        execute_batch(batch).await
    }
}
```

---

## Performance Tuning

### VRAM Management

```rust
// Monitor VRAM usage before spawning workers
pub fn check_vram_availability(required_gb: u32) -> Result<()> {
    let devices = nvml::Device::all()?;
    
    for (idx, device) in devices.iter().enumerate() {
        let memory = device.memory_info()?;
        let available_gb = memory.free / (1024 * 1024 * 1024);
        
        if available_gb >= required_gb as u64 {
            tracing::info!(
                "GPU {} has {}GB available ({}GB required)",
                idx, available_gb, required_gb
            );
            return Ok(());
        }
    }
    
    Err(Error::InsufficientVRAM {
        required_gb,
        available_gb: devices.iter()
            .map(|d| d.memory_info().unwrap().free / (1024*1024*1024))
            .max()
            .unwrap_or(0) as u32,
    })
}
```

### Connection Pooling

```rust
// Reuse HTTP connections to workers
pub struct WorkerPool {
    clients: HashMap<WorkerId, Client>,
}

impl WorkerPool {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
        }
    }
    
    pub fn get_client(&mut self, worker_id: &WorkerId, url: &str) -> &Client {
        self.clients.entry(worker_id.clone()).or_insert_with(|| {
            Client::builder()
                .pool_max_idle_per_host(10)
                .pool_idle_timeout(Duration::from_secs(90))
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap()
        })
    }
}
```

### Async Task Spawning

```rust
// Spawn workers concurrently for faster startup
pub async fn spawn_workers_parallel(
    models: Vec<ModelRef>,
) -> Result<Vec<WorkerId>> {
    let tasks: Vec<_> = models
        .into_iter()
        .map(|model| {
            tokio::spawn(async move {
                spawn_worker(model).await
            })
        })
        .collect();
    
    // Wait for all workers to start
    let results = futures::future::join_all(tasks).await;
    
    results
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
}
```

---

## Debugging & Troubleshooting

### Enable Debug Logging

```bash
# Set log level
export RUST_LOG=debug
export RUST_BACKTRACE=1

# Run with verbose logging
rbee-keeper infer --node mac --model tinyllama --prompt "hello"

# Filter specific modules
export RUST_LOG=rbee_hive=debug,llm_worker=trace
```

### Inspect Proof Bundles

```bash
# Run test with proof bundle
LLORCH_RUN_ID=debug-001 cargo test --bin bdd-runner

# Inspect artifacts
cd test-harness/bdd/.proof_bundle/integration/debug-001/
cat seeds.json         # RNG seeds used
cat metadata.json      # Model, device, versions
jq . transcript.ndjson # SSE events (pretty-print)
cat result.txt         # Final output
```

### Common Issues & Solutions

#### Issue 1: Worker Hangs During Model Load

**Symptoms:** Worker stuck in "loading" state, no progress

**Debug:**
```bash
# Check worker logs
journalctl -u llm-worker-rbee -f

# Check VRAM usage
nvidia-smi

# Check worker HTTP endpoint
curl http://localhost:8001/v1/loading/progress
```

**Solutions:**
- Insufficient VRAM → Use smaller model or free VRAM
- Corrupted model file → Delete and re-download
- CUDA driver issue → Update NVIDIA drivers

#### Issue 2: Tests Hang Indefinitely

**Symptoms:** BDD tests never exit, processes remain

**Debug:**
```bash
# Check for orphaned processes
ps aux | grep rbee

# Check port bindings
lsof -i :8080
lsof -i :8001
```

**Solutions:**
- Blocking in Drop → Use explicit cleanup before drop
- Missing cascading shutdown → Implement SIGTERM handlers
- Port conflicts → Kill existing processes

#### Issue 3: VRAM Leaks

**Symptoms:** VRAM not freed after worker shutdown

**Debug:**
```bash
# Monitor VRAM over time
watch -n 1 nvidia-smi

# Check worker state
curl http://localhost:8001/v1/health
```

**Solutions:**
- Worker not unloading model → Call explicit unload before exit
- Process killed before cleanup → Use graceful shutdown
- CUDA context not released → Ensure Drop is called

#### Issue 4: High Latency

**Symptoms:** Slow token generation, high p95 latency

**Debug:**
```bash
# Profile with perf
perf record -g cargo run --release
perf report

# Check CPU/GPU usage
htop
nvidia-smi dmon
```

**Solutions:**
- CPU bottleneck → Use GPU backend
- Model too large → Use quantized model
- Network latency → Use local workers
- Too many concurrent requests → Increase worker count

---

## Key Documentation

**For Engineers:**
- `ERROR_HANDLING_RESEARCH.md` - Industry patterns
- `ERROR_PATTERNS.md` - Implementation guide
- `EDGE_CASES_CATALOG.md` - MVP edge cases

**For Contributors:**
- `TEAM_XXX_SUMMARY.md` files - Handoff documents
- `tests/features/*.feature` - Gherkin specs
- `.docs/workflow.md` - Development process

---

---

## Rhai Scripting (User-Defined Routing)

### What is Rhai?

**Rhai** is an embedded scripting language for Rust. It allows users to write custom routing logic **without recompiling rbee**.

### Basic Routing Script

```rhai
// Route based on model size
fn route_task(task, workers) {
    if task.model.contains("70b") {
        // Large models need multi-GPU
        return workers.filter(|w| w.gpus > 1).first();
    } else if task.model.contains("7b") {
        // Medium models use single GPU
        return workers.filter(|w| w.backend == "cuda").least_loaded();
    } else {
        // Small models can use CPU
        return workers.filter(|w| w.backend == "cpu").first();
    }
}
```

### Advanced Routing: Priority + Load Balancing

```rhai
fn route_task(task, workers) {
    // High priority tasks get best GPUs
    if task.priority == "high" {
        let cuda_workers = workers.filter(|w| w.backend == "cuda");
        return cuda_workers.least_loaded();
    }
    
    // EU compliance mode
    if task.compliance == "gdpr" {
        let eu_workers = workers.filter(|w| w.region == "EU");
        if eu_workers.is_empty() {
            return error("No EU workers available");
        }
        return eu_workers.least_loaded();
    }
    
    // Default: round-robin
    return workers.round_robin();
}
```

### Admission Control Script

```rhai
fn should_admit(task, queue_depth, workers) {
    // Reject if queue too deep
    if queue_depth > 100 {
        return false;
    }
    
    // Reject if no workers available
    if workers.is_empty() {
        return false;
    }
    
    // Reject if all workers busy
    let idle_workers = workers.filter(|w| w.state == "idle");
    if idle_workers.is_empty() && queue_depth > 10 {
        return false;
    }
    
    return true;
}
```

### Marketplace Routing Script

```rhai
fn route_task(task, local_workers, marketplace_providers) {
    // Try local workers first
    let available_local = local_workers.filter(|w| w.state == "idle");
    if !available_local.is_empty() {
        return available_local.least_loaded();
    }
    
    // Fallback to marketplace
    let providers = marketplace_providers.filter(|p| {
        p.pricing.per_token < 0.001 &&  // Max price
        p.geo.region == "EU" &&          // GDPR compliance
        p.sla.uptime > 0.99              // High availability
    });
    
    if providers.is_empty() {
        return error("No suitable providers");
    }
    
    // Select cheapest provider
    return providers.sort_by(|a, b| a.pricing.per_token < b.pricing.per_token).first();
}
```

---

## Security Considerations

### API Key Management

```rust
// Store API keys securely
use secrets_management::SecretStore;

let store = SecretStore::new("~/.rbee/secrets.db")?;
store.set("openai_api_key", api_key)?;

// Retrieve at runtime
let api_key = store.get("openai_api_key")?;
```

### SSH Key Permissions

```bash
# Correct SSH key permissions
chmod 600 ~/.ssh/rbee_key
chmod 700 ~/.ssh

# Test SSH connection
ssh -i ~/.ssh/rbee_key user@hostname "echo 'Connection OK'"
```

### GDPR Compliance Mode

```bash
# Enable audit logging
export LLORCH_EU_AUDIT=true
export LLORCH_AUDIT_LOG_PATH=/var/log/rbee/audit.log

# Start queen-rbee with compliance mode
queen-rbee daemon --compliance-mode gdpr

# Audit log format (immutable, append-only)
{
  "timestamp": "2025-10-11T12:00:00Z",
  "event_type": "inference_request",
  "user_id": "user-123",
  "model": "llama-3.1-8b",
  "prompt_hash": "sha256:abc123...",
  "worker_id": "worker-456",
  "region": "EU"
}
```

### Network Security

```rust
// TLS for production
use rustls::ServerConfig;

let config = ServerConfig::builder()
    .with_safe_defaults()
    .with_no_client_auth()
    .with_single_cert(certs, key)?;

let listener = TlsListener::new(config, addr);
```

---

## Deployment Patterns

### Single-Node (Development)

```bash
# Start all components locally
queen-rbee daemon &
rbee-hive daemon &
rbee-hive worker spawn cuda --model llama-7b --gpu 0

# Use SDK
curl -X POST http://localhost:8080/v2/tasks \
  -d '{"prompt": "hello", "model": "llama-7b"}'
```

### Multi-Node (Production)

```bash
# Orchestrator node
queen-rbee daemon --bind 0.0.0.0:8080 &

# Worker nodes (via SSH)
ssh node1 "rbee-hive daemon --bind 0.0.0.0:9200 &"
ssh node2 "rbee-hive daemon --bind 0.0.0.0:9200 &"

# Configure nodes
rbee-keeper setup add-node --name node1 --ssh-host node1.internal
rbee-keeper setup add-node --name node2 --ssh-host node2.internal

# Spawn workers
rbee-keeper worker start --node node1 --model llama-7b
rbee-keeper worker start --node node2 --model llama-70b
```

### Docker Deployment

```dockerfile
# Dockerfile for llm-worker-rbee
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin llm-worker-rbee

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
COPY --from=builder /app/target/release/llm-worker-rbee /usr/local/bin/
ENTRYPOINT ["llm-worker-rbee"]
```

```bash
# Run worker in Docker
docker run --gpus all \
  -p 8001:8001 \
  -v /models:/models \
  rbee/llm-worker-rbee:latest \
  --model /models/llama-7b.gguf \
  --backend cuda \
  --port 8001
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-worker
  template:
    metadata:
      labels:
        app: llm-worker
    spec:
      containers:
      - name: worker
        image: rbee/llm-worker-rbee:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: MODEL_PATH
          value: /models/llama-7b.gguf
        - name: BACKEND
          value: cuda
        ports:
        - containerPort: 8001
```

---

## Monitoring & Observability

### Metrics Collection

```rust
// Emit Prometheus metrics
use prometheus::{Counter, Histogram, Registry};

lazy_static! {
    static ref INFERENCE_REQUESTS: Counter = Counter::new(
        "rbee_inference_requests_total",
        "Total inference requests"
    ).unwrap();
    
    static ref INFERENCE_LATENCY: Histogram = Histogram::new(
        "rbee_inference_latency_seconds",
        "Inference latency in seconds"
    ).unwrap();
}

// Record metrics
INFERENCE_REQUESTS.inc();
let timer = INFERENCE_LATENCY.start_timer();
// ... execute inference ...
timer.observe_duration();
```

### Health Checks

```rust
// Health check endpoint
#[get("/health")]
async fn health_check(state: State<AppState>) -> Result<Json<HealthStatus>> {
    let workers = state.workers.read().await;
    let healthy = workers.values().filter(|w| w.is_healthy()).count();
    
    Ok(Json(HealthStatus {
        status: if healthy > 0 { "healthy" } else { "unhealthy" },
        workers_total: workers.len(),
        workers_healthy: healthy,
        uptime_seconds: state.start_time.elapsed().as_secs(),
    }))
}
```

### Distributed Tracing

```rust
// OpenTelemetry integration
use opentelemetry::trace::{Tracer, Span};

async fn execute_inference(tracer: &Tracer, request: Request) -> Result<Response> {
    let mut span = tracer.start("execute_inference");
    span.set_attribute("model", request.model.clone());
    span.set_attribute("prompt_length", request.prompt.len() as i64);
    
    let result = do_inference(request).await;
    
    span.end();
    result
}
```

---

## Performance Benchmarks

### Target Metrics (M0)

| Metric | Target | Current | Notes |
|--------|--------|---------|-------|
| First token latency | <100ms p95 | TBD | After model loaded |
| Per-token latency | <50ms p95 | TBD | Streaming |
| Admission latency | <10ms p95 | TBD | Queue admission |
| Scheduling latency | <50ms p95 | TBD | Worker selection |
| Model load time | <30s | TBD | 7B model to VRAM |

### Scalability Targets

| Resource | Target | Notes |
|----------|--------|-------|
| Workers per hive | 100+ | Single rbee-hive instance |
| Hives per queen | 1000+ | Single queen-rbee instance |
| Concurrent jobs | 10,000+ | Across all workers |
| Tokens/second | 1M+ | Aggregate throughput |

### Benchmark Commands

```bash
# Benchmark inference latency
cargo bench --bench inference_latency

# Load test with wrk
wrk -t 12 -c 400 -d 30s \
  --script load_test.lua \
  http://localhost:8080/v2/tasks

# Profile with flamegraph
cargo flamegraph --bin llm-worker-rbee
```

---

## Migration & Upgrade Guide

### Upgrading from M0 to M1

**Breaking Changes:**
- rbee-hive becomes HTTP daemon (was CLI)
- Worker registry moves from SQLite to in-memory
- API endpoints change from v1 to v2

**Migration Steps:**

```bash
# 1. Backup databases
cp ~/.rbee/beehives.db ~/.rbee/beehives.db.backup
cp ~/.rbee/models.db ~/.rbee/models.db.backup

# 2. Stop all services
rbee-keeper daemon stop
ssh node1 "pkill rbee-hive"

# 3. Upgrade binaries
cargo build --release
cargo install --path bin/queen-rbee
cargo install --path bin/rbee-hive

# 4. Migrate database schema
rbee-hive migrate --from v0.1 --to v0.2

# 5. Restart services
rbee-keeper daemon start
```

### Rollback Procedure

```bash
# 1. Stop new version
rbee-keeper daemon stop

# 2. Restore backups
cp ~/.rbee/beehives.db.backup ~/.rbee/beehives.db

# 3. Reinstall old version
git checkout v0.1.0
cargo build --release
cargo install --path bin/queen-rbee

# 4. Restart
rbee-keeper daemon start
```

---

**Built with 🍯 by AI engineering teams**  
**Last Updated:** 2025-10-11  
**Status:** M0 in progress, 42/62 scenarios passing  
**Document Version:** 2.0 (In-Depth Edition)
