# llama-orch Orchestration Overview
**Complete List of Actions & Behaviors**

**Created by:** TEAM-024  
**Date:** 2025-10-09  
**Purpose:** High-level overview of everything you can orchestrate with llama-orch

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Control Plane Operations (SSH)](#control-plane-operations-ssh)
3. [Data Plane Operations (HTTP)](#data-plane-operations-http)
4. [Complete Workflow Examples](#complete-workflow-examples)
5. [Component Responsibilities](#component-responsibilities)
6. [Quick Reference Commands](#quick-reference-commands)

---

## System Architecture

### IMPORTANT: The Four Binaries (TEAM-024)

**llama-orch has 4 binaries:**

1. **`rbees-orcd`** = HTTP daemon (THE BRAIN) - `bin/rbees-orcd/` [M1 - not built]
   - Rhai scripting, worker registry (SQLite), scheduling, routing
   
2. **`rbees-workerd`** = HTTP daemon (WORKER) - `bin/rbees-workerd/` [M0 ✅ DONE]
   - Loads ONE model, generates tokens, stateless
   
3. **`rbees`** = CLI tool (REMOTE CONTROL) - `bin/rbees-ctl/` [M0 ✅ DONE]
   - SSH to pools, precise commands, operator tool
   
4. **`rbees-pool`** = CLI tool (LOCAL POOL) - `bin/rbees-pool/` [M0 ✅ DONE]
   - Model catalog, worker spawning, backend detection

**ARCHITECTURAL CHANGE (2025-10-09):**
- ❌ **pool-managerd daemon is NOT NEEDED**
- ✅ Pool management is CLI-based (`rbees-pool`)
- ✅ Only 2 daemons: rbees-orcd + rbees-workerd
- ✅ 2 CLIs: rbees + rbees-pool
- See: `/bin/.specs/ARCHITECTURE_DECISION_NO_POOL_DAEMON.md`

### The Four-Binary System

```
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATORD (The Brain) - HTTP DAEMON                         │
│ Binary: rbees-orcd (NOT llorch!)                             │
│ Location: blep.home.arpa:8080                                    │
│ Makes ALL intelligent decisions                                  │
│ - Admission, Queue, Scheduling, Worker Selection                 │
│ - Eviction, Retry, Timeout, Cancellation Policies               │
│ - Client-facing API                                              │
│ - SSE Streaming Relay                                            │
│ Status: M1 (not built yet)                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─ HTTP POST /execute (direct to workers)
                     │
┌────────────────────┴────────────────────────────────────────────┐
│ WORKERS (Dumb Executors) - HTTP DAEMONS                         │
│ Binary: rbees-workerd (and variants)                            │
│ - rbees-workerd (NVIDIA CUDA, Metal, CPU)                       │
│ - worker-orcd (Bespoke NVIDIA - future)                          │
│ - worker-aarmd (Apple ARM - future)                              │
│ Each worker:                                                     │
│ - Loads ONE model into VRAM/RAM                                  │
│ - Runs HTTP server on assigned port                             │
│ - Executes inference requests                                    │
│ - Streams tokens via SSE                                         │
│ Status: M0 (BUILT and WORKING)                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ OPERATOR TOOLS (CLI for humans) - SSH-BASED                     │
│                                                                   │
│ rbees (remote control) - bin/rbees-ctl/                        │
│   Purpose: SSH to pools, control remotely                        │
│   Usage: rbees pool worker spawn metal --host mac ...           │
│   Status: M0 (BUILT and WORKING)                                 │
│                                                                   │
│ rbees-pool (local control) - bin/rbees-pool/                      │
│   Purpose: Local pool operations (models, workers, catalog)      │
│   Usage: rbees-pool models download qwen-0.5b                   │
│   Status: M0 (BUILT and WORKING)                                 │
│   Note: This REPLACES pool-managerd daemon!                      │
└─────────────────────────────────────────────────────────────────┘
```

### Two Communication Channels

**Control Plane (SSH + CLI - 2 binaries):**
- Operator → Pools (via SSH)
- Model management (rbees-pool CLI)
- Worker lifecycle (rbees-pool CLI)
- System updates (llorch CLI)
- NO DAEMON NEEDED!
- **Binaries:** rbees + rbees-pool

**Data Plane (HTTP + Daemons - 2 binaries):**
- Client → Orchestrator → Worker
- Inference requests (rbees-orcd routes)
- Token streaming (rbees-workerd generates)
- Health checks (both daemons)
- **Binaries:** rbees-orcd + rbees-workerd

---

## Control Plane Operations (SSH)

### 1. Pool Management

#### 1.1 Check Pool Status
```bash
# Check status of a remote pool
llorch pool status --host mac.home.arpa

# Check status of all pools
llorch pool status --host mac.home.arpa
llorch pool status --host workstation.home.arpa
```

**What happens:**
- SSH to pool machine
- Run `rbees-pool status`
- Report GPU info, workers, models

#### 1.2 Update Pool Software
```bash
# Pull latest code on remote pool
llorch pool git pull --host mac.home.arpa

# Check git status
llorch pool git status --host mac.home.arpa

# Build binaries on remote pool
llorch pool git build --host mac.home.arpa
```

**What happens:**
- SSH to pool machine
- Execute git commands
- Build Rust binaries
- Update submodules

---

### 2. Model Management

#### 2.1 Register Model in Catalog
```bash
# Register a model (local pool)
rbees-pool models register \
  --id qwen-0.5b \
  --name "Qwen2.5 0.5B Instruct" \
  --repo "Qwen/Qwen2.5-0.5B-Instruct" \
  --architecture qwen

# Register on remote pool
llorch pool models register \
  --host mac.home.arpa \
  --id qwen-0.5b \
  --name "Qwen2.5 0.5B Instruct" \
  --repo "Qwen/Qwen2.5-0.5B-Instruct" \
  --architecture qwen
```

**What happens:**
- Create/update `.test-models/catalog.json`
- Add model entry with metadata
- Mark as not downloaded

#### 2.2 Download Model
```bash
# Download model (local pool)
rbees-pool models download qwen-0.5b

# Download on remote pool
llorch pool models download qwen-0.5b --host mac.home.arpa
```

**What happens:**
- SSH to pool (if remote)
- Run `hf download <repo> --include "*.safetensors" "*.json" --local-dir .test-models/<id>`
- Calculate model size
- Update catalog: `downloaded: true`, `size_gb: X.X`

#### 2.3 View Model Catalog
```bash
# View catalog (local)
rbees-pool models catalog

# View catalog (remote)
llorch pool models catalog --host mac.home.arpa
```

**Output:**
```
Model Catalog for mac.lan
================================================================================
ID              Name                           Downloaded   Size      
--------------------------------------------------------------------------------
tinyllama       TinyLlama 1.1B Chat            ✅            2.2 GB
qwen-0.5b       Qwen2.5 0.5B Instruct          ✅            0.9 GB
phi3            Phi-3 Mini 4K Instruct         ❌            0.0 GB
mistral         Mistral 7B Instruct v0.2       ❌            0.0 GB
================================================================================
Total models: 4
```

#### 2.4 List Downloaded Models
```bash
# List models (local)
rbees-pool models list

# List models (remote)
llorch pool models list --host mac.home.arpa
```

#### 2.5 Unregister Model
```bash
# Remove model from catalog
rbees-pool models unregister qwen-0.5b

# Remote
llorch pool models unregister qwen-0.5b --host mac.home.arpa
```

**Note:** Does NOT delete model files, only removes from catalog

---

### 3. Worker Lifecycle Management

#### 3.1 Spawn Worker
```bash
# Spawn worker on local pool
rbees-pool worker spawn metal \
  --model qwen-0.5b \
  --gpu 0

# Spawn worker on remote pool
llorch pool worker spawn metal \
  --host mac.home.arpa \
  --model qwen-0.5b \
  --gpu 0

# Spawn CUDA worker
llorch pool worker spawn cuda \
  --host workstation.home.arpa \
  --model mistral \
  --gpu 1

# Spawn CPU worker (no GPU)
llorch pool worker spawn cpu \
  --host blep.home.arpa \
  --model tinyllama
```

**What happens:**
1. Find model path in catalog
2. Generate worker ID (UUID)
3. Assign port (8001, 8002, etc.)
4. Spawn background process:
   ```bash
   rbees-workerd \
     --worker-id <uuid> \
     --model .test-models/qwen-0.5b \
     --port 8001 \
     --callback-url http://pool-manager:9200/callback
   ```
5. Worker loads model into VRAM/RAM (30-60 seconds)
6. Worker starts HTTP server
7. Worker calls ready callback
8. Save worker info to `.runtime/workers/<worker-id>.json`

#### 3.2 List Workers
```bash
# List workers (local)
rbees-pool worker list

# List workers (remote)
llorch pool worker list --host mac.home.arpa
```

**Output:**
```
Active Workers
================================================================================
ID                                   Backend  Model      Port   PID    Status
--------------------------------------------------------------------------------
worker-metal-abc123                  metal    qwen-0.5b  8001   12345  Running
worker-cuda-def456                   cuda     mistral    8002   12346  Running
================================================================================
Total workers: 2
```

#### 3.3 Stop Worker
```bash
# Stop worker (local)
rbees-pool worker stop worker-metal-abc123

# Stop worker (remote)
llorch pool worker stop worker-metal-abc123 --host mac.home.arpa
```

**What happens:**
1. Read worker info from `.runtime/workers/<id>.json`
2. Send SIGTERM to process (PID)
3. Wait for graceful shutdown
4. If timeout, send SIGKILL
5. Delete worker info file
6. Worker unloads model from VRAM

#### 3.4 Stop All Workers
```bash
# Stop all workers on pool
rbees-pool worker stop-all

# Remote
llorch pool worker stop-all --host mac.home.arpa
```

---

### 4. Development Operations

#### 4.1 Run Tests
```bash
# Run tests on remote pool
llorch pool test --host mac.home.arpa

# Run specific test
llorch pool test integration --host workstation.home.arpa
```

#### 4.2 Setup Development Environment
```bash
# Install dependencies on pool
llorch pool dev setup --host mac.home.arpa
```

**What happens:**
- Install Rust toolchain
- Install Python + pip
- Install `hf` CLI (`pip install huggingface_hub[cli]`)
- Install system dependencies

---

## Data Plane Operations (HTTP)

### 1. Inference Requests

#### 1.1 Submit Inference Job (Client → Orchestrator)
```bash
# Using llama-orch-sdk (Rust)
let client = Client::new("http://localhost:8080");
let request = InferenceRequest {
    prompt: "Once upon a time",
    max_tokens: 100,
    temperature: 0.7,
    model: Some("qwen-0.5b".to_string()),
};
let response = client.enqueue(request).await?;
```

```bash
# Using curl
curl -X POST http://localhost:8080/v2/tasks \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7,
    "model": "qwen-0.5b"
  }'
```

**What happens:**
1. **Admission:** Orchestrator validates request
2. **Queue:** Add to priority queue (interactive/batch)
3. **Scheduling:** Select worker based on:
   - Model availability
   - GPU load
   - Worker slots
   - Placement policy
4. **Dispatch:** Orchestrator → Worker HTTP call
5. **Stream:** Worker → Orchestrator → Client (SSE)

#### 1.2 Direct Worker Inference (Orchestrator → Worker)
```bash
# Orchestrator makes direct HTTP call to worker
curl -X POST http://mac.home.arpa:8001/execute \
  -H 'Content-Type: application/json' \
  -d '{
    "job_id": "job-123",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Response (SSE stream):**
```
data: {"type":"started","job_id":"job-123","model":"qwen-0.5b","started_at":"2025-10-09T14:30:00Z"}

data: {"type":"token","t":"I","i":0}

data: {"type":"token","t":"'m","i":1}

data: {"type":"token","t":" doing","i":2}

data: {"type":"token","t":" well","i":3}

data: {"type":"end","tokens_out":50,"decode_time_ms":1234,"stop_reason":"MAX_TOKENS"}
```

#### 1.3 Stream Tokens (SSE)
```rust
// Client receives SSE stream
let mut stream = client.stream(request).await?;
while let Some(event) = stream.next().await {
    match event {
        InferenceEvent::Started { job_id, model, .. } => {
            println!("Started: {} on {}", job_id, model);
        }
        InferenceEvent::Token { t, i } => {
            print!("{}", t);
        }
        InferenceEvent::End { tokens_out, .. } => {
            println!("\nGenerated {} tokens", tokens_out);
        }
    }
}
```

---

### 2. Health & Status

#### 2.1 Check Orchestrator Health
```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "healthy",
  "pools": 2,
  "workers": 5,
  "queue_depth": 3
}
```

#### 2.2 Check Worker Health
```bash
curl http://mac.home.arpa:8001/health
```

**Response:**
```json
{
  "status": "healthy",
  "vram_bytes": 2147483648
}
```

#### 2.3 Check Pool Manager Health
```bash
curl http://mac.home.arpa:9200/health
```

**Response:**
```json
{
  "status": "healthy",
  "workers": 2,
  "gpus": 1
}
```

---

### 3. Cancellation

#### 3.1 Cancel Job
```bash
# Cancel via orchestrator
curl -X POST http://localhost:8080/v2/tasks/job-123/cancel

# Cancel directly on worker
curl -X POST http://mac.home.arpa:8001/cancel \
  -H 'Content-Type: application/json' \
  -d '{"job_id": "job-123"}'
```

**What happens:**
1. Orchestrator receives cancel request
2. Orchestrator calls worker cancel endpoint
3. Worker stops token generation
4. Worker sends `end` event with `stop_reason: "CANCELLED"`
5. Stream closes

---

## Complete Workflow Examples

### Example 1: Deploy New Model on Remote Pool

```bash
# Step 1: Register model in catalog
llorch pool models register \
  --host mac.home.arpa \
  --id llama3-8b \
  --name "Llama 3 8B Instruct" \
  --repo "meta-llama/Meta-Llama-3-8B-Instruct" \
  --architecture llama

# Step 2: Download model (takes 5-10 minutes)
llorch pool models download llama3-8b --host mac.home.arpa

# Step 3: Verify download
llorch pool models catalog --host mac.home.arpa

# Step 4: Spawn worker
llorch pool worker spawn metal \
  --host mac.home.arpa \
  --model llama3-8b \
  --gpu 0

# Step 5: Verify worker is running
llorch pool worker list --host mac.home.arpa

# Step 6: Test inference
curl -X POST http://localhost:8080/v2/tasks \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 200,
    "temperature": 0.7,
    "model": "llama3-8b"
  }'
```

---

### Example 2: Multi-Pool Setup with Different Models

```bash
# Pool 1 (Mac - Metal GPU) - Small models
llorch pool models download qwen-0.5b --host mac.home.arpa
llorch pool models download tinyllama --host mac.home.arpa
llorch pool worker spawn metal --host mac.home.arpa --model qwen-0.5b --gpu 0
llorch pool worker spawn metal --host mac.home.arpa --model tinyllama --gpu 0

# Pool 2 (Workstation - CUDA GPU) - Large models
llorch pool models download mistral --host workstation.home.arpa
llorch pool models download llama3-8b --host workstation.home.arpa
llorch pool worker spawn cuda --host workstation.home.arpa --model mistral --gpu 0
llorch pool worker spawn cuda --host workstation.home.arpa --model llama3-8b --gpu 1

# Pool 3 (Blep - CPU) - Fallback
llorch pool models download tinyllama --host blep.home.arpa
llorch pool worker spawn cpu --host blep.home.arpa --model tinyllama

# Verify all workers
llorch pool worker list --host mac.home.arpa
llorch pool worker list --host workstation.home.arpa
llorch pool worker list --host blep.home.arpa

# Orchestrator automatically routes to best worker
curl -X POST http://localhost:8080/v2/tasks \
  -d '{"prompt": "Hello", "model": "qwen-0.5b"}'  # → mac.home.arpa

curl -X POST http://localhost:8080/v2/tasks \
  -d '{"prompt": "Hello", "model": "mistral"}'    # → workstation.home.arpa
```

---

### Example 3: Update and Restart Workers

```bash
# Step 1: Pull latest code on all pools
llorch pool git pull --host mac.home.arpa
llorch pool git pull --host workstation.home.arpa

# Step 2: Build binaries
llorch pool git build --host mac.home.arpa
llorch pool git build --host workstation.home.arpa

# Step 3: Stop all workers
llorch pool worker stop-all --host mac.home.arpa
llorch pool worker stop-all --host workstation.home.arpa

# Step 4: Restart workers with new binary
llorch pool worker spawn metal --host mac.home.arpa --model qwen-0.5b --gpu 0
llorch pool worker spawn cuda --host workstation.home.arpa --model mistral --gpu 0

# Step 5: Verify
llorch pool worker list --host mac.home.arpa
llorch pool worker list --host workstation.home.arpa
```

---

### Example 4: Test Model on Different Backends

```bash
# Test Qwen on Metal (Mac)
llorch pool worker spawn metal --host mac.home.arpa --model qwen-0.5b --gpu 0
curl -X POST http://mac.home.arpa:8001/execute \
  -d '{"job_id":"test-1","prompt":"Hello","max_tokens":20}'

# Test Qwen on CUDA (Workstation)
llorch pool worker spawn cuda --host workstation.home.arpa --model qwen-0.5b --gpu 0
curl -X POST http://workstation.home.arpa:8001/execute \
  -d '{"job_id":"test-2","prompt":"Hello","max_tokens":20}'

# Test Qwen on CPU (Blep)
llorch pool worker spawn cpu --host blep.home.arpa --model qwen-0.5b
curl -X POST http://blep.home.arpa:8001/execute \
  -d '{"job_id":"test-3","prompt":"Hello","max_tokens":20}'

# Compare performance
```

---

## Component Responsibilities

### Orchestratord (The Brain)

**Makes ALL intelligent decisions:**
- ✅ Admission control (accept/reject requests)
- ✅ Queue management (priority, ordering)
- ✅ Scheduling (which worker gets which job)
- ✅ Worker selection (based on load, model, GPU)
- ✅ Eviction policy (which worker to stop)
- ✅ Retry policy (when to retry failed jobs)
- ✅ Timeout policy (when to cancel slow jobs)
- ✅ Cancellation handling
- ✅ SSE stream relay (worker → client)
- ✅ Session management (TTL, budgets)
- ✅ Multi-tenant isolation (platform mode)

**Does NOT:**
- ❌ Spawn workers (pool manager does this)
- ❌ Download models (pool manager does this)
- ❌ Execute inference (workers do this)
- ❌ Manage GPU state (pool manager reports, orchestrator decides)

---

### Pool-Managerd (Control Plane)

**Executes commands, reports state (DUMB):**
- ✅ GPU discovery (NVML, Metal)
- ✅ Worker spawning (start rbees-workerd process)
- ✅ Worker monitoring (PID, health checks)
- ✅ Worker stopping (SIGTERM, SIGKILL)
- ✅ Model provisioning (download via hf CLI)
- ✅ Heartbeat to orchestrator (GPU state, worker status)
- ✅ Ready callback handling (worker → pool manager)

**Does NOT:**
- ❌ Decide which worker to spawn (orchestrator decides)
- ❌ Decide when to stop workers (orchestrator decides)
- ❌ Route inference requests (orchestrator does this)
- ❌ Make scheduling decisions (orchestrator decides)

---

### Workers (Dumb Executors)

**Executes inference, streams tokens:**
- ✅ Load ONE model into VRAM/RAM
- ✅ Run HTTP server on assigned port
- ✅ Execute inference requests
- ✅ Stream tokens via SSE
- ✅ Handle cancellation
- ✅ Report VRAM usage
- ✅ Call ready callback when loaded

**Does NOT:**
- ❌ Decide which requests to accept (orchestrator routes)
- ❌ Manage multiple models (one worker = one model)
- ❌ Download models (pool manager does this)
- ❌ Spawn other workers (pool manager does this)

---

## Quick Reference Commands

### Pool Management
```bash
# Status
llorch pool status --host <pool>

# Update
llorch pool git pull --host <pool>
llorch pool git build --host <pool>
```

### Model Management
```bash
# Register
llorch pool models register --host <pool> --id <id> --name <name> --repo <repo> --architecture <arch>

# Download
llorch pool models download <id> --host <pool>

# View
llorch pool models catalog --host <pool>
llorch pool models list --host <pool>

# Remove
llorch pool models unregister <id> --host <pool>
```

### Worker Management
```bash
# Spawn
llorch pool worker spawn <backend> --host <pool> --model <id> --gpu <n>
# backends: metal, cuda, cpu

# List
llorch pool worker list --host <pool>

# Stop
llorch pool worker stop <worker-id> --host <pool>
llorch pool worker stop-all --host <pool>
```

### Inference
```bash
# Submit job
curl -X POST http://localhost:8080/v2/tasks \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"...","max_tokens":100,"model":"<id>"}'

# Direct worker call
curl -X POST http://<pool>:<port>/execute \
  -H 'Content-Type: application/json' \
  -d '{"job_id":"...","prompt":"...","max_tokens":100}'

# Cancel
curl -X POST http://localhost:8080/v2/tasks/<job-id>/cancel
```

### Health Checks
```bash
# Orchestrator
curl http://localhost:8080/health

# Pool Manager
curl http://<pool>:9200/health

# Worker
curl http://<pool>:<port>/health
```

---

## Architecture Principles

### Smart/Dumb Boundary
- **Orchestratord = SMART** (makes all decisions)
- **Pool-Managerd = DUMB** (executes commands, reports facts)
- **Workers = DUMB** (executes inference)

### Process Isolation
- Each worker runs in separate process
- Each worker has isolated memory context
- Workers can crash without affecting others

### Direct Communication
- Orchestrator → Worker (HTTP, direct)
- NOT: Orchestrator → Pool Manager → Worker
- Pool manager is NOT in inference path

### Two Planes
- **Control Plane:** SSH (operator → pools)
- **Data Plane:** HTTP (client → orchestrator → worker)

---

## Current Status (M0)

### ✅ Implemented
- ✅ Worker spawning (rbees-workerd)
- ✅ Model downloads (hf CLI)
- ✅ Token generation (CPU, Metal, CUDA)
- ✅ SSE streaming
- ✅ Health checks
- ✅ Model catalog
- ✅ Worker lifecycle
- ✅ SSH control (rbees-ctl)

### 🚧 In Progress (M1)
- 🚧 Pool-managerd daemon
- 🚧 Heartbeat protocol
- 🚧 Worker adapters (llamacpp, vllm)
- 🚧 Preflight validation

### 📋 Future (M2+)
- 📋 Orchestratord daemon
- 📋 Queue management
- 📋 Scheduling policies
- 📋 Multi-tenant support
- 📋 Metrics & observability

---

**For more details, see:**
- [`.specs/00_llama-orch.md`](.specs/00_llama-orch.md) - Full system spec
- [`README.md`](README.md) - Project overview
- [`bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md`](bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md) - Architecture decision
