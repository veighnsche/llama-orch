# Final Architecture: SSH Control + HTTP Inference

**Status**: Normative  
**Version**: 1.0  
**Date**: 2025-10-09

---

## Executive Summary

**Control Plane:** SSH (operator → pools)  
**Data Plane:** HTTP (orchestrator → workers for inference)

**Key Insight:** Workers MUST be HTTP daemons because they need to:
1. Stay running with model loaded in VRAM
2. Accept inference requests from orchestrator
3. Be directly accessible over network (not through pool manager)

---

## The Correct Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ CONTROL PLANE (Operator → System)                               │
│ Protocol: SSH                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ rbees-ctl (blep)                                                │
│     ↓ SSH (control operations)                                   │
│ rbees-pool (mac/workstation)                                       │
│     ↓ spawn process                                              │
│ rbees-workerd (worker daemon, HTTP server)                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ DATA PLANE (Inference Requests)                                 │
│ Protocol: HTTP                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ llama-orch-sdk (client)                                          │
│     ↓ HTTP POST /v2/tasks                                        │
│ rbees-orcd (daemon :8080)                                     │
│     ↓ HTTP POST /execute (DIRECT to worker, not via pool)       │
│ rbees-workerd (worker daemon :8001)                             │
│     ↓ SSE stream                                                 │
│ rbees-orcd (relays)                                           │
│     ↓ SSE stream                                                 │
│ llama-orch-sdk (client)                                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Workers MUST Be HTTP Daemons

### Requirement 1: Keep Model Loaded in VRAM

**Problem:** Loading model into VRAM is expensive (seconds to minutes)

**Solution:** Worker daemon keeps model loaded
```rust
// rbees-workerd lifecycle
1. Spawn: rbees-workerd --model tinyllama.gguf --gpu 0 --port 8001
2. Load model into VRAM (30 seconds)
3. HTTP server starts, listens on :8001
4. Ready callback: POST http://pool-manager:9200/workers/ready
5. Wait for inference requests (model stays in VRAM)
6. Receive: POST http://worker:8001/execute
7. Execute inference (model already loaded, fast)
8. Stream tokens via SSE
9. Return to waiting (model still in VRAM)
10. Repeat steps 6-9 for multiple requests
11. Shutdown: Unload model, exit
```

**Why HTTP:** Need to send requests to running process

### Requirement 2: Direct Orchestrator → Worker Communication

**From specs (SYS-5.4.x):**
> Orchestrator → Worker (Direct)
> 
> The orchestrator MUST be able to call worker endpoints directly to proxy/relay requests.

**Architecture:**
```
Client (llama-orch-sdk)
    ↓ POST /v2/tasks
Orchestratord (makes scheduling decision)
    ↓ Selects worker: worker-metal-0 at mac.home.arpa:8001
    ↓ POST http://mac.home.arpa:8001/execute (DIRECT, not via pool)
Worker (rbees-workerd)
    ↓ Executes inference
    ↓ SSE stream tokens
Orchestratord (relays stream)
    ↓ SSE stream
Client
```

**Why direct:** Pool manager is not in the inference path (SYS-4.3.2)

### Requirement 3: Network Accessibility

**Workers live on different machines:**
- Worker on mac.home.arpa:8001 (Metal GPU)
- Worker on workstation.home.arpa:8002 (CUDA GPU)
- Worker on blep.home.arpa:8003 (CPU)

**Orchestrator (on blep) needs to reach workers on other machines:**
```bash
# Orchestrator makes HTTP calls to workers on network
curl http://mac.home.arpa:8001/execute
curl http://workstation.home.arpa:8002/execute
```

**Why HTTP:** Standard network protocol, works across machines

### Requirement 4: Multiple Concurrent Requests

**Worker can handle multiple requests (if slots available):**
```
Worker (4 slots available)
    ├─ Request 1 (slot 0) - generating tokens
    ├─ Request 2 (slot 1) - generating tokens
    ├─ Request 3 (slot 2) - generating tokens
    └─ Slot 3 - available
```

**Why HTTP:** Async request handling, connection pooling

---

## What Uses HTTP vs SSH

### HTTP (Data Plane - Inference)

**1. Client → Orchestrator**
```rust
// llama-orch-sdk (HTTP client)
let client = Client::new("http://localhost:8080");
let response = client.enqueue(request).await?;
```

**2. Orchestrator → Worker (Direct)**
```rust
// rbees-orcd makes HTTP call to worker
let response = reqwest::Client::new()
    .post("http://mac.home.arpa:8001/execute")
    .json(&inference_request)
    .send()
    .await?;
```

**3. Worker → Orchestrator (Callback)**
```rust
// rbees-workerd calls back when ready
reqwest::Client::new()
    .post("http://orchestrator:8080/workers/ready")
    .json(&ready_payload)
    .send()
    .await?;
```

**Why HTTP:**
- Workers are long-running daemons
- Need to keep model in VRAM
- Need network accessibility
- Need async request handling
- Need streaming (SSE)

### SSH (Control Plane - Operations)

**1. Orchestrator → Pool (Control)**
```bash
# rbees-ctl uses SSH to control pools
llorch pool models download tinyllama --host mac
  → ssh mac.home.arpa "rbees-pool models download tinyllama"

llorch pool worker spawn metal --host mac --model tinyllama
  → ssh mac.home.arpa "rbees-pool worker spawn metal --model tinyllama"
```

**2. Pool → Worker (Spawn)**
```bash
# rbees-pool spawns worker as background process
rbees-pool worker spawn metal --model tinyllama
  → rbees-workerd --model tinyllama.gguf --gpu 0 --port 8001 &
  → Worker starts HTTP server
  → Worker loads model into VRAM
  → Worker calls ready callback
```

**Why SSH:**
- Control operations (not inference)
- Spawn processes
- Download models
- Git operations
- Trusted network

---

## Worker Registry

### Problem: Orchestrator Needs to Know Workers

**Orchestrator needs:**
1. Which workers exist
2. Where they are (host:port)
3. What models they have loaded
4. How many slots available
5. Worker status (ready, busy, stopping)

### Solution: Worker Registration

**When worker starts:**
```rust
// rbees-workerd startup
1. Load model into VRAM
2. Start HTTP server on :8001
3. Call ready callback:
   POST http://orchestrator:8080/workers/register
   {
     "worker_id": "worker-metal-0",
     "host": "mac.home.arpa",
     "port": 8001,
     "backend": "metal",
     "model_ref": "file:.test-models/tinyllama/...",
     "gpu_id": 0,
     "slots_total": 4,
     "slots_available": 4
   }
4. Orchestrator adds to registry
5. Worker is now available for inference
```

**When orchestrator needs to dispatch job:**
```rust
// rbees-orcd scheduling
1. Receive job: POST /v2/tasks
2. Look up workers in registry
3. Find worker with:
   - Matching model
   - Available slots
   - Lowest load
4. Select worker: worker-metal-0 at mac.home.arpa:8001
5. Make HTTP call: POST http://mac.home.arpa:8001/execute
6. Stream response back to client
```

---

## Pool Registry

### Problem: Orchestrator Needs to Know Pools

**Orchestrator needs:**
1. Which pools exist (mac, workstation, blep)
2. Where they are (for SSH)
3. What GPUs they have
4. What models they have downloaded
5. Pool status (online, offline)

### Solution: Pool Registration (SSH-based)

**Option 1: Static Configuration**
```toml
# orchestrator.toml
[[pools]]
id = "mac"
host = "mac.home.arpa"
ssh_user = "vinceliem"
ssh_key = "~/.ssh/id_ed25519"
repo_path = "~/Projects/llama-orch"

[[pools]]
id = "workstation"
host = "workstation.home.arpa"
ssh_user = "vince"
ssh_key = "~/.ssh/id_ed25519"
repo_path = "~/Projects/llama-orch"

[[pools]]
id = "blep-cpu"
host = "localhost"
repo_path = "~/Projects/llama-orch"
```

**Option 2: Dynamic Registration (SSH probe)**
```bash
# rbees-ctl discovers pools
llorch pool register mac --host mac.home.arpa --user vinceliem

# Orchestrator probes pool via SSH
ssh mac.home.arpa "rbees-pool info"
  → Returns: GPUs, models, disk space, etc.
  → Orchestrator stores in registry
```

**Option 3: Heartbeat (SSH-based, M1+)**
```bash
# Orchestrator polls pools periodically
ssh mac.home.arpa "rbees-pool status"
  → Returns: Workers running, GPU usage, models available
  → Orchestrator updates registry
```

---

## Complete Flow Example

### Scenario: Submit Inference Job

**Step 1: Client submits job**
```typescript
// Client (web app)
const client = new Client('http://blep.home.arpa:8080');
const response = await client.enqueue({
  prompt: 'Hello world',
  model: 'tinyllama',
  maxTokens: 100,
});
```

**Step 2: Orchestrator receives job**
```rust
// rbees-orcd (HTTP server on blep:8080)
POST /v2/tasks
  → Parse request
  → Validate model exists
  → Check worker registry for workers with model "tinyllama"
  → Find: worker-metal-0 at mac.home.arpa:8001 (4 slots available)
  → Select worker
```

**Step 3: Orchestrator dispatches to worker (DIRECT)**
```rust
// rbees-orcd makes HTTP call to worker
POST http://mac.home.arpa:8001/execute
{
  "job_id": "job-123",
  "prompt": "Hello world",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42
}
```

**Step 4: Worker executes inference**
```rust
// rbees-workerd (HTTP server on mac:8001)
POST /execute
  → Model already loaded in VRAM (fast)
  → Execute inference
  → Stream tokens via SSE
```

**Step 5: Orchestrator relays stream**
```rust
// rbees-orcd relays SSE stream
Worker SSE stream → Orchestrator → Client
  data: {"type":"token","text":"Hello"}
  data: {"type":"token","text":" there"}
  data: {"type":"end"}
```

**Step 6: Client receives tokens**
```typescript
// Client receives stream
for await (const event of stream) {
  if (event.type === 'token') {
    console.log(event.text);  // "Hello", " there", ...
  }
}
```

---

## Control Flow Example

### Scenario: Download Model and Spawn Worker

**Step 1: Operator downloads model on pool**
```bash
# On blep (orchestrator host)
llorch pool models download tinyllama --host mac
```

**Step 2: SSH to pool, execute download**
```bash
# rbees-ctl executes via SSH
ssh mac.home.arpa "cd ~/Projects/llama-orch && rbees-pool models download tinyllama"
```

**Step 3: rbees-pool downloads model**
```bash
# rbees-pool (on mac) executes
hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --local-dir .test-models/tinyllama
```

**Step 4: Operator spawns worker**
```bash
# On blep
llorch pool worker spawn metal --host mac --model tinyllama
```

**Step 5: SSH to pool, spawn worker**
```bash
# rbees-ctl executes via SSH
ssh mac.home.arpa "cd ~/Projects/llama-orch && rbees-pool worker spawn metal --model tinyllama"
```

**Step 6: rbees-pool spawns worker daemon**
```bash
# rbees-pool (on mac) spawns background process
rbees-workerd \
  --worker-id worker-metal-0 \
  --model .test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --backend metal \
  --gpu 0 \
  --port 8001 \
  --orchestrator-url http://blep.home.arpa:8080 &
```

**Step 7: Worker starts and registers**
```rust
// rbees-workerd (on mac)
1. Load model into VRAM (30 seconds)
2. Start HTTP server on :8001
3. Register with orchestrator:
   POST http://blep.home.arpa:8080/workers/register
   {
     "worker_id": "worker-metal-0",
     "host": "mac.home.arpa",
     "port": 8001,
     "model_ref": "file:.test-models/tinyllama/...",
     "backend": "metal",
     "gpu_id": 0,
     "slots_total": 4
   }
4. Wait for inference requests
```

**Step 8: Orchestrator adds worker to registry**
```rust
// rbees-orcd (on blep)
POST /workers/register
  → Add worker to in-memory registry
  → Worker is now available for scheduling
```

---

## Binary Responsibilities

### rbees-workerd (Worker Daemon)

**MUST be HTTP daemon because:**
- Keeps model loaded in VRAM
- Accepts inference requests over network
- Streams tokens via SSE
- Handles multiple concurrent requests (slots)

**Responsibilities:**
- Load model into VRAM
- Start HTTP server
- Register with orchestrator
- Execute inference requests
- Stream tokens via SSE
- Report status (slots available)

**NOT responsible for:**
- Downloading models (rbees-pool does this)
- Spawning itself (rbees-pool does this)
- Scheduling (orchestrator does this)

### rbees-pool (CLI)

**Uses SSH for control, spawns HTTP daemons:**

**Responsibilities:**
- Download models (hf CLI)
- Git operations (git CLI)
- Spawn workers (rbees-workerd as background process)
- Stop workers (kill process)
- List workers (ps/pidof)

**NOT responsible for:**
- Inference (workers do this)
- Scheduling (orchestrator does this)
- Keeping workers alive (systemd/launchd does this in production)

### rbees-ctl (CLI)

**Uses SSH for control:**

**Responsibilities:**
- Command pools via SSH
- Query pool status (via SSH)
- Query worker registry (via orchestrator HTTP API, M2+)
- Submit jobs (via orchestrator HTTP API, M2+)

**NOT responsible for:**
- Inference (workers do this)
- Spawning workers directly (rbees-pool does this)

### rbees-orcd (Daemon, M2+)

**HTTP daemon for scheduling:**

**Responsibilities:**
- Receive job submissions (HTTP)
- Maintain worker registry
- Make scheduling decisions
- Dispatch jobs to workers (HTTP)
- Relay SSE streams
- Maintain job queue (SQLite)

**NOT responsible for:**
- Spawning workers (rbees-pool does this)
- Downloading models (rbees-pool does this)
- Executing inference (workers do this)

---

## Network Topology

```
┌─────────────────────────────────────────────────────────────────┐
│ blep.home.arpa (Orchestrator + Pool)                             │
├─────────────────────────────────────────────────────────────────┤
│ rbees-orcd :8080 (HTTP server, M2+)                           │
│   ├─ Worker registry (in-memory)                                 │
│   ├─ Pool registry (in-memory)                                   │
│   └─ Job queue (SQLite)                                          │
│                                                                   │
│ rbees-ctl (CLI)                                                 │
│   └─ SSH client → pools                                          │
│                                                                   │
│ rbees-pool (CLI)                                                   │
│   └─ Spawns workers locally                                      │
│                                                                   │
│ rbees-workerd :8003 (HTTP server, CPU worker)                   │
└─────────────────────────────────────────────────────────────────┘
         ↑ SSH (control)                    ↓ HTTP (inference)
         │                                  │
┌────────┴──────────────────────────────────┴─────────────────────┐
│ mac.home.arpa (Pool)                                             │
├─────────────────────────────────────────────────────────────────┤
│ rbees-pool (CLI)                                                   │
│   └─ Spawns workers locally                                      │
│                                                                   │
│ rbees-workerd :8001 (HTTP server, Metal worker)                 │
│   └─ Model loaded in VRAM                                        │
└─────────────────────────────────────────────────────────────────┘
         ↑ SSH (control)                    ↓ HTTP (inference)
         │                                  │
┌────────┴──────────────────────────────────┴─────────────────────┐
│ workstation.home.arpa (Pool)                                     │
├─────────────────────────────────────────────────────────────────┤
│ rbees-pool (CLI)                                                   │
│   └─ Spawns workers locally                                      │
│                                                                   │
│ rbees-workerd :8002 (HTTP server, CUDA worker)                  │
│   └─ Model loaded in VRAM                                        │
└─────────────────────────────────────────────────────────────────┘
```

**Control plane:** SSH (rbees-ctl → rbees-pool)  
**Data plane:** HTTP (rbees-orcd → workers, clients → rbees-orcd)

---

## Summary

### Workers MUST Be HTTP Daemons

**Reasons:**
1. ✅ Keep model loaded in VRAM (expensive to reload)
2. ✅ Accept inference requests over network
3. ✅ Direct orchestrator → worker communication (not via pool)
4. ✅ Stream tokens via SSE
5. ✅ Handle multiple concurrent requests (slots)
6. ✅ Workers live on different machines (network accessibility)

**This is NOT negotiable.** Workers are long-running HTTP daemons.

### Control Plane Uses SSH

**Reasons:**
1. ✅ Secure (SSH keys)
2. ✅ Simple (no pool-managerd daemon needed)
3. ✅ Homelab-friendly (already configured)
4. ✅ Trusted network

**Control operations:**
- Download models (hf CLI)
- Git operations (git CLI)
- Spawn workers (background process)
- Stop workers (kill process)

### Data Plane Uses HTTP

**Reasons:**
1. ✅ Workers are HTTP daemons
2. ✅ Streaming (SSE)
3. ✅ Network accessibility
4. ✅ Async request handling

**Inference operations:**
- Submit jobs (POST /v2/tasks)
- Execute inference (POST /execute)
- Stream tokens (SSE)
- Query status (GET /status)

### Registries

**Worker Registry (in rbees-orcd):**
- Which workers exist
- Where they are (host:port)
- What models loaded
- Slots available
- Updated via HTTP callbacks

**Pool Registry (in rbees-orcd):**
- Which pools exist
- Where they are (for SSH)
- What GPUs available
- What models downloaded
- Updated via SSH probes or static config

---

**Version**: 1.0  
**Status**: Normative (MUST follow)  
**Last Updated**: 2025-10-09

---

**End of Architecture**
