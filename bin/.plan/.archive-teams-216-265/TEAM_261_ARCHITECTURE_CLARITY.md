# TEAM-261: Architecture Clarity - Job Client/Server Relationships

**Date:** Oct 23, 2025  
**Updated:** Oct 23, 2025 (Added simplification decision)  
**Status:** 🚨 CRITICAL DOCUMENTATION  
**Purpose:** Document the UNINTUITIVE but CORRECT architecture

---

## 🎯 ARCHITECTURAL DECISION (Oct 23, 2025)

**Decision:** Keep hive as daemon, remove hive heartbeat

**Why?**
- ✅ **Performance:** Daemon is 10-100x faster than CLI (1-5ms vs 80-350ms)
- ✅ **UX:** Daemon provides real-time SSE streaming
- ✅ **Security:** Daemon avoids command injection risks
- ✅ **Simplification:** Remove hive heartbeat (workers → queen direct)

**Changes:**
1. ✅ Keep hive as HTTP daemon
2. ❌ Remove hive heartbeat task
3. ✅ Workers send heartbeats directly to queen
4. ✅ Queen is single source of truth for worker state

**See:** `TEAM_261_SIMPLIFICATION_AUDIT.md` for implementation details

---

## ⚠️ THE UNINTUITIVE TRUTH

### What Seems Intuitive (BUT IS WRONG!)

```
❌ WRONG MENTAL MODEL:

rbee-keeper (job-client)
    ↓
queen-rbee (job-server)
    ↓
rbee-hive (job-client)  ← Seems like hive should forward to worker
    ↓
llm-worker-rbee (job-server)  ← Seems like worker should be a job server
```

### What Is Actually True (CORRECT!)

```
✅ CORRECT ARCHITECTURE:

rbee-keeper (job-client)
    ↓
queen-rbee (job-server + MANY job-clients)
    ├─→ rbee-hive (job-server) ← For worker LIFECYCLE only
    │   └─→ spawn/stop/list workers (NOT inference!)
    │
    └─→ llm-worker-rbee (simple HTTP) ← Direct connection, NO job-server!
        └─→ POST /v1/inference (simple request/response)
```

---

## 🎯 Key Architectural Decisions

### 1. Queen Circumvents Hive for Inference

**Why?**
- **Performance:** Direct queen → worker connection eliminates hop
- **Simplicity:** Worker only does ONE thing (inference)
- **Scheduling:** Queen needs direct control for load balancing

**What This Means:**
- Hive manages worker LIFECYCLE (spawn, stop, list)
- Queen manages worker INFERENCE (routing, load balancing)
- Worker is NOT a job-server (too complex for single purpose)

### 2. Worker Uses Simple HTTP (NOT job-server)

**Why?**
- Worker has ONE job: inference
- Job architecture is overkill for single-purpose daemon
- Simpler = faster = better for inference hot path

**What This Means:**
- Worker has simple endpoints:
  - `POST /v1/inference` - Submit inference request
  - `GET /v1/inference/{job_id}/stream` - Stream tokens (SSE)
  - `GET /health` - Health check
  - `GET /v1/ready` - Readiness check
- NO job-server dependency
- NO complex routing
- JUST inference

### 3. Hive is Job-Server for Lifecycle Operations

**Why?**
- Hive manages MULTIPLE workers
- Worker operations (spawn, stop, list) need job pattern
- Consistency with queen-rbee pattern

**What This Means:**
- Hive has job-server for:
  - WorkerSpawn
  - WorkerStop
  - WorkerList
  - WorkerGet
  - WorkerDelete
  - ModelDownload
  - ModelList
  - ModelGet
  - ModelDelete
- Hive does NOT handle Infer (that's queen → worker direct)

---

## 📋 Operation Routing Table

| Operation | Handled By | Uses job-server? | Why? |
|-----------|------------|------------------|------|
| **Hive Lifecycle** | | | |
| HiveInstall | queen-rbee | ✅ Yes | Complex SSH/daemon operations |
| HiveUninstall | queen-rbee | ✅ Yes | Complex cleanup operations |
| HiveStart | queen-rbee | ✅ Yes | Daemon lifecycle management |
| HiveStop | queen-rbee | ✅ Yes | Daemon lifecycle management |
| HiveList | queen-rbee | ✅ Yes | Query config + registry |
| HiveGet | queen-rbee | ✅ Yes | Query config + registry |
| HiveStatus | queen-rbee | ✅ Yes | Health check + heartbeat |
| HiveRefreshCapabilities | queen-rbee | ✅ Yes | GPU detection via SSH |
| **Worker Lifecycle** | | | |
| WorkerSpawn | rbee-hive | ✅ Yes | Spawn daemon, register |
| WorkerStop | rbee-hive | ✅ Yes | Stop daemon, cleanup |
| WorkerList | rbee-hive | ✅ Yes | Query worker registry |
| WorkerGet | rbee-hive | ✅ Yes | Query worker registry |
| WorkerDelete | rbee-hive | ✅ Yes | Stop + cleanup |
| **Model Management** | | | |
| ModelDownload | rbee-hive | ✅ Yes | Download + register |
| ModelList | rbee-hive | ✅ Yes | Query model catalog |
| ModelGet | rbee-hive | ✅ Yes | Query model catalog |
| ModelDelete | rbee-hive | ✅ Yes | Delete + cleanup |
| **Inference** | | | |
| Infer | queen-rbee → worker | ❌ NO! | Direct HTTP, no job-server |

---

## 🔥 Critical Notes for Code

### In queen-rbee/src/job_router.rs

```rust
// ============================================================================
// INFERENCE ROUTING - CRITICAL ARCHITECTURE NOTE
// ============================================================================
//
// ⚠️  UNINTUITIVE BUT CORRECT: Infer is handled in QUEEN, not forwarded to HIVE!
//
// Why?
// - Queen needs direct control for scheduling/load balancing
// - Hive only manages worker LIFECYCLE (spawn/stop/list)
// - Queen → Worker is direct HTTP (no job-server on worker)
// - This eliminates a hop and simplifies the inference hot path
//
// DO NOT forward Infer to hive! Queen circumvents hive for performance.
//
Operation::Infer { .. } => {
    // TODO: IMPLEMENT INFERENCE SCHEDULING
    // - Select worker based on load/availability
    // - Direct HTTP POST to worker's /v1/inference endpoint
    // - Stream tokens back to client via SSE
    // - NO job-server on worker side (simple HTTP only)
}
```

### In rbee-hive/src/job_router.rs

```rust
// ============================================================================
// INFERENCE REJECTION - CRITICAL ARCHITECTURE NOTE
// ============================================================================
//
// ⚠️  INFER SHOULD NOT BE IN HIVE!
//
// Why?
// - Hive only manages worker LIFECYCLE (spawn/stop/list)
// - Queen handles inference routing directly to workers
// - Queen → Worker is direct HTTP (circumvents hive)
// - This is INTENTIONAL for performance and simplicity
//
// If you see Infer here, something is wrong with the routing!
//
Operation::Infer { .. } => {
    return Err(anyhow::anyhow!(
        "Infer operation should NOT be routed to hive! \
         Queen should route inference directly to workers. \
         This indicates a routing bug in queen-rbee."
    ));
}
```

---

## 🏗️ Component Responsibilities

### rbee-keeper (CLI)
- **Role:** User interface
- **Uses:** job-client to talk to queen
- **Handles:** All user commands

### queen-rbee (Orchestrator)
- **Role:** Central coordinator
- **Uses:** 
  - job-server (for keeper → queen)
  - job-client (for queen → hive)
  - Simple HTTP client (for queen → worker)
- **Handles:**
  - Hive lifecycle operations
  - Worker lifecycle operations (forwards to hive)
  - Model management (forwards to hive)
  - **Inference scheduling (direct to worker!)**

### rbee-hive (Worker Pool Manager)
- **Role:** Manage workers on one machine
- **Uses:** job-server (for queen → hive)
- **Handles:**
  - Worker spawning/stopping
  - Worker registry
  - Model downloading
  - Model catalog
  - **NOT inference!**

### llm-worker-rbee (Inference Engine)
- **Role:** Execute inference
- **Uses:** Simple HTTP (NO job-server!)
- **Handles:**
  - Load model
  - Execute inference
  - Stream tokens
  - **ONLY inference!**

---

## 🔍 Why Worker Doesn't Need job-server

### Current Worker Implementation

**File:** `bin/30_llm_worker_rbee/Cargo.toml`

```toml
# TEAM-154: Job registry shared crate (dual-call pattern)
job-server = { path = "../99_shared_crates/job-server" }
```

**Wait, it DOES have job-server!** Let's check if it's actually needed...

### Worker Endpoints

```
POST /v1/inference → Create job, return job_id + sse_url
GET /v1/inference/{job_id}/stream → Stream tokens via SSE
GET /health → Health check
GET /v1/ready → Readiness check
GET /v1/loading/progress → Model loading progress
```

### Analysis

**Worker DOES use job-server for:**
- Dual-call pattern (POST creates job, GET streams)
- Job registry for inference requests
- SSE streaming with job isolation

**But this is DIFFERENT from queen/hive:**
- Worker's job-server is INTERNAL (not exposed to network routing)
- Worker's jobs are SIMPLE (just inference, no routing)
- Worker's job-server is for SSE streaming pattern, not operation routing

### Verdict: Worker SHOULD Keep job-server

**Why?**
- Dual-call pattern is good for SSE streaming
- Job isolation prevents cross-contamination
- Consistent with industry patterns (OpenAI, Anthropic)

**But:**
- Worker's job-server is SIMPLER (no operation routing)
- Worker's job-server is INTERNAL (not part of queen → worker protocol)
- Queen talks to worker via simple HTTP POST, not job-client

---

## 🎭 The Three Patterns

### Pattern 1: keeper → queen (Full job-client/job-server)
```
keeper uses job-client
    ↓
    POST /v1/jobs (with Operation enum)
    ↓
queen uses job-server
    ↓
    Parse Operation, route to handler
    ↓
    Stream results via SSE
```

### Pattern 2: queen → hive (Full job-client/job-server)
```
queen uses job-client
    ↓
    POST /v1/jobs (with Operation enum)
    ↓
hive uses job-server
    ↓
    Parse Operation, route to handler
    ↓
    Stream results via SSE
```

### Pattern 3: queen → worker (Simple HTTP, NO job-client!)
```
queen uses simple HTTP client (reqwest)
    ↓
    POST /v1/inference (with InferenceRequest)
    ↓
worker uses job-server INTERNALLY
    ↓
    Create job, execute inference
    ↓
    Stream tokens via SSE
```

**Key Difference:** Queen doesn't use job-client to talk to worker!

---

## 📝 Action Items

### 1. Update queen-rbee/src/job_router.rs ✅ TODO
- Add CRITICAL ARCHITECTURE NOTE for Infer operation
- Explain why it's NOT forwarded to hive
- Document direct queen → worker connection

### 2. Update rbee-hive/src/job_router.rs ✅ TODO
- Add CRITICAL ARCHITECTURE NOTE for Infer operation
- Reject Infer with clear error message
- Explain this is intentional, not a bug

### 3. Document queen → worker protocol ✅ TODO
- Create separate doc for inference routing
- Explain why it's simple HTTP, not job-client
- Document InferenceRequest format

### 4. Keep worker's job-server ✅ DECISION
- Worker DOES use job-server internally
- But it's for SSE streaming, not operation routing
- Queen doesn't use job-client to talk to worker

---

## 🚨 Common Mistakes to Avoid

### ❌ DON'T: Forward Infer to hive
```rust
// WRONG!
Operation::Infer { .. } => {
    hive_forwarder::forward_to_hive(&job_id, operation, config).await?
}
```

### ✅ DO: Handle Infer in queen directly
```rust
// CORRECT!
Operation::Infer { .. } => {
    // Select worker
    // Direct HTTP POST to worker
    // Stream tokens back
}
```

### ❌ DON'T: Use job-client for queen → worker
```rust
// WRONG!
let client = JobClient::new(&worker_url);
client.submit_and_stream(operation, ...).await?
```

### ✅ DO: Use simple HTTP for queen → worker
```rust
// CORRECT!
let client = reqwest::Client::new();
let response = client
    .post(format!("{}/v1/inference", worker_url))
    .json(&inference_request)
    .send()
    .await?;
```

---

## 🎯 Summary

1. **Queen circumvents hive for inference** (direct to worker)
2. **Hive manages worker lifecycle** (spawn, stop, list)
3. **Worker uses job-server internally** (for SSE streaming)
4. **Queen uses simple HTTP to talk to worker** (not job-client)
5. **This is UNINTUITIVE but CORRECT** (performance + simplicity)

**Remember:** If you're confused, re-read this document!

---

**TEAM-261 Architecture Clarity**  
**Date:** Oct 23, 2025  
**Status:** 🚨 CRITICAL REFERENCE
