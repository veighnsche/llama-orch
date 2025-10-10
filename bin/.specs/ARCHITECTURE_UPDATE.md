# Architecture Update: queen-rbee Orchestration

**Date:** 2025-10-10T13:54  
**Updated:** 2025-10-10T13:58 - Added rbee-keeper purpose clarification  
**Status:** CRITICAL UPDATE  
**Impact:** ALL test scenarios

---

## CRITICAL: rbee-keeper Purpose

**rbee-keeper is a TESTING/INTEGRATION TOOL, NOT for production!**

### Purpose
- **Integration testing** of the entire rbee system
- **Quick one-off inferences** for development/testing
- **Validates** that queen-rbee, rbee-hive, and workers work together
- **NOT meant for production workloads**

### Production Use Case
- Users should use **llama-orch SDK** (HTTP client) → queen-rbee
- **NOT** rbee-keeper (CLI)

### Testing Use Case
- Developers use **rbee-keeper** to test the system
- rbee-keeper spawns queen-rbee, orchestrates everything, then kills it all

---

## What Changed

### ❌ OLD (INCORRECT)
```
rbee-keeper → rbee-hive → worker
```

### ✅ NEW (CORRECT)
```
rbee-keeper (TESTING TOOL)
    ↓ spawns for testing
queen-rbee → rbee-hive → worker
    ↓
(relays SSE stream)
```

---

## The 4 Components

| Component | Type | Port | Purpose | Production? |
|-----------|------|------|---------|-------------|
| **rbee-keeper** | CLI | N/A | **TESTING TOOL** - Integration tester | ❌ NO |
| **queen-rbee** | HTTP Daemon | 8080 | Orchestrator, uses SSH to control rbee-hive | ✅ YES |
| **rbee-hive** | HTTP Daemon | 9200 | Pool manager, spawns workers | ✅ YES |
| **llm-worker-rbee** | HTTP Daemon | 8001+ | Worker, executes inference | ✅ YES |

---

## Complete Flow

### 1. Startup (Ephemeral Mode)

```
User runs: rbee-keeper infer --node mac --model tinyllama --prompt "hello"

rbee-keeper
    ↓ spawns as child process
queen-rbee (starts HTTP daemon on :8080)
    ↓ SSH to mac.home.arpa
rbee-hive (starts HTTP daemon on :9200)
    ↓ spawns as child process
llm-worker-rbee (starts HTTP daemon on :8001)
```

### 2. Worker Ready Callback

```
llm-worker-rbee
    ↓ HTTP POST /v1/workers/ready
rbee-hive (registers worker in-memory)
    ↓ HTTP POST /v1/orchestrator/worker-ready
queen-rbee (knows worker is ready)
```

### 3. Inference Execution

```
queen-rbee
    ↓ HTTP POST /execute
llm-worker-rbee (generates tokens)
    ↓ SSE stream
queen-rbee (relays stream)
    ↓ SSE to stdout
rbee-keeper (displays tokens to user)
```

### 4. Shutdown (Ephemeral Mode)

**CRITICAL RULE: WHENEVER QUEEN-RBEE DIES, ALL HIVES AND WORKERS STOP GRACEFULLY!**

```
User presses Ctrl+C
    ↓
rbee-keeper (receives SIGTERM)
    ↓ sends SIGTERM (ONLY if rbee-keeper spawned queen-rbee)
queen-rbee (receives SIGTERM)
    ↓ SSH: kill ALL rbee-hive instances on ALL nodes
rbee-hive on mac (receives SIGTERM)
    ↓ HTTP POST /v1/admin/shutdown to ALL workers
llm-worker-rbee (unloads model, exits)
    ↓
rbee-hive on workstation (receives SIGTERM)
    ↓ HTTP POST /v1/admin/shutdown to ALL workers
llm-worker-rbee (unloads model, exits)
    ↓
ALL rbee-hive instances exit
    ↓
queen-rbee exits
    ↓
rbee-keeper exits
```

**ALWAYS: queen-rbee death → ALL hives die → ALL workers die**

---

## Key Insights

### 1. rbee-keeper is a TESTING TOOL

- **rbee-keeper is NOT for production** - it's an integration tester
- **rbee-keeper spawns queen-rbee** for testing (ephemeral mode)
- **rbee-keeper validates** that the entire system works together
- **rbee-keeper kills queen-rbee** when done (which cascades to everything)
- **Production users** should use llama-orch SDK → queen-rbee directly

### 2. queen-rbee is the Orchestrator

- **queen-rbee controls ALL rbee-hive instances** via SSH
- **queen-rbee sends inference requests** directly to workers
- **queen-rbee relays SSE streams** from workers to rbee-keeper
- **queen-rbee does NOT exit** after inference (stays running)
- **CRITICAL: When queen-rbee dies, ALL hives and workers die gracefully**

### 3. rbee-hive is the Pool Manager

- **rbee-hive is controlled by queen-rbee** via SSH
- **rbee-hive spawns workers** on the local node
- **rbee-hive notifies queen-rbee** when workers are ready
- **rbee-hive does NOT exit** after spawning workers
- **rbee-hive ALWAYS dies** when queen-rbee dies

### 4. Workers are Dumb Executors

- **Workers are spawned by rbee-hive**
- **Workers send ready callback to rbee-hive**
- **Workers receive inference requests from queen-rbee**
- **Workers stream tokens via SSE to queen-rbee**
- **Workers ALWAYS die** when rbee-hive dies

---

## Process Ownership

### Ephemeral Mode

```
rbee-keeper owns queen-rbee
    ↓
queen-rbee owns rbee-hive (via SSH)
    ↓
rbee-hive owns workers
```

### Persistent Mode

```
Operator owns queen-rbee (pre-started)
Operator owns rbee-hive (pre-started on each node)
rbee-hive owns workers
```

---

## Communication Protocols

| From | To | Protocol | Purpose |
|------|-----|----------|---------|
| rbee-keeper | queen-rbee | HTTP | Submit inference requests |
| queen-rbee | rbee-hive | SSH | Start/stop pool manager |
| queen-rbee | rbee-hive | HTTP | Check status, get worker list |
| queen-rbee | worker | HTTP | Send inference requests |
| rbee-hive | worker | Process spawn | Start worker process |
| worker | rbee-hive | HTTP | Ready callback |
| rbee-hive | queen-rbee | HTTP | Worker ready notification |
| worker | queen-rbee | SSE | Stream tokens |
| queen-rbee | rbee-keeper | SSE | Relay token stream |

---

## Port Assignments

| Component | Port | Why |
|-----------|------|-----|
| queen-rbee | 8080 | Orchestrator HTTP API |
| rbee-hive | 9200 | Pool manager HTTP API (avoids conflict with queen-rbee) |
| llm-worker-rbee | 8001+ | Worker HTTP API (one port per worker) |

---

## Impact on Test Scenarios

### ALL scenarios must be updated to include queen-rbee:

**Before:**
```gherkin
When rbee-keeper sends inference request to rbee-hive
Then rbee-hive spawns worker
And worker streams tokens to rbee-keeper
```

**After:**
```gherkin
When rbee-keeper sends inference request to queen-rbee
Then queen-rbee uses SSH to ensure rbee-hive is running
And rbee-hive spawns worker
And worker sends ready callback to rbee-hive
And rbee-hive notifies queen-rbee
And queen-rbee sends inference request to worker
And worker streams tokens to queen-rbee
And queen-rbee relays stream to rbee-keeper
```

---

## Updated Lifecycle Rules

### RULE 1: rbee-keeper is EPHEMERAL CLI
- Spawns queen-rbee (or connects to existing)
- Streams output from queen-rbee to stdout
- Exits after command completes

### RULE 2: queen-rbee is PERSISTENT HTTP DAEMON
- Orchestrates everything
- Uses SSH to control rbee-hive
- Relays SSE streams
- Does NOT exit after inference

### RULE 3: rbee-hive is PERSISTENT HTTP DAEMON
- Controlled by queen-rbee via SSH
- Spawns workers
- Notifies queen-rbee when workers ready
- Does NOT exit after spawning workers

### RULE 4: llm-worker-rbee is PERSISTENT HTTP DAEMON
- Spawned by rbee-hive
- Sends ready callback to rbee-hive
- Receives inference requests from queen-rbee
- Streams tokens via SSE to queen-rbee
- Does NOT exit after inference (stays idle)

---

## Next Steps

1. **Update all test scenarios** to include queen-rbee
2. **Update lifecycle scenarios** (MVP-008 through MVP-013)
3. **Update architecture diagrams** in feature files
4. **Update README.md** with correct flow
5. **Update LIFECYCLE_CLARIFICATION.md** ✅ DONE

---

**Created by:** TEAM-037 (Testing Team)  
**Status:** CRITICAL - Must update all test scenarios  
**Priority:** P0 - Blocks implementation

---
Verified by Testing Team 🔍
