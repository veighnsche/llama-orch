# Architecture Summary - TEAM-025 (Post-Rebranding)

**Date**: 2025-10-09T23:00  
**Status**: NORMATIVE  
**Source**: test-001-mvp.md (THE source of truth)

---

## The 4 Fundamental Binaries

### 1. queen-rbee (HTTP Daemon) - M1 ❌ NOT BUILT

**Port**: 8080  
**Purpose**: Orchestrator - Routes inference requests to workers  
**Location**: `bin/queen-rbee/`

**Responsibilities:**
- Accept client inference requests (POST /v2/tasks)
- Maintain worker registry
- Make scheduling decisions
- Dispatch jobs to workers (direct HTTP)
- Relay SSE streams
- Maintain job queue (SQLite)

---

### 2. rbee-hive (HTTP Daemon) - M1 ❌ NOT BUILT

**Port**: 8080 or 9200  
**Purpose**: Pool manager - Manages workers on each pool  
**Location**: `bin/rbee-hive/`

**HTTP API Endpoints:**
- `GET /v1/health` - Health check (MVP Phase 2, lines 42-55)
- `POST /v1/models/download` - Download models
- `POST /v1/workers/spawn` - Spawn worker processes
- `GET /v1/workers/list` - List running workers
- `POST /v1/workers/ready` - Callback from workers

**Background Tasks:**
- Monitor worker health every 30s (MVP Phase 5, line 171)
- Enforce idle timeout (5 minutes) (MVP Phase 5, line 172)
- Download models (hf CLI)
- Git operations
- Spawn/stop worker processes

**Current State**: CLI exists at `bin/rbee-hive/`, needs daemon features added

---

### 3. llm-worker-rbee (HTTP Daemon) - M0 ✅ WORKING

**Port**: 8001-8999  
**Purpose**: Worker - Executes inference, keeps model in VRAM  
**Location**: `bin/llm-worker-rbee/`

**HTTP API:**
- `POST /v1/execute` - Execute inference
- `GET /v1/ready` - Health check
- `GET /v1/loading/progress` - Model loading progress (SSE)

**Lifecycle:**
1. Spawned by rbee-hive
2. Load model into VRAM
3. Start HTTP server
4. Send ready callback to rbee-hive
5. Accept inference requests
6. Stream tokens via SSE
7. Auto-shutdown after 5 min idle

---

### 4. rbee-keeper (CLI) - M0 ✅ WORKING

**Purpose**: CLI tool - Calls HTTP APIs  
**Location**: `bin/rbee-keeper/`

**Commands:**
```bash
# Pool operations (calls rbee-hive HTTP API)
rbee-keeper pool models download <model> --host <pool>
  → POST http://mac.home.arpa:8080/v1/models/download

rbee-keeper pool worker spawn <backend> --host <pool>
  → POST http://mac.home.arpa:8080/v1/workers/spawn

# Inference (calls worker HTTP API directly)
rbee-keeper infer --worker <host:port> --prompt <text>
  → POST http://worker:8001/v1/inference
```

**Communication:**
- Calls HTTP APIs (NOT SSH command execution)
- Can use SSH tunneling for remote access
- No REPL, no conversation

---

## Communication Flow

### Control Plane (Pool Operations)

```
rbee-keeper (CLI)
    ↓ HTTP
rbee-hive (HTTP daemon :8080)
    ↓ spawn process
llm-worker-rbee (HTTP daemon :8001)
```

**Example:**
```bash
rbee-keeper pool worker spawn metal --host mac --model tinyllama
```
1. rbee-keeper makes HTTP call: `POST http://mac.home.arpa:8080/v1/workers/spawn`
2. rbee-hive spawns: `llm-worker-rbee --model tinyllama.gguf --port 8001`
3. Worker loads model, starts HTTP server
4. Worker sends callback: `POST http://mac.home.arpa:8080/v1/workers/ready`

---

### Data Plane (Inference)

```
Client (llama-orch-sdk)
    ↓ HTTP POST /v2/tasks
queen-rbee (HTTP daemon :8080)
    ↓ HTTP POST /execute (DIRECT to worker)
llm-worker-rbee (HTTP daemon :8001)
    ↓ SSE stream
queen-rbee (relays)
    ↓ SSE stream
Client
```

---

## What SSH Is Used For

**SSH is ONLY used for:**
1. Starting/stopping daemons remotely
2. SSH tunneling to reach HTTP endpoints
3. **NOT** for executing pool operations

**Example (starting rbee-hive remotely):**
```bash
ssh mac.home.arpa "rbee-hive daemon start"
```

**NOT this (wrong):**
```bash
# ❌ WRONG - This is NOT how it works
ssh mac.home.arpa "rbee-hive models download tinyllama"
```

**Correct:**
```bash
# ✅ CORRECT - rbee-keeper calls HTTP API
rbee-keeper pool models download tinyllama --host mac
  → POST http://mac.home.arpa:8080/v1/models/download
```

---

## MVP Proof

**From test-001-mvp.md:**

**Phase 2 (lines 42-55)**: rbee-hive HTTP API
```
GET http://mac.home.arpa:8080/v1/health
Authorization: Bearer <api_key>

Response:
{
  "status": "alive",
  "version": "0.1.0",
  "api_version": "v1"
}
```

**Phase 5 (lines 169-173)**: Pool manager lifecycle
```
Pool manager lifecycle:
- Remains running as persistent daemon
- Monitors worker health every 30s
- Enforces idle timeout (5 minutes)
```

---

## What Needs to Be Built (M1)

### 1. queen-rbee daemon
- HTTP server on :8080
- Job queue (SQLite)
- Scheduling logic
- Worker registry
- SSE relay

### 2. rbee-hive daemon features
- HTTP server on :8080 or :9200
- Health monitoring loop (every 30s)
- Idle timeout enforcement (5 minutes)
- Worker lifecycle tracking
- HTTP API endpoints

**Current rbee-hive CLI can be extended** - add daemon mode with HTTP server.

---

## Outdated Documents

**These documents are WRONG and should be ignored:**

1. `ARCHITECTURE_DECISION_NO_POOL_DAEMON.md` - Says rbee-hive is not needed (WRONG)
2. `CONTROL_PLANE_ARCHITECTURE_DECISION.md` - Says use SSH for pool operations (WRONG)

**All have been updated with warnings pointing to this document and test-001-mvp.md.**

---

## Summary

**3 HTTP Daemons:**
- queen-rbee (orchestrator)
- rbee-hive (pool manager)
- llm-worker-rbee (workers)

**1 CLI:**
- rbee-keeper (calls HTTP APIs)

**ALL communication is HTTP. SSH is only for starting daemons remotely.**

**Source of truth**: `bin/.specs/.gherkin/test-001-mvp.md`
