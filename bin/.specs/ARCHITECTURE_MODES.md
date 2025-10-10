# Architecture: Two Usage Modes

**Created by:** TEAM-030  
**Updated by:** TEAM-037 (2025-10-10T14:02)  
**Date:** 2025-10-10  
**Status:** Implemented

---

## 🚨 CRITICAL: rbee-keeper is a TESTING TOOL 🚨

**rbee-keeper is NOT for production!**
- Testing: rbee-keeper spawns queen-rbee, runs test, kills everything
- Production: llama-orch SDK → queen-rbee directly

**WHENEVER queen-rbee DIES, ALL hives and workers DIE gracefully!**

---

## Overview

The llama-orch system supports two distinct usage modes:

1. **Ephemeral Mode (Testing)** - rbee-keeper spawns queen-rbee, everything dies after
2. **Persistent Mode (Production)** - queen-rbee pre-started, long-running

## Mode 1: Ephemeral (rbee-keeper → queen-rbee → rbee-hive)

**Purpose:** Testing, integration validation, development

### Lifecycle

```
Developer runs: rbee-keeper infer --node mac --model tinyllama --prompt "hello"

1. rbee-keeper spawns queen-rbee as child process
2. queen-rbee starts HTTP daemon on port 8080
3. queen-rbee uses SSH to start rbee-hive on mac
4. rbee-hive starts HTTP daemon on port 9200
5. rbee-hive spawns llm-worker-rbee process
6. Worker starts HTTP daemon on port 8001
7. Worker loads model into VRAM
8. Worker becomes ready
9. Worker sends ready callback to rbee-hive (HTTP POST /v1/workers/ready)
   - Includes: worker_id, url (http://mac:8001), model_ref, backend, device
10. rbee-hive registers worker locally (in-memory)
11. rbee-hive notifies queen-rbee (HTTP POST /v1/orchestrator/worker-ready)
    - Includes: worker_id, url, model_ref, backend, device
12. queen-rbee adds worker to global worker registry
13. queen-rbee looks up worker in registry (gets url: http://mac:8001)
14. queen-rbee sends inference request DIRECTLY to worker (HTTP POST http://mac:8001/execute)
    - Bypasses rbee-hive for inference (by design)
15. Worker streams tokens via SSE to queen-rbee
16. queen-rbee relays SSE stream to rbee-keeper stdout
17. User gets result
18. User sends SIGTERM to rbee-keeper (Ctrl+C)
19. rbee-keeper sends SIGTERM to queen-rbee
20. queen-rbee uses SSH to send SIGTERM to ALL rbee-hive instances
21. rbee-hive cascades shutdown to all workers
22. All processes exit, VRAM cleaned
```

**Key Points:**
- rbee-keeper controls queen-rbee lifecycle
- queen-rbee controls ALL hives via SSH
- **Worker HTTP server must start BEFORE ready callback**
- **rbee-hive is bypassed for inference** - queen-rbee connects directly to worker
- Worker registry in queen-rbee contains worker URLs for direct connection

### Storage

- **Worker registry:** In-memory HashMap (ephemeral - lost on exit)
- **Model catalog:** SQLite database (persistent - survives restarts)
- **Database file:** `~/.rbee/models.db` (tracks downloaded models)

### Use Cases

- Testing new models
- One-off inferences
- CI/CD pipelines
- Development

### Current Implementation (MVP)

✅ **Implemented:**
- In-memory worker registry in rbee-hive (ephemeral)
- SQLite model catalog (persistent)
- Model provisioner with catalog integration
- Cascading shutdown (rbee-hive → workers)

⏳ **Not Yet Implemented:**
- queen-rbee spawning (M1+ feature)
- Automatic cleanup on rbee-keeper exit

## Mode 2: Persistent (queen-rbee pre-started)

**Purpose:** Production, worker reuse, performance

### Lifecycle

```
# Operator starts queen-rbee
queen-rbee daemon &

# Operator starts rbee-hive on each node
ssh mac "rbee-hive daemon &"
ssh workstation "rbee-hive daemon &"

# Production application uses SDK
llama-orch-sdk → queen-rbee (HTTP POST /v2/tasks)
    ↓
queen-rbee → rbee-hive (check workers)
    ↓
queen-rbee → worker (HTTP POST /execute)
    ↓
worker → queen-rbee (SSE stream)
    ↓
queen-rbee → SDK (SSE stream)

# Workers stay alive, reused across requests
# Operator manages lifecycle manually
```

### Storage

- **Worker registry:** In-memory HashMap in queen-rbee (ephemeral - lost on restart)
- **Model catalog:** SQLite database in rbee-hive (persistent - survives restarts)
- **Database file:** `~/.rbee/models.db` (tracks downloaded models)

### Use Cases

- Production workloads
- Multi-tenant systems
- High-throughput inference
- Long-running services
- **NOT rbee-keeper** - use SDK instead

### Implementation Status (M1+)

⏳ **Planned:**
- queen-rbee HTTP daemon
- Multi-hive coordination via SSH
- Worker reuse across requests
- Graceful shutdown cascade (queen → ALL hives → ALL workers)

## Design Decisions

### Why This Storage Strategy?

**Worker Registry (In-Memory):**
- Ephemeral mode: Everything dies after task → no persistence needed
- Persistent mode: In-memory is faster, simpler
- Health monitoring rebuilds state anyway
- **No SQLite** - workers are transient

**Model Catalog (SQLite):**
- Models are large files → need to track what's downloaded
- Avoid re-downloading same model multiple times
- Persistent across rbee-hive restarts
- **SQLite is perfect** - simple, persistent, no server needed

### Cascading Shutdown

**Signal Flow:**
```
SIGTERM → rbee-hive
    ↓
rbee-hive sends POST /v1/shutdown to all workers
    ↓
workers exit gracefully
    ↓
rbee-hive clears registry and exits
```

**Implementation:**
- rbee-hive tracks worker URLs in registry
- Graceful shutdown with 5s timeout per worker
- Registry cleared after all workers notified

### Ephemeral vs Persistent

**rbee-hive (current):**
- Can run standalone (ephemeral)
- Can be managed by queen-rbee (persistent - M1+)
- In-memory registry works for both modes

**queen-rbee (future M1+):**
- Will track multiple hives
- Will cascade shutdown to all hives
- Will use in-memory registry (no SQLite)

## Architecture Evolution

**TEAM-029 → TEAM-030 Changes:**

### Removed
- ❌ `bin/shared-crates/worker-registry` (SQLite-based - workers are ephemeral!)

### Kept
- ✅ `bin/shared-crates/model-catalog` (SQLite-based - models are persistent!)
- ✅ In-memory `WorkerRegistry` in rbee-hive (already existed)
- ✅ `ModelProvisioner` with catalog integration

### Key Insight
- **Workers:** Ephemeral → In-memory registry
- **Models:** Persistent → SQLite catalog
- Different lifecycles require different storage strategies

### Benefits
- Worker registry is fast (no DB overhead)
- Model catalog prevents re-downloads
- Clean separation of concerns
- Optimal storage for each use case

## Testing

### Ephemeral Mode Test

```bash
# Terminal 1: Start pool manager
./target/debug/rbee-hive daemon

# Terminal 2: Run inference
./target/debug/rbee infer \
  --node localhost \
  --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
  --prompt "hello world" \
  --max-tokens 10

# Terminal 1: Ctrl+C to shutdown
# Verify: No .db files, no lingering processes
```

### Verification

```bash
# Model catalog database exists (persistent)
ls ~/.rbee/
# Should show: models.db

# No lingering workers
ps aux | grep llm-worker-rbee
# Should be empty

# VRAM freed
nvidia-smi  # or metal activity monitor
# Should show no model loaded
```

## References

- `bin/.plan/TEAM_029_HANDOFF_FINAL.md` - Architecture redesign rationale
- `bin/.specs/.gherkin/test-001-mvp.md` - MVP test specification
- `bin/rbee-hive/src/registry.rs` - In-memory worker registry
- `bin/rbee-hive/src/provisioner.rs` - Filesystem-based model cache
