# Architecture: Two Usage Modes

**Created by:** TEAM-030  
**Date:** 2025-10-10  
**Status:** Implemented

## Overview

The llama-orch system supports two distinct usage modes:

1. **Ephemeral Mode** - Single inference, everything dies after (testing/development)
2. **Persistent Mode** - Long-running, worker reuse (production - M1+)

## Mode 1: Ephemeral (rbee-keeper → rbee-hive)

**Purpose:** Testing, single inference, clean environment

### Lifecycle

```
User runs: rbee infer --node localhost --model tinyllama --prompt "hello"

1. rbee-keeper connects to rbee-hive (pool manager) via HTTP
2. rbee-hive spawns llm-worker-rbee process
3. Worker loads model and becomes ready
4. rbee-keeper streams inference via SSE
5. User gets result
6. User sends SIGTERM to rbee-hive (Ctrl+C)
7. rbee-hive cascades shutdown to all workers
8. All processes exit, VRAM cleaned
```

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

## Mode 2: Persistent (queen-rbee daemon)

**Purpose:** Production, worker reuse, performance

### Lifecycle

```
# On control node
queen-rbee daemon &

# Workers stay alive, reused across requests
# User manages lifecycle manually
```

### Storage

- **Worker registry:** In-memory HashMap (ephemeral - lost on restart)
- **Model catalog:** SQLite database (persistent - shared across restarts)
- **Database file:** `~/.rbee/models.db` (tracks downloaded models)

### Use Cases

- Production workloads
- Multi-tenant systems
- High-throughput inference
- Long-running services

### Future Implementation (M1+)

⏳ **Planned:**
- queen-rbee HTTP daemon
- Multi-hive coordination
- Worker reuse across requests
- Graceful shutdown cascade (queen → hives → workers)

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
