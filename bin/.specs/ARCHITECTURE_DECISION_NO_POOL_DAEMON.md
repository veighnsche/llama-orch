# Architecture Decision: No Pool Daemon Needed

**Date:** 2025-10-09T17:17:00+02:00  
**Decision By:** User (Vince)  
**Documented By:** TEAM-024  
**Status:** ⚠️ COMPLETELY WRONG - SUPERSEDED BY MVP  
**Impact:** HIGH - Changes M1 milestone

---

## ⚠️ THIS DECISION WAS COMPLETELY INCORRECT

**The MVP (test-001-mvp.md) is normative and requires rbee-hive as a persistent HTTP daemon.**

From test-001-mvp.md:
- **Phase 2 (lines 42-55)**: rbee-hive has HTTP API at `GET http://mac.home.arpa:8080/v1/health`
- **Phase 5 (lines 169-173)**: Pool manager remains running as persistent daemon, monitors worker health every 30s, enforces idle timeout

**CORRECT ARCHITECTURE:**
- **rbee-hive** is an **HTTP daemon** (not a CLI)
- **rbee-keeper** calls rbee-hive's HTTP API
- **SSH is only used** to start/stop the rbee-hive daemon remotely or for SSH tunneling

**TEAM-025 NOTE**: This entire document is wrong. Ignore it completely and follow test-001-mvp.md.

---

## ~~Decision~~ (OUTDATED)

~~**pool-managerd (daemon) is NOT needed.**~~

~~The pool manager functionality is fully provided by `rbee-hive` CLI (`rbee-hive` binary).~~

---

## Rationale

### Why Pool Manager Doesn't Need to Be a Daemon

**Key Insight:** Pool management is **control operations**, not **data plane operations**.

**Control operations (SSH-based):**
- Download models
- Spawn workers
- Stop workers
- Check status
- Git operations


### What Needs to Be a Daemon vs CLI

| Component | Type | Why |
|-----------|------|-----|
| **queen-rbee** | HTTP Daemon | Accepts inference requests 24/7, routes to workers |
| **llm-worker-rbee** | HTTP Daemon | Keeps model in VRAM, accepts inference requests |
| **rbee-hive** | HTTP Daemon | Pool management HTTP API, monitors workers, enforces timeouts |
| **rbee-keeper** | CLI | Calls HTTP APIs of queen-rbee and rbee-hive |

### The CORRECT Architecture (Per MVP)

```
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR (HTTP Daemon) - M1                                 │
│ Binary: queen-rbee                                              │
│ Port: 8080                                                       │
│ Purpose: Routes inference requests to workers                    │
│ Runs: 24/7 as HTTP daemon                                        │
│ Why daemon: Accepts client requests continuously                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─ HTTP POST /execute (direct to workers)
                     │
┌────────────────────┴────────────────────────────────────────────┐
│ WORKERS (HTTP Daemons) - M0 ✅                                   │
│ Binary: llm-worker-rbee                                         │
│ Ports: 8001, 8002, 8003, etc.                                    │
│ Purpose: Execute inference, stream tokens                        │
│ Runs: 24/7 as HTTP daemon (one per model)                        │
│ Why daemon: Keep model loaded in VRAM                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ POOL MANAGER (HTTP Daemon) - M1 ❌ NOT BUILT YET                │
│ Binary: rbee-hive                                               │
│ Port: 8080 or 9200                                               │
│ Purpose: Pool management HTTP API                                │
│ Runs: 24/7 as HTTP daemon                                        │
│ Why daemon: Monitors worker health (30s), enforces timeouts      │
│                                                                   │
│ HTTP API:                                                        │
│ - GET  /v1/health                                                │
│ - POST /v1/models/download                                       │
│ - POST /v1/workers/spawn                                         │
│ - GET  /v1/workers/list                                          │
│ - POST /v1/workers/ready (callback from workers)                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ CLI TOOL - M0 ✅                                                 │
│ Binary: rbee-keeper                                             │
│ Purpose: Calls HTTP APIs of queen-rbee and rbee-hive            │
│ Runs: On-demand when operator calls it                           │
│                                                                   │
│ Commands (via HTTP):                                             │
│ - rbee-keeper pool models download <model> --host <pool>        │
│   → POST http://mac.home.arpa:8080/v1/models/download           │
│ - rbee-keeper pool worker spawn <backend> --host <pool>         │
│   → POST http://mac.home.arpa:8080/v1/workers/spawn             │
│ - rbee-keeper infer --worker <host:port> --prompt <text>        │
│   → POST http://worker:8001/v1/inference                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## What This Changes

### ❌ REMOVED: M1 Milestone (pool-managerd daemon)

**Old Plan:**
- M0: Workers + CLIs
- M1: Build pool-managerd daemon (HTTP server)
- M2: Build queen-rbee daemon

**New Plan:**
- M0: Workers + CLIs ✅ COMPLETE
- ~~M1: pool-managerd~~ ❌ NOT NEEDED
- M1: Build queen-rbee daemon (moved up from M2)

### ✅ SIMPLIFIED: Two-Binary System

**Only 2 daemon binaries needed:**
1. **queen-rbee** - Routes inference requests
2. **llm-worker-rbee** - Executes inference

**Plus 2 CLI tools:**
1. **rbee-hive** - Local pool management
2. **llorch** - Remote pool control

---

## Architecture Comparison

### Old (Incorrect) Architecture
```
Client → queen-rbee → pool-managerd → worker
         (daemon)        (daemon)         (daemon)
```

### New (Correct) Architecture
```
Client → queen-rbee → worker
         (daemon)         (daemon)

Operator → rbee → rbee-hive → spawn worker
           (CLI)    (CLI via SSH)
```

**Key Insight:** Control plane (SSH) and data plane (HTTP) are separate!

---

## Why This Makes Sense

### 1. Pool Manager Has No Long-Running State
- Doesn't need to accept requests 24/7
- Doesn't keep connections open
- Doesn't maintain in-memory state
- All state is in filesystem (catalog.json, worker PIDs)

### 2. Workers Are Already Daemons
- Workers keep model in VRAM
- Workers accept HTTP requests
- Workers are the long-running processes

### 3. Orchestrator Needs to Be a Daemon
- Accepts client requests 24/7
- Maintains queue state
- Routes to workers
- Relays SSE streams

### 4. Pool Operations Are On-Demand
- Download model: Run once, exits
- Spawn worker: Run once, spawns daemon, exits
- Stop worker: Run once, exits
- List workers: Run once, exits

**None of these need a daemon!**

---

## Impact on Specs

### Files That Need Updating

**High Priority (user-facing):**
- [ ] `/ORCHESTRATION_OVERVIEW.md` - Remove pool-managerd references
- [ ] `/QUICK_STATUS.md` - Remove M1 milestone
- [ ] `/bin/.plan/TEAM_025_HANDOFF.md` - Remove pool-managerd tasks
- [ ] `/README.md` - Update architecture section

**Medium Priority (specs):**
- [ ] `/bin/.specs/00_llama-orch.md` - Update Section 6.2 (pool manager)
- [ ] `/bin/.specs/COMPLETE_BINARY_ARCHITECTURE.md` - Remove pool-managerd
- [ ] `/bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md` - Update diagrams

**Low Priority (historical):**
- [ ] Other architecture decision docs (mark as outdated)

---

## New Milestone Plan

### M0: Foundation ✅ COMPLETE
**Deliverables:**
- ✅ llm-worker-rbee (worker daemon)
- ✅ rbee-hive (local pool CLI)
- ✅ rbee (remote control CLI)
- ✅ Model catalog system
- ✅ Worker spawning
- ✅ Token generation

**Status:** DONE by TEAM-022, 023, 024

### CP4: Multi-Model Testing ⏳ NEXT
**Deliverables:**
- [ ] Download all 4 models
- [ ] Test all backends
- [ ] Document results

**Status:** IN PROGRESS

### M1: Orchestrator Daemon 🔜 AFTER CP4
**Deliverables:**
- [ ] queen-rbee binary (HTTP daemon)
- [ ] Client API (`POST /v2/tasks`)
- [ ] Admission control
- [ ] Queue management
- [ ] Scheduling
- [ ] SSE relay

**Status:** NOT STARTED (was M2, moved up)

**Note:** This is now the ONLY daemon left to build!

---

## Updated System Architecture

### Two Daemons (HTTP)
1. **queen-rbee** (port 8080) - Routes inference
2. **llm-worker-rbee** (ports 8001+) - Executes inference

### Two CLIs (SSH/Local)
1. **llorch** - Remote control via SSH
2. **rbee-hive** - Local pool operations

### Communication Flow

**Control Plane (Operator):**
```
Operator (human)
    ↓ runs
llorch (CLI)
    ↓ SSH
rbee-hive (CLI on remote machine)
    ↓ spawns
llm-worker-rbee (daemon)
```

**Data Plane (Inference):**
```
Client (SDK)
    ↓ HTTP POST /v2/tasks
queen-rbee (daemon)
    ↓ HTTP POST /execute
llm-worker-rbee (daemon)
    ↓ SSE stream
queen-rbee (relay)
    ↓ SSE stream
Client
```

**No pool-managerd in either flow!**

---

## Action Items for TEAM-025

### Immediate (Before Starting Work):
1. [ ] Read this decision document
2. [ ] Update TEAM_025_HANDOFF.md (remove pool-managerd)
3. [ ] Update ORCHESTRATION_OVERVIEW.md (remove pool-managerd)
4. [ ] Update QUICK_STATUS.md (remove M1 pool-managerd milestone)

### Then Proceed With:
1. [ ] CP4: Multi-model testing (as planned)
2. [ ] M1: Build queen-rbee (not pool-managerd!)

---

## Benefits of This Architecture

### Simpler
- ✅ Only 2 daemons instead of 3
- ✅ Fewer moving parts
- ✅ Easier to understand

### More Maintainable
- ✅ Less code to maintain
- ✅ Fewer HTTP APIs
- ✅ Fewer integration points

### More Reliable
- ✅ Fewer daemons to crash
- ✅ Fewer network calls
- ✅ Simpler failure modes

### More Flexible
- ✅ CLI can be called from scripts
- ✅ CLI can be called via SSH
- ✅ No daemon lifecycle management

---

## Migration Notes

### Code That Doesn't Need to Change
- ✅ llm-worker-rbee (workers) - No changes
- ✅ rbee-hive (pool CLI) - No changes
- ✅ rbee (remote CLI) - No changes

### Code That Never Needs to Be Written
- ❌ pool-managerd HTTP server
- ❌ pool-managerd heartbeat
- ❌ pool-managerd worker lifecycle API
- ❌ pool-managerd GPU discovery API

### Code That Still Needs to Be Written
- ⏳ queen-rbee HTTP server (M1)
- ⏳ queen-rbee admission control
- ⏳ queen-rbee queue management
- ⏳ queen-rbee scheduling
- ⏳ queen-rbee SSE relay

---

## Verification

### Test Current Architecture Works:
```bash
# 1. Spawn worker (CLI)
rbee-hive worker spawn cpu --model qwen-0.5b

# 2. Test inference (CLI)
llorch infer --worker localhost:8001 --prompt "Hello" --max-tokens 20

# 3. Stop worker (CLI)
rbee-hive worker stop-all
```

**Result:** ✅ Everything works without pool-managerd!

### Test Remote Control Works:
```bash
# 1. Remote spawn (SSH + CLI)
llorch pool worker spawn metal --host mac.home.arpa --model qwen-0.5b --gpu 0

# 2. Remote test (CLI to remote worker)
llorch infer --worker mac.home.arpa:8001 --prompt "Hello" --max-tokens 20

# 3. Remote stop (SSH + CLI)
llorch pool worker stop-all --host mac.home.arpa
```

**Result:** ✅ Everything works without pool-managerd!

---

## Conclusion

**pool-managerd is NOT needed.**

The pool manager functionality is fully provided by:
- `rbee-hive` (local CLI)
- `rbee` (remote CLI via SSH)

This simplifies the architecture and removes an entire milestone (M1).

**Next milestone is now building queen-rbee (M1, was M2).**

---

**Signed:** TEAM-024  
**Approved By:** User (Vince)  
**Status:** NORMATIVE - Follow this architecture going forward
