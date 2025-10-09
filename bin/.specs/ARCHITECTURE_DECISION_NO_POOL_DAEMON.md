# Architecture Decision: No Pool Daemon Needed

**Date:** 2025-10-09T17:17:00+02:00  
**Decision By:** User (Vince)  
**Documented By:** TEAM-024  
**Status:** NORMATIVE  
**Impact:** HIGH - Changes M1 milestone

---

## Decision

**pool-managerd (daemon) is NOT needed.**

The pool manager functionality is fully provided by `pool-ctl` CLI (`llorch-pool` binary).

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

**These are all CLI commands, not long-running services!**

### What Needs to Be a Daemon vs CLI

| Component | Type | Why |
|-----------|------|-----|
| **orchestratord** | Daemon | Accepts inference requests 24/7, routes to workers |
| **llorch-candled** (worker) | Daemon | Keeps model in VRAM, accepts inference requests |
| **pool-ctl** (llorch-pool) | CLI | Control operations on-demand via SSH |
| **llorch-ctl** (llorch) | CLI | Remote control operations via SSH |

### The Correct Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATORD (HTTP Daemon) - M2                                │
│ Binary: orchestratord                                            │
│ Port: 8080                                                       │
│ Purpose: Routes inference requests to workers                    │
│ Runs: 24/7 as daemon                                             │
│ Why daemon: Accepts client requests continuously                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─ HTTP POST /execute (direct to workers)
                     │
┌────────────────────┴────────────────────────────────────────────┐
│ WORKERS (HTTP Daemons) - M0 ✅                                   │
│ Binary: llorch-candled                                           │
│ Ports: 8001, 8002, 8003, etc.                                    │
│ Purpose: Execute inference, stream tokens                        │
│ Runs: 24/7 as daemon (one per model)                             │
│ Why daemon: Keep model loaded in VRAM                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ POOL MANAGER (CLI Tool) - M0 ✅                                  │
│ Binary: llorch-pool                                              │
│ Purpose: Local pool operations (models, workers)                 │
│ Runs: On-demand when operator calls it                           │
│ Why CLI: Control operations don't need 24/7 daemon               │
│                                                                   │
│ Commands:                                                        │
│ - llorch-pool models download <model>                            │
│ - llorch-pool worker spawn <backend> --model <model>             │
│ - llorch-pool worker list                                        │
│ - llorch-pool worker stop <id>                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR CLI (CLI Tool) - M0 ✅                              │
│ Binary: llorch                                                   │
│ Purpose: Remote pool control via SSH                             │
│ Runs: On-demand when operator calls it                           │
│ Why CLI: Control operations don't need 24/7 daemon               │
│                                                                   │
│ Commands:                                                        │
│ - llorch pool models download <model> --host <pool>              │
│ - llorch pool worker spawn <backend> --host <pool> --model <m>   │
│ - llorch infer --worker <host:port> --prompt <text>              │
└─────────────────────────────────────────────────────────────────┘
```

---

## What This Changes

### ❌ REMOVED: M1 Milestone (pool-managerd daemon)

**Old Plan:**
- M0: Workers + CLIs
- M1: Build pool-managerd daemon (HTTP server)
- M2: Build orchestratord daemon

**New Plan:**
- M0: Workers + CLIs ✅ COMPLETE
- ~~M1: pool-managerd~~ ❌ NOT NEEDED
- M1: Build orchestratord daemon (moved up from M2)

### ✅ SIMPLIFIED: Two-Binary System

**Only 2 daemon binaries needed:**
1. **orchestratord** - Routes inference requests
2. **llorch-candled** - Executes inference

**Plus 2 CLI tools:**
1. **llorch-pool** - Local pool management
2. **llorch** - Remote pool control

---

## Architecture Comparison

### Old (Incorrect) Architecture
```
Client → orchestratord → pool-managerd → worker
         (daemon)        (daemon)         (daemon)
```

### New (Correct) Architecture
```
Client → orchestratord → worker
         (daemon)         (daemon)

Operator → llorch → llorch-pool → spawn worker
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
- ✅ llorch-candled (worker daemon)
- ✅ llorch-pool (local pool CLI)
- ✅ llorch (remote control CLI)
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
- [ ] orchestratord binary (HTTP daemon)
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
1. **orchestratord** (port 8080) - Routes inference
2. **llorch-candled** (ports 8001+) - Executes inference

### Two CLIs (SSH/Local)
1. **llorch** - Remote control via SSH
2. **llorch-pool** - Local pool operations

### Communication Flow

**Control Plane (Operator):**
```
Operator (human)
    ↓ runs
llorch (CLI)
    ↓ SSH
llorch-pool (CLI on remote machine)
    ↓ spawns
llorch-candled (daemon)
```

**Data Plane (Inference):**
```
Client (SDK)
    ↓ HTTP POST /v2/tasks
orchestratord (daemon)
    ↓ HTTP POST /execute
llorch-candled (daemon)
    ↓ SSE stream
orchestratord (relay)
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
2. [ ] M1: Build orchestratord (not pool-managerd!)

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
- ✅ llorch-candled (workers) - No changes
- ✅ llorch-pool (pool CLI) - No changes
- ✅ llorch (remote CLI) - No changes

### Code That Never Needs to Be Written
- ❌ pool-managerd HTTP server
- ❌ pool-managerd heartbeat
- ❌ pool-managerd worker lifecycle API
- ❌ pool-managerd GPU discovery API

### Code That Still Needs to Be Written
- ⏳ orchestratord HTTP server (M1)
- ⏳ orchestratord admission control
- ⏳ orchestratord queue management
- ⏳ orchestratord scheduling
- ⏳ orchestratord SSE relay

---

## Verification

### Test Current Architecture Works:
```bash
# 1. Spawn worker (CLI)
llorch-pool worker spawn cpu --model qwen-0.5b

# 2. Test inference (CLI)
llorch infer --worker localhost:8001 --prompt "Hello" --max-tokens 20

# 3. Stop worker (CLI)
llorch-pool worker stop-all
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
- `llorch-pool` (local CLI)
- `llorch` (remote CLI via SSH)

This simplifies the architecture and removes an entire milestone (M1).

**Next milestone is now building orchestratord (M1, was M2).**

---

**Signed:** TEAM-024  
**Approved By:** User (Vince)  
**Status:** NORMATIVE - Follow this architecture going forward
