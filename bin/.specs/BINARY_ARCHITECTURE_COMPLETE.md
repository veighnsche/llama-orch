# Complete Binary Architecture

**Status**: Normative  
**Version**: 1.0  
**Date**: 2025-10-09

---

## Executive Summary

**6 Binaries + 2 Shared Crates:**

### Binaries

**Orchestrator:**
1. `orchestratord` - Daemon (HTTP server, scheduling, state)
2. `llorch-ctl` - CLI (HTTP client, controls daemon + pools)

**Pool Manager:**
3. `pool-managerd` - Daemon (HTTP server, worker lifecycle, GPU inventory)
4. `pool-ctl` - CLI (HTTP client, controls daemon + local operations)

**Worker:**
5. `llorch-candled` - Daemon (HTTP server, inference execution)

**Test Harness:**
6. `bdd-runner` - BDD test runner (already exists)

### Shared Crates

7. `bin/shared-crates/orchestrator-core/` - Shared logic for orchestratord + llorch-ctl
8. `bin/shared-crates/pool-core/` - Shared logic for pool-managerd + pool-ctl

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR (blep.home.arpa)                                    │
├─────────────────────────────────────────────────────────────────┤
│ orchestratord (daemon)                                           │
│   ├── HTTP Server :8080                                          │
│   ├── Uses: orchestrator-core (shared)                           │
│   ├── Scheduling, admission, job queue                           │
│   └── SQLite state store                                         │
│                                                                   │
│ llorch-ctl (CLI)                                                 │
│   ├── HTTP Client → orchestratord API                            │
│   ├── SSH Client → pool-ctl on remote hosts                      │
│   ├── Uses: orchestrator-core (shared)                           │
│   ├── Commands: llorch jobs submit/list/cancel                   │
│   ├── Commands: llorch pool <cmd> --host <host>                  │
│   └── NO REPL, NO CONVERSATION (HARD RULE)                       │
└─────────────────────────────────────────────────────────────────┘
                     │ SSH or HTTP
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ POOL MANAGER (mac.home.arpa, workstation.home.arpa)             │
├─────────────────────────────────────────────────────────────────┤
│ pool-managerd (daemon)                                           │
│   ├── HTTP Server :9200                                          │
│   ├── Uses: pool-core (shared)                                   │
│   ├── Worker lifecycle, GPU inventory                            │
│   └── Model caching (optional)                                   │
│                                                                   │
│ pool-ctl (CLI)                                                   │
│   ├── HTTP Client → pool-managerd API (if running)               │
│   ├── Direct execution (M0, no daemon)                           │
│   ├── Uses: pool-core (shared)                                   │
│   ├── Commands: llorch-pool models download/list                 │
│   ├── Commands: llorch-pool git pull/sync                        │
│   ├── Commands: llorch-pool worker spawn/stop                    │
│   └── NO REPL, NO CONVERSATION (HARD RULE)                       │
└─────────────────────────────────────────────────────────────────┘
                     │ spawn
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ WORKER (llorch-candled)                                          │
│   ├── HTTP Server :8001-8999                                     │
│   ├── Inference execution                                        │
│   ├── CUDA/Metal/CPU backend                                     │
│   └── NO CLI (controlled by pool-managerd)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Shared Crates

### orchestrator-core (Shared)

**Location:** `bin/shared-crates/orchestrator-core/`

**Purpose:** Shared logic between orchestratord daemon and llorch-ctl CLI

**Responsibilities:**
- Job queue data structures
- Scheduling algorithms (Rhai integration)
- Pool registry (pool manager tracking)
- API types (request/response)
- Configuration parsing
- State management types (not SQLite itself)

**Used by:**
- `orchestratord` - Daemon uses for runtime logic
- `llorch-ctl` - CLI uses for validation, types, client logic

**Does NOT include:**
- HTTP server (daemon only)
- SQLite persistence (daemon only)
- SSH client (CLI only)
- REPL/conversation (NEVER)

### pool-core (Shared)

**Location:** `bin/shared-crates/pool-core/`

**Purpose:** Shared logic between pool-managerd daemon and pool-ctl CLI

**Responsibilities:**
- Worker registry data structures
- GPU inventory types (NVML wrappers)
- Model catalog types
- Worker lifecycle logic
- API types (request/response)
- Configuration parsing

**Used by:**
- `pool-managerd` - Daemon uses for runtime logic
- `pool-ctl` - CLI uses for validation, types, spawning logic

**Does NOT include:**
- HTTP server (daemon only)
- Heartbeat protocol (daemon only)
- SSH operations (neither, that's llorch-ctl)
- REPL/conversation (NEVER)

---

## Control Relationships

### llorch-ctl Controls

**1. orchestratord daemon (M2+)**
```bash
# Via HTTP API
llorch orchestrator start
llorch orchestrator stop
llorch orchestrator status
llorch orchestrator logs
```

**2. Pool managers (M0+)**
```bash
# Via SSH (M0) or HTTP (M2+)
llorch pool status --host mac
llorch pool models download tinyllama --host mac
llorch pool worker spawn metal --host mac
```

**3. Jobs (M2+)**
```bash
# Via orchestratord HTTP API
llorch jobs submit --model llama3 --prompt "hello"
llorch jobs list
llorch jobs cancel job-123
llorch jobs status job-123
```

### pool-ctl Controls

**1. pool-managerd daemon (M1+)**
```bash
# Via HTTP API (if daemon running)
llorch-pool daemon start
llorch-pool daemon stop
llorch-pool daemon status
```

**2. Local operations (M0)**
```bash
# Direct execution (no daemon)
llorch-pool models download tinyllama
llorch-pool git pull
llorch-pool worker spawn metal --model tinyllama
```

**3. Workers (M0+)**
```bash
# Direct spawn (M0) or via daemon (M1+)
llorch-pool worker spawn metal --model tinyllama --gpu 0
llorch-pool worker stop worker-metal-0
llorch-pool worker list
```

---

## HARD RULES

### [RULE-001] NO REPL, NO CONVERSATION

**NEVER implement:**
```bash
# ❌ FORBIDDEN
llorch chat
llorch repl
llorch interactive
llorch conversation

# ❌ FORBIDDEN
$ llorch
> submit job with llama3
> list jobs
> cancel job-123
```

**Why forbidden:**
- Agentic API is HTTP-based (POST /v2/tasks)
- Conversations happen via HTTP SSE streams
- CLI is for CONTROL, not CONVERSATION
- UI/UX is web-based, not terminal-based

**Correct approach:**
```bash
# ✅ CORRECT: Single commands
llorch jobs submit --model llama3 --prompt "hello"
llorch jobs list
llorch jobs cancel job-123

# ✅ CORRECT: Scripting
for job in $(llorch jobs list --format json | jq -r '.[] | .id'); do
  llorch jobs cancel $job
done
```

### [RULE-002] CLI Can Do What Daemon Does

**The CLI MUST be able to perform daemon operations:**

**orchestratord daemon:**
- Schedules jobs → `llorch jobs submit` (calls HTTP API)
- Manages pools → `llorch pool status` (calls HTTP API or SSH)
- Stores state → `llorch jobs list` (queries HTTP API)

**llorch-ctl CLI:**
- Can submit jobs (via HTTP API)
- Can query state (via HTTP API)
- Can command pools (via SSH or HTTP)
- **BUT**: Does not run as daemon, does not persist state

**pool-managerd daemon:**
- Spawns workers → `llorch-pool worker spawn` (direct or HTTP)
- Manages GPU inventory → `llorch-pool gpu status` (NVML query)
- Caches models → `llorch-pool models cache` (RAM cache)

**pool-ctl CLI:**
- Can spawn workers (direct spawn or HTTP)
- Can query GPU (NVML query)
- Can download models (hf CLI)
- **BUT**: Does not run as daemon, does not maintain heartbeat

### [RULE-003] Shared Logic, Separate Execution

**Shared crates contain:**
- Data structures
- Algorithms
- Validation logic
- API types
- Configuration parsing

**Daemons add:**
- HTTP servers
- State persistence
- Background tasks
- Heartbeat protocols
- Metrics emission

**CLIs add:**
- Argument parsing (clap)
- SSH clients
- Progress indicators
- Colored output
- Interactive prompts

---

## Crate Dependency Graph

```
orchestratord (daemon)
    ├── depends on: orchestrator-core
    ├── adds: HTTP server, SQLite, state persistence
    └── produces: orchestratord binary

llorch-ctl (CLI)
    ├── depends on: orchestrator-core
    ├── adds: clap, SSH client, colored output
    └── produces: llorch binary

orchestrator-core (shared)
    ├── Job queue types
    ├── Scheduling algorithms
    ├── Pool registry types
    ├── API types
    └── Configuration types

pool-managerd (daemon)
    ├── depends on: pool-core
    ├── adds: HTTP server, NVML, heartbeat
    └── produces: pool-managerd binary

pool-ctl (CLI)
    ├── depends on: pool-core
    ├── adds: clap, hf CLI wrapper, git wrapper
    └── produces: llorch-pool binary

pool-core (shared)
    ├── Worker registry types
    ├── GPU inventory types
    ├── Model catalog types
    ├── Worker lifecycle logic
    └── Configuration types

llorch-candled (worker daemon)
    ├── depends on: candle, cuda/metal
    ├── HTTP server, inference
    └── produces: llorch-{cpu,cuda,metal}-candled binaries
```

---

## Implementation Order

### M0 (Current)

**Priority 1: pool-ctl**
```
bin/pool-ctl/
bin/shared-crates/pool-core/
```
- Model download (hf CLI)
- Git operations (submodules)
- Worker spawn (direct)
- Replaces: scripts/llorch-models, scripts/llorch-git

**Priority 2: llorch-ctl**
```
bin/llorch-ctl/
bin/shared-crates/orchestrator-core/
```
- Pool commands via SSH
- Calls pool-ctl on remote hosts
- Replaces: scripts/homelab/llorch-remote

**Priority 3: Delete bash scripts**
```
rm -rf scripts/llorch-models
rm -rf scripts/llorch-git
rm -rf scripts/homelab/llorch-remote
rm -rf .docs/testing/download_*.sh
```

### M1 (Future)

**Priority 4: pool-managerd daemon**
```
bin/pool-managerd/
```
- HTTP server :9200
- Worker lifecycle via HTTP
- Heartbeat to orchestrator
- Uses: pool-core (shared)

**Priority 5: pool-ctl HTTP mode**
- Add HTTP client to pool-ctl
- Can call pool-managerd API
- Fallback to direct execution

### M2 (Future)

**Priority 6: orchestratord daemon**
```
bin/orchestratord/
```
- HTTP server :8080
- Job scheduling
- SQLite state
- Uses: orchestrator-core (shared)

**Priority 7: llorch-ctl HTTP mode**
- Add HTTP client to llorch-ctl
- Call orchestratord API
- Job submission, listing, cancellation

---

## Summary

**6 Binaries:**
1. `orchestratord` - Daemon (HTTP server)
2. `llorch-ctl` - CLI (controls orchestratord + pools)
3. `pool-managerd` - Daemon (HTTP server)
4. `pool-ctl` - CLI (controls pool-managerd + local ops)
5. `llorch-candled` - Worker daemon (HTTP server)
6. `bdd-runner` - Test runner (already exists)

**2 Shared Crates:**
7. `orchestrator-core` - Shared between orchestratord + llorch-ctl
8. `pool-core` - Shared between pool-managerd + pool-ctl

**HARD RULES:**
- ✅ CLI can control daemons (start/stop/status)
- ✅ CLI can do what daemon does (via HTTP or direct)
- ✅ Shared crates contain common logic
- ❌ CLI NEVER starts REPL/conversation
- ❌ CLI NEVER embeds daemon logic
- ❌ Agentic API is HTTP-based, not CLI-based

**Implementation Order:**
1. M0: pool-ctl + pool-core (local operations)
2. M0: llorch-ctl + orchestrator-core (SSH to pools)
3. M1: pool-managerd (daemon)
4. M2: orchestratord (daemon)

---

**Version**: 1.0  
**Status**: Normative (MUST follow)  
**Last Updated**: 2025-10-09

---

**End of Architecture**
