# Final Architecture - The 4 Binaries

**Date:** 2025-10-09T17:40:00+02:00  
**Status:** NORMATIVE  
**This is THE definitive architecture document**

---

## The 4 Binaries

llama-orch consists of **exactly 4 binaries**:

### 1. orchestratord (Daemon - HTTP)
**Crate:** `bin/orchestratord/`  
**Binary:** `orchestratord`  
**Status:** M1 ❌ NOT BUILT  
**Port:** 8080

**Purpose:** THE BRAIN
- Makes ALL intelligent decisions
- Rhai scripting (user-defined orchestration)
- Worker registry (SQLite, global)
- Admission control
- Queue management
- Scheduling
- SSE relay

**Why daemon:** Long-running, stateful, needs to accept client requests 24/7

---

### 2. llorch-candled (Daemon - HTTP)
**Crate:** `bin/llorch-candled/`  
**Binary:** `llorch-candled` (and variants: llorch-cpu-candled, llorch-cuda-candled, llorch-metal-candled)  
**Status:** M0 ✅ DONE  
**Ports:** 8001, 8002, 8003, etc.

**Purpose:** WORKER (dumb executor)
- Loads ONE model into VRAM/RAM
- Executes inference requests
- Streams tokens via SSE
- Stateless (can be killed anytime)
- Needs cancellation support (M0 task)

**Why daemon:** Long-running, keeps model in memory, accepts HTTP requests

---

### 3. llorch (CLI - SSH)
**Crate:** `bin/llorch-ctl/`  
**Binary:** `llorch`  
**Status:** M0 ✅ DONE

**Purpose:** REMOTE CONTROL (operator tool)
- SSH to pools
- Execute precise commands
- No intelligence, no scheduling
- Stateless (runs on-demand, exits)

**Commands:**
```bash
llorch pool models download <model> --host <pool>
llorch pool worker spawn <backend> --host <pool> --model <model>
llorch infer --worker <host:port> --prompt <text>
```

**Why CLI:** Control operations are on-demand, no daemon needed

---

### 4. llorch-pool (CLI - Local)
**Crate:** `bin/pool-ctl/`  
**Binary:** `llorch-pool`  
**Status:** M0 ✅ DONE

**Purpose:** LOCAL POOL MANAGEMENT
- Model catalog (tracks models)
- Backend catalog (CPU, Metal, CUDA)
- Worker spawning (local)
- Worker metadata (local, for cleanup)
- Orphan cleanup
- Stateless (runs on-demand, exits)

**Commands:**
```bash
llorch-pool models download <model>
llorch-pool worker spawn <backend> --model <model>
llorch-pool worker cleanup
```

**Why CLI:** Pool operations are on-demand, no daemon needed

**Note:** This REPLACES pool-managerd daemon!

---

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│ CLIENT (SDK)                             │
└────────────┬────────────────────────────┘
             │ HTTP POST /v2/tasks
             ↓
┌─────────────────────────────────────────┐
│ 1. orchestratord (DAEMON)                │
│    THE BRAIN                             │
│    - Rhai scripting                      │
│    - Worker registry (SQLite)            │
│    - Scheduling, routing                 │
│    Status: M1 ❌ NOT BUILT               │
└────────────┬────────────────────────────┘
             │ HTTP POST /execute
             ↓
┌─────────────────────────────────────────┐
│ 2. llorch-candled (DAEMON)               │
│    WORKER                                │
│    - Loads ONE model                     │
│    - Generates tokens                    │
│    - Stateless                           │
│    Status: M0 ✅ DONE                    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ OPERATOR (Human)                         │
└────────────┬────────────────────────────┘
             │ runs
             ↓
┌─────────────────────────────────────────┐
│ 3. llorch (CLI)                          │
│    REMOTE CONTROL                        │
│    - SSH to pools                        │
│    - Precise commands                    │
│    Status: M0 ✅ DONE                    │
└────────────┬────────────────────────────┘
             │ SSH
             ↓
┌─────────────────────────────────────────┐
│ 4. llorch-pool (CLI)                     │
│    LOCAL POOL                            │
│    - Model catalog                       │
│    - Worker spawning                     │
│    - Backend detection                   │
│    Status: M0 ✅ DONE                    │
└─────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | Type | Registry? | Catalog? | Stateful? | Scripting? |
|-----------|------|-----------|----------|-----------|------------|
| orchestratord | Daemon | ✅ Global (SQLite) | ❌ | ✅ YES | ✅ Rhai |
| llorch-candled | Daemon | ❌ | ❌ | ❌ NO | ❌ |
| llorch | CLI | ❌ | ❌ | ❌ NO | ❌ |
| llorch-pool | CLI | ✅ Local (files) | ✅ Model + Backend | ❌ NO | ❌ |

---

## Key Clarifications

### 1. Only 2 Daemons
- orchestratord (THE BRAIN)
- llorch-candled (WORKER)

**NOT 3!** pool-managerd is NOT a daemon!

### 2. Only 2 CLIs
- llorch (remote control via SSH)
- llorch-pool (local pool management)

### 3. orchestratord is THE BRAIN
- Rhai scripting for user-defined logic
- Makes ALL intelligent decisions
- Stateful (SQLite)

### 4. llorch-ctl has PRECISE COMMANDS
- No intelligence
- No scheduling
- No scripting
- Stateless

### 5. Worker Registry Ownership
- **orchestratord:** Global worker registry (SQLite)
- **llorch-pool:** Local worker metadata (files, for cleanup only)

### 6. Prompt Constructor
- Shared crate: `bin/shared-crates/prompt-constructor/`
- Used by: orchestratord + llorch-pool
- Formats chat templates (Qwen, Llama, Mistral, Phi)

---

## M0 Status (3 of 4 binaries done)

### ✅ DONE:
1. llorch-candled (worker daemon) ✅
2. llorch (remote CLI) ✅
3. llorch-pool (local CLI) ✅

### ❌ NOT DONE:
1. orchestratord (brain daemon) ❌

### M0 Remaining Tasks:
- Backend catalog detection (llorch-pool)
- Worker cancellation (llorch-candled)
- Orphan cleanup (llorch-pool)

---

## M1 Goal

Build orchestratord (4th binary):
- HTTP server (port 8080)
- Worker registry (SQLite)
- Rhai scripting engine
- Admission control
- Queue management
- Scheduling
- SSE relay

**Estimated time:** 2-3 weeks

---

## Why This Architecture Works

### 2 Daemons (Data Plane)
- orchestratord: Routes requests, maintains state
- llorch-candled: Executes inference, keeps model in memory

### 2 CLIs (Control Plane)
- llorch: Remote control via SSH
- llorch-pool: Local pool operations

### Clear Separation
- Daemons: Long-running, HTTP servers
- CLIs: On-demand, exit after command
- No confusion about what's a daemon vs CLI

### No pool-managerd Needed
- Pool management is control operations (on-demand)
- Not data plane operations (24/7)
- CLI is sufficient

---

## Summary

**4 binaries total:**
- 2 daemons: orchestratord + llorch-candled
- 2 CLIs: llorch + llorch-pool

**M0 complete:** 3 of 4 (missing orchestratord)

**Next:** CP4 (test models), then M1 (build orchestratord)

---

**This is the definitive architecture. All other documents updated to match.**

**Signed:** TEAM-024  
**Date:** 2025-10-09T17:40:00+02:00
