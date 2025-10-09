# Binary Structure Clarification

**Date**: 2025-10-09  
**Status**: Clarification Document

---

## The Confusion

We had bash scripts that were already implementing orchestrator patterns:
- `llorch-remote` (on blep) → orchestrator telling pools what to do
- `llorch-models` (on pools) → pool manager provisioning models
- `llorch-git` (on pools) → pool manager git operations

**Question**: Are we building a CLI or are we building the orchestrator?

**Answer**: We're building BOTH, but they are SEPARATE binaries.

---

## The Correct Architecture

### Runtime Binaries (Daemons)

**These are HTTP servers that run as daemons:**

```
bin/rbees-orcd/
├── src/
│   ├── main.rs              # HTTP server
│   ├── api/                 # HTTP endpoints (SYS-5.1.x, SYS-5.2.x)
│   ├── scheduler/           # Scheduling logic (SYS-6.1.4)
│   ├── queue/               # Job queue (SYS-6.1.2)
│   └── state/               # SQLite state (SYS-6.1.3)
└── Cargo.toml

bin/pool-managerd/
├── src/
│   ├── main.rs              # HTTP server
│   ├── api/                 # HTTP endpoints (SYS-5.3.x)
│   ├── worker_lifecycle/    # Worker spawning (SYS-6.2.1)
│   ├── gpu_inventory/       # NVML queries (SYS-6.2.2)
│   └── model_cache/         # Model caching (optional)
└── Cargo.toml

bin/rbees-workerd/
├── src/
│   ├── main.rs              # HTTP server
│   ├── api/                 # HTTP endpoints (SYS-5.4.x)
│   ├── backend/             # CUDA/Metal/CPU
│   └── inference/           # Inference execution
└── Cargo.toml
```

**Lifecycle**: Managed by systemd/launchd  
**Communication**: HTTP APIs only  
**State**: Persistent (SQLite for orchestrator)  
**Purpose**: Production runtime

### CLI Binaries (Operator Tooling)

**These are HTTP clients for operator convenience:**

```
bin/rbees-ctl/
├── src/
│   ├── main.rs              # CLI entry point (command: llorch)
│   ├── commands/
│   │   ├── jobs.rs          # rbees jobs submit/list/cancel
│   │   ├── pools.rs         # rbees pool status/register
│   │   └── dev.rs           # rbees dev setup/doctor
│   └── client.rs            # HTTP client (calls rbees-orcd API)
└── Cargo.toml

bin/rbees-pool/
├── src/
│   ├── main.rs              # CLI entry point (command: rbees-pool)
│   ├── commands/
│   │   ├── models.rs        # rbees-pool models download/list
│   │   ├── git.rs           # rbees-pool git pull/sync
│   │   ├── workers.rs       # rbees-pool worker spawn/stop
│   │   └── dev.rs           # rbees-pool dev setup
│   └── client.rs            # HTTP client (calls pool-managerd API)
└── Cargo.toml
```

**Lifecycle**: Short-lived invocations  
**Communication**: HTTP client → daemon HTTP API  
**State**: None (stateless)  
**Purpose**: Operator convenience + development tooling

---

## How They Work Together

### Scenario 1: Submit Job (M2+)

**Operator on blep (orchestrator host):**
```bash
llorch jobs submit --model llama3 --prompt "hello"
```

**What happens:**
1. `rbees-ctl` (CLI binary) parses arguments
2. `rbees-ctl` makes HTTP call: `POST http://localhost:8080/v2/tasks`
3. `rbees-orcd` (daemon) receives request
4. `rbees-orcd` makes scheduling decision
5. `rbees-orcd` makes HTTP call: `POST http://mac.home.arpa:9200/workers/spawn`
6. `pool-managerd` (on mac) Spawns worker
7. Worker calls back when ready
8. `rbees-orcd` dispatches job to worker
9. `rbees-ctl` streams SSE response back to operator

### Scenario 2: Download Model on Pool (M0)

**Operator on blep (orchestrator host):**
```bash
llorch pool models download tinyllama --host mac
```

**What happens:**
1. `rbees-ctl` (CLI binary) parses arguments
2. `rbees-ctl` makes SSH call: `ssh mac.home.arpa "rbees-pool models download tinyllama"`
3. `rbees-pool` (on mac) parses arguments
4. `rbees-pool` executes: `hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF ...`
5. Model downloaded to `.test-models/tinyllama/`
6. Output streamed back via SSH

**Alternative (if pool-managerd daemon running):**
```bash
llorch pool models download tinyllama --host mac
```
1. `rbees-ctl` makes HTTP call: `POST http://mac.home.arpa:9200/models/download`
2. `pool-managerd` (daemon on mac) executes download
3. Response streamed back via HTTP

### Scenario 3: Local Development (M0)

**Developer on mac (pool manager host):**
```bash
rbees-pool models download tinyllama
rbees-pool git pull
rbees-pool worker spawn metal --model tinyllama --gpu 0
```

**What happens:**
1. `rbees-pool` (local CLI) executes commands
2. For models/git: Direct shell commands (hf, git)
3. For worker spawn: Either:
   - Direct spawn: `rbees-workerd --model ... &`
   - OR HTTP call: `POST http://localhost:9200/workers/spawn` (if daemon running)

---

## The Correct Mental Model

### Three Layers

**Layer 1: Daemons (Runtime)**
```
rbees-orcd (HTTP server on blep:8080)
pool-managerd (HTTP server on mac:9200, workstation:9200, blep:9200)
rbees-workerd (HTTP server on various ports)
```
- Long-running processes
- Managed by systemd/launchd
- HTTP APIs for communication
- State persistence
- Production runtime

**Layer 2: CLI Clients (Operator Tooling)**
```
rbees-ctl (llorch command on blep)
rbees-pool (rbees-pool command on pools)
```
- Short-lived invocations
- HTTP clients (call daemon APIs)
- Also: development tooling (git, models, builds)
- No state
- Operator convenience

**Layer 3: Development Scripts (Being Replaced)**
```
scripts/llorch-remote → rbees-ctl
scripts/llorch-models → rbees-pool
scripts/llorch-git → rbees-pool
```
- Bash scripts (being replaced)
- Will be deleted after Rust CLI complete

---

## What We're Actually Building

### M0: Worker + Pool Manager CLI

**Priority 1: Pool Manager CLI**
```
bin/rbees-pool/
```
**Commands:**
- `rbees-pool models download <model>` - Download model locally
- `rbees-pool models list` - List local models
- `rbees-pool git pull` - Pull latest code
- `rbees-pool git sync` - Hard reset
- `rbees-pool worker spawn <backend>` - Spawn worker locally
- `rbees-pool dev setup` - Setup pool manager environment

**Why first**: Pools need local tooling for model provisioning and worker spawning.

**Priority 2: Orchestrator CLI**
```
bin/rbees-ctl/
```
**Commands:**
- `llorch pool models download <model> --host <host>` - Tell pool to download model
- `llorch pool status --host <host>` - Get pool status
- `llorch pool worker spawn --backend <backend> --host <host>` - Tell pool to spawn worker
- `llorch dev setup` - Setup orchestrator environment

**Why second**: Orchestrator needs to command pools remotely.

### M2: Full Orchestrator Daemon

**Priority 3: Orchestrator Daemon**
```
bin/rbees-orcd/
```
**HTTP API:**
- `POST /v2/tasks` - Submit job (SYS-5.1.x)
- `GET /v2/tasks/{id}` - Get job status
- `POST /pools/{id}/workers/spawn` - Command pool (SYS-5.2.x)

**Why later**: Need working pools first before orchestrating them.

**Priority 4: Pool Manager Daemon**
```
bin/pool-managerd/
```
**HTTP API:**
- `POST /workers/spawn` - Spawn worker (SYS-5.3.x)
- `GET /status` - Report state
- `POST /models/download` - Download model (optional)

**Why later**: CLI can spawn workers directly for M0, daemon for M1+.

---

## Summary

**We're building BOTH:**

1. **Daemons** (rbees-orcd, pool-managerd, rbees-workerd)
   - HTTP servers
   - Long-running
   - Production runtime
   - Implement specs (SYS-5.x, SYS-6.x)

2. **CLI Clients** (rbees-ctl, rbees-pool)
   - HTTP clients
   - Short-lived
   - Operator convenience
   - Development tooling

**They are separate binaries with separate purposes.**

**Current bash scripts map to:**
- `llorch-remote` → `rbees-ctl` (orchestrator commands pools)
- `llorch-models` → `rbees-pool` (pool provisions models)
- `llorch-git` → `rbees-pool` (pool manages git)

**Implementation order:**
1. M0: `rbees-pool` (local pool operations)
2. M0: `rbees-ctl` (orchestrator commands pools via SSH)
3. M1: `pool-managerd` (daemon with HTTP API)
4. M2: `rbees-orcd` (daemon with HTTP API)

---

**Version**: 1.0  
**Status**: Clarification (not a decision, just explaining structure)  
**Last Updated**: 2025-10-09

---

**End of Clarification**
