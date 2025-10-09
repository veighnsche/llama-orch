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
bin/queen-rbee/
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

bin/llm-worker-rbee/
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
bin/rbee-keeper/
├── src/
│   ├── main.rs              # CLI entry point (command: llorch)
│   ├── commands/
│   │   ├── jobs.rs          # rbee jobs submit/list/cancel
│   │   ├── pools.rs         # rbee pool status/register
│   │   └── dev.rs           # rbee dev setup/doctor
│   └── client.rs            # HTTP client (calls queen-rbee API)
└── Cargo.toml

bin/rbee-hive/
├── src/
│   ├── main.rs              # CLI entry point (command: rbee-hive)
│   ├── commands/
│   │   ├── models.rs        # rbee-hive models download/list
│   │   ├── git.rs           # rbee-hive git pull/sync
│   │   ├── workers.rs       # rbee-hive worker spawn/stop
│   │   └── dev.rs           # rbee-hive dev setup
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
1. `rbee-keeper` (CLI binary) parses arguments
2. `rbee-keeper` makes HTTP call: `POST http://localhost:8080/v2/tasks`
3. `queen-rbee` (daemon) receives request
4. `queen-rbee` makes scheduling decision
5. `queen-rbee` makes HTTP call: `POST http://mac.home.arpa:9200/workers/spawn`
6. `pool-managerd` (on mac) Spawns worker
7. Worker calls back when ready
8. `queen-rbee` dispatches job to worker
9. `rbee-keeper` streams SSE response back to operator

### Scenario 2: Download Model on Pool (M0)

**Operator on blep (orchestrator host):**
```bash
llorch pool models download tinyllama --host mac
```

**What happens:**
1. `rbee-keeper` (CLI binary) parses arguments
2. `rbee-keeper` makes SSH call: `ssh mac.home.arpa "rbee-hive models download tinyllama"`
3. `rbee-hive` (on mac) parses arguments
4. `rbee-hive` executes: `hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF ...`
5. Model downloaded to `.test-models/tinyllama/`
6. Output streamed back via SSH

**Alternative (if pool-managerd daemon running):**
```bash
llorch pool models download tinyllama --host mac
```
1. `rbee-keeper` makes HTTP call: `POST http://mac.home.arpa:9200/models/download`
2. `pool-managerd` (daemon on mac) executes download
3. Response streamed back via HTTP

### Scenario 3: Local Development (M0)

**Developer on mac (pool manager host):**
```bash
rbee-hive models download tinyllama
rbee-hive git pull
rbee-hive worker spawn metal --model tinyllama --gpu 0
```

**What happens:**
1. `rbee-hive` (local CLI) executes commands
2. For models/git: Direct shell commands (hf, git)
3. For worker spawn: Either:
   - Direct spawn: `llm-worker-rbee --model ... &`
   - OR HTTP call: `POST http://localhost:9200/workers/spawn` (if daemon running)

---

## The Correct Mental Model

### Three Layers

**Layer 1: Daemons (Runtime)**
```
queen-rbee (HTTP server on blep:8080)
pool-managerd (HTTP server on mac:9200, workstation:9200, blep:9200)
llm-worker-rbee (HTTP server on various ports)
```
- Long-running processes
- Managed by systemd/launchd
- HTTP APIs for communication
- State persistence
- Production runtime

**Layer 2: CLI Clients (Operator Tooling)**
```
rbee-keeper (llorch command on blep)
rbee-hive (rbee-hive command on pools)
```
- Short-lived invocations
- HTTP clients (call daemon APIs)
- Also: development tooling (git, models, builds)
- No state
- Operator convenience

**Layer 3: Development Scripts (Being Replaced)**
```
scripts/llorch-remote → rbee-keeper
scripts/llorch-models → rbee-hive
scripts/llorch-git → rbee-hive
```
- Bash scripts (being replaced)
- Will be deleted after Rust CLI complete

---

## What We're Actually Building

### M0: Worker + Pool Manager CLI

**Priority 1: Pool Manager CLI**
```
bin/rbee-hive/
```
**Commands:**
- `rbee-hive models download <model>` - Download model locally
- `rbee-hive models list` - List local models
- `rbee-hive git pull` - Pull latest code
- `rbee-hive git sync` - Hard reset
- `rbee-hive worker spawn <backend>` - Spawn worker locally
- `rbee-hive dev setup` - Setup pool manager environment

**Why first**: Pools need local tooling for model provisioning and worker spawning.

**Priority 2: Orchestrator CLI**
```
bin/rbee-keeper/
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
bin/queen-rbee/
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

1. **Daemons** (queen-rbee, pool-managerd, llm-worker-rbee)
   - HTTP servers
   - Long-running
   - Production runtime
   - Implement specs (SYS-5.x, SYS-6.x)

2. **CLI Clients** (rbee-keeper, rbee-hive)
   - HTTP clients
   - Short-lived
   - Operator convenience
   - Development tooling

**They are separate binaries with separate purposes.**

**Current bash scripts map to:**
- `llorch-remote` → `rbee-keeper` (orchestrator commands pools)
- `llorch-models` → `rbee-hive` (pool provisions models)
- `llorch-git` → `rbee-hive` (pool manages git)

**Implementation order:**
1. M0: `rbee-hive` (local pool operations)
2. M0: `rbee-keeper` (orchestrator commands pools via SSH)
3. M1: `pool-managerd` (daemon with HTTP API)
4. M2: `queen-rbee` (daemon with HTTP API)

---

**Version**: 1.0  
**Status**: Clarification (not a decision, just explaining structure)  
**Last Updated**: 2025-10-09

---

**End of Clarification**
