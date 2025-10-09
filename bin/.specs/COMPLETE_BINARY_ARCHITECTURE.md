# Complete Binary Architecture — 4 Binaries + 2 Shared Crates

**Status**: Normative (Updated post-rebranding)  
**Version**: 2.0  
**Date**: 2025-10-09  
**Updated**: 2025-10-09T23:00 (TEAM-025)

---

## The Complete Picture

### 4 Binaries (THE FUNDAMENTAL ARCHITECTURE)

**Orchestrator (blep.home.arpa):**
1. **`queen-rbee`** - HTTP Daemon (:8080) - M1
   - Scheduling, admission, job queue
   - SQLite state store
   - Makes ALL intelligent decisions
   - Routes inference requests to workers
   - Uses: `orchestrator-core`

**Pool Manager (mac.home.arpa, workstation.home.arpa, blep.home.arpa):**
2. **`rbee-hive`** - HTTP Daemon (:8080 or :9200) - M1
   - Worker lifecycle management via HTTP API
   - Model downloads (hf CLI in background)
   - Git operations (submodules in background)
   - Worker spawn (process spawn)
   - Health monitoring (every 30s)
   - Idle timeout enforcement (5 minutes)
   - GPU inventory (NVML)
   - Uses: `pool-core`

**Worker (spawned by rbee-hive):**
3. **`llm-worker-rbee`** - HTTP Daemon (:8001-8999) - M0 ✅
   - Inference execution
   - CUDA/Metal/CPU backends
   - Keeps model in VRAM
   - SSE streaming
   - Controlled by rbee-hive

**CLI Tool:**
4. **`rbee-keeper`** - CLI - M0 ✅
   - Calls HTTP APIs of queen-rbee and rbee-hive
   - Job submission, listing, cancellation
   - Pool operations (models, workers)
   - Can use SSH tunneling for remote access
   - Uses: `orchestrator-core`
   - **NO REPL, NO CONVERSATION**

**Test Harness:**
5. **`bdd-runner`** - BDD test runner (already exists)

### 2 Shared Crates

6. **`orchestrator-core`** - Shared between queen-rbee + rbee-keeper
   - Job queue types
   - Scheduling algorithms
   - Pool registry types
   - API types
   - Configuration

7. **`pool-core`** - Shared between rbee-hive (daemon) + rbee-keeper (CLI)
   - Worker registry types
   - GPU inventory types
   - Model catalog types
   - Worker lifecycle logic
   - API types
   - Configuration

---

## Directory Structure

```
bin/
├── .specs/
│   ├── 00_llama-orch.md                    # System spec (existing)
│   ├── ARCHITECTURE_DECISION_CLI_VS_HTTP.md
│   ├── BINARY_STRUCTURE_CLARIFICATION.md
│   └── COMPLETE_BINARY_ARCHITECTURE.md     # This document
│
├── queen-rbee/                           # Daemon (M2+)
│   ├── .specs/
│   │   └── 00_queen-rbee.md
│   ├── src/
│   │   ├── main.rs                         # HTTP server
│   │   ├── api/                            # HTTP endpoints
│   │   ├── scheduler/                      # Scheduling logic
│   │   ├── queue/                          # Job queue
│   │   └── state/                          # SQLite store
│   └── Cargo.toml
│
├── rbee-keeper/                              # CLI (M0+)
│   ├── .specs/
│   │   └── 00_llorch-cli.md
│   ├── catalog.toml                        # Model catalog (shared)
│   ├── src/
│   │   ├── main.rs                         # CLI entry (command: llorch)
│   │   ├── commands/
│   │   │   ├── jobs.rs                     # rbee jobs ...
│   │   │   ├── pools.rs                    # rbee pool ...
│   │   │   └── dev.rs                      # rbee dev ...
│   │   ├── ssh.rs                          # SSH client
│   │   └── http_client.rs                  # HTTP client
│   └── Cargo.toml
│
├── pool-managerd/                           # Daemon (M1+)
│   ├── .specs/
│   │   └── 00_pool-managerd.md
│   ├── src/
│   │   ├── main.rs                         # HTTP server
│   │   ├── api/                            # HTTP endpoints
│   │   ├── worker_lifecycle/               # Worker spawning
│   │   ├── gpu_inventory/                  # NVML queries
│   │   └── heartbeat.rs                    # Heartbeat to orchestrator
│   └── Cargo.toml
│
├── rbee-hive/                                # CLI (M0+)
│   ├── .specs/
│   │   └── 00_rbee-hive.md
│   ├── catalog.toml                        # Model catalog (shared)
│   ├── src/
│   │   ├── main.rs                         # CLI entry (command: rbee-hive)
│   │   ├── commands/
│   │   │   ├── models.rs                   # rbee-hive models ...
│   │   │   ├── git.rs                      # rbee-hive git ...
│   │   │   ├── worker.rs                   # rbee-hive worker ...
│   │   │   └── dev.rs                      # rbee-hive dev ...
│   │   ├── hf_wrapper.rs                   # hf CLI wrapper
│   │   └── git_wrapper.rs                  # git CLI wrapper
│   └── Cargo.toml
│
├── llm-worker-rbee/                          # Worker daemon (M0+, exists)
│   └── ...
│
└── shared-crates/
    ├── orchestrator-core/                   # Shared orchestrator logic
    │   ├── .specs/
    │   │   └── 00_orchestrator-core.md
    │   ├── src/
    │   │   ├── lib.rs
    │   │   ├── job.rs                      # Job types
    │   │   ├── queue.rs                    # Queue trait
    │   │   ├── scheduler.rs                # Scheduling algorithms
    │   │   ├── pool.rs                     # Pool registry
    │   │   ├── api/                        # API types
    │   │   └── config.rs                   # Configuration
    │   └── Cargo.toml
    │
    ├── pool-core/                           # Shared pool manager logic
    │   ├── .specs/
    │   │   └── 00_pool-core.md
    │   ├── src/
    │   │   ├── lib.rs
    │   │   ├── worker.rs                   # Worker types
    │   │   ├── registry.rs                 # Worker registry
    │   │   ├── gpu.rs                      # GPU inventory
    │   │   ├── nvml.rs                     # NVML wrapper
    │   │   ├── catalog/                    # Model catalog
    │   │   ├── lifecycle/                  # Worker spawn logic
    │   │   ├── api/                        # API types
    │   │   └── config.rs                   # Configuration
    │   └── Cargo.toml
    │
    └── ... (other shared crates)
```

---

## Control Relationships

### rbee-keeper Controls

**1. queen-rbee daemon (M2+)**
```bash
llorch orchestrator start [--config PATH]
llorch orchestrator stop
llorch orchestrator status
llorch orchestrator logs
```
**How**: Spawns/stops process, queries HTTP API

**2. Pool managers (M0+)**
```bash
llorch pool status --host mac
llorch pool models download tinyllama --host mac
llorch pool worker spawn metal --host mac
llorch pool git pull --host workstation
```
**How**: SSH to pool host, executes `rbee-hive` command

**3. Jobs (M2+)**
```bash
llorch jobs submit --model llama3 --prompt "hello"
llorch jobs list
llorch jobs cancel job-123
llorch jobs status job-123
```
**How**: HTTP calls to queen-rbee API

### rbee-hive Controls

**1. pool-managerd daemon (M1+)**
```bash
rbee-hive daemon start [--config PATH]
rbee-hive daemon stop
rbee-hive daemon status
```
**How**: Spawns/stops process, queries HTTP API

**2. Local operations (M0)**
```bash
rbee-hive models download tinyllama
rbee-hive git pull
rbee-hive worker spawn metal --model tinyllama
```
**How**: Direct execution (hf CLI, git CLI, process spawn)

**3. Workers (M0+)**
```bash
rbee-hive worker spawn metal --model tinyllama --gpu 0
rbee-hive worker stop worker-metal-0
rbee-hive worker list
```
**How**: Direct spawn (M0) or HTTP call to pool-managerd (M1+)

---

## Shared Logic Examples

### orchestrator-core (Shared)

**Job Queue:**
```rust
// Used by BOTH queen-rbee and rbee-keeper
pub struct Job {
    pub id: String,
    pub model_ref: String,
    pub prompt: String,
    pub status: JobStatus,
}

pub trait JobQueue {
    fn enqueue(&mut self, job: Job) -> Result<()>;
    fn dequeue(&mut self) -> Result<Option<Job>>;
    fn peek(&self) -> Result<Option<&Job>>;
}
```

**Daemon uses it:**
```rust
// queen-rbee/src/main.rs
use orchestrator_core::{JobQueue, Job};

struct Orchestratord {
    queue: Box<dyn JobQueue>,
    state: SqliteStore,  // Daemon-specific
}
```

**CLI uses it:**
```rust
// rbee-keeper/src/commands/jobs.rs
use orchestrator_core::{Job, TaskSubmitRequest};

fn submit_job(model: &str, prompt: &str) -> Result<()> {
    let req = TaskSubmitRequest { model, prompt, /* ... */ };
    req.validate()?;  // Shared validation
    
    // CLI-specific: HTTP call
    let response = http_client.post("/v2/tasks").json(&req).send()?;
    println!("Job submitted: {}", response.job_id);
    Ok(())
}
```

### pool-core (Shared)

**Worker Spawn Logic:**
```rust
// Used by BOTH pool-managerd and rbee-hive
pub fn spawn_worker(req: &WorkerSpawnRequest) -> Result<WorkerInfo> {
    // Shared validation
    validate_backend(&req.backend)?;
    validate_model_exists(&req.model_ref)?;
    validate_gpu_available(req.gpu_id)?;
    
    // Shared spawn logic
    let cmd = build_worker_command(req)?;
    let child = Command::new("llm-worker-rbee")
        .args(&cmd.args)
        .spawn()?;
    
    Ok(WorkerInfo {
        id: req.worker_id.clone(),
        pid: child.id(),
        backend: req.backend.clone(),
        /* ... */
    })
}
```

**Daemon uses it:**
```rust
// pool-managerd/src/worker_lifecycle.rs
use pool_core::{spawn_worker, WorkerSpawnRequest};

async fn handle_spawn_request(&self, req: WorkerSpawnRequest) -> Result<()> {
    let worker_info = spawn_worker(&req)?;  // Shared logic
    
    self.registry.register(worker_info.clone())?;
    self.emit_metrics(&worker_info);  // Daemon-specific
    
    Ok(())
}
```

**CLI uses it:**
```rust
// rbee-hive/src/commands/worker.rs
use pool_core::{spawn_worker, WorkerSpawnRequest};

fn spawn_worker_cmd(backend: &str, model: &str, gpu: u32) -> Result<()> {
    let req = WorkerSpawnRequest { backend, model, gpu, /* ... */ };
    let worker_info = spawn_worker(&req)?;  // Shared logic
    
    // CLI-specific: colored output
    println!("✅ Worker spawned: {}", worker_info.id);
    println!("   PID: {}", worker_info.pid);
    
    Ok(())
}
```

---

## Evolution Path

### M0 (Current)

**What exists:**
- `llm-worker-rbee` (worker daemon)
- Bash scripts (being replaced)

**What we build:**
1. `pool-core` (shared crate)
2. `rbee-hive` (CLI)
   - Direct model downloads
   - Direct git operations
   - Direct worker spawning
3. `orchestrator-core` (shared crate)
4. `rbee-keeper` (CLI)
   - SSH to pools
   - Commands rbee-hive remotely
   - No daemon yet (CLI does scheduling)

**Architecture:**
```
rbee-keeper (on blep)
    ↓ SSH
rbee-hive (on mac/workstation)
    ↓ spawn
llm-worker-rbee (worker)
```

### M1 (Future)

**What we add:**
5. `pool-managerd` (daemon)
   - HTTP server
   - Worker lifecycle via HTTP
   - Heartbeat to orchestrator
   - Uses: pool-core

**What changes:**
- `rbee-hive` can call pool-managerd HTTP API
- `rbee-keeper` can call pool-managerd HTTP API (via SSH tunnel or direct)

**Architecture:**
```
rbee-keeper (on blep)
    ↓ SSH or HTTP
pool-managerd (daemon on mac/workstation)
    ↓ spawn
llm-worker-rbee (worker)
```

### M2 (Future)

**What we add:**
6. `queen-rbee` (daemon)
   - HTTP server
   - Job scheduling via HTTP
   - State persistence
   - Uses: orchestrator-core

**What changes:**
- `rbee-keeper` calls queen-rbee HTTP API
- `queen-rbee` calls pool-managerd HTTP API

**Architecture:**
```
rbee-keeper (CLI)
    ↓ HTTP
queen-rbee (daemon on blep)
    ↓ HTTP
pool-managerd (daemon on mac/workstation)
    ↓ spawn
llm-worker-rbee (worker)
```

---

## Command Examples

### M0: CLI-Only Mode

**On blep (orchestrator host):**
```bash
# Command pool to download model
llorch pool models download tinyllama --host mac
  → SSH: ssh mac "rbee-hive models download tinyllama"
  → Executes: hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF ...

# Command pool to spawn worker
llorch pool worker spawn metal --host mac --model tinyllama
  → SSH: ssh mac "rbee-hive worker spawn metal --model tinyllama"
  → Executes: llm-worker-rbee --backend metal --model .test-models/tinyllama/...
```

**On mac (pool host):**
```bash
# Local pool operations
rbee-hive models download tinyllama
  → Executes: hf download ...

rbee-hive worker spawn metal --model tinyllama
  → Executes: llm-worker-rbee --backend metal ...
```

### M1: Pool Daemon Mode

**On blep (orchestrator host):**
```bash
# Command pool daemon via HTTP
llorch pool worker spawn metal --host mac --model tinyllama
  → HTTP: POST http://mac.home.arpa:9200/workers/spawn
  → pool-managerd spawns worker
```

**On mac (pool host):**
```bash
# Start pool daemon
rbee-hive daemon start

# CLI can still work directly OR via daemon
rbee-hive worker spawn metal --model tinyllama
  → Option 1: Direct spawn (if no daemon)
  → Option 2: HTTP POST http://localhost:9200/workers/spawn (if daemon running)
```

### M2: Full Daemon Mode

**On blep (orchestrator host):**
```bash
# Start orchestrator daemon
llorch orchestrator start

# Submit job via daemon
llorch jobs submit --model llama3 --prompt "hello"
  → HTTP: POST http://localhost:8080/v2/tasks
  → queen-rbee schedules job
  → queen-rbee calls pool-managerd: POST http://mac:9200/workers/spawn
  → pool-managerd spawns worker
  → queen-rbee dispatches job to worker
  → rbee-keeper streams SSE response
```

---

## HARD RULES

### [RULE-001] NO REPL, NO CONVERSATION

**FORBIDDEN in rbee-keeper:**
```bash
# ❌ NEVER
llorch chat
llorch repl
llorch interactive

# ❌ NEVER
$ llorch
> submit job
> list jobs
```

**FORBIDDEN in rbee-hive:**
```bash
# ❌ NEVER
rbee-hive chat
rbee-hive repl
rbee-hive interactive
```

**WHY:**
- Agentic API is HTTP-based (POST /v2/tasks)
- Conversations are SSE streams over HTTP
- CLI is for CONTROL and AUTOMATION
- Terminal is not the UX for LLM interactions
- Web UI provides conversation interface

**CORRECT:**
```bash
# ✅ Single commands for automation
llorch jobs submit --model llama3 --prompt "hello"
llorch jobs list --format json
llorch jobs cancel job-123

# ✅ Scripting
for job in $(llorch jobs list --format json | jq -r '.[].id'); do
  rbee jobs cancel $job
done
```

### [RULE-002] CTL Can Control Daemon

**rbee-keeper MUST be able to:**
- Start/stop queen-rbee daemon
- Query queen-rbee status
- Submit jobs (via HTTP API)
- List jobs (via HTTP API)
- Cancel jobs (via HTTP API)
- Command pools (via SSH or HTTP)

**rbee-hive MUST be able to:**
- Start/stop pool-managerd daemon
- Query pool-managerd status
- Spawn workers (direct or via HTTP)
- Download models (direct hf CLI)
- Manage git (direct git CLI)

### [RULE-003] Shared Logic, Separate Execution

**orchestrator-core contains:**
- Job queue types and algorithms
- Scheduling algorithms (Rhai integration)
- Pool registry types
- API types (requests, responses, errors)
- Configuration types

**queen-rbee adds:**
- HTTP server (axum/actix)
- SQLite persistence
- Background tasks
- Metrics emission

**rbee-keeper adds:**
- Clap CLI parsing
- SSH client
- HTTP client
- Colored output
- Progress indicators

**pool-core contains:**
- Worker registry types
- GPU inventory types (NVML wrapper)
- Model catalog types
- Worker spawn logic
- API types

**pool-managerd adds:**
- HTTP server
- Heartbeat protocol
- Background GPU monitoring
- Metrics emission

**rbee-hive adds:**
- Clap CLI parsing
- hf CLI wrapper
- git CLI wrapper
- Colored output
- Progress indicators

---

## Dependency Graph

```
queen-rbee
    ├── orchestrator-core
    ├── axum (HTTP server)
    ├── sqlx (SQLite)
    └── prometheus (metrics)

rbee-keeper
    ├── orchestrator-core
    ├── clap (CLI)
    ├── reqwest (HTTP client)
    ├── ssh2 (SSH client)
    └── colored (output)

orchestrator-core
    ├── serde (serialization)
    ├── rhai (scheduler)
    └── anyhow (errors)

pool-managerd
    ├── pool-core
    ├── axum (HTTP server)
    ├── nvml-wrapper (GPU)
    └── prometheus (metrics)

rbee-hive
    ├── pool-core
    ├── clap (CLI)
    ├── reqwest (HTTP client, optional)
    └── colored (output)

pool-core
    ├── serde (serialization)
    ├── nvml-wrapper (GPU)
    └── anyhow (errors)

llm-worker-rbee
    ├── candle (inference)
    ├── axum (HTTP server)
    └── cuda/metal (backends)
```

---

## Implementation Priority

### Phase 1: Shared Crates + CLIs (M0)

**Week 1-2:**
1. Create `pool-core` crate
   - Worker types
   - GPU inventory types
   - Model catalog types
   - Worker spawn logic

2. Create `rbee-hive` CLI
   - Model downloads (hf CLI)
   - Git operations (submodules)
   - Worker spawn (direct)
   - Uses: pool-core

**Week 3-4:**
3. Create `orchestrator-core` crate
   - Job queue types
   - Scheduling algorithms
   - Pool registry types

4. Create `rbee-keeper` CLI
   - Pool commands via SSH
   - Calls rbee-hive remotely
   - Uses: orchestrator-core

**Deliverable:** Working CLIs that replace bash scripts

### Phase 2: Pool Daemon (M1)

**Week 5-6:**
5. Create `pool-managerd` daemon
   - HTTP server :9200
   - Worker lifecycle via HTTP
   - Heartbeat protocol
   - Uses: pool-core

6. Update `rbee-hive` to support HTTP mode
   - Can call pool-managerd API
   - Fallback to direct execution

**Deliverable:** Pool manager daemon with HTTP API

### Phase 3: Orchestrator Daemon (M2)

**Week 7-8:**
7. Create `queen-rbee` daemon
   - HTTP server :8080
   - Job scheduling
   - SQLite state store
   - Uses: orchestrator-core

8. Update `rbee-keeper` to support HTTP mode
   - Call queen-rbee API for jobs
   - Call pool-managerd API via orchestrator

**Deliverable:** Full orchestrator daemon with HTTP API

---

## Summary

**6 Binaries:**
1. `queen-rbee` - Daemon (M2+)
2. `rbee-keeper` - CLI (M0+)
3. `pool-managerd` - Daemon (M1+)
4. `rbee-hive` - CLI (M0+)
5. `llm-worker-rbee` - Worker daemon (M0+, exists)
6. `bdd-runner` - Test runner (exists)

**2 Shared Crates:**
7. `orchestrator-core` - Shared orchestrator logic
8. `pool-core` - Shared pool manager logic

**HARD RULES:**
- ✅ CTL controls daemons (start/stop/status)
- ✅ CTL can do what daemon does (via HTTP or direct)
- ✅ Shared crates contain common logic
- ❌ CTL NEVER starts REPL or conversation
- ❌ Agentic API is HTTP-based, not CLI-based
- ❌ Terminal is not the UX for LLM interactions

**Current bash scripts map to:**
- `llorch-remote` → `rbee-keeper` (orchestrator CLI)
- `llorch-models` → `rbee-hive` (pool CLI)
- `llorch-git` → `rbee-hive` (pool CLI)

**Implementation order:**
1. M0: pool-core + rbee-hive (local operations)
2. M0: orchestrator-core + rbee-keeper (SSH to pools)
3. M1: pool-managerd (daemon)
4. M2: queen-rbee (daemon)

---

**Version**: 1.0  
**Status**: Normative (MUST follow)  
**Last Updated**: 2025-10-09

---

**End of Complete Architecture**
