# Complete Binary Architecture — 6 Binaries + 2 Shared Crates

**Status**: Normative  
**Version**: 1.0  
**Date**: 2025-10-09

---

## The Complete Picture

### 6 Binaries

**Orchestrator (blep.home.arpa):**
1. **`rbees-orcd`** - Daemon (HTTP server :8080)
   - Scheduling, admission, job queue
   - SQLite state store
   - Makes ALL intelligent decisions
   - Uses: `orchestrator-core`

2. **`rbees-ctl`** (command: `rbees`) - CLI
   - Controls rbees-orcd daemon (M2+)
   - Commands pools via SSH (M0) or HTTP (M2+)
   - Job submission, listing, cancellation
   - Uses: `orchestrator-core`
   - **NO REPL, NO CONVERSATION**

**Pool Manager (mac.home.arpa, workstation.home.arpa, blep.home.arpa):**
3. **`pool-managerd`** - Daemon (HTTP server :9200)
   - Worker lifecycle management
   - GPU inventory (NVML)
   - Heartbeat to orchestrator
   - Uses: `pool-core`

4. **`rbees-pool`** (command: `rbees-pool`) - CLI
   - Controls pool-managerd daemon (M1+)
   - Model downloads (hf CLI)
   - Git operations (submodules)
   - Worker spawn (direct or via daemon)
   - Uses: `pool-core`
   - **NO REPL, NO CONVERSATION**

**Worker (spawned by pool-managerd):**
5. **`rbees-workerd`** - Daemon (HTTP server :8001-8999)
   - Inference execution
   - CUDA/Metal/CPU backends
   - No CLI (controlled by pool-managerd)

**Test Harness:**
6. **`bdd-runner`** - BDD test runner (already exists)

### 2 Shared Crates

7. **`orchestrator-core`** - Shared between rbees-orcd + rbees-ctl
   - Job queue types
   - Scheduling algorithms
   - Pool registry types
   - API types
   - Configuration

8. **`pool-core`** - Shared between pool-managerd + rbees-pool
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
├── rbees-orcd/                           # Daemon (M2+)
│   ├── .specs/
│   │   └── 00_rbees-orcd.md
│   ├── src/
│   │   ├── main.rs                         # HTTP server
│   │   ├── api/                            # HTTP endpoints
│   │   ├── scheduler/                      # Scheduling logic
│   │   ├── queue/                          # Job queue
│   │   └── state/                          # SQLite store
│   └── Cargo.toml
│
├── rbees-ctl/                              # CLI (M0+)
│   ├── .specs/
│   │   └── 00_llorch-cli.md
│   ├── catalog.toml                        # Model catalog (shared)
│   ├── src/
│   │   ├── main.rs                         # CLI entry (command: llorch)
│   │   ├── commands/
│   │   │   ├── jobs.rs                     # rbees jobs ...
│   │   │   ├── pools.rs                    # rbees pool ...
│   │   │   └── dev.rs                      # rbees dev ...
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
├── rbees-pool/                                # CLI (M0+)
│   ├── .specs/
│   │   └── 00_rbees-pool.md
│   ├── catalog.toml                        # Model catalog (shared)
│   ├── src/
│   │   ├── main.rs                         # CLI entry (command: rbees-pool)
│   │   ├── commands/
│   │   │   ├── models.rs                   # rbees-pool models ...
│   │   │   ├── git.rs                      # rbees-pool git ...
│   │   │   ├── worker.rs                   # rbees-pool worker ...
│   │   │   └── dev.rs                      # rbees-pool dev ...
│   │   ├── hf_wrapper.rs                   # hf CLI wrapper
│   │   └── git_wrapper.rs                  # git CLI wrapper
│   └── Cargo.toml
│
├── rbees-workerd/                          # Worker daemon (M0+, exists)
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

### rbees-ctl Controls

**1. rbees-orcd daemon (M2+)**
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
**How**: SSH to pool host, executes `rbees-pool` command

**3. Jobs (M2+)**
```bash
llorch jobs submit --model llama3 --prompt "hello"
llorch jobs list
llorch jobs cancel job-123
llorch jobs status job-123
```
**How**: HTTP calls to rbees-orcd API

### rbees-pool Controls

**1. pool-managerd daemon (M1+)**
```bash
rbees-pool daemon start [--config PATH]
rbees-pool daemon stop
rbees-pool daemon status
```
**How**: Spawns/stops process, queries HTTP API

**2. Local operations (M0)**
```bash
rbees-pool models download tinyllama
rbees-pool git pull
rbees-pool worker spawn metal --model tinyllama
```
**How**: Direct execution (hf CLI, git CLI, process spawn)

**3. Workers (M0+)**
```bash
rbees-pool worker spawn metal --model tinyllama --gpu 0
rbees-pool worker stop worker-metal-0
rbees-pool worker list
```
**How**: Direct spawn (M0) or HTTP call to pool-managerd (M1+)

---

## Shared Logic Examples

### orchestrator-core (Shared)

**Job Queue:**
```rust
// Used by BOTH rbees-orcd and rbees-ctl
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
// rbees-orcd/src/main.rs
use orchestrator_core::{JobQueue, Job};

struct Orchestratord {
    queue: Box<dyn JobQueue>,
    state: SqliteStore,  // Daemon-specific
}
```

**CLI uses it:**
```rust
// rbees-ctl/src/commands/jobs.rs
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
// Used by BOTH pool-managerd and rbees-pool
pub fn spawn_worker(req: &WorkerSpawnRequest) -> Result<WorkerInfo> {
    // Shared validation
    validate_backend(&req.backend)?;
    validate_model_exists(&req.model_ref)?;
    validate_gpu_available(req.gpu_id)?;
    
    // Shared spawn logic
    let cmd = build_worker_command(req)?;
    let child = Command::new("rbees-workerd")
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
// rbees-pool/src/commands/worker.rs
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
- `rbees-workerd` (worker daemon)
- Bash scripts (being replaced)

**What we build:**
1. `pool-core` (shared crate)
2. `rbees-pool` (CLI)
   - Direct model downloads
   - Direct git operations
   - Direct worker spawning
3. `orchestrator-core` (shared crate)
4. `rbees-ctl` (CLI)
   - SSH to pools
   - Commands rbees-pool remotely
   - No daemon yet (CLI does scheduling)

**Architecture:**
```
rbees-ctl (on blep)
    ↓ SSH
rbees-pool (on mac/workstation)
    ↓ spawn
rbees-workerd (worker)
```

### M1 (Future)

**What we add:**
5. `pool-managerd` (daemon)
   - HTTP server
   - Worker lifecycle via HTTP
   - Heartbeat to orchestrator
   - Uses: pool-core

**What changes:**
- `rbees-pool` can call pool-managerd HTTP API
- `rbees-ctl` can call pool-managerd HTTP API (via SSH tunnel or direct)

**Architecture:**
```
rbees-ctl (on blep)
    ↓ SSH or HTTP
pool-managerd (daemon on mac/workstation)
    ↓ spawn
rbees-workerd (worker)
```

### M2 (Future)

**What we add:**
6. `rbees-orcd` (daemon)
   - HTTP server
   - Job scheduling via HTTP
   - State persistence
   - Uses: orchestrator-core

**What changes:**
- `rbees-ctl` calls rbees-orcd HTTP API
- `rbees-orcd` calls pool-managerd HTTP API

**Architecture:**
```
rbees-ctl (CLI)
    ↓ HTTP
rbees-orcd (daemon on blep)
    ↓ HTTP
pool-managerd (daemon on mac/workstation)
    ↓ spawn
rbees-workerd (worker)
```

---

## Command Examples

### M0: CLI-Only Mode

**On blep (orchestrator host):**
```bash
# Command pool to download model
llorch pool models download tinyllama --host mac
  → SSH: ssh mac "rbees-pool models download tinyllama"
  → Executes: hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF ...

# Command pool to spawn worker
llorch pool worker spawn metal --host mac --model tinyllama
  → SSH: ssh mac "rbees-pool worker spawn metal --model tinyllama"
  → Executes: rbees-workerd --backend metal --model .test-models/tinyllama/...
```

**On mac (pool host):**
```bash
# Local pool operations
rbees-pool models download tinyllama
  → Executes: hf download ...

rbees-pool worker spawn metal --model tinyllama
  → Executes: rbees-workerd --backend metal ...
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
rbees-pool daemon start

# CLI can still work directly OR via daemon
rbees-pool worker spawn metal --model tinyllama
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
  → rbees-orcd schedules job
  → rbees-orcd calls pool-managerd: POST http://mac:9200/workers/spawn
  → pool-managerd spawns worker
  → rbees-orcd dispatches job to worker
  → rbees-ctl streams SSE response
```

---

## HARD RULES

### [RULE-001] NO REPL, NO CONVERSATION

**FORBIDDEN in rbees-ctl:**
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

**FORBIDDEN in rbees-pool:**
```bash
# ❌ NEVER
rbees-pool chat
rbees-pool repl
rbees-pool interactive
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
  rbees jobs cancel $job
done
```

### [RULE-002] CTL Can Control Daemon

**rbees-ctl MUST be able to:**
- Start/stop rbees-orcd daemon
- Query rbees-orcd status
- Submit jobs (via HTTP API)
- List jobs (via HTTP API)
- Cancel jobs (via HTTP API)
- Command pools (via SSH or HTTP)

**rbees-pool MUST be able to:**
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

**rbees-orcd adds:**
- HTTP server (axum/actix)
- SQLite persistence
- Background tasks
- Metrics emission

**rbees-ctl adds:**
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

**rbees-pool adds:**
- Clap CLI parsing
- hf CLI wrapper
- git CLI wrapper
- Colored output
- Progress indicators

---

## Dependency Graph

```
rbees-orcd
    ├── orchestrator-core
    ├── axum (HTTP server)
    ├── sqlx (SQLite)
    └── prometheus (metrics)

rbees-ctl
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

rbees-pool
    ├── pool-core
    ├── clap (CLI)
    ├── reqwest (HTTP client, optional)
    └── colored (output)

pool-core
    ├── serde (serialization)
    ├── nvml-wrapper (GPU)
    └── anyhow (errors)

rbees-workerd
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

2. Create `rbees-pool` CLI
   - Model downloads (hf CLI)
   - Git operations (submodules)
   - Worker spawn (direct)
   - Uses: pool-core

**Week 3-4:**
3. Create `orchestrator-core` crate
   - Job queue types
   - Scheduling algorithms
   - Pool registry types

4. Create `rbees-ctl` CLI
   - Pool commands via SSH
   - Calls rbees-pool remotely
   - Uses: orchestrator-core

**Deliverable:** Working CLIs that replace bash scripts

### Phase 2: Pool Daemon (M1)

**Week 5-6:**
5. Create `pool-managerd` daemon
   - HTTP server :9200
   - Worker lifecycle via HTTP
   - Heartbeat protocol
   - Uses: pool-core

6. Update `rbees-pool` to support HTTP mode
   - Can call pool-managerd API
   - Fallback to direct execution

**Deliverable:** Pool manager daemon with HTTP API

### Phase 3: Orchestrator Daemon (M2)

**Week 7-8:**
7. Create `rbees-orcd` daemon
   - HTTP server :8080
   - Job scheduling
   - SQLite state store
   - Uses: orchestrator-core

8. Update `rbees-ctl` to support HTTP mode
   - Call rbees-orcd API for jobs
   - Call pool-managerd API via orchestrator

**Deliverable:** Full orchestrator daemon with HTTP API

---

## Summary

**6 Binaries:**
1. `rbees-orcd` - Daemon (M2+)
2. `rbees-ctl` - CLI (M0+)
3. `pool-managerd` - Daemon (M1+)
4. `rbees-pool` - CLI (M0+)
5. `rbees-workerd` - Worker daemon (M0+, exists)
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
- `llorch-remote` → `rbees-ctl` (orchestrator CLI)
- `llorch-models` → `rbees-pool` (pool CLI)
- `llorch-git` → `rbees-pool` (pool CLI)

**Implementation order:**
1. M0: pool-core + rbees-pool (local operations)
2. M0: orchestrator-core + rbees-ctl (SSH to pools)
3. M1: pool-managerd (daemon)
4. M2: rbees-orcd (daemon)

---

**Version**: 1.0  
**Status**: Normative (MUST follow)  
**Last Updated**: 2025-10-09

---

**End of Complete Architecture**
