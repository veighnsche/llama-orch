# Complete Binary Architecture — 6 Binaries + 2 Shared Crates

**Status**: Normative  
**Version**: 1.0  
**Date**: 2025-10-09

---

## The Complete Picture

### 6 Binaries

**Orchestrator (blep.home.arpa):**
1. **`orchestratord`** - Daemon (HTTP server :8080)
   - Scheduling, admission, job queue
   - SQLite state store
   - Makes ALL intelligent decisions
   - Uses: `orchestrator-core`

2. **`llorch-ctl`** (command: `llorch`) - CLI
   - Controls orchestratord daemon (M2+)
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

4. **`pool-ctl`** (command: `llorch-pool`) - CLI
   - Controls pool-managerd daemon (M1+)
   - Model downloads (hf CLI)
   - Git operations (submodules)
   - Worker spawn (direct or via daemon)
   - Uses: `pool-core`
   - **NO REPL, NO CONVERSATION**

**Worker (spawned by pool-managerd):**
5. **`llorch-candled`** - Daemon (HTTP server :8001-8999)
   - Inference execution
   - CUDA/Metal/CPU backends
   - No CLI (controlled by pool-managerd)

**Test Harness:**
6. **`bdd-runner`** - BDD test runner (already exists)

### 2 Shared Crates

7. **`orchestrator-core`** - Shared between orchestratord + llorch-ctl
   - Job queue types
   - Scheduling algorithms
   - Pool registry types
   - API types
   - Configuration

8. **`pool-core`** - Shared between pool-managerd + pool-ctl
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
├── orchestratord/                           # Daemon (M2+)
│   ├── .specs/
│   │   └── 00_orchestratord.md
│   ├── src/
│   │   ├── main.rs                         # HTTP server
│   │   ├── api/                            # HTTP endpoints
│   │   ├── scheduler/                      # Scheduling logic
│   │   ├── queue/                          # Job queue
│   │   └── state/                          # SQLite store
│   └── Cargo.toml
│
├── llorch-ctl/                              # CLI (M0+)
│   ├── .specs/
│   │   └── 00_llorch-cli.md
│   ├── catalog.toml                        # Model catalog (shared)
│   ├── src/
│   │   ├── main.rs                         # CLI entry (command: llorch)
│   │   ├── commands/
│   │   │   ├── jobs.rs                     # llorch jobs ...
│   │   │   ├── pools.rs                    # llorch pool ...
│   │   │   └── dev.rs                      # llorch dev ...
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
├── pool-ctl/                                # CLI (M0+)
│   ├── .specs/
│   │   └── 00_pool-ctl.md
│   ├── catalog.toml                        # Model catalog (shared)
│   ├── src/
│   │   ├── main.rs                         # CLI entry (command: llorch-pool)
│   │   ├── commands/
│   │   │   ├── models.rs                   # llorch-pool models ...
│   │   │   ├── git.rs                      # llorch-pool git ...
│   │   │   ├── worker.rs                   # llorch-pool worker ...
│   │   │   └── dev.rs                      # llorch-pool dev ...
│   │   ├── hf_wrapper.rs                   # hf CLI wrapper
│   │   └── git_wrapper.rs                  # git CLI wrapper
│   └── Cargo.toml
│
├── llorch-candled/                          # Worker daemon (M0+, exists)
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

### llorch-ctl Controls

**1. orchestratord daemon (M2+)**
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
**How**: SSH to pool host, executes `llorch-pool` command

**3. Jobs (M2+)**
```bash
llorch jobs submit --model llama3 --prompt "hello"
llorch jobs list
llorch jobs cancel job-123
llorch jobs status job-123
```
**How**: HTTP calls to orchestratord API

### pool-ctl Controls

**1. pool-managerd daemon (M1+)**
```bash
llorch-pool daemon start [--config PATH]
llorch-pool daemon stop
llorch-pool daemon status
```
**How**: Spawns/stops process, queries HTTP API

**2. Local operations (M0)**
```bash
llorch-pool models download tinyllama
llorch-pool git pull
llorch-pool worker spawn metal --model tinyllama
```
**How**: Direct execution (hf CLI, git CLI, process spawn)

**3. Workers (M0+)**
```bash
llorch-pool worker spawn metal --model tinyllama --gpu 0
llorch-pool worker stop worker-metal-0
llorch-pool worker list
```
**How**: Direct spawn (M0) or HTTP call to pool-managerd (M1+)

---

## Shared Logic Examples

### orchestrator-core (Shared)

**Job Queue:**
```rust
// Used by BOTH orchestratord and llorch-ctl
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
// orchestratord/src/main.rs
use orchestrator_core::{JobQueue, Job};

struct Orchestratord {
    queue: Box<dyn JobQueue>,
    state: SqliteStore,  // Daemon-specific
}
```

**CLI uses it:**
```rust
// llorch-ctl/src/commands/jobs.rs
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
// Used by BOTH pool-managerd and pool-ctl
pub fn spawn_worker(req: &WorkerSpawnRequest) -> Result<WorkerInfo> {
    // Shared validation
    validate_backend(&req.backend)?;
    validate_model_exists(&req.model_ref)?;
    validate_gpu_available(req.gpu_id)?;
    
    // Shared spawn logic
    let cmd = build_worker_command(req)?;
    let child = Command::new("llorch-candled")
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
// pool-ctl/src/commands/worker.rs
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
- `llorch-candled` (worker daemon)
- Bash scripts (being replaced)

**What we build:**
1. `pool-core` (shared crate)
2. `pool-ctl` (CLI)
   - Direct model downloads
   - Direct git operations
   - Direct worker spawning
3. `orchestrator-core` (shared crate)
4. `llorch-ctl` (CLI)
   - SSH to pools
   - Commands pool-ctl remotely
   - No daemon yet (CLI does scheduling)

**Architecture:**
```
llorch-ctl (on blep)
    ↓ SSH
pool-ctl (on mac/workstation)
    ↓ spawn
llorch-candled (worker)
```

### M1 (Future)

**What we add:**
5. `pool-managerd` (daemon)
   - HTTP server
   - Worker lifecycle via HTTP
   - Heartbeat to orchestrator
   - Uses: pool-core

**What changes:**
- `pool-ctl` can call pool-managerd HTTP API
- `llorch-ctl` can call pool-managerd HTTP API (via SSH tunnel or direct)

**Architecture:**
```
llorch-ctl (on blep)
    ↓ SSH or HTTP
pool-managerd (daemon on mac/workstation)
    ↓ spawn
llorch-candled (worker)
```

### M2 (Future)

**What we add:**
6. `orchestratord` (daemon)
   - HTTP server
   - Job scheduling via HTTP
   - State persistence
   - Uses: orchestrator-core

**What changes:**
- `llorch-ctl` calls orchestratord HTTP API
- `orchestratord` calls pool-managerd HTTP API

**Architecture:**
```
llorch-ctl (CLI)
    ↓ HTTP
orchestratord (daemon on blep)
    ↓ HTTP
pool-managerd (daemon on mac/workstation)
    ↓ spawn
llorch-candled (worker)
```

---

## Command Examples

### M0: CLI-Only Mode

**On blep (orchestrator host):**
```bash
# Command pool to download model
llorch pool models download tinyllama --host mac
  → SSH: ssh mac "llorch-pool models download tinyllama"
  → Executes: hf download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF ...

# Command pool to spawn worker
llorch pool worker spawn metal --host mac --model tinyllama
  → SSH: ssh mac "llorch-pool worker spawn metal --model tinyllama"
  → Executes: llorch-candled --backend metal --model .test-models/tinyllama/...
```

**On mac (pool host):**
```bash
# Local pool operations
llorch-pool models download tinyllama
  → Executes: hf download ...

llorch-pool worker spawn metal --model tinyllama
  → Executes: llorch-candled --backend metal ...
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
llorch-pool daemon start

# CLI can still work directly OR via daemon
llorch-pool worker spawn metal --model tinyllama
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
  → orchestratord schedules job
  → orchestratord calls pool-managerd: POST http://mac:9200/workers/spawn
  → pool-managerd spawns worker
  → orchestratord dispatches job to worker
  → llorch-ctl streams SSE response
```

---

## HARD RULES

### [RULE-001] NO REPL, NO CONVERSATION

**FORBIDDEN in llorch-ctl:**
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

**FORBIDDEN in pool-ctl:**
```bash
# ❌ NEVER
llorch-pool chat
llorch-pool repl
llorch-pool interactive
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
  llorch jobs cancel $job
done
```

### [RULE-002] CTL Can Control Daemon

**llorch-ctl MUST be able to:**
- Start/stop orchestratord daemon
- Query orchestratord status
- Submit jobs (via HTTP API)
- List jobs (via HTTP API)
- Cancel jobs (via HTTP API)
- Command pools (via SSH or HTTP)

**pool-ctl MUST be able to:**
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

**orchestratord adds:**
- HTTP server (axum/actix)
- SQLite persistence
- Background tasks
- Metrics emission

**llorch-ctl adds:**
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

**pool-ctl adds:**
- Clap CLI parsing
- hf CLI wrapper
- git CLI wrapper
- Colored output
- Progress indicators

---

## Dependency Graph

```
orchestratord
    ├── orchestrator-core
    ├── axum (HTTP server)
    ├── sqlx (SQLite)
    └── prometheus (metrics)

llorch-ctl
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

pool-ctl
    ├── pool-core
    ├── clap (CLI)
    ├── reqwest (HTTP client, optional)
    └── colored (output)

pool-core
    ├── serde (serialization)
    ├── nvml-wrapper (GPU)
    └── anyhow (errors)

llorch-candled
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

2. Create `pool-ctl` CLI
   - Model downloads (hf CLI)
   - Git operations (submodules)
   - Worker spawn (direct)
   - Uses: pool-core

**Week 3-4:**
3. Create `orchestrator-core` crate
   - Job queue types
   - Scheduling algorithms
   - Pool registry types

4. Create `llorch-ctl` CLI
   - Pool commands via SSH
   - Calls pool-ctl remotely
   - Uses: orchestrator-core

**Deliverable:** Working CLIs that replace bash scripts

### Phase 2: Pool Daemon (M1)

**Week 5-6:**
5. Create `pool-managerd` daemon
   - HTTP server :9200
   - Worker lifecycle via HTTP
   - Heartbeat protocol
   - Uses: pool-core

6. Update `pool-ctl` to support HTTP mode
   - Can call pool-managerd API
   - Fallback to direct execution

**Deliverable:** Pool manager daemon with HTTP API

### Phase 3: Orchestrator Daemon (M2)

**Week 7-8:**
7. Create `orchestratord` daemon
   - HTTP server :8080
   - Job scheduling
   - SQLite state store
   - Uses: orchestrator-core

8. Update `llorch-ctl` to support HTTP mode
   - Call orchestratord API for jobs
   - Call pool-managerd API via orchestrator

**Deliverable:** Full orchestrator daemon with HTTP API

---

## Summary

**6 Binaries:**
1. `orchestratord` - Daemon (M2+)
2. `llorch-ctl` - CLI (M0+)
3. `pool-managerd` - Daemon (M1+)
4. `pool-ctl` - CLI (M0+)
5. `llorch-candled` - Worker daemon (M0+, exists)
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
- `llorch-remote` → `llorch-ctl` (orchestrator CLI)
- `llorch-models` → `pool-ctl` (pool CLI)
- `llorch-git` → `pool-ctl` (pool CLI)

**Implementation order:**
1. M0: pool-core + pool-ctl (local operations)
2. M0: orchestrator-core + llorch-ctl (SSH to pools)
3. M1: pool-managerd (daemon)
4. M2: orchestratord (daemon)

---

**Version**: 1.0  
**Status**: Normative (MUST follow)  
**Last Updated**: 2025-10-09

---

**End of Complete Architecture**
