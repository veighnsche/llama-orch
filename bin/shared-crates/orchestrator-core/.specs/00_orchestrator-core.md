# orchestrator-core SPEC — Shared Orchestrator Logic

**Status**: Draft  
**Version**: 0.1.0  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## Purpose

`orchestrator-core` is a shared library crate containing logic used by BOTH:
- `orchestratord` (daemon) - Uses for runtime scheduling, state management
- `llorch-ctl` (CLI) - Uses for types, validation, client logic

**Key Principle:** Shared logic, separate execution contexts.

---

## Responsibilities

### [ORCH-CORE-001] Job Queue Types
**MUST** provide:
- `Job` struct (job_id, model_ref, prompt, params, status)
- `JobQueue` trait (enqueue, dequeue, peek)
- `JobStatus` enum (queued, running, completed, failed, cancelled)
- `Priority` enum (interactive, batch)

### [ORCH-CORE-002] Scheduling Algorithms
**MUST** provide:
- Rhai script integration (SYS-6.1.5)
- Scheduler trait
- Default scheduling policies (FIFO, priority-based)
- Eviction policies

### [ORCH-CORE-003] Pool Registry Types
**MUST** provide:
- `PoolInfo` struct (pool_id, host, status, capabilities)
- `PoolRegistry` trait (register, unregister, list, get)
- `PoolStatus` enum (online, offline, degraded)

### [ORCH-CORE-004] API Types
**MUST** provide:
- Request types (TaskSubmitRequest, WorkerSpawnRequest)
- Response types (TaskSubmitResponse, JobStatusResponse)
- Error types (aligned with SYS-5.5.x)

### [ORCH-CORE-005] Configuration
**MUST** provide:
- `OrchestratorConfig` struct
- Config parsing (TOML)
- Validation logic

---

## What It Does NOT Include

**NOT included (daemon-only):**
- HTTP server implementation
- SQLite persistence
- Background tasks (heartbeat processing)
- Metrics emission (Prometheus)
- SSE streaming server

**NOT included (CLI-only):**
- Clap argument parsing
- SSH client
- Colored output
- Progress indicators
- Interactive prompts

---

## Crate Structure

```
bin/shared-crates/orchestrator-core/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── job.rs              # Job types
│   ├── queue.rs            # Queue trait + implementations
│   ├── scheduler.rs        # Scheduling algorithms
│   ├── pool.rs             # Pool registry types
│   ├── api/
│   │   ├── mod.rs
│   │   ├── requests.rs     # API request types
│   │   ├── responses.rs    # API response types
│   │   └── errors.rs       # API error types
│   ├── config.rs           # Configuration types
│   └── rhai_integration.rs # Rhai scheduler integration
└── tests/
    └── unit/
```

---

## Usage Examples

### In orchestratord (daemon)

```rust
use orchestrator_core::{JobQueue, Job, Scheduler};

struct Orchestratord {
    queue: Box<dyn JobQueue>,
    scheduler: Box<dyn Scheduler>,
    state: SqliteStore,  // Daemon-specific
}

impl Orchestratord {
    async fn submit_job(&mut self, req: TaskSubmitRequest) -> Result<TaskSubmitResponse> {
        // Use shared types and logic
        let job = Job::from_request(req)?;
        self.queue.enqueue(job)?;
        
        // Daemon-specific: persist to SQLite
        self.state.save_job(&job).await?;
        
        Ok(TaskSubmitResponse { job_id: job.id, status: "queued" })
    }
}
```

### In llorch-ctl (CLI)

```rust
use orchestrator_core::{Job, TaskSubmitRequest};

struct JobsCommand {
    client: HttpClient,  // CLI-specific
}

impl JobsCommand {
    async fn submit(&self, model: &str, prompt: &str) -> Result<()> {
        // Use shared types for validation
        let req = TaskSubmitRequest {
            model: model.to_string(),
            prompt: prompt.to_string(),
            max_tokens: 100,
            temperature: 0.7,
        };
        
        // Validate using shared logic
        req.validate()?;
        
        // CLI-specific: HTTP call to daemon
        let response = self.client
            .post("http://localhost:8080/v2/tasks")
            .json(&req)
            .send()
            .await?;
        
        println!("Job submitted: {}", response.job_id);
        Ok(())
    }
}
```

---

**Version**: 0.1.0  
**Last Updated**: 2025-10-09  
**Status**: Draft

---

**End of Specification**
