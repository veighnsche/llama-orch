# pool-core SPEC — Shared Pool Manager Logic

**Status**: Draft  
**Version**: 0.1.0  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## Purpose

`pool-core` is a shared library crate containing logic used by BOTH:
- `pool-managerd` (daemon) - Uses for runtime worker management, GPU inventory
- `rbees-pool` (CLI) - Uses for types, validation, spawning logic

**Key Principle:** Shared logic, separate execution contexts.

---

## Responsibilities

### [POOL-CORE-001] Worker Registry Types
**MUST** provide:
- `WorkerInfo` struct (worker_id, backend, model_ref, gpu_id, port, status)
- `WorkerRegistry` trait (register, unregister, list, get)
- `WorkerStatus` enum (spawning, ready, busy, idle, stopping, stopped)

### [POOL-CORE-002] GPU Inventory Types
**MUST** provide:
- `GpuInfo` struct (gpu_id, name, vram_total, vram_free, vram_used)
- `GpuInventory` trait (list_gpus, get_gpu, refresh)
- NVML wrapper (read-only queries)

### [POOL-CORE-003] Model Catalog Types
**MUST** provide:
- `ModelMetadata` struct (id, name, repo, file, size, verified)
- `ModelCatalog` trait (list, get, verify)
- Catalog parser (TOML)

### [POOL-CORE-004] Worker Lifecycle Logic
**MUST** provide:
- Worker spawn logic (command construction)
- Worker validation (model exists, GPU available)
- Worker process management (PID tracking)

### [POOL-CORE-005] API Types
**MUST** provide:
- Request types (WorkerSpawnRequest, WorkerStopRequest)
- Response types (WorkerSpawnResponse, PoolStatusResponse)
- Error types (aligned with SYS-5.5.x)

### [POOL-CORE-006] Configuration
**MUST** provide:
- `PoolConfig` struct
- Config parsing (TOML)
- Validation logic

---

## What It Does NOT Include

**NOT included (daemon-only):**
- HTTP server implementation
- Heartbeat protocol (sending to orchestrator)
- Background tasks (GPU monitoring loop)
- Metrics emission (Prometheus)
- Model caching in RAM

**NOT included (CLI-only):**
- Clap argument parsing
- `hf` CLI wrapper (model downloads)
- Git wrapper (git operations)
- Colored output
- Progress indicators

---

## Crate Structure

```
bin/shared-crates/pool-core/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── worker.rs           # Worker types
│   ├── registry.rs         # Worker registry trait
│   ├── gpu.rs              # GPU inventory types
│   ├── nvml.rs             # NVML wrapper
│   ├── catalog/
│   │   ├── mod.rs
│   │   ├── parser.rs       # Catalog TOML parser
│   │   └── models.rs       # Model metadata types
│   ├── lifecycle/
│   │   ├── mod.rs
│   │   ├── spawn.rs        # Worker spawn logic
│   │   └── validate.rs     # Preflight validation
│   ├── api/
│   │   ├── mod.rs
│   │   ├── requests.rs     # API request types
│   │   ├── responses.rs    # API response types
│   │   └── errors.rs       # API error types
│   └── config.rs           # Configuration types
└── tests/
    └── unit/
```

---

## Usage Examples

### In pool-managerd (daemon)

```rust
use pool_core::{WorkerRegistry, WorkerInfo, GpuInventory};

struct PoolManagerd {
    registry: Box<dyn WorkerRegistry>,
    gpu_inventory: Box<dyn GpuInventory>,
    heartbeat_task: JoinHandle<()>,  // Daemon-specific
}

impl PoolManagerd {
    async fn spawn_worker(&mut self, req: WorkerSpawnRequest) -> Result<WorkerSpawnResponse> {
        // Use shared validation
        req.validate()?;
        
        // Use shared GPU inventory
        let gpu = self.gpu_inventory.get_gpu(req.gpu_id)?;
        if gpu.vram_free < req.required_vram {
            return Err(InsufficientVram);
        }
        
        // Use shared spawn logic
        let worker_info = pool_core::lifecycle::spawn_worker(&req)?;
        
        // Use shared registry
        self.registry.register(worker_info.clone())?;
        
        // Daemon-specific: emit metrics
        metrics::worker_spawned(worker_info.backend);
        
        Ok(WorkerSpawnResponse { worker_id: worker_info.id })
    }
}
```

### In rbees-pool (CLI)

```rust
use pool_core::{WorkerSpawnRequest, ModelCatalog};

struct WorkerCommand {
    catalog: ModelCatalog,  // Shared
}

impl WorkerCommand {
    fn spawn(&self, backend: &str, model: &str, gpu: u32) -> Result<()> {
        // Use shared validation
        let model_meta = self.catalog.get(model)?;
        
        // Use shared spawn logic
        let req = WorkerSpawnRequest {
            backend: backend.to_string(),
            model_ref: format!("file:.test-models/{}/{}", model, model_meta.file),
            gpu_id: gpu,
            port: 8001,
        };
        
        req.validate()?;
        
        let worker_info = pool_core::lifecycle::spawn_worker(&req)?;
        
        // CLI-specific: colored output
        println!("✅ Worker spawned: {}", worker_info.id);
        println!("   PID: {}", worker_info.pid);
        println!("   Port: {}", worker_info.port);
        
        Ok(())
    }
}
```

---

## Shared vs Specific Logic

### Shared in pool-core

```rust
// ✅ Shared: Types, validation, algorithms
pub struct WorkerInfo { /* ... */ }
pub trait WorkerRegistry { /* ... */ }
pub fn spawn_worker(req: &WorkerSpawnRequest) -> Result<WorkerInfo> { /* ... */ }
pub fn validate_gpu_availability(gpu_id: u32, required_vram: u64) -> Result<()> { /* ... */ }
```

### Daemon-specific (pool-managerd)

```rust
// Daemon only: HTTP server, persistence, background tasks
pub struct HttpServer { /* ... */ }
pub struct HeartbeatTask { /* ... */ }
pub struct MetricsEmitter { /* ... */ }
```

### CLI-specific (rbees-pool)

```rust
// CLI only: Argument parsing, colored output, progress
pub struct Cli { /* ... */ }  // clap
pub fn show_progress(msg: &str) { /* ... */ }  // indicatif
pub fn colored_output(msg: &str, color: Color) { /* ... */ }  // colored
```

---

**Version**: 0.1.0  
**Last Updated**: 2025-10-09  
**Status**: Draft

---

**End of Specification**
