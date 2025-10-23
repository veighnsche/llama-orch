# rbee-hive-worker-lifecycle

**TEAM-271: Worker lifecycle management**

**Category:** Orchestration  
**Pattern:** Command Pattern  
**Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`

## Purpose

Lifecycle management for LLM worker instances.
Uses daemon-lifecycle for process spawning and worker-catalog for binary resolution.

## Module Structure

- `types` - Request/Response types for all operations
- `spawn` - Worker spawning operations (TEAM-271)
- `list` - List workers (TEAM-272) - queries queen
- `get` - Get worker details (TEAM-272) - queries queen
- `delete` - Delete worker (TEAM-272) - kills process

## Usage

### Worker Spawning

```rust
use rbee_hive_worker_lifecycle::{spawn_worker, WorkerSpawnConfig};

let config = WorkerSpawnConfig {
    worker_id: "worker-123".to_string(),
    model_id: "meta-llama/Llama-3-8b".to_string(),
    device: "cuda:0".to_string(),
    port: 9001,
    queen_url: "http://localhost:8500".to_string(),
    job_id: "job-456".to_string(),
};

let result = spawn_worker(config).await?;
println!("Worker spawned: PID {}", result.pid);
```

## Architecture

### Stateless Spawn Pattern

```
Hive receives WorkerSpawn → spawn_worker() → Process spawned → Return PID
                                                    ↓
                                            Worker sends heartbeat to QUEEN
                                            (Hive does NOT track it)
```

**Key Principles:**
- ✅ Hive is STATELESS executor
- ✅ Worker sends heartbeat to queen (not hive)
- ✅ No worker registry in hive
- ✅ Just spawn and return

### Worker Types Supported

| Worker Type | Binary Name | Device | Platform |
|-------------|-------------|--------|----------|
| **CpuLlm** | cpu-llm-worker-rbee | cpu, CPU-0 | All |
| **CudaLlm** | cuda-llm-worker-rbee | cuda:0, GPU-0 | Linux, Windows |
| **MetalLlm** | metal-llm-worker-rbee | metal | macOS |

## Testing

```bash
cargo test --package rbee-hive-worker-lifecycle
```

## Dependencies

- `daemon-lifecycle` - Shared daemon management utilities
- `worker-catalog` - Worker binary catalog
- `artifact-catalog` - Artifact trait
- `narration-core` - Observability

## Created By

**TEAM-271** - Worker spawning implementation

## Implementation Status

- [ ] Core functionality
- [ ] Tests
- [ ] Documentation
- [ ] Examples

## Notes

Uses shared daemon-lifecycle crate. Expected ~386 LOC.
