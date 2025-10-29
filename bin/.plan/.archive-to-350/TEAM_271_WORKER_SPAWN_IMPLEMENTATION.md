## TEAM-271: Worker Spawn Implementation

**Status:** ✅ COMPLETE  
**Date:** Oct 23, 2025  
**Effort:** 2-3 hours

---

## 🎯 Mission

Implement worker spawning using `daemon-lifecycle` and `worker-catalog` to support multiple worker types (CPU, CUDA, Metal).

**Deliverables:**
1. ✅ Worker lifecycle crate using daemon-lifecycle
2. ✅ spawn_worker() function (stateless)
3. ✅ Worker type detection from device string
4. ✅ Binary resolution (catalog → target/debug → target/release)
5. ✅ Multiple worker type support (CpuLlm, CudaLlm, MetalLlm)
6. ✅ Unit tests
7. ✅ Narration events

---

## 📁 Files Created/Modified

```
bin/25_rbee_hive_crates/worker-lifecycle/
├── src/
│   ├── lib.rs          ← Module exports
│   ├── spawn.rs        ← spawn_worker() implementation
│   └── types.rs        ← WorkerSpawnConfig, SpawnResult
└── Cargo.toml          ← Updated dependencies
```

**Total:** ~250 LOC

---

## 🏗️ Architecture

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

### Worker Type Detection

```rust
fn determine_worker_type(device: &str) -> Result<WorkerType> {
    match device.to_lowercase() {
        "cpu" | "CPU-0" → WorkerType::CpuLlm
        "cuda:0" | "GPU-0" → WorkerType::CudaLlm
        "metal" → WorkerType::MetalLlm
    }
}
```

### Binary Resolution Strategy

```
1. Try worker-catalog (production)
   ~/.cache/rbee/workers/{worker-type}-v{version}-{platform}/

2. Fallback to target/debug/{binary-name} (development)

3. Fallback to target/release/{binary-name} (development)

4. Error if not found
```

---

## 📊 Implementation

### spawn_worker() Function

```rust
pub async fn spawn_worker(config: WorkerSpawnConfig) -> Result<SpawnResult> {
    // Step 1: Determine worker type from device
    let worker_type = determine_worker_type(&config.device)?;
    
    // Step 2: Find worker binary
    let binary_path = find_worker_binary(worker_type, &config.job_id)?;
    
    // Step 3: Build command-line arguments
    let args = vec![
        "--worker-id", &config.worker_id,
        "--model", &config.model_id,
        "--device", &config.device,
        "--port", &config.port.to_string(),
        "--queen-url", &config.queen_url,  // ← Worker sends heartbeat HERE
    ];
    
    // Step 4: Spawn using daemon-lifecycle
    let manager = DaemonManager::new(binary_path.clone(), args)
        .enable_auto_update(
            worker_type.binary_name(),
            format!("bin/30_{}", worker_type.binary_name().replace("-", "_")),
        );
    
    let child = manager.spawn().await?;
    let pid = child.id().ok_or_else(|| anyhow!("Failed to get PID"))?;
    
    // Step 5: Return spawn result (hive doesn't track it)
    Ok(SpawnResult {
        worker_id: config.worker_id,
        pid,
        port: config.port,
        binary_path: binary_path.display().to_string(),
    })
}
```

### Worker Types Supported

| Worker Type | Binary Name | Device | Platform |
|-------------|-------------|--------|----------|
| **CpuLlm** | cpu-llm-worker-rbee | cpu, CPU-0 | All |
| **CudaLlm** | cuda-llm-worker-rbee | cuda:0, GPU-0 | Linux, Windows |
| **MetalLlm** | metal-llm-worker-rbee | metal | macOS |

### Command-Line Arguments

```bash
# Example: CPU worker
./cpu-llm-worker-rbee \
  --worker-id worker-abc123 \
  --model meta-llama/Llama-3-8b \
  --device cpu \
  --port 9001 \
  --queen-url http://localhost:8500

# Example: CUDA worker
./cuda-llm-worker-rbee \
  --worker-id worker-def456 \
  --model meta-llama/Llama-3-70b \
  --device cuda:0 \
  --port 9002 \
  --queen-url http://localhost:8500
```

---

## 🔄 Integration with daemon-lifecycle

### Auto-Update Support

```rust
let manager = DaemonManager::new(binary_path, args)
    .enable_auto_update(
        "cpu-llm-worker-rbee",
        "bin/30_cpu_llm_worker_rbee",
    );
```

**Benefits:**
- Automatically rebuilds worker if source changed
- Checks dependencies (Cargo.toml, shared crates)
- Transparent to caller

### Process Management

```rust
// daemon-lifecycle handles:
- Process spawning (tokio::process::Command)
- PID tracking
- Stdout/stderr redirection
- Error handling
```

---

## 🧪 Testing

**Unit Tests:** 1 test
- test_determine_worker_type

**Test Coverage:**
- ✅ CPU device detection
- ✅ CUDA device detection
- ✅ Metal device detection
- ✅ Unknown device error

```bash
cargo test --package rbee-hive-worker-lifecycle
# ✅ 1 passed
```

---

## 📝 Narration Events

### worker_spawn_start
```
🚀 Spawning worker 'worker-abc123' for model 'meta-llama/Llama-3-8b' on device 'cuda:0' port 9001
```

### worker_type_determined
```
Worker type: CudaLlm (device: cuda:0)
```

### worker_binary_found
```
Worker binary: /home/user/.cache/rbee/workers/cuda-llm-worker-rbee-v0.1.0-linux/cuda-llm-worker-rbee
```

### worker_spawn_command
```
Command: /path/to/cuda-llm-worker-rbee --worker-id worker-abc123 --model meta-llama/Llama-3-8b --device cuda:0 --port 9001 --queen-url http://localhost:8500
```

### worker_spawned
```
✅ Worker 'worker-abc123' spawned (PID: 12345, port: 9001)
```

---

## 🔗 Next Steps

### TEAM-272: Worker Management Operations

Now that spawning is implemented, TEAM-272 will implement:
1. **WorkerList** - Query queen's registry for workers
2. **WorkerGet** - Get worker info from queen
3. **WorkerDelete** - Kill worker process by PID (get PID from queen)

### Integration with rbee-hive

```rust
// bin/20_rbee_hive/src/job_router.rs

Operation::WorkerSpawn { hive_id, model, worker, device } => {
    use rbee_hive_worker_lifecycle::{spawn_worker, WorkerSpawnConfig};
    
    let config = WorkerSpawnConfig {
        worker_id: format!("worker-{}", uuid::Uuid::new_v4()),
        model_id: model.clone(),
        device: device.clone(),
        port: allocate_port()?,
        queen_url: "http://localhost:8500".to_string(),
        job_id: job_id.clone(),
    };
    
    let result = spawn_worker(config).await?;
    
    NARRATE
        .action("worker_spawn_complete")
        .job_id(&job_id)
        .context(&result.worker_id)
        .context(&result.pid.to_string())
        .human("✅ Worker spawned: {} (PID: {})")
        .emit();
}
```

---

## ✅ Acceptance Criteria

- [x] Uses daemon-lifecycle for process spawning
- [x] Uses worker-catalog for binary resolution
- [x] Supports multiple worker types (CPU, CUDA, Metal)
- [x] Stateless operation (no tracking in hive)
- [x] Worker sends heartbeat to queen
- [x] Auto-update support enabled
- [x] Narration events for observability
- [x] Unit tests passing
- [x] Fallback to target directory for development

---

## 📚 Key Design Decisions

### 1. Stateless Hive

**Decision:** Hive does NOT track workers  
**Rationale:** Queen is the source of truth for worker state (via heartbeats)  
**Impact:** Hive is simpler, more scalable, easier to test

### 2. Worker Type from Device

**Decision:** Infer worker type from device string  
**Rationale:** User specifies device, we determine appropriate worker  
**Impact:** Simpler API, automatic worker selection

### 3. Binary Resolution Strategy

**Decision:** Catalog → debug → release  
**Rationale:** Production uses catalog, development uses target directory  
**Impact:** Works in both environments without configuration

### 4. daemon-lifecycle Integration

**Decision:** Use shared daemon-lifecycle crate  
**Rationale:** Eliminates duplication, consistent process management  
**Impact:** ~200 LOC savings, auto-update support

---

## 🎯 Architecture Alignment

### Corrected Architecture (CORRECTION_269_TO_272)

```
┌─────────────────────────────────────────────────────────────┐
│ QUEEN (Orchestrator)                                        │
│ - Tracks workers via heartbeats                             │
│ - Routes inference requests to workers                      │
│ - Worker registry (who's alive, what they serve)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP (WorkerSpawn operation)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ HIVE (Executor)                                             │
│ - Executes WorkerSpawn operation                            │
│ - NO worker tracking (stateless)                            │
│ - Uses worker-lifecycle to spawn process                    │
│ - Returns PID and port                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Process spawn
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ WORKER (Inference Engine)                                   │
│ - Sends heartbeat to QUEEN (not hive!)                      │
│ - Serves inference requests                                 │
│ - Implements worker contract                                │
└─────────────────────────────────────────────────────────────┘
```

**✅ This implementation follows the corrected architecture exactly.**

---

**TEAM-271: Worker spawn implementation complete! 🎉**

**Impact:** Stateless worker spawning with multi-type support, ~250 LOC implementation.
