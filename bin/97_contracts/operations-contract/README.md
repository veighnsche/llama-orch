# operations-contract

**TEAM-186:** Shared operation types for job submissions  
**TEAM-CLEANUP:** Updated to reflect NO PROXYING architecture

## Purpose

This crate provides a **single source of truth** for all operation types in the rbee system, ensuring type safety between:
- **rbee-keeper** (CLI client) - Creates operation payloads
- **queen-rbee** (HTTP server) - Handles orchestration operations
- **rbee-hive** (HTTP server) - Handles worker/model lifecycle operations

## Architecture

**CRITICAL:** rbee-keeper talks directly to BOTH queen AND hive (NO proxying)

```
rbee-keeper
    ↓
Operation enum
    ↓
    ├─→ Queen Operations (Status, Infer)
    │   POST http://localhost:7833/v1/jobs
    │   ├─ Status - Query registries
    │   └─ Infer - Schedule and route to workers
    │
    └─→ Hive Operations (Worker/Model lifecycle)
        POST http://localhost:7835/v1/jobs
        ├─ WorkerSpawn, WorkerProcessList, etc.
        └─ ModelDownload, ModelList, etc.
```

## Operation Types

### Queen Operations (http://localhost:7833/v1/jobs)

**Orchestration operations:**
- `Status` - Query hive and worker registries
- `Infer { ... }` - Schedule inference and route to worker

**RHAI script operations:**
- `RhaiScriptSave { name, content, id }` - Save RHAI script
- `RhaiScriptTest { content }` - Test RHAI script
- `RhaiScriptGet { id }` - Get RHAI script by ID
- `RhaiScriptList` - List all RHAI scripts
- `RhaiScriptDelete { id }` - Delete RHAI script

**Diagnostic operations:**
- `QueenCheck` - Test queen SSE streaming

### Hive Operations (http://localhost:7835/v1/jobs)

**Worker process operations:**
- `WorkerSpawn(WorkerSpawnRequest)` - Spawn worker process
- `WorkerProcessList(WorkerProcessListRequest)` - List worker processes
- `WorkerProcessGet(WorkerProcessGetRequest)` - Get worker process details
- `WorkerProcessDelete(WorkerProcessDeleteRequest)` - Kill worker process

**Model operations:**
- `ModelDownload(ModelDownloadRequest)` - Download model
- `ModelList(ModelListRequest)` - List models
- `ModelGet(ModelGetRequest)` - Get model details
- `ModelDelete(ModelDeleteRequest)` - Delete model

**Diagnostic operations:**
- `HiveCheck { alias }` - Test hive SSE streaming

## Response Format

**Important:** rbee-hive currently returns **narration events** via SSE, not structured JSON responses.

**Example narration output:**
```
data: 🚀 Spawning worker 'cpu' with model 'llama-3.2-1b' on device 0
data: ✅ Worker 'worker-cpu-9301' spawned (PID: 12345, port: 9301)
data: [DONE]
```

Response types (WorkerSpawnResponse, ModelListResponse, etc.) are defined for:
- Type safety and documentation
- Future structured API support
- Programmatic client integration (future)

See `src/responses.rs` for full documentation on response types.

## Usage

### Client (rbee-keeper)

```rust
use rbee_operations::Operation;

// Create operation
let op = Operation::HiveList;

// Serialize to JSON
let payload = serde_json::to_value(&op)?;

// Send to queen-rbee
client.post("/v1/jobs").json(&payload).send().await?;
```

### Server (queen-rbee)

```rust
use operations_contract::{Operation, TargetServer};

// Receive JSON payload
async fn handle_create_job(Json(payload): Json<serde_json::Value>) {
    // Deserialize to Operation enum
    let operation: Operation = serde_json::from_value(payload)?;
    
    // Pattern match and route (queen only handles Status and Infer)
    match operation {
        Operation::Status => handle_status().await,
        Operation::Infer(req) => handle_infer(req).await,
        _ => Err("Operation not supported by queen"),
    }
}
```

### Server (rbee-hive)

```rust
use operations_contract::Operation;

// Receive JSON payload
async fn handle_create_job(Json(payload): Json<serde_json::Value>) {
    // Deserialize to Operation enum
    let operation: Operation = serde_json::from_value(payload)?;
    
    // Pattern match and route (hive handles worker/model operations)
    match operation {
        Operation::WorkerSpawn(req) => handle_worker_spawn(req).await,
        Operation::ModelDownload(req) => handle_model_download(req).await,
        // ... etc
    }
}
```

## Benefits

✅ **Type Safety** - Compile-time guarantees that client and server agree on operation structure  
✅ **Single Source of Truth** - Operation definitions live in one place  
✅ **Exhaustive Matching** - Compiler ensures all operations are handled  
✅ **Automatic Serialization** - Serde handles JSON conversion  
✅ **Documentation** - Operation types are self-documenting  

## JSON Format

All operations use tagged enum serialization with `"operation"` field:

```json
{"operation": "hive_list"}

{"operation": "hive_start", "hive_id": "localhost"}

{
  "operation": "worker_spawn",
  "hive_id": "localhost",
  "model": "test-model",
  "worker": "cpu",
  "device": 0
}

{
  "operation": "infer",
  "hive_id": "localhost",
  "model": "test-model",
  "prompt": "hello",
  "max_tokens": 20,
  "temperature": 0.7,
  "stream": true
}
```

## Routing Helper

Use `target_server()` to determine which server to send an operation to:

```rust
use operations_contract::{Operation, TargetServer};

let op = Operation::Status;
match op.target_server() {
    TargetServer::Queen => {
        // Send to http://localhost:7833/v1/jobs
    }
    TargetServer::Hive => {
        // Send to http://localhost:7835/v1/jobs
    }
}
```

## Migration Path

1. ✅ Create shared crate with Operation enum
2. Update rbee-keeper to use Operation enum
3. Update queen-rbee to use Operation enum
4. Remove string-based operation constants
5. Enjoy type safety! 🎉
