# rbee-operations

**TEAM-186:** Shared operation types for rbee-keeper ↔ queen-rbee contract

## Purpose

This crate provides a **single source of truth** for all operation types in the rbee system, ensuring type safety between:
- **rbee-keeper** (CLI client) - Creates operation payloads
- **queen-rbee** (HTTP server) - Parses and routes operations

## Architecture

```
rbee-keeper                    queen-rbee
    ↓                              ↓
Operation enum              Operation enum
    ↓                              ↓
serde_json::to_value()      serde_json::from_value()
    ↓                              ↓
POST /v1/jobs  ────────────→  Pattern match & route
```

## Operation Types

### Hive Operations
- `HiveStart { hive_id }`
- `HiveStop { hive_id }`
- `HiveList`
- `HiveGet { id }`
- `HiveCreate { host, port }`
- `HiveUpdate { id }`
- `HiveDelete { id }`

### Worker Operations
- `WorkerSpawn { hive_id, model, worker, device }`
- `WorkerList { hive_id }`
- `WorkerGet { hive_id, id }`
- `WorkerDelete { hive_id, id }`

### Model Operations
- `ModelDownload { hive_id, model }`
- `ModelList { hive_id }`
- `ModelGet { hive_id, id }`
- `ModelDelete { hive_id, id }`

### Inference
- `Infer { hive_id, model, prompt, max_tokens, temperature, top_p?, top_k?, device?, worker_id?, stream }`

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
use rbee_operations::Operation;

// Receive JSON payload
async fn handle_create_job(Json(payload): Json<serde_json::Value>) {
    // Deserialize to Operation enum
    let operation: Operation = serde_json::from_value(payload)?;
    
    // Pattern match and route
    match operation {
        Operation::HiveList => handle_hive_list().await,
        Operation::WorkerSpawn { hive_id, model, worker, device } => {
            handle_worker_spawn(hive_id, model, worker, device).await
        }
        Operation::Infer { hive_id, model, prompt, .. } => {
            handle_infer(hive_id, model, prompt).await
        }
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

## Backward Compatibility

For code that still uses string constants, we provide:

```rust
use rbee_operations::constants::*;

const OP_HIVE_LIST: &str = "hive_list";
const OP_WORKER_SPAWN: &str = "worker_spawn";
// ... etc
```

## Migration Path

1. ✅ Create shared crate with Operation enum
2. Update rbee-keeper to use Operation enum
3. Update queen-rbee to use Operation enum
4. Remove string-based operation constants
5. Enjoy type safety! 🎉
