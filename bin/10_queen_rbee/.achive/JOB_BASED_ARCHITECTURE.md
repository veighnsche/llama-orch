# Job-Based Architecture Update

**Date:** 2025-10-21  
**Team:** TEAM-186  
**Status:** ✅ Complete

## Summary

Updated `handle_create_job` to support the new job-based architecture where ALL operations (not just inference) go through `POST /v1/jobs`.

## Problem

The old implementation assumed jobs were only for inference:

```rust
// OLD: Inference-only
#[derive(Debug, Deserialize)]
pub struct HttpJobRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}
```

This didn't support:
- Hive operations (start, stop, list, get, create, update, delete)
- Worker operations (spawn, list, get, delete)
- Model operations (download, list, get, delete)
- Inference with all parameters

## Solution

Accept generic JSON payloads with an `"operation"` field:

```rust
// NEW: Generic job payload
pub async fn handle_create_job(
    State(state): State<SchedulerState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<HttpJobResponse>, (StatusCode, String)> {
    // Extract operation field
    let operation = payload["operation"]
        .as_str()
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "Missing 'operation' field".to_string()))?;
    
    // Route based on operation
    // TODO: Implement routing logic
}
```

## Supported Operations

### Hive Operations
```json
{"operation": "hive_start", "hive_id": "..."}
{"operation": "hive_stop", "hive_id": "..."}
{"operation": "hive_list"}
{"operation": "hive_get", "id": "..."}
{"operation": "hive_create", "host": "...", "port": 8600}
{"operation": "hive_update", "id": "..."}
{"operation": "hive_delete", "id": "..."}
```

### Worker Operations
```json
{"operation": "worker_spawn", "hive_id": "localhost", "model": "...", "worker": "cpu|cuda|metal", "device": 0}
{"operation": "worker_list", "hive_id": "localhost"}
{"operation": "worker_get", "hive_id": "localhost", "id": "..."}
{"operation": "worker_delete", "hive_id": "localhost", "id": "..."}
```

### Model Operations
```json
{"operation": "model_download", "hive_id": "localhost", "model": "..."}
{"operation": "model_list", "hive_id": "localhost"}
{"operation": "model_get", "hive_id": "localhost", "id": "..."}
{"operation": "model_delete", "hive_id": "localhost", "id": "..."}
```

### Inference
```json
{
  "operation": "infer",
  "hive_id": "localhost",
  "model": "...",
  "prompt": "...",
  "max_tokens": 20,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "device": "cuda",
  "worker_id": "...",
  "stream": true
}
```

## Changes Made

### 1. Updated `handle_create_job` Function

**File:** `src/http.rs`

- Changed from typed `HttpJobRequest` to generic `serde_json::Value`
- Extract `operation` field from payload
- Generate job ID using `uuid::Uuid::new_v4()`
- Register job in registry for SSE streaming
- Return job_id and sse_url

### 2. Use JobRegistry API

**File:** `src/http.rs`

- Use `registry.create_job()` to generate job IDs (no external uuid needed)
- Use `registry.set_token_receiver()` to store the receiver for SSE streaming
- JobRegistry internally uses uuid for job ID generation

### 3. Updated Documentation

- Added TEAM-185 comments explaining job-based architecture
- Documented all supported operation types
- Referenced API_REFERENCE.md for full specifications

## Current Implementation

The current implementation is a **stub** that:
1. ✅ Accepts any JSON payload
2. ✅ Validates `operation` field exists
3. ✅ Generates unique job ID
4. ✅ Registers job for SSE streaming
5. ✅ Returns proper response format
6. ⚠️  **TODO:** Route to appropriate handler based on operation

## Next Steps

1. **Implement operation routing:**
   ```rust
   match operation {
       "hive_start" => handle_hive_start_job(...),
       "hive_stop" => handle_hive_stop_job(...),
       "hive_list" => handle_hive_list_job(...),
       "worker_spawn" => handle_worker_spawn_job(...),
       "infer" => handle_infer_job(...),
       _ => Err((StatusCode::BAD_REQUEST, format!("Unknown operation: {}", operation)))
   }
   ```

2. **Create handler functions** for each operation type

3. **Implement async job execution** that streams results via SSE

4. **Add payload validation** for each operation type

## Testing

```bash
# Compile check
cargo check -p queen-rbee

# Test endpoint
curl -X POST http://localhost:8500/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"operation": "hive_list"}'

# Expected response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "sse_url": "/v1/jobs/550e8400-e29b-41d4-a716-446655440000/stream"
}

# Stream results:
curl http://localhost:8500/v1/jobs/550e8400-e29b-41d4-a716-446655440000/stream
```

## Related Files

- `bin/10_queen_rbee/src/http.rs` - Job endpoint implementation
- `bin/10_queen_rbee/Cargo.toml` - Dependencies
- `bin/API_REFERENCE.md` - API specification
- `bin/00_rbee_keeper/src/main.rs` - Client implementation
- `bin/00_rbee_keeper/src/job_client.rs` - Job submission client

## Architecture Alignment

✅ **rbee-keeper** sends generic job payloads with `operation` field  
✅ **queen-rbee** accepts generic payloads and validates `operation`  
✅ **API_REFERENCE.md** documents all operation types  
⚠️  **TODO:** Implement operation routing and execution
