# Worker Callback Fix Summary

**Date:** 2025-10-18  
**Issue:** Worker unable to callback to rbee-hive, causing system failure  
**Status:** Fixed (pending full test)

## Problems Identified

### 1. Hostname Resolution Issue (FIXED ✅)
**Problem:** Workers were spawned with hostname `blep` instead of `127.0.0.1`  
**Location:** `/home/vince/Projects/llama-orch/bin/rbee-hive/src/http/workers.rs:152-157`  
**Root Cause:** `hostname::get()` returned system hostname, which couldn't be resolved  
**Fix:** Default to `127.0.0.1` for local workers, use `RBEE_WORKER_HOST` env var for remote

```rust
// Before (BROKEN):
let hostname = std::env::var("RBEE_WORKER_HOST").unwrap_or_else(|_| {
    hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "127.0.0.1".to_string())
});

// After (FIXED):
let hostname = std::env::var("RBEE_WORKER_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
```

### 2. Callback Port Mismatch (FIXED ✅)
**Problem:** Callback URL hardcoded to port 8080 (queen-rbee) instead of rbee-hive's actual port  
**Location:** `/home/vince/Projects/llama-orch/bin/rbee-hive/src/http/workers.rs:168`  
**Root Cause:** Hardcoded port instead of using server's actual bind address  
**Fix:** Pass `server_addr` through AppState and use `state.server_addr.port()`

```rust
// Before (BROKEN):
let callback_url = format!("http://{}:8080/v1/workers/ready", hostname);

// After (FIXED):
let callback_url = format!("http://{}:{}/v1/workers/ready", hostname, state.server_addr.port());
```

**Changes:**
- Added `server_addr: SocketAddr` to `AppState` in `routes.rs`
- Updated `create_router()` to accept and pass `server_addr`
- Updated `daemon.rs` to pass actual bind address to router

### 3. Callback Payload Mismatch (FIXED ✅)
**Problem:** Worker sends wrong payload structure (422 Unprocessable Entity)  
**Location:** `/home/vince/Projects/llama-orch/bin/llm-worker-rbee/src/common/startup.rs`  
**Root Cause:** Worker callback payload didn't match rbee-hive's expected structure

**Worker was sending:**
```rust
{
    worker_id: String,
    vram_bytes: u64,
    uri: String,  // ← Wrong field name
}
```

**rbee-hive expects:**
```rust
{
    worker_id: String,
    url: String,        // ← Different name
    model_ref: String,  // ← Missing
    backend: String,    // ← Missing
    device: u32,        // ← Missing
}
```

**Fix:**
1. Updated worker CLI args to accept `--model-ref`, `--backend`, `--device`
2. Updated rbee-hive spawn to pass these args to worker
3. Updated `callback_ready()` function signature and payload structure

## Architecture Preserved ✅

The fix maintains the critical design decision:
- **Worker → rbee-hive:** Callback for registration (port 9200)
- **queen-rbee → worker:** Direct inference requests (performance path)

Inference does NOT go through rbee-hive, preserving performance.

## Files Modified

1. **`bin/rbee-hive/src/http/workers.rs`**
   - Fixed hostname default to `127.0.0.1`
   - Fixed callback URL to use actual server port
   - Added model_ref, backend, device args to worker spawn

2. **`bin/rbee-hive/src/http/routes.rs`**
   - Added `server_addr` to `AppState`
   - Updated `create_router()` signature

3. **`bin/rbee-hive/src/commands/daemon.rs`**
   - Pass `addr` to `create_router()`

4. **`bin/llm-worker-rbee/src/main.rs`**
   - Added `model_ref`, `backend`, `device` CLI args
   - Updated `callback_ready()` call with new parameters

5. **`bin/llm-worker-rbee/src/common/startup.rs`**
   - Updated `ReadyCallback` struct to match rbee-hive expectations
   - Updated `callback_ready()` function signature

## Testing

### Isolation Test Created
**Location:** `/home/vince/Projects/llama-orch/bin/llm-worker-rbee/test_worker_isolation.sh`

Tests:
- Worker startup
- Model loading
- Callback to mock server
- HTTP endpoints (health, ready)
- Inference execution

### Integration Test
**Location:** `/home/vince/Projects/llama-orch/ASK_SKY_BLUE.sh`

Full system test with queen-rbee orchestration.

## Next Steps

1. ✅ Fix compilation (done)
2. ⏳ Run isolation test to verify worker callback
3. ⏳ Run full integration test (ASK_SKY_BLUE.sh)
4. ⏳ Verify inference completes successfully
5. ⏳ Update BDD tests if needed

## Team Signature

**TEAM-091:** Fixed callback URL port issue  
**TEAM-092:** Fixed callback payload structure mismatch
