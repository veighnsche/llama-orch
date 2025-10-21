# Queen-rbee HTTP API Update

**Date:** 2025-10-21  
**Team:** TEAM-186  
**Status:** ✅ Complete

## Summary

Updated queen-rbee HTTP API to match rbee-keeper client expectations and API_REFERENCE.md specification.

## Changes Made

### 1. Added `/v1/` Prefix to API Endpoints

**Before:**
```rust
.route("/jobs", post(http::handle_create_job))
.route("/jobs/{job_id}/stream", get(http::handle_stream_job))
.route("/shutdown", post(handle_shutdown))
.route("/heartbeat", post(http::handle_heartbeat))
```

**After:**
```rust
.route("/v1/jobs", post(http::handle_create_job))
.route("/v1/jobs/:job_id/stream", get(http::handle_stream_job))
.route("/v1/shutdown", post(handle_shutdown))
.route("/v1/heartbeat", post(http::handle_heartbeat))
```

### 2. Created Operations Module

**File:** `src/operations.rs`

Centralized action constants for narration:
- `ACTOR_QUEEN_RBEE`
- `ACTION_START`
- `ACTION_LISTEN`
- `ACTION_READY`
- `ACTION_ERROR`
- `ACTION_SHUTDOWN`

### 3. Updated main.rs

- Removed duplicate constant definitions
- Imported operations from `operations` module
- Updated router to use `/v1/` prefix for API endpoints
- Kept `/health` without prefix for compatibility

## API Endpoints (Final)

### Health & Monitoring
```
GET  /health                     # Health check (no /v1 prefix)
GET  /narration/stream           # Global narration stream (SSE)
```

### V1 API
```
POST /v1/shutdown                # Graceful shutdown
POST /v1/heartbeat               # Receive hive heartbeat (callback)
POST /v1/jobs                    # Submit job (ALL operations)
GET  /v1/jobs/:job_id/stream     # Stream job narration (SSE)
```

### Internal
```
POST /hive/start                 # Internal hive lifecycle management
```

## Compatibility

✅ **rbee-keeper client** - Expects `/v1/jobs` and `/v1/jobs/{job_id}/stream`  
✅ **API_REFERENCE.md** - All endpoints match specification  
✅ **Health check** - Kept at `/health` for backward compatibility  

## Testing

```bash
# Compile check
cargo check -p queen-rbee

# Expected endpoints work:
curl http://localhost:8500/health
curl -X POST http://localhost:8500/v1/jobs -d '{"operation":"hive_list"}'
curl http://localhost:8500/v1/jobs/{job_id}/stream
curl -X POST http://localhost:8500/v1/shutdown
```

## Related Files

- `bin/10_queen_rbee/src/main.rs` - Router configuration
- `bin/10_queen_rbee/src/operations.rs` - Operation constants (NEW)
- `bin/00_rbee_keeper/src/job_client.rs` - Client expectations
- `bin/API_REFERENCE.md` - API specification
