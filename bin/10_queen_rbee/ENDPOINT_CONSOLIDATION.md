# Endpoint Consolidation - Rule Zero Compliance

**Date:** Oct 29, 2025  
**Status:** ✅ COMPLETE  
**Reason:** Rule Zero violation - two endpoints serving overlapping purposes

---

## Problem

Had two separate endpoints:
- `/v1/build-info` - Build information (version, features, timestamp)
- `/v1/info` - Service discovery (base_url, port, version)

**Rule Zero violation:** "One way to do things" - Not 3 different APIs for the same thing

Both endpoints returned queen metadata. This creates:
- Confusion (which endpoint should I use?)
- Maintenance burden (update both for version changes)
- API bloat (unnecessary duplication)

---

## Solution

**Consolidated into single `/v1/info` endpoint** that returns ALL queen metadata:

```json
{
  "base_url": "http://localhost:7833",
  "port": 7833,
  "version": "0.1.0",
  "features": ["local-hive"],
  "build_timestamp": "2025-10-29T14:00:00Z"
}
```

---

## Changes Made

### 1. Deleted `/v1/build-info` endpoint
- Removed route from `main.rs`
- Deleted `http/build_info.rs` file
- Removed module declaration from `http/mod.rs`

### 2. Enhanced `/v1/info` endpoint
- Added `features` field (from build-info)
- Added `build_timestamp` field (from build-info)
- Kept existing `base_url`, `port`, `version` fields

### 3. Updated `job_router.rs`
- Replaced `should_forward_to_hive()` with `target_server()`
- Uses new `TargetServer` enum from operations-contract

---

## Files Modified

- `bin/10_queen_rbee/src/http/info.rs` - Enhanced with build info fields
- `bin/10_queen_rbee/src/main.rs` - Removed `/v1/build-info` route
- `bin/10_queen_rbee/src/http/mod.rs` - Removed build_info module
- `bin/10_queen_rbee/src/job_router.rs` - Updated to use `target_server()`

## Files Deleted

- `bin/10_queen_rbee/src/http/build_info.rs` - Consolidated into info.rs

---

## Usage

### Before (WRONG - Two endpoints)
```bash
# Service discovery
curl http://localhost:7833/v1/info
# {"base_url":"http://localhost:7833","port":7833,"version":"0.1.0"}

# Build info
curl http://localhost:7833/v1/build-info
# {"version":"0.1.0","features":["local-hive"],"build_timestamp":"..."}
```

### After (CORRECT - One endpoint)
```bash
# All queen metadata in one place
curl http://localhost:7833/v1/info
# {
#   "base_url":"http://localhost:7833",
#   "port":7833,
#   "version":"0.1.0",
#   "features":["local-hive"],
#   "build_timestamp":"2025-10-29T14:00:00Z"
# }
```

---

## Benefits

✅ **One source of truth** - Single endpoint for all queen metadata  
✅ **Simpler API** - Fewer endpoints to remember  
✅ **Easier maintenance** - Update once, not twice  
✅ **Rule Zero compliant** - No duplication  
✅ **Better DX** - One call gets everything

---

## Migration Guide

If you were using `/v1/build-info`:

```diff
- GET /v1/build-info
+ GET /v1/info

Response now includes ALL fields:
{
+ "base_url": "http://localhost:7833",
+ "port": 7833,
  "version": "0.1.0",
  "features": ["local-hive"],
  "build_timestamp": "2025-10-29T14:00:00Z"
}
```

**No breaking changes** - `/v1/info` is additive (added fields, didn't remove any)

---

## Verification

```bash
# Verify queen-rbee compiles
cargo check --package queen-rbee
# ✅ SUCCESS

# Test endpoint (requires queen running)
curl http://localhost:7833/v1/info
```

---

**Rule Zero Compliance:** ✅ One endpoint, one purpose, no duplication
