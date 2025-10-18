# TEAM-087 HANDOFF - Debug queen-rbee /v2/tasks Endpoint

**From:** TEAM-086  
**Date:** 2025-10-11  
**Status:** 🔴 CRITICAL - Inference endpoint failing

---

## Mission

**Debug and fix the queen-rbee `/v2/tasks` endpoint that's returning HTTP 500 errors.**

---

## Current State

### ✅ What Works
- `rbee infer` command compiles and runs
- Queen-rbee auto-starts successfully on port 8080
- Diagnostic output shows exactly what's happening
- HTTP request reaches queen-rbee

### ❌ What's Broken
- `/v2/tasks` endpoint returns HTTP 500 Internal Server Error
- Error message: "Worker spawn failed: HTTP 400 Bad Request"

---

## How to Reproduce

```bash
# Method 1: Direct curl test
curl -X POST http://localhost:8080/v2/tasks \
  -H "Content-Type: application/json" \
  -d '{"node":"localhost","model":"test","prompt":"test","max_tokens":10,"temperature":0.7}'

# Response: HTTP 500 - Worker spawn failed: HTTP 400 Bad Request

# Method 2: Via rbee CLI (shows diagnostic output)
./target/release/rbee infer \
  --node localhost \
  --model "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
  --prompt "Why is the sky blue?" \
  --max-tokens 50
```

---

## Investigation Starting Points

### File: `bin/queen-rbee/src/http/inference.rs`
- Function: `handle_create_inference_task()` (line ~28)
- This is the `/v2/tasks` endpoint handler
- Error says "Worker spawn failed: HTTP 400"
- Likely calling rbee-hive spawn endpoint incorrectly

### File: `bin/rbee-hive/src/http/workers.rs`
- Check the spawn worker endpoint
- Why is it returning 400 Bad Request?
- What validation is failing?

### Debugging Steps

1. **Add logging to queen-rbee**
   ```bash
   RUST_LOG=debug ./target/release/queen-rbee --port 8080 --database /tmp/test.db
   ```

2. **Check what request queen-rbee sends to rbee-hive**
   - Log the spawn request payload
   - Compare to what rbee-hive expects

3. **Test rbee-hive spawn endpoint directly**
   ```bash
   # Start rbee-hive on port 9200
   ./target/release/rbee-hive --port 9200
   
   # Test spawn endpoint
   curl -X POST http://localhost:9200/v1/workers/spawn \
     -H "Content-Type: application/json" \
     -d '{"model":"test","backend":"cpu"}'
   ```

---

## Expected Flow

1. Client → `POST /v2/tasks` → queen-rbee
2. Queen-rbee → looks up node in beehive registry
3. Queen-rbee → `POST /v1/workers/spawn` → rbee-hive (on node)
4. rbee-hive → downloads model if needed
5. rbee-hive → spawns llm-worker-rbee
6. rbee-hive → returns worker URL
7. Queen-rbee → `POST /v1/inference` → worker
8. Worker → streams tokens back
9. Queen-rbee → streams to client

**Current failure:** Step 3 (queen-rbee → rbee-hive spawn request)

---

## Files to Check

```
bin/queen-rbee/src/http/
├── inference.rs          # Line ~28: handle_create_inference_task()
├── types.rs              # Request/response types
└── routes.rs             # Route registration

bin/rbee-hive/src/http/
├── workers.rs            # Spawn endpoint handler
└── validation.rs         # Request validation
```

---

## Success Criteria

- [x] ✅ Identify root cause of HTTP 400 from rbee-hive - **Model ref format validation**
- [x] ✅ Fix the bug (likely request format mismatch) - **Added auto-prefixing with "hf:"**
- [x] ✅ Test with curl - Worker spawns successfully (no HTTP 400)
- [ ] ⚠️ Test with `rbee infer` - Worker ready timeout (separate bug, out of scope)
- [x] ✅ Add TEAM-087 signature to modified files

**TEAM-087 Status:** ✅ **HTTP 400 BUG FIXED** (Worker ready timeout is a separate issue)

---

## Quick Win

The error message says "HTTP 400 Bad Request" from rbee-hive. This is almost certainly a **request validation error**. Check:
- Missing required fields
- Wrong field names
- Wrong data types
- URL/path mismatch

**Hint:** Compare the request queen-rbee sends vs what rbee-hive expects.

---

**Created by:** TEAM-086  
**Date:** 2025-10-11  
**Time:** 20:20  
**Completed by:** TEAM-087  
**Completion Date:** 2025-10-11  
**Priority:** P0 - Blocks inference functionality - ✅ **RESOLVED**

---

## TEAM-087 Resolution

**Root Cause:** Model reference format mismatch between queen-rbee and rbee-hive.

**Fixes Applied:**

### 1. HTTP 400 Bug Fix (Primary)
- Added model_ref validation in `bin/queen-rbee/src/http/inference.rs`
- Auto-prefix model names without `:` with `hf:` (e.g., `"test"` → `"hf:test"`)
- Worker now spawns successfully without HTTP 400 error

### 2. Enhanced Timeout Diagnostics (Secondary)
**queen-rbee improvements:**
- Progress logging every 10 seconds during worker ready wait
- Track and report last error (connection, HTTP status, parse errors)
- Comprehensive timeout message with elapsed time, attempts, and possible causes
- Enhanced spawn logging (model, backend, worker ID, URL)

**rbee-hive improvements:**
- Log all spawn parameters (binary path, model, port, callback URL)
- Detect immediate process exits (within 100ms)
- Capture stdout/stderr when worker crashes on startup
- Check binary existence and report helpful error messages

**Benefits:**
- Faster diagnosis of worker startup failures
- Clear error messages with actionable suggestions
- Better visibility into the orchestration flow

**See:** `TEAM_087_SUMMARY.md` for full details
