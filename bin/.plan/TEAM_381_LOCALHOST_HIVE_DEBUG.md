# TEAM-381: Localhost Hive Loading Issue - Deep Dive

**Date:** 2025-11-01  
**Status:** 🔍 INVESTIGATING

## Current State

**Backend:** ✅ Running on port 7835
```bash
ps aux | grep rbee-hive
# vince 1095628 rbee-hive --port 7835 --queen-url http://localhost:7833 --hive-id localhost
```

**API Test:** ✅ Working
```bash
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"ModelList":{"hive_id":"localhost"}}'
# Returns: {"job_id":"job-xxx","sse_url":"/v1/jobs/job-xxx/stream"}
```

**SSE Stream:** ⚠️ Empty (no data)
```bash
curl -N http://localhost:7835/v1/jobs/job-xxx/stream
# Returns: (nothing, stream is empty)
```

## Problem: SSE Stream is Empty

### Why the UI Hangs

The React hook is waiting for data from the SSE stream, but the stream never sends any data:

```tsx
// useModels() hook flow:
1. POST /v1/jobs → Get job_id ✅
2. Connect to /v1/jobs/{job_id}/stream → Wait for data ⏳
3. Parse JSON from stream → NEVER HAPPENS (stream is empty)
4. UI shows "Loading..." forever ❌
```

### Root Cause Hypothesis

The backend is creating the job but **not executing it**. Possible reasons:

1. **Job router not dispatching** - Job created but handler never called
2. **Operation not recognized** - Backend doesn't know how to handle `ModelList`
3. **Hive ID mismatch** - Backend looking for different hive_id
4. **Silent error** - Exception caught but not logged to SSE

## Investigation Steps

### Step 1: Check Backend Logs

```bash
# Look at rbee-hive terminal output
# Should see narration like:
# "📋 Received job: job-xxx"
# "🔍 Executing ModelList operation"
# etc.
```

**If you see nothing:** Job router isn't dispatching the job.

### Step 2: Test with curl + SSE

```bash
# Create job
JOB_ID=$(curl -s -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"ModelList":{"hive_id":"localhost"}}' \
  | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Stream results (with timeout)
timeout 10 curl -N "http://localhost:7835/v1/jobs/$JOB_ID/stream"
```

**Expected output:**
```
data: 📋 Listing models...
data: 🔍 Found 3 models
data: [{"id":"model-1","name":"llama-3.2-1b","size_bytes":1234}]
data: [DONE]
```

**If you see nothing:** Backend isn't sending SSE events.

### Step 3: Check Job Router Implementation

The job router should:
1. Receive POST /v1/jobs
2. Create job in registry
3. Dispatch operation to handler
4. Handler sends SSE events
5. Stream closes with [DONE]

Let me check the implementation:

```bash
# Check if ModelList operation is handled
grep -r "ModelList" bin/20_rbee_hive/src/
```

### Step 4: Check if Operation is Registered

```bash
# Check job_router.rs for operation dispatch
cat bin/20_rbee_hive/src/job_router.rs | grep -A 20 "match operation"
```

## Likely Issues

### Issue 1: Operation Not Implemented

**Symptom:** SSE stream is empty, no errors in logs

**Cause:** Backend receives `ModelList` operation but has no handler for it

**Fix:** Implement the handler in `job_router.rs`:

```rust
// bin/20_rbee_hive/src/job_router.rs
match operation {
    Operation::ModelList(req) => {
        // Handler code here
        handle_model_list(req, &model_catalog, &job_id).await?;
    }
    // ... other operations
}
```

### Issue 2: Hive ID Validation Failing

**Symptom:** SSE stream is empty, error logged but not sent to stream

**Cause:** Backend validates `hive_id` and rejects "localhost"

**Fix:** Check validation logic:

```rust
// Should accept "localhost" as valid hive_id
if req.hive_id != "localhost" && req.hive_id != hive_config.id {
    // Error: hive_id mismatch
}
```

### Issue 3: Job Router Not Dispatching

**Symptom:** Job created but never executed

**Cause:** Job router creates job but doesn't spawn task to execute it

**Fix:** Ensure job execution task is spawned:

```rust
// After creating job
tokio::spawn(async move {
    execute_operation(operation, job_id, catalogs).await;
});
```

## Quick Diagnostic

**Run this in browser console (on http://localhost:7836):**

```javascript
// Test the SDK directly
import { init, HiveClient, OperationBuilder } from '@rbee/rbee-hive-sdk'

init()

const client = new HiveClient('http://localhost:7835', 'localhost')
const op = OperationBuilder.modelList('localhost')

console.log('Operation:', op)

const lines = []
const jobId = await client.submitAndStream(op, (line) => {
  console.log('Line:', line)
  lines.push(line)
})

console.log('Job ID:', jobId)
console.log('Lines received:', lines.length)
console.log('All lines:', lines)
```

**Expected:**
- Job ID returned
- Multiple lines logged
- Last line is `[DONE]`

**If you see:**
- Job ID but no lines → SSE stream is empty (backend issue)
- No job ID → POST /v1/jobs failed (network/CORS issue)
- Error in console → Check error message

## Next Steps

1. **Check backend terminal** - Look for narration output when job is created
2. **Run curl test** - See if SSE stream has data
3. **Check job_router.rs** - Verify ModelList operation is handled
4. **Add debug logging** - Add `n!()` calls to trace execution
5. **Check browser Network tab** - See actual HTTP requests/responses

## Temporary Workaround

If the backend isn't implementing ModelList yet, you can mock it in the frontend:

```tsx
// packages/rbee-hive-react/src/index.ts
export function useModels() {
  return useQuery({
    queryKey: ['hive-models'],
    queryFn: async () => {
      // TEMPORARY: Return mock data until backend is ready
      return [
        { id: 'model-1', name: 'llama-3.2-1b', size_bytes: 1234567890 },
        { id: 'model-2', name: 'llama-3.2-3b', size_bytes: 3234567890 },
      ]
    },
    staleTime: 30000,
  })
}
```

This will unblock UI development while we fix the backend.

## Summary

**Problem:** ModelManagement hangs at "Loading models..."

**Root Cause:** SSE stream from `/v1/jobs/{job_id}/stream` is empty

**Most Likely Issue:** Backend doesn't have a handler for `ModelList` operation

**Next Action:** Check `bin/20_rbee_hive/src/job_router.rs` for ModelList implementation
