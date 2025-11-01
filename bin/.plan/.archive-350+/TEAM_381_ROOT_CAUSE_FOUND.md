# TEAM-381: Root Cause - SSE Stream Empty

**Date:** 2025-11-01  
**Status:** 🎯 ROOT CAUSE IDENTIFIED

## The Problem

ModelManagement hangs at "Loading models..." because the SSE stream from `/v1/jobs/{job_id}/stream` is completely empty - no data, no errors, nothing.

## Root Cause

The job execution is **silently failing** somewhere in the pipeline. Here's what's happening:

### The Flow (What SHOULD Happen)

```
1. POST /v1/jobs with ModelList operation
   ↓
2. create_job() → Creates job, returns job_id ✅
   ↓
3. GET /v1/jobs/{job_id}/stream
   ↓
4. execute_job() → Spawns background task ✅
   ↓
5. route_operation() → Parses operation ✅
   ↓
6. Match Operation::ModelList → Execute handler ✅
   ↓
7. n!() calls → Send to SSE stream ❌ NOT HAPPENING
   ↓
8. Client receives data → Parse JSON ❌ NEVER GETS HERE
```

### Where It's Failing

The `n!()` narration calls in the ModelList handler are **not reaching the SSE stream**. This means one of two things:

1. **Narration context is not set** - The `n!()` calls don't know which job_id to send to
2. **Error before narration** - The handler is erroring before any `n!()` calls

## Investigation

### Test 1: Check if narration context is set

Looking at the code:

```rust
// job_router.rs - route_operation()
match operation {
    Operation::ModelList(request) => {
        // ❌ NO narration context set!
        n!("model_list_start", "📋 Listing models on hive '{}'", hive_id);
        // ...
    }
}
```

**Problem:** Unlike `HiveCheck` operation which explicitly sets narration context:

```rust
Operation::HiveCheck { .. } => {
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, handle_hive_check()).await?;
}
```

The `ModelList` operation **doesn't set narration context**, so `n!()` calls go to stdout instead of SSE!

## The Fix

### Option 1: Set Narration Context for Each Operation (CORRECT)

Wrap each operation handler with narration context:

```rust
// bin/20_rbee_hive/src/job_router.rs
match operation {
    Operation::ModelList(request) => {
        use observability_narration_core::{with_narration_context, NarrationContext};
        
        let ctx = NarrationContext::new().with_job_id(&job_id);
        with_narration_context(ctx, async {
            let hive_id = request.hive_id.clone();
            n!("model_list_start", "📋 Listing models on hive '{}'", hive_id);

            let models = state.model_catalog.list();
            n!("model_list_result", "Found {} model(s)", models.len());

            let json = serde_json::to_string(&models)
                .unwrap_or_else(|_| "[]".to_string());
            n!("model_list_json", "{}", json);
            
            Ok(())
        }).await?;
    }
    
    // Same for ALL other operations...
}
```

### Option 2: Set Context Once at Route Level (BETTER)

Set narration context once in `route_operation()`:

```rust
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState,
) -> Result<()> {
    use observability_narration_core::{with_narration_context, NarrationContext};
    
    // Set narration context for ALL operations
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async {
        // Parse payload
        let operation: Operation = serde_json::from_value(payload)?;
        let operation_name = operation.name();
        
        n!("route_job", "Executing operation: {}", operation_name);
        
        // Match and execute (all n!() calls now go to SSE)
        match operation {
            Operation::ModelList(request) => {
                // n!() calls now work!
                n!("model_list_start", "📋 Listing models...");
                // ...
            }
            // ... other operations
        }
        
        Ok(())
    }).await
}
```

## Why This Happened

Looking at the code history:

1. **HiveCheck was implemented first** - It explicitly sets narration context (TEAM-313)
2. **Other operations were added later** - They copied the `n!()` pattern but **forgot to set context**
3. **It worked in testing** - Because narration went to stdout (visible in terminal)
4. **It fails in production** - Because UI expects SSE stream, not stdout

## Verification

After applying the fix, test with curl:

```bash
# Create job
JOB_ID=$(curl -s -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"ModelList":{"hive_id":"localhost"}}' \
  | jq -r '.job_id')

# Stream results
curl -N "http://localhost:7835/v1/jobs/$JOB_ID/stream"
```

**Expected output:**
```
data: 📋 Listing models on hive 'localhost'
data: Found 3 model(s)
data: [{"id":"model-1","name":"llama-3.2-1b","size_bytes":1234}]
data: [DONE]
```

## Impact

**Affected Operations:**
- ✅ `HiveCheck` - Works (has context)
- ❌ `ModelList` - Broken (no context)
- ❌ `ModelGet` - Broken (no context)
- ❌ `ModelDownload` - Broken (no context)
- ❌ `ModelDelete` - Broken (no context)
- ❌ `ModelLoad` - Broken (no context)
- ❌ `ModelUnload` - Broken (no context)
- ❌ `WorkerSpawn` - Broken (no context)
- ❌ `WorkerProcessList` - Broken (no context)
- ❌ `WorkerProcessGet` - Broken (no context)
- ❌ `WorkerProcessDelete` - Broken (no context)

**Basically everything except HiveCheck is broken!**

## Next Steps

1. Apply Option 2 fix (set context once in `route_operation`)
2. Test with curl
3. Test with UI
4. Add regression test to prevent this in future

## Summary

**Root Cause:** Narration context not set for most operations

**Symptom:** SSE stream is empty, UI hangs at "Loading..."

**Fix:** Set narration context in `route_operation()` before matching operations

**Impact:** All operations except HiveCheck are affected

**Priority:** CRITICAL - Blocks all Hive UI functionality
