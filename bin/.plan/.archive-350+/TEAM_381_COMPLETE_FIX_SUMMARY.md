# TEAM-381: Complete Fix Summary - ModelManagement Loading Issue

**Date:** 2025-11-01  
**Status:** ✅ FIXES APPLIED, ⚠️ RESTART REQUIRED

## Problem

ModelManagement UI shows "Loading models..." forever, even when there are 0 models.

## Root Cause

**Backend:** Narration context not set for operations → SSE stream is empty → Client waits forever

**Frontend:** No timeout → Infinite wait for `[DONE]` marker that never comes

## Fixes Applied

### 1. Backend Fix (CRITICAL)
**File:** `bin/20_rbee_hive/src/job_router.rs`

**Change:** Set narration context once for ALL operations

```rust
// BEFORE (broken)
async fn route_operation(job_id: String, payload: serde_json::Value, state: JobState) -> Result<()> {
    let operation: Operation = serde_json::from_value(payload)?;
    n!("route_job", "Executing operation: {}", operation.name());
    // ❌ n!() goes to stdout, not SSE!
    match operation {
        Operation::ModelList(req) => {
            n!("model_list_start", "📋 Listing models...");
            // ❌ This never reaches the client!
        }
    }
}

// AFTER (fixed)
async fn route_operation(job_id: String, payload: serde_json::Value, state: JobState) -> Result<()> {
    use observability_narration_core::{with_narration_context, NarrationContext};
    
    // ✅ Set context ONCE for all operations
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async move {
        let operation: Operation = serde_json::from_value(payload)?;
        n!("route_job", "Executing operation: {}", operation.name());
        // ✅ n!() now goes to SSE stream!
        execute_operation(operation, operation_name, job_id, state).await
    }).await
}
```

**Impact:** ALL operations now send narration to SSE (not just HiveCheck)

### 2. Frontend Fix (DEFENSIVE)
**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts`

**Change:** Add 10-second timeout to prevent infinite hanging

```typescript
// BEFORE (broken)
await client.submitAndStream(op, callback)
// ❌ Waits forever if backend doesn't respond

// AFTER (fixed)
const timeoutPromise = new Promise((_, reject) => {
  setTimeout(() => reject(new Error('Request timeout: Backend did not respond within 10 seconds. Is rbee-hive running?')), 10000)
})

const streamPromise = client.submitAndStream(op, callback)

await Promise.race([streamPromise, timeoutPromise])
// ✅ Fails after 10s with helpful error message
```

**Impact:** UI shows error instead of hanging forever

### 3. API Consistency Fix
**File:** `bin/20_rbee_hive/src/main.rs`

**Change:** Moved `/capabilities` to `/v1/capabilities`

```rust
// BEFORE
.route("/capabilities", get(get_capabilities))

// AFTER
.route("/v1/capabilities", get(get_capabilities))
```

**Impact:** All API endpoints now under `/v1/` prefix

## Code Flow (After Fix)

### Backend Flow
```
1. POST /v1/jobs with ModelList operation
   ↓
2. create_job() → Creates job, returns job_id ✅
   ↓
3. GET /v1/jobs/{job_id}/stream
   ↓
4. execute_job() → Spawns background task ✅
   ↓
5. route_operation() → Sets narration context ✅ NEW!
   ↓
6. execute_operation() → Parses operation ✅
   ↓
7. Match Operation::ModelList → Execute handler ✅
   ↓
8. n!() calls → Send to SSE stream ✅ NOW WORKS!
   ↓
9. Client receives:
   data: Executing operation: ModelList
   data: 📋 Listing models on hive 'localhost'
   data: Found 0 model(s)
   data: []
   data: [DONE]
```

### Frontend Flow
```
1. useModels() hook calls queryFn
   ↓
2. client.submitAndStream(op, callback)
   ↓
3. POST /v1/jobs → Get job_id ✅
   ↓
4. Connect to SSE stream ✅
   ↓
5. Receive lines (with 10s timeout) ✅ NEW!
   ↓
6. Parse JSON from last line: []
   ↓
7. Return empty array to React Query ✅
   ↓
8. UI shows "No models" (0 Downloaded) ✅
```

## How to Test

### Step 1: Rebuild Frontend (if needed)
```bash
cd bin/20_rbee_hive/ui/packages/rbee-hive-react
pnpm build
```

### Step 2: Restart rbee-hive Backend
```bash
# Kill old process
pkill rbee-hive

# Start new one
cargo run -p rbee-hive -- --port 7835 --queen-url http://localhost:7833 --hive-id localhost
```

### Step 3: Test with curl
```bash
JOB_ID=$(curl -s -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"ModelList":{"hive_id":"localhost"}}' \
  | jq -r '.job_id')

curl -N "http://localhost:7835/v1/jobs/$JOB_ID/stream"
```

**Expected output:**
```
data: Executing operation: ModelList
data: 📋 Listing models on hive 'localhost'
data: Found 0 model(s)
data: []
data: [DONE]
```

### Step 4: Test in UI
1. Open http://localhost:7835 (or http://localhost:7836 if using Vite dev server)
2. Navigate to Model Management
3. Should see "0 Downloaded" instead of "Loading models..."
4. Should show empty state (no models)

## Expected Behavior

**With 0 models:**
- ✅ Shows "0 Downloaded" badge
- ✅ Shows empty state message
- ✅ No "Loading models..." spinner

**With models:**
- ✅ Shows "X Downloaded" badge
- ✅ Lists models in table
- ✅ Can click to view details

**On error:**
- ✅ Shows error message after 10s timeout
- ✅ Error says "Backend did not respond within 10 seconds"
- ✅ Can retry

## Files Changed

### Backend (3 files)
1. `bin/20_rbee_hive/src/job_router.rs` - Narration context fix
2. `bin/20_rbee_hive/src/main.rs` - API consistency fix
3. `bin/20_rbee_hive/build.rs` - Dev server detection fix

### Frontend (2 files)
4. `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/index.ts` - Timeout fix
5. `bin/10_queen_rbee/build.rs` - Dev server detection fix

## Why It Was Hanging

**Old behavior:**
1. Client connects to SSE stream
2. Backend creates job but doesn't set narration context
3. n!() calls go to stdout (terminal) instead of SSE
4. SSE stream is empty (no data sent)
5. Client waits for `[DONE]` marker forever
6. React Query shows "Loading..." forever
7. UI appears frozen

**New behavior:**
1. Client connects to SSE stream
2. Backend creates job AND sets narration context ✅
3. n!() calls go to SSE stream ✅
4. SSE stream has data ✅
5. Client receives `[DONE]` marker ✅
6. React Query resolves with data ✅
7. UI shows results ✅

## Prevention

**Added safeguards:**
- ✅ Timeout on frontend (10s)
- ✅ Reduced retry count (2 instead of 3)
- ✅ Fixed retry delay (1s instead of exponential)
- ✅ Helpful error messages

**Future improvements:**
- Add health check before making requests
- Add connection status indicator in UI
- Add "Retry" button on error
- Add backend version check

## Summary

**Problem:** UI hangs at "Loading models..." forever

**Root Cause:** Backend not sending SSE data (narration context missing)

**Fix:** Set narration context for all operations + add frontend timeout

**Status:** ✅ Code fixed, ⚠️ Restart required

**Next Step:** Restart rbee-hive backend to load new code
