# WHO BROKE THE DUAL-CALL PATTERN?

**Investigation:** TEAM-153  
**Date:** 2025-10-20  
**Status:** 🔥 CRITICAL FINDING

---

## 🚨 THE PROBLEM

**The worker bee uses DIRECT POST → SSE instead of the dual-call pattern from `a_human_wrote_this.md`**

### What Should Have Been Implemented:
```
POST /v1/inference → { "job_id": "...", "sse_url": "/v1/inference/{job_id}/stream" }
GET /v1/inference/{job_id}/stream → SSE stream
```

### What Was Actually Implemented:
```
POST /v1/inference → SSE stream directly (NO intermediate response)
```

---

## 🔍 THE CULPRIT

### **TEAM-147** - Acknowledged but Ignored

**File:** `bin/30_llm_worker_rbee/TEAM-147-STREAMING-BACKEND.md` (lines 191-206)

**Evidence:**
```markdown
### The "POST → GET SSE link" Pattern

**User mentioned this should be a shared crate:**
```
POST /v1/inference → Returns: { "sse_url": "/v1/inference/job-123/stream" }
GET /v1/inference/job-123/stream → SSE stream
```

**Current implementation:**
- POST /v1/inference → Returns SSE stream directly
- No separate GET endpoint

**This is VALID but different from the dual-call pattern.**

**TODO:** Decide if we want the dual-call pattern or keep current approach.
```

**TEAM-147 KNEW about the dual-call pattern but decided to keep the direct approach!**

They left a TODO but **NEVER IMPLEMENTED IT**.

---

## 📋 Timeline of Changes

### Original Implementation
- **Unknown team** created `/execute` endpoint with direct SSE response
- No dual-call pattern from the start

### TEAM-017
- Updated to use Mutex-wrapped backend
- Kept direct SSE pattern

### TEAM-035
- Renamed `/execute` to `/v1/inference`
- Added `[DONE]` marker
- **Still kept direct SSE pattern**

### TEAM-039
- Added narration channel
- **Still kept direct SSE pattern**

### TEAM-147 (October 19, 2025)
- **ACKNOWLEDGED the dual-call pattern should exist**
- **EXPLICITLY CHOSE to keep direct pattern**
- Left TODO but never implemented
- **THIS IS WHERE THE DECISION WAS MADE**

### TEAM-149
- Real-time streaming with request queue
- **Still direct SSE pattern**

### TEAM-150
- Fixed streaming hang
- **Still direct SSE pattern**

---

## 💥 THE IMPACT

### Worker Bee (Current - WRONG)
```rust
// xtask/src/tasks/worker.rs line 364
let payload = serde_json::json!({
    "job_id": "test-job-001",  // Client provides job_id
    "prompt": "The capital of France is",
    "max_tokens": 50,
});

match ureq::post(&inference_url)
    .send_json(&payload)
{
    Ok(response) => {
        // Response IS the SSE stream - NO intermediate JSON!
        let reader = std::io::BufReader::new(response.into_reader());
```

### What Should Be (From a_human_wrote_this.md)
```
1. POST /v1/inference → { "job_id": "job-123", "sse_url": "/v1/inference/job-123/stream" }
2. GET /v1/inference/job-123/stream → SSE stream
```

---

## 🎯 ROOT CAUSE

**NO TEAM EVER IMPLEMENTED THE DUAL-CALL PATTERN**

The worker bee was built with direct SSE from the beginning, and every team that touched it:
1. Either didn't know about the dual-call requirement
2. Or chose to ignore it (TEAM-147)

**TEAM-147 is the most culpable because they:**
- ✅ KNEW about the dual-call pattern
- ✅ DOCUMENTED it in their handoff
- ❌ CHOSE not to implement it
- ❌ Left it as a TODO for "future teams"

---

## 📝 EVIDENCE FROM a_human_wrote_this.md

**Lines 21-24:**
```
Then the bee keeper sends the user task to the queen bee through post.
The queen bee sends a GET link back to the bee keeper.
The bee keeper makes a SSE connection with the queen bee.
```

**This clearly describes:**
1. POST (send task)
2. GET link back (response with SSE URL)
3. SSE connection (separate GET request)

**This is the DUAL-CALL pattern, NOT direct SSE!**

---

## 🔧 WHAT NEEDS TO BE FIXED

### Worker Bee Needs Refactoring

**Current:**
```rust
POST /v1/inference
  Request: { job_id, prompt, max_tokens, ... }
  Response: SSE stream
```

**Should Be:**
```rust
POST /v1/inference
  Request: { prompt, max_tokens, ... }  // NO job_id from client
  Response: { "job_id": "job-123", "sse_url": "/v1/inference/job-123/stream" }

GET /v1/inference/{job_id}/stream
  Response: SSE stream
```

### Changes Required:

1. **Create job registry** - Track active jobs
2. **Generate job_id** - Server generates, not client
3. **Split endpoints:**
   - `POST /v1/inference` - Create job, return job_id + sse_url
   - `GET /v1/inference/{job_id}/stream` - Stream results
4. **Update xtask** - Use dual-call pattern
5. **Update all callers** - Queen, tests, etc.

---

## 🎯 RESPONSIBILITY

### Primary: TEAM-147
- **Date:** 2025-10-19
- **Action:** Acknowledged dual-call pattern but chose direct SSE
- **Impact:** Left worker with wrong pattern
- **Evidence:** TEAM-147-STREAMING-BACKEND.md lines 191-206

### Secondary: All Previous Teams
- **Teams:** TEAM-017, TEAM-035, TEAM-039, TEAM-149, TEAM-150
- **Action:** Never checked against `a_human_wrote_this.md`
- **Impact:** Perpetuated wrong pattern

### Tertiary: Original Implementation
- **Team:** Unknown
- **Action:** Built direct SSE from the start
- **Impact:** Set wrong precedent

---

## 🚨 CONCLUSION

**TEAM-147 IS THE PRIMARY CULPRIT**

They:
1. ✅ Read the requirement
2. ✅ Documented it
3. ❌ Chose not to implement it
4. ❌ Passed the problem to future teams

**The worker bee has NEVER implemented the dual-call pattern from `a_human_wrote_this.md`**

**This needs to be fixed IMMEDIATELY to match the happy flow.**

---

**Signed:** TEAM-153  
**Date:** 2025-10-20  
**Status:** 🔥 CRITICAL - Pattern violation identified
