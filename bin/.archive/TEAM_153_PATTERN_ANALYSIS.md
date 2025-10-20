# TEAM-153 Pattern Analysis: Worker vs Queen Job Submission

**Team:** TEAM-153  
**Date:** 2025-10-20  
**Status:** Analysis Complete

---

## 🔍 Findings

### Worker Bee Pattern (Current Implementation)

**File:** `bin/30_llm_worker_rbee/src/http/execute.rs`

**Pattern:** Direct POST → SSE stream

```
Client → POST /v1/inference → Worker
                              ↓
                        SSE stream starts immediately
                              ↓
                        Client ← SSE events
```

**Request:**
```json
{
  "job_id": "job-123",
  "prompt": "hello",
  "max_tokens": 20,
  "temperature": 0.7
}
```

**Response:** SSE stream directly (no intermediate response)
```
data: {"type":"started","job_id":"job-123",...}\n\n
data: {"type":"token","t":"hello","i":0}\n\n
data: {"type":"token","t":"world","i":1}\n\n
data: [DONE]\n\n
```

**Key characteristics:**
- ✅ Client provides `job_id` in request
- ❌ No intermediate response with SSE URL
- ❌ POST endpoint returns SSE stream directly
- ✅ SSE events include job_id in payload

---

### Queen Bee Pattern (From Happy Flow)

**File:** `bin/a_human_wrote_this.md` lines 21-27

**Pattern:** POST → GET link → SSE

```
Client → POST /jobs → Queen
                      ↓
        Response: {job_id, sse_url}
                      ↓
Client → GET /jobs/{job_id}/stream → Queen
                                      ↓
                                SSE stream
                                      ↓
                                Client ← SSE events
```

**Request:**
```json
{
  "model": "HF:author/minillama",
  "prompt": "hello",
  "max_tokens": 20,
  "temperature": 0.7
}
```

**Response:** Job created with SSE link
```json
{
  "job_id": "job-abc123",
  "sse_url": "/jobs/job-abc123/stream"
}
```

**Then:** Client makes GET request to SSE URL
```
GET /jobs/job-abc123/stream
```

**Key characteristics:**
- ❌ Client does NOT provide job_id (queen generates it)
- ✅ Intermediate response with job_id and sse_url
- ✅ Separate GET endpoint for SSE stream
- ✅ Two-step process: create job, then stream

---

## 🆚 Comparison

| Aspect | Worker Bee | Queen Bee (Happy Flow) |
|--------|-----------|------------------------|
| **Job ID** | Client provides | Server generates |
| **POST response** | SSE stream | JSON with job_id + sse_url |
| **SSE endpoint** | Same as POST | Separate GET endpoint |
| **Steps** | 1 (POST → SSE) | 2 (POST → response, GET → SSE) |
| **Pattern** | Direct streaming | Job creation + streaming |

---

## 💡 Analysis

### Why the Difference?

**Worker Bee (Direct Pattern):**
- Worker is a **synchronous executor** - it processes one job at a time
- Job ID is provided by caller (queen) for tracking
- No job queue or registry needed
- Stream starts immediately because worker processes synchronously

**Queen Bee (Two-Step Pattern):**
- Queen is an **orchestrator** - it manages multiple jobs across multiple workers
- Queen generates job_id for tracking across the system
- Needs job registry to track job state
- Two-step allows:
  1. Job creation/validation/queueing
  2. Separate streaming connection (can reconnect if dropped)

### Should Worker Use Two-Step Pattern?

**NO** - Worker's direct pattern is correct because:
1. Worker doesn't need job registry (queen tracks jobs)
2. Worker processes synchronously (no queue)
3. Job ID comes from queen (worker doesn't generate)
4. Simpler is better for leaf nodes

**YES for Queen** - Queen needs two-step because:
1. Queen manages job lifecycle (create → queue → dispatch → stream)
2. Multiple clients may want to connect to same job stream
3. Job can be created before worker is available
4. Allows reconnection to existing job streams

---

## 🏗️ Shared Crate Opportunity?

### Can We Create a Shared Crate?

**Answer: PARTIAL** - We can share some components, but not the full pattern.

### What Can Be Shared?

#### 1. **SSE Event Types** ✅
Both use similar event structures:
```rust
// Shared: bin/99_shared_crates/sse-events/
pub enum InferenceEvent {
    Started { job_id: String, ... },
    Token { t: String, i: u32 },
    Error { code: String, message: String },
}
```

#### 2. **SSE Streaming Utilities** ✅
```rust
// Shared: bin/99_shared_crates/sse-stream/
pub fn create_sse_stream<T>(events: impl Stream<Item = T>) -> Sse<...>
pub fn parse_sse_client(response: Response) -> impl Stream<Item = String>
```

#### 3. **Job ID Generation** ✅
```rust
// Shared: bin/99_shared_crates/job-id/
pub fn generate_job_id() -> String {
    format!("job-{}", uuid::Uuid::new_v4())
}
```

### What CANNOT Be Shared?

#### 1. **Job Submission Pattern** ❌
- Worker: Direct POST → SSE
- Queen: POST → response → GET → SSE
- These are fundamentally different architectures

#### 2. **Job Registry** ❌
- Worker: No registry (stateless executor)
- Queen: Needs registry (stateful orchestrator)

#### 3. **Endpoint Structure** ❌
- Worker: `/v1/inference` (single endpoint)
- Queen: `/jobs` (create) + `/jobs/{id}/stream` (stream)

---

## 📝 Recommendations

### For TEAM-154 (Queen Implementation)

1. **DO NOT copy worker pattern** - Queen needs the two-step pattern from happy flow
2. **DO use shared SSE utilities** - Create `sse-events` crate for event types
3. **DO generate job_id** - Queen creates job_id, not client
4. **DO implement job registry** - Track job state (created, running, complete)

### For Future Shared Crates

#### Priority 1: SSE Event Types
```
bin/99_shared_crates/sse-events/
├── src/
│   ├── lib.rs
│   ├── inference.rs  // InferenceEvent enum
│   └── serialization.rs
```

#### Priority 2: SSE Streaming Utilities
```
bin/99_shared_crates/sse-stream/
├── src/
│   ├── lib.rs
│   ├── server.rs  // SSE server helpers
│   └── client.rs  // SSE client helpers
```

#### Priority 3: Job ID Generation
```
bin/99_shared_crates/job-id/
├── src/
│   └── lib.rs  // generate_job_id()
```

---

## 🎯 Conclusion

**Worker and Queen use DIFFERENT patterns, and that's CORRECT.**

**Shared crate opportunities:**
- ✅ SSE event types (both use similar events)
- ✅ SSE streaming utilities (both stream over SSE)
- ✅ Job ID generation (both need unique IDs)
- ❌ Job submission pattern (fundamentally different)

**Worker pattern:** Direct POST → SSE (synchronous executor)  
**Queen pattern:** POST → response → GET → SSE (async orchestrator)

Both patterns are correct for their use case. Don't force them to be the same.

---

**Signed:** TEAM-153  
**Date:** 2025-10-20  
**Status:** Analysis Complete ✅
