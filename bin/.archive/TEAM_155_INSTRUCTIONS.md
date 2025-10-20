# TEAM-155 Handoff Document

**From:** TEAM-153  
**To:** TEAM-155  
**Date:** 2025-10-20  
**Focus:** Job Submission & SSE Streaming - Submit inference jobs to queen and stream results

**⚠️ DEPENDENCY:** TEAM-154 must fix worker bee dual-call pattern first!  
See: `bin/30_llm_worker_rbee/TEAM_154_FIX_DUAL_CALL_PATTERN.md`

---

## 🎯 Mission

Implement job submission from rbee-keeper to queen-rbee and establish SSE connection for streaming results.

## 📋 Quick Summary

**The Pattern (from `a_human_wrote_this.md`):**

1. **rbee-keeper sends POST** → `POST /jobs` with job details
2. **queen-rbee sends GET link back** → Response with `job_id` and `sse_url`
3. **rbee-keeper makes GET request** → `GET /jobs/{job_id}/stream` (SSE connection)
4. **Narration flows through SSE** → All events stream to shell via SSE

**Example:**
```bash
# Step 1: POST
rbee-keeper → POST http://localhost:8500/jobs
            { "model": "...", "prompt": "..." }

# Step 2: Response with link
queen-rbee → { "job_id": "job-123", "sse_url": "/jobs/job-123/stream" }

# Step 3: GET SSE stream
rbee-keeper → GET http://localhost:8500/jobs/job-123/stream

# Step 4: Events stream to stdout
queen-rbee → data: Starting inference...\n\n
           → data: Token: hello\n\n
           → data: [DONE]\n\n
```

### Happy Flow Target

From `a_human_wrote_this.md` lines 21-27:

> **"Then the bee keeper sends the user task to the queen bee through post."**  
> **"The queen bee sends a GET link back to the bee keeper."**  
> **"The bee keeper makes a SSE connection with the queen bee. Everything getting from SSE I except on the shell"**  
> **"narration (bee keeper): having a sse connection from the bee keeper to the queen bee"**

**Flow Diagram:**
```
┌─────────────┐                                    ┌─────────────┐
│ rbee-keeper │                                    │ queen-rbee  │
└──────┬──────┘                                    └──────┬──────┘
       │                                                  │
       │ 1. POST /jobs                                   │
       │    { model, prompt, max_tokens, temp }          │
       ├────────────────────────────────────────────────>│
       │                                                  │
       │ 2. Response: { job_id, sse_url }                │
       │<────────────────────────────────────────────────┤
       │                                                  │
       │ 3. GET /jobs/{job_id}/stream (SSE)              │
       ├────────────────────────────────────────────────>│
       │                                                  │
       │ 4. SSE Events:                                  │
       │    data: Starting...\n\n                        │
       │<────────────────────────────────────────────────┤
       │    data: Token: hello\n\n                       │
       │<────────────────────────────────────────────────┤
       │    data: [DONE]\n\n                             │
       │<────────────────────────────────────────────────┤
       │                                                  │
       └─> stdout                                         │
```

**Summary:**
1. rbee-keeper → POST /jobs → queen-rbee
2. queen-rbee → Response with job_id and SSE link → rbee-keeper
3. rbee-keeper → GET /jobs/{job_id}/stream → queen-rbee (SSE)
4. queen-rbee → SSE events → rbee-keeper → stdout

---

## ✅ What TEAM-152 Completed

### 1. Queen Auto-Start ✅
**Location:** `bin/05_rbee_keeper_crates/queen-lifecycle/src/lib.rs`

**Function:**
```rust
pub async fn ensure_queen_running(base_url: &str) -> Result<()>
```

**Already integrated in rbee-keeper infer command** (line 287):
```rust
rbee_keeper_queen_lifecycle::ensure_queen_running("http://localhost:8500").await?;
```

**Test:**
```bash
./target/debug/rbee-keeper infer "hello" --model HF:author/minillama
# ⚠️  queen is asleep, waking queen.
# ✅ queen is awake and healthy.
# TODO: Implement infer command (submit job to queen)
```

### 2. daemon-lifecycle Shared Crate ✅
**Location:** `bin/99_shared_crates/daemon-lifecycle/`

Available for spawning processes (you'll need this for hive/worker later).

---

## 🚀 Your Mission: Job Submission & SSE

### Step 1: Implement Queen POST /jobs Endpoint

**Location:** `bin/10_queen_rbee/src/http/jobs.rs` (create this file)

**Endpoint:** `POST /jobs`

**Request Body:**
```json
{
  "model": "HF:author/minillama",
  "prompt": "hello",
  "max_tokens": 20,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "job_id": "job-123",
  "sse_url": "/jobs/job-123/stream"
}
```

**Implementation:**
```rust
// bin/10_queen_rbee/src/http/jobs.rs
// TEAM-153: Create this file

use axum::{Json, extract::State};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct JobRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

pub async fn create_job(
    Json(req): Json<JobRequest>,
) -> Json<JobResponse> {
    // Generate job ID
    let job_id = format!("job-{}", uuid::Uuid::new_v4());
    
    // TODO: Store job in registry
    // TODO: Start processing job
    
    Json(JobResponse {
        job_id: job_id.clone(),
        sse_url: format!("/jobs/{}/stream", job_id),
    })
}
```

**Wire into router:**
```rust
// bin/10_queen_rbee/src/http/mod.rs
mod jobs;

pub fn create_router() -> Router {
    Router::new()
        .route("/health", get(health::health_check))
        .route("/jobs", post(jobs::create_job))  // TEAM-153: Add this
}
```

---

### Step 2: Implement rbee-keeper Job Submission

**Location:** `bin/00_rbee_keeper/src/main.rs` (line 285-302)

**Current code:**
```rust
Commands::Infer { model, prompt, max_tokens, temperature, node, backend, device } => {
    // TEAM-152: Ensure queen is running before submitting inference job
    rbee_keeper_queen_lifecycle::ensure_queen_running("http://localhost:8500").await?;
    
    // TODO: Submit inference job to queen
    println!("TODO: Implement infer command (submit job to queen)");
    // ...
}
```

**Replace TODO with:**
```rust
Commands::Infer { model, prompt, max_tokens, temperature, node, backend, device } => {
    // TEAM-153: Ensure queen is running and get handle for cleanup
    let queen_handle = rbee_keeper_queen_lifecycle::ensure_queen_running("http://localhost:8500").await?;
    
    // TEAM-153: Step 1 - Submit job to queen via POST
    let client = reqwest::Client::new();
    let job_request = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    });
    
    let response = client
        .post("http://localhost:8500/jobs")
        .json(&job_request)
        .send()
        .await?;
    
    // TEAM-153: Step 2 - Queen sends back GET link with job_id
    let job_response: JobResponse = response.json().await?;
    println!("📝 Job submitted: {}", job_response.job_id);
    println!("🔗 SSE URL: {}", job_response.sse_url);
    
    // TEAM-153: Step 3 - Make SSE connection with GET request
    let sse_url = format!("http://localhost:8500{}", job_response.sse_url);
    
    // Narration from happy flow
    println!("🔗 Having a SSE connection from the bee keeper to the queen bee");
    
    // TEAM-153: Step 4 - Stream SSE events to stdout
    let mut event_source = client
        .get(&sse_url)
        .send()
        .await?
        .bytes_stream();
    
    while let Some(chunk) = event_source.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);
        
        // Parse SSE format: "data: message\n\n"
        for line in text.lines() {
            if line.starts_with("data: ") {
                let data = &line[6..];
                if data == "[DONE]" {
                    break;
                }
                print!("{}", data);  // Stream to stdout
                std::io::stdout().flush()?;
            }
        }
    }
    
    // TEAM-153: Cleanup - shutdown queen ONLY if we started it
    queen_handle.shutdown().await?;
    
    Ok(())
}
```

**Key points:**
1. **POST /jobs** - Submit job, get job_id and SSE link back
2. **GET /jobs/{job_id}/stream** - Establish SSE connection
3. **Stream to stdout** - Everything from SSE goes to shell
4. **Narration** - "Having a SSE connection from the bee keeper to the queen bee"

---

### Step 3: Implement SSE Streaming

**Queen side:** `bin/10_queen_rbee/src/http/jobs.rs`

```rust
use axum::response::sse::{Event, Sse};
use futures::stream::Stream;

pub async fn stream_job(
    Path(job_id): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // TODO: Get job from registry
    // TODO: Stream job events
    
    let stream = async_stream::stream! {
        // Example events:
        yield Ok(Event::default().data("Starting inference..."));
        yield Ok(Event::default().data("Token: hello"));
        yield Ok(Event::default().data("Token: world"));
        yield Ok(Event::default().data("[DONE]"));
    };
    
    Sse::new(stream)
}
```

**rbee-keeper side:** Use `reqwest::get()` with streaming:

```rust
use futures::StreamExt;

let mut event_source = client
    .get(&sse_url)
    .send()
    .await?
    .bytes_stream();

while let Some(chunk) = event_source.next().await {
    let chunk = chunk?;
    let text = String::from_utf8_lossy(&chunk);
    
    // Parse SSE format
    for line in text.lines() {
        if line.starts_with("data: ") {
            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }
            print!("{}", data);  // Stream to stdout
            std::io::stdout().flush()?;
        }
    }
}
```

---

## 📋 Implementation Checklist

### Phase 1: Basic Job Submission
- [ ] Create `bin/10_queen_rbee/src/http/jobs.rs`
- [ ] Implement `POST /jobs` endpoint
- [ ] Wire endpoint into router
- [ ] Add `uuid` dependency to queen-rbee
- [ ] Test: `curl -X POST http://localhost:8500/jobs -d '{"model":"test","prompt":"hello","max_tokens":20,"temperature":0.7}'`

### Phase 2: rbee-keeper Integration
- [ ] Add `serde_json` dependency to rbee-keeper
- [ ] Implement job submission in infer command
- [ ] Parse job response
- [ ] Test: `./target/debug/rbee-keeper infer "hello" --model HF:author/minillama`

### Phase 3: SSE Streaming
- [ ] Implement `GET /jobs/:id/stream` endpoint in queen
- [ ] Add SSE dependencies (`axum::response::sse`, `futures`)
- [ ] Implement SSE client in rbee-keeper
- [ ] Stream events to stdout
- [ ] Handle `[DONE]` signal

### Phase 4: BDD Tests
- [ ] Create `bin/00_rbee_keeper/bdd/tests/features/job_submission.feature`
- [ ] Implement step definitions
- [ ] Test scenarios:
  - Job submission succeeds
  - SSE connection established
  - Tokens streamed to stdout

---

## 🚨 Important Notes

### Narration Messages

From happy flow (line 24):
```
narration (bee keeper): having a sse connection from the bee keeper to the queen bee
```

Add this message when SSE connection is established:
```rust
println!("🔗 SSE connection established with queen-rbee");
```

### SSE Format

Server-Sent Events format:
```
data: message content\n\n
```

Example:
```
data: Starting inference...\n\n
data: Token: hello\n\n
data: Token: world\n\n
data: [DONE]\n\n
```

### Error Handling

- If queen is not running → auto-start (already implemented)
- If job submission fails → return error with helpful message
- If SSE connection drops → retry or fail gracefully

---

## 📚 Reference Documents

### Architecture
- `bin/a_human_wrote_this.md` - Happy flow (lines 21-27)
- `bin/10_queen_rbee/src/http/` - Existing HTTP endpoints

### Dependencies to Add

**queen-rbee:**
```toml
uuid = { version = "1.0", features = ["v4"] }
axum = { version = "0.7", features = ["sse"] }
futures = "0.3"
async-stream = "0.3"
```

**rbee-keeper:**
```toml
serde_json = "1.0"
futures = "0.3"
```

---

## 🧪 Testing Strategy

### Manual Testing

1. **Start queen manually:**
   ```bash
   ./target/debug/queen-rbee --port 8500
   ```

2. **Test job submission:**
   ```bash
   curl -X POST http://localhost:8500/jobs \
     -H "Content-Type: application/json" \
     -d '{"model":"test","prompt":"hello","max_tokens":20,"temperature":0.7}'
   ```

3. **Test SSE streaming:**
   ```bash
   curl -N http://localhost:8500/jobs/job-123/stream
   ```

4. **Test end-to-end:**
   ```bash
   ./target/debug/rbee-keeper infer "hello" --model HF:author/minillama
   ```

### BDD Testing

Create scenarios for:
- Job submission with valid parameters
- Job submission with invalid parameters
- SSE connection establishment
- Token streaming
- `[DONE]` signal handling

---

## 💡 Tips for TEAM-153

1. **Start with job submission** - Get POST /jobs working first
2. **Test incrementally** - Test queen endpoint, then rbee-keeper client
3. **SSE is simple** - Just HTTP with `Content-Type: text/event-stream`
4. **Use curl for testing** - Test queen endpoints before integrating
5. **Follow narration** - Add exact messages from happy flow
6. **Write BDD tests** - They'll guide your implementation

---

## 🚀 Ready to Start!

TEAM-153, you have everything you need:
- ✅ Queen auto-starts (TEAM-152)
- ✅ Health checks working (TEAM-151)
- ✅ HTTP server running (TEAM-151)
- ✅ Clear requirements (happy flow)
- ✅ Reference code (existing endpoints)

**Your mission:** Connect rbee-keeper to queen and stream those tokens! 🐝

Good luck, TEAM-153! 🎉

---

**Signed:** TEAM-152  
**Date:** 2025-10-20  
**Status:** Handoff Complete ✅
