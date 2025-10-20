# TEAM-154: FIX WORKER BEE TO USE DUAL-CALL PATTERN

**Team:** TEAM-154  
**Date:** 2025-10-20  
**Priority:** 🔥 CRITICAL - Pattern violation  
**Status:** TODO

---

## 🚨 THE PROBLEM

**Worker bee uses DIRECT POST → SSE instead of the dual-call pattern from `a_human_wrote_this.md`**

### Current Implementation (WRONG):
```
POST /v1/inference → SSE stream directly
```

**Request:**
```json
{
  "job_id": "test-job-001",  // ← Client provides job_id
  "prompt": "hello",
  "max_tokens": 50
}
```

**Response:** SSE stream immediately (no intermediate JSON)

### Required Implementation (From a_human_wrote_this.md):
```
POST /v1/inference → { "job_id": "...", "sse_url": "/v1/inference/{job_id}/stream" }
GET /v1/inference/{job_id}/stream → SSE stream
```

**Request:**
```json
{
  "prompt": "hello",  // ← NO job_id from client
  "max_tokens": 50
}
```

**Response 1 (POST):**
```json
{
  "job_id": "job-abc123",  // ← Server generates job_id
  "sse_url": "/v1/inference/job-abc123/stream"
}
```

**Response 2 (GET):**
```
SSE stream
```

---

## 🎯 YOUR MISSION

**Convert worker bee from single-call to dual-call pattern to match `a_human_wrote_this.md`**

### Flow Diagram (Target):
```
┌─────────┐                                    ┌─────────────┐
│ Client  │                                    │ Worker Bee  │
└────┬────┘                                    └──────┬──────┘
     │                                                │
     │ 1. POST /v1/inference                         │
     │    { prompt, max_tokens, ... }                │
     ├──────────────────────────────────────────────>│
     │                                                │
     │                                                ├─ Generate job_id
     │                                                ├─ Store in registry
     │                                                ├─ Queue for processing
     │                                                │
     │ 2. Response: { job_id, sse_url }              │
     │<──────────────────────────────────────────────┤
     │                                                │
     │ 3. GET /v1/inference/{job_id}/stream          │
     ├──────────────────────────────────────────────>│
     │                                                │
     │ 4. SSE Events:                                │
     │    data: {"type":"started",...}               │
     │<──────────────────────────────────────────────┤
     │    data: {"type":"token","t":"hello"}         │
     │<──────────────────────────────────────────────┤
     │    data: [DONE]                               │
     │<──────────────────────────────────────────────┤
```

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1: Job Registry (Infrastructure)

**Create:** `src/job_registry.rs`

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::UnboundedSender;

/// Job state in the registry
#[derive(Debug, Clone)]
pub enum JobState {
    Queued,
    Running,
    Completed,
    Failed(String),
}

/// Job information
pub struct Job {
    pub job_id: String,
    pub prompt: String,
    pub config: SamplingConfig,
    pub state: JobState,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub token_sender: Option<UnboundedSender<TokenResponse>>,
}

/// Job registry - tracks all active jobs
pub struct JobRegistry {
    jobs: Arc<Mutex<HashMap<String, Job>>>,
}

impl JobRegistry {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Create a new job and return job_id
    pub fn create_job(
        &self,
        prompt: String,
        config: SamplingConfig,
    ) -> String {
        let job_id = format!("job-{}", uuid::Uuid::new_v4());
        
        let job = Job {
            job_id: job_id.clone(),
            prompt,
            config,
            state: JobState::Queued,
            created_at: chrono::Utc::now(),
            token_sender: None,
        };
        
        self.jobs.lock().unwrap().insert(job_id.clone(), job);
        job_id
    }
    
    /// Get job by ID
    pub fn get_job(&self, job_id: &str) -> Option<Job> {
        self.jobs.lock().unwrap().get(job_id).cloned()
    }
    
    /// Update job state
    pub fn update_state(&self, job_id: &str, state: JobState) {
        if let Some(job) = self.jobs.lock().unwrap().get_mut(job_id) {
            job.state = state;
        }
    }
    
    /// Set token sender for streaming
    pub fn set_token_sender(
        &self,
        job_id: &str,
        sender: UnboundedSender<TokenResponse>,
    ) {
        if let Some(job) = self.jobs.lock().unwrap().get_mut(job_id) {
            job.token_sender = Some(sender);
        }
    }
}
```

**Wire into main.rs:**
```rust
// Add to imports
use crate::job_registry::JobRegistry;

// In main()
let job_registry = Arc::new(JobRegistry::new());

// Pass to router
let app = create_router(queue, job_registry, expected_token);
```

---

### Phase 2: Split POST Endpoint

**File:** `src/http/execute.rs`

**Create new function:**
```rust
/// Handle POST /v1/inference - Create job and return job_id + sse_url
///
/// TEAM-154: Changed from direct SSE to dual-call pattern
pub async fn handle_create_job(
    State(registry): State<Arc<JobRegistry>>,
    State(queue): State<Arc<RequestQueue>>,
    Json(req): Json<CreateJobRequest>,
) -> Result<Json<CreateJobResponse>, ValidationErrorResponse> {
    // Validate request
    if let Err(validation_errors) = req.validate_all() {
        return Err(validation_errors);
    }
    
    // Convert to sampling config
    let config = SamplingConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
        min_p: req.min_p,
        stop_sequences: vec![],
        stop_strings: req.stop.clone(),
        seed: req.seed.unwrap_or(42),
        max_tokens: req.max_tokens,
    };
    
    // Create job in registry (generates job_id)
    let job_id = registry.create_job(req.prompt.clone(), config.clone());
    
    // Narration
    narration::narrate_dual(NarrationFields {
        actor: ACTOR_HTTP_SERVER,
        action: "job_created",
        target: job_id.clone(),
        human: format!("Job {} created and queued", job_id),
        job_id: Some(job_id.clone()),
        ..Default::default()
    });
    
    // Queue the job (will be processed by background worker)
    let (response_tx, _response_rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_sender(&job_id, response_tx.clone());
    
    let generation_request = GenerationRequest {
        request_id: job_id.clone(),
        prompt: req.prompt,
        config,
        response_tx,
    };
    
    if let Err(e) = queue.add_request(generation_request) {
        registry.update_state(&job_id, JobState::Failed(e.to_string()));
        return Err(ValidationErrorResponse::single_error(
            "queue",
            "Failed to queue job",
        ));
    }
    
    registry.update_state(&job_id, JobState::Queued);
    
    // Return job_id and SSE URL
    Ok(Json(CreateJobResponse {
        job_id: job_id.clone(),
        sse_url: format!("/v1/inference/{}/stream", job_id),
    }))
}
```

**Add request/response types:**
```rust
/// Request to create a job (NO job_id from client)
#[derive(Debug, Deserialize)]
pub struct CreateJobRequest {
    pub prompt: String,
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: u32,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default)]
    pub stop: Vec<String>,
    #[serde(default)]
    pub min_p: f32,
}

/// Response from creating a job
#[derive(Debug, Serialize)]
pub struct CreateJobResponse {
    pub job_id: String,
    pub sse_url: String,
}
```

---

### Phase 3: Create GET Stream Endpoint

**File:** `src/http/stream.rs` (NEW FILE)

```rust
//! GET /v1/inference/{job_id}/stream - Stream job results via SSE
//!
//! Created by: TEAM-154 (dual-call pattern implementation)

use crate::backend::request_queue::TokenResponse;
use crate::http::sse::InferenceEvent;
use crate::job_registry::{JobRegistry, JobState};
use crate::narration::{self, ACTION_ERROR, ACTOR_HTTP_SERVER};
use axum::{
    extract::{Path, State},
    response::{sse::Event, Sse},
};
use futures::stream::{self, Stream, StreamExt};
use observability_narration_core::NarrationFields;
use std::{convert::Infallible, sync::Arc};

type EventStream = Box<dyn Stream<Item = Result<Event, Infallible>> + Send + Unpin>;

/// Handle GET /v1/inference/{job_id}/stream
///
/// TEAM-154: New endpoint for dual-call pattern
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(registry): State<Arc<JobRegistry>>,
) -> Result<Sse<EventStream>, (axum::http::StatusCode, String)> {
    // Get job from registry
    let job = registry.get_job(&job_id).ok_or_else(|| {
        (
            axum::http::StatusCode::NOT_FOUND,
            format!("Job {} not found", job_id),
        )
    })?;
    
    // Check job state
    match job.state {
        JobState::Failed(ref error) => {
            return Err((
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Job {} failed: {}", job_id, error),
            ));
        }
        _ => {}
    }
    
    // Get token sender from job
    let token_sender = job.token_sender.ok_or_else(|| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Job {} has no token sender", job_id),
        )
    })?;
    
    // Create receiver for this stream
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    
    // Forward tokens from job to this stream
    // (In real implementation, you'd subscribe to the job's token stream)
    
    narration::narrate_dual(NarrationFields {
        actor: ACTOR_HTTP_SERVER,
        action: "stream_started",
        target: job_id.clone(),
        human: format!("SSE stream started for job {}", job_id),
        job_id: Some(job_id.clone()),
        ..Default::default()
    });
    
    // Build SSE stream
    let started_event = InferenceEvent::Started {
        job_id: job_id.clone(),
        model: "model".to_string(),
        started_at: chrono::Utc::now().timestamp().to_string(),
    };
    
    let mut token_count = 0u32;
    let token_events = Box::pin(async_stream::stream! {
        while let Some(token_response) = rx.recv().await {
            match token_response {
                TokenResponse::Token(token) => {
                    yield Ok(Event::default().json_data(&InferenceEvent::Token {
                        t: token,
                        i: token_count,
                    }).unwrap());
                    token_count += 1;
                }
                TokenResponse::Error(e) => {
                    yield Ok(Event::default().json_data(&InferenceEvent::Error {
                        code: "GENERATION_ERROR".to_string(),
                        message: e,
                    }).unwrap());
                }
                TokenResponse::Done => {
                    break;
                }
            }
        }
    });
    
    let started_stream = stream::once(futures::future::ready(
        Ok(Event::default().json_data(&started_event).unwrap())
    ));
    
    let stream_with_done: EventStream = Box::new(
        started_stream
            .chain(token_events)
            .chain(stream::once(futures::future::ready(Ok(Event::default().data("[DONE]"))))),
    );
    
    Ok(Sse::new(stream_with_done))
}
```

---

### Phase 4: Update Routes

**File:** `src/http/routes.rs`

```rust
// Add stream module
mod stream;

// Update router
let worker_routes = Router::new()
    .route("/v1/inference", post(execute::handle_create_job))  // ← Changed!
    .route("/v1/inference/:job_id/stream", get(stream::handle_stream_job));  // ← New!
```

---

### Phase 5: Update xtask Test

**File:** `xtask/src/tasks/worker.rs`

**Replace lines 353-413 with:**
```rust
// Step 7: Test inference with dual-call pattern
println!("\n🤔 Testing inference with dual-call pattern...");

// Step 7a: POST to create job
let inference_url = format!("http://127.0.0.1:{}/v1/inference", config.port);
let payload = serde_json::json!({
    // NO job_id - server generates it
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
});

let create_response = ureq::post(&inference_url)
    .set("Content-Type", "application/json")
    .send_json(&payload)
    .expect("Failed to create job");

#[derive(serde::Deserialize)]
struct CreateJobResponse {
    job_id: String,
    sse_url: String,
}

let job_response: CreateJobResponse = create_response
    .into_json()
    .expect("Failed to parse job response");

println!("✅ Job created: {}", job_response.job_id);
println!("📡 SSE URL: {}", job_response.sse_url);

// Step 7b: GET to stream results
let stream_url = format!("http://127.0.0.1:{}{}", config.port, job_response.sse_url);

match ureq::get(&stream_url).call() {
    Ok(response) => {
        println!("✅ SSE connection established");
        
        let reader = std::io::BufReader::new(response.into_reader());
        let mut token_count = 0;
        let mut done_received = false;
        
        println!("📡 Streaming tokens (30s timeout):");
        
        let stream_start = std::time::Instant::now();
        let stream_timeout = Duration::from_secs(30);
        
        for line in reader.lines() {
            if stream_start.elapsed() > stream_timeout {
                println!("\n\n❌ TIMEOUT: No tokens after 30 seconds!");
                break;
            }
            
            if let Ok(line) = line {
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    if data == "[DONE]" {
                        done_received = true;
                        println!("\n✅ Received [DONE] signal");
                        break;
                    }
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(event_type) = json.get("type").and_then(|t| t.as_str()) {
                            if event_type == "token" {
                                if let Some(text) = json.get("t").and_then(|t| t.as_str()) {
                                    print!("{}", text);
                                    std::io::Write::flush(&mut std::io::stdout()).ok();
                                    token_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        println!("\n\n📊 Inference Test Results");
        println!("==================================");
        println!("Tokens received: {}", token_count);
        println!("[DONE] signal: {}", if done_received { "✅" } else { "❌" });
        
        if token_count > 0 && done_received {
            println!("\n✅ DUAL-CALL PATTERN TEST PASSED!");
        } else {
            println!("\n❌ DUAL-CALL PATTERN TEST FAILED!");
        }
    }
    Err(e) => {
        println!("❌ SSE connection failed: {}", e);
    }
}
```

---

## 📝 DEPENDENCIES TO ADD

**Cargo.toml:**
```toml
[dependencies]
uuid = { version = "1.0", features = ["v4"] }
chrono = "0.4"
```

---

## 🧪 TESTING STRATEGY

### Manual Test:

1. **Start worker:**
   ```bash
   cargo run --bin llm-worker-rbee -- \
     --model .test-models/tinyllama/tinyllama.gguf \
     --port 28081 \
     --local-mode
   ```

2. **Create job:**
   ```bash
   curl -X POST http://localhost:28081/v1/inference \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Hello","max_tokens":50,"temperature":0.7}'
   ```
   
   **Expected response:**
   ```json
   {
     "job_id": "job-abc123",
     "sse_url": "/v1/inference/job-abc123/stream"
   }
   ```

3. **Stream results:**
   ```bash
   curl -N http://localhost:28081/v1/inference/job-abc123/stream
   ```
   
   **Expected:** SSE events streaming

### Automated Test:

```bash
cargo xtask worker:test
```

Should now use dual-call pattern.

---

## 🚨 BREAKING CHANGES

### For Queen Bee

**OLD (will break):**
```rust
let response = client
    .post("http://worker:28081/v1/inference")
    .json(&json!({
        "job_id": "job-123",  // ← Worker no longer accepts this
        "prompt": "hello"
    }))
    .send()
    .await?;

// response is SSE stream
```

**NEW (required):**
```rust
// Step 1: Create job
let response = client
    .post("http://worker:28081/v1/inference")
    .json(&json!({
        // NO job_id
        "prompt": "hello"
    }))
    .send()
    .await?;

let job_response: CreateJobResponse = response.json().await?;

// Step 2: Stream results
let stream_url = format!("http://worker:28081{}", job_response.sse_url);
let stream = client.get(&stream_url).send().await?;
```

---

## 📋 ACCEPTANCE CRITERIA

- [ ] Job registry implemented
- [ ] POST /v1/inference returns JSON (not SSE)
- [ ] Response includes job_id and sse_url
- [ ] Server generates job_id (client doesn't provide it)
- [ ] GET /v1/inference/{job_id}/stream returns SSE
- [ ] xtask test uses dual-call pattern
- [ ] All tests pass
- [ ] Documentation updated
- [ ] TEAM-147 TODO resolved

---

## 🎯 SUCCESS CRITERIA

**When complete, worker bee will:**
1. ✅ Match `a_human_wrote_this.md` pattern
2. ✅ Use dual-call: POST → JSON, GET → SSE
3. ✅ Generate job_id server-side
4. ✅ Be ready for shared crate extraction
5. ✅ Work with queen bee (after queen is updated)

---

## 💡 TIPS

1. **Start with job registry** - Foundation for everything
2. **Test incrementally** - Test POST, then GET separately
3. **Keep old endpoint temporarily** - Add `/v1/inference/legacy` for backwards compat during migration
4. **Update xtask first** - It's your integration test
5. **Follow the happy flow** - `a_human_wrote_this.md` is the spec

---

## 🔥 PRIORITY

**THIS IS CRITICAL** - Worker bee has been wrong since the beginning!

**TEAM-147 left this as a TODO. You're fixing their mistake.**

**This MUST be done before queen bee implementation (TEAM-155).**

---

**Signed:** TEAM-153  
**Date:** 2025-10-20  
**Status:** TODO - Fix pattern violation  
**Priority:** 🔥 CRITICAL
