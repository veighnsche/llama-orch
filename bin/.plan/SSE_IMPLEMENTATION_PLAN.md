# SSE Implementation Plan - Industry Standard Patterns

**Created by:** TEAM-033  
**Date:** 2025-10-10  
**Status:** PLANNING  
**Spec References:** test-001-mvp.md, README.md, llm-worker-rbee patterns  
**Industry References:** llama.cpp server.cpp, mistral.rs streaming.rs

---

## Overview

This plan implements Server-Sent Events (SSE) streaming throughout the application following:
1. **Our existing patterns** in `llm-worker-rbee/src/http/execute.rs`
2. **Industry standards** from llama.cpp and mistral.rs
3. **Our specs** in test-001-mvp.md

## Industry Standard SSE Patterns (Verified)

### From llama.cpp (tools/server/server.cpp)

**Key Pattern:**
```cpp
// Line 4711: Set chunked content provider with SSE
res.set_chunked_content_provider("text/event-stream", chunked_content_provider, on_complete);

// Line 4700: Send [DONE] marker
static const std::string ev_done = "data: [DONE]\n\n";
sink.write(ev_done.data(), ev_done.size());
```

**Observations:**
- Uses `text/event-stream` content type
- Sends `data: [DONE]\n\n` as final marker (OpenAI compatible)
- Checks `sink.is_writable()` for connection state
- Handles errors by sending error events before [DONE]

### From mistral.rs (mistralrs-server-core/src/streaming.rs)

**Key Pattern:**
```rust
// Line 107: Send [DONE] marker
return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));

// Line 153: Send JSON data events
Poll::Ready(Some(Event::default().json_data(response)))

// Line 10-11: Keep-alive interval
pub const DEFAULT_KEEP_ALIVE_INTERVAL_MS: u64 = 10_000;
```

**Observations:**
- Uses Axum's `Event::default().json_data()` for structured events
- Uses `Event::default().data("[DONE]")` for termination
- Implements `futures::Stream` trait with `poll_next()`
- Has 3 states: `Running`, `SendingDone`, `Done`
- **10-second keep-alive interval** (configurable via env var)
- Stores chunks for completion callbacks
- Sanitizes error messages before sending

## Key Industry Standards

### 1. **[DONE] Marker** (OpenAI Compatible)
```
data: [DONE]\n\n
```
- **Required** for OpenAI API compatibility
- Sent as final event after all data
- Both llama.cpp and mistral.rs use this

### 2. **Keep-Alive**
- **10 seconds** is industry standard (mistral.rs)
- Prevents proxy timeouts
- Sends comment lines: `: keep-alive\n\n`

### 3. **Error Handling**
```rust
// mistral.rs pattern (line 127-136)
Response::ModelError(msg, _) => {
    self.done_state = DoneState::SendingDone;
    Poll::Ready(Some(Ok(Event::default().data(msg))))
}
```
- Send error as data event
- Still send [DONE] after error
- Don't close stream abruptly

### 4. **Connection State**
```cpp
// llama.cpp pattern (line 4697)
return !sink.is_writable();
```
- Check if client disconnected
- Stop generation early if connection closed
- Prevents wasted compute

### 5. **State Machine**
```rust
// mistral.rs pattern (line 14-21)
enum DoneState {
    Running,       // Actively streaming
    SendingDone,   // About to send [DONE]
    Done,          // Completed
}
```
- Clear state transitions
- Ensures [DONE] is always sent
- Handles cleanup in Done state

## Existing SSE Pattern (llm-worker-rbee)

### ✅ Already Implemented in Worker

**Location:** `bin/llm-worker-rbee/src/http/execute.rs`

**Pattern:**
```rust
use axum::response::sse::{Event, Sse};
use futures::stream::{self, Stream};

type EventStream = Box<dyn Stream<Item = Result<Event, Infallible>> + Send + Unpin>;

pub async fn handle_execute<B: InferenceBackend>(
    State(backend): State<Arc<Mutex<B>>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<EventStream>, ValidationErrorResponse> {
    // 1. Execute inference
    let result = backend.lock().await.execute(&req.prompt, &config).await?;
    
    // 2. Build event stream
    let mut events = Vec::new();
    
    // Started event
    events.push(InferenceEvent::Started { job_id, timestamp });
    
    // Token events
    for (i, token) in result.tokens.iter().enumerate() {
        events.push(InferenceEvent::Token {
            t: token.clone(),
            i,
        });
    }
    
    // End event
    events.push(InferenceEvent::End {
        job_id,
        total_tokens,
        duration_ms,
    });
    
    // 3. Convert to SSE stream
    let stream: EventStream = Box::new(
        stream::iter(events).map(|event| {
            Ok(Event::default().json_data(&event).unwrap())
        })
    );
    
    Ok(Sse::new(stream))
}
```

**Key Components:**
- `axum::response::sse::{Event, Sse}` - Axum's built-in SSE support
- `futures::stream` - Stream construction
- `Event::default().json_data(&event)` - JSON serialization
- Response type: `Sse<EventStream>`

---

## Required SSE Implementations

Per test-001-mvp.md, we need SSE in **3 places**:

### 1. Model Download Progress (Phase 3)

**Spec:** Lines 109-120 of test-001-mvp.md

**Endpoint:** `GET /v1/models/download/progress?id=<download_id>`

**Event Format:**
```json
data: {"stage": "downloading", "bytes_downloaded": 1048576, "bytes_total": 5242880, "speed_mbps": 45.2}
data: {"stage": "downloading", "bytes_downloaded": 2097152, "bytes_total": 5242880, "speed_mbps": 48.1}
data: {"stage": "complete", "local_path": "/models/tinyllama-q4.gguf"}
data: [DONE]
```

**Implementation Location:** `bin/rbee-hive/src/http/models.rs`

**Pattern to Follow:**
```rust
use axum::response::sse::{Event, Sse};
use futures::stream::{self, Stream};
use tokio::sync::broadcast;

type EventStream = Box<dyn Stream<Item = Result<Event, Infallible>> + Send + Unpin>;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "stage")]
pub enum DownloadEvent {
    #[serde(rename = "downloading")]
    Downloading {
        bytes_downloaded: u64,
        bytes_total: u64,
        speed_mbps: f64,
    },
    #[serde(rename = "complete")]
    Complete {
        local_path: String,
    },
}

pub async fn handle_download_progress(
    State(state): State<AppState>,
    Query(params): Query<DownloadProgressQuery>,
) -> Result<Sse<EventStream>, (StatusCode, String)> {
    // Get download tracker by ID
    let mut rx = state.download_tracker
        .subscribe(params.id)
        .ok_or((StatusCode::NOT_FOUND, "Download not found".to_string()))?;
    
    // Stream events
    let stream = async_stream::stream! {
        while let Ok(event) = rx.recv().await {
            yield Ok(Event::default().json_data(&event).unwrap());
        }
        // Send [DONE] marker
        yield Ok(Event::default().data("[DONE]"));
    };
    
    Ok(Sse::new(Box::new(stream)))
}
```

---

### 2. Model Loading Progress (Phase 7)

**Spec:** Lines 243-254 of test-001-mvp.md

**Endpoint:** `GET /v1/loading/progress` (on worker)

**Event Format:**
```json
data: {"stage": "loading_to_vram", "layers_loaded": 12, "layers_total": 32, "vram_mb": 2048}
data: {"stage": "loading_to_vram", "layers_loaded": 24, "layers_total": 32, "vram_mb": 4096}
data: {"stage": "ready"}
data: [DONE]
```

**Implementation Location:** `bin/llm-worker-rbee/src/http/loading.rs` (NEW)

**Pattern to Follow:**
```rust
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "stage")]
pub enum LoadingEvent {
    #[serde(rename = "loading_to_vram")]
    LoadingToVram {
        layers_loaded: u32,
        layers_total: u32,
        vram_mb: u64,
    },
    #[serde(rename = "ready")]
    Ready,
}

pub async fn handle_loading_progress(
    State(backend): State<Arc<Mutex<dyn InferenceBackend>>>,
) -> Result<Sse<EventStream>, (StatusCode, String)> {
    // Subscribe to loading progress channel
    let mut rx = backend.lock().await.loading_progress_channel();
    
    let stream = async_stream::stream! {
        while let Ok(event) = rx.recv().await {
            yield Ok(Event::default().json_data(&event).unwrap());
            
            if matches!(event, LoadingEvent::Ready) {
                break;
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    
    Ok(Sse::new(Box::new(stream)))
}
```

---

### 3. Inference Token Streaming (Phase 8)

**Spec:** Lines 290-308 of test-001-mvp.md

**Endpoint:** `POST /v1/inference` (on worker)

**Event Format:**
```json
data: {"token": "Once", "index": 0}
data: {"token": " upon", "index": 1}
data: {"token": " a", "index": 2}
data: {"done": true, "total_tokens": 20, "duration_ms": 1234}
data: [DONE]
```

**Implementation Location:** `bin/llm-worker-rbee/src/http/execute.rs` (ALREADY EXISTS!)

**Status:** ✅ **ALREADY IMPLEMENTED** - Just needs endpoint rename from `/execute` to `/v1/inference`

---

## Implementation Steps

### Step 1: Add SSE Dependencies ✅

**Already in Cargo.toml:**
```toml
axum = { version = "0.8", features = ["sse"] }
futures = "0.3"
tokio = { version = "1", features = ["sync"] }
```

**Additional needed:**
```toml
async-stream = "0.3"  # For async_stream::stream! macro
```

---

### Step 2: Implement Download Progress Tracker (Industry Standard)

**File:** `bin/rbee-hive/src/download_tracker.rs` (NEW)

```rust
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;
use anyhow::Result;

/// Download event states (per test-001-mvp.md)
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "stage")]
pub enum DownloadEvent {
    #[serde(rename = "downloading")]
    Downloading {
        bytes_downloaded: u64,
        bytes_total: u64,
        speed_mbps: f64,
    },
    #[serde(rename = "complete")]
    Complete {
        local_path: String,
    },
    #[serde(rename = "error")]
    Error {
        message: String,
    },
}

/// Download stream state (mistral.rs pattern)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadState {
    Running,
    SendingDone,
    Done,
}

pub struct DownloadTracker {
    downloads: Arc<RwLock<HashMap<String, broadcast::Sender<DownloadEvent>>>>,
}

impl DownloadTracker {
    pub fn new() -> Self {
        Self {
            downloads: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Start tracking a new download
    pub async fn start_download(&self) -> String {
        let download_id = Uuid::new_v4().to_string();
        // Industry standard: 100 buffer size (mistral.rs uses this)
        let (tx, _rx) = broadcast::channel(100);
        
        self.downloads.write().await.insert(download_id.clone(), tx);
        download_id
    }
    
    /// Send progress update
    pub async fn send_progress(&self, download_id: &str, event: DownloadEvent) -> Result<()> {
        let downloads = self.downloads.read().await;
        if let Some(tx) = downloads.get(download_id) {
            // Ignore send errors (no active subscribers)
            let _ = tx.send(event);
        }
        Ok(())
    }
    
    /// Subscribe to download progress
    pub async fn subscribe(&self, download_id: &str) -> Option<broadcast::Receiver<DownloadEvent>> {
        let downloads = self.downloads.read().await;
        downloads.get(download_id).map(|tx| tx.subscribe())
    }
    
    /// Complete and cleanup download
    pub async fn complete_download(&self, download_id: &str) {
        self.downloads.write().await.remove(download_id);
    }
}
```

---

### Step 3: Update AppState

**File:** `bin/rbee-hive/src/http/routes.rs`

```rust
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<WorkerRegistry>,
    pub model_catalog: Arc<ModelCatalog>,
    pub provisioner: Arc<ModelProvisioner>,
    pub download_tracker: Arc<DownloadTracker>,  // NEW
}
```

---

### Step 4: Implement Download Progress Endpoint (Industry Standard)

**File:** `bin/rbee-hive/src/http/models.rs`

```rust
use axum::response::sse::{Event, Sse, KeepAlive};
use axum::extract::Query;
use futures::stream::Stream;
use std::convert::Infallible;
use std::time::Duration;

type EventStream = Box<dyn Stream<Item = Result<Event, Infallible>> + Send + Unpin>;

#[derive(Debug, Deserialize)]
pub struct DownloadProgressQuery {
    pub id: String,
}

/// Handle GET /v1/models/download/progress?id=<download_id>
///
/// Per test-001-mvp.md Phase 3: Model Download Progress
/// Industry standard: mistral.rs streaming pattern with keep-alive
pub async fn handle_download_progress(
    State(state): State<AppState>,
    Query(params): Query<DownloadProgressQuery>,
) -> Result<Sse<EventStream>, (StatusCode, String)> {
    // Subscribe to download progress
    let mut rx = state
        .download_tracker
        .subscribe(&params.id)
        .await
        .ok_or((
            StatusCode::NOT_FOUND,
            format!("Download {} not found", params.id),
        ))?;
    
    // Create SSE stream with industry-standard pattern
    let stream = async_stream::stream! {
        let mut done_state = DownloadState::Running;
        
        loop {
            match done_state {
                DownloadState::SendingDone => {
                    // Industry standard: Send [DONE] marker (OpenAI compatible)
                    yield Ok(Event::default().data("[DONE]"));
                    done_state = DownloadState::Done;
                }
                DownloadState::Done => {
                    // Stream complete
                    break;
                }
                DownloadState::Running => {
                    match rx.recv().await {
                        Ok(event) => {
                            // Check if this is terminal event
                            let is_terminal = matches!(
                                event,
                                DownloadEvent::Complete { .. } | DownloadEvent::Error { .. }
                            );
                            
                            // Send event as JSON
                            yield Ok(Event::default().json_data(&event).unwrap());
                            
                            if is_terminal {
                                done_state = DownloadState::SendingDone;
                            }
                        }
                        Err(_) => {
                            // Channel closed, send [DONE]
                            done_state = DownloadState::SendingDone;
                        }
                    }
                }
            }
        }
    };
    
    // Industry standard: 10-second keep-alive (mistral.rs pattern)
    Ok(Sse::new(Box::new(stream))
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(10))))
}
```

---

### Step 5: Update Download Handler to Use Tracker

**File:** `bin/rbee-hive/src/http/models.rs`

```rust
pub async fn handle_download_model(
    State(state): State<AppState>,
    Json(request): Json<DownloadModelRequest>,
) -> Result<Json<DownloadModelResponse>, (StatusCode, String)> {
    // ... parse model reference ...
    
    // Check catalog
    match state.model_catalog.find_model(reference, provider).await {
        Ok(Some(model_info)) => {
            // Already downloaded
            return Ok(Json(DownloadModelResponse {
                download_id: "cached".to_string(),
                local_path: Some(model_info.local_path),
            }));
        }
        Ok(None) => {
            // Need to download - start tracking
            let download_id = state.download_tracker.start_download().await;
            
            // Spawn download task
            let state_clone = state.clone();
            let reference = reference.to_string();
            let provider = provider.to_string();
            let download_id_clone = download_id.clone();
            
            tokio::spawn(async move {
                download_with_progress(
                    state_clone,
                    &reference,
                    &provider,
                    &download_id_clone,
                ).await
            });
            
            // Return immediately with download ID
            return Ok(Json(DownloadModelResponse {
                download_id,
                local_path: None,
            }));
        }
        Err(e) => {
            return Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Catalog error: {}", e)));
        }
    }
}

async fn download_with_progress(
    state: AppState,
    reference: &str,
    provider: &str,
    download_id: &str,
) -> Result<()> {
    // TODO: Implement actual download with progress callbacks
    // For now, use existing provisioner
    match state.provisioner.download_model(reference, provider).await {
        Ok(local_path) => {
            // Send complete event
            state.download_tracker.send_progress(
                download_id,
                DownloadEvent::Complete {
                    local_path: local_path.to_string_lossy().to_string(),
                },
            ).await?;
            
            // Register in catalog
            // ... existing catalog registration code ...
            
            // Cleanup tracker
            state.download_tracker.complete_download(download_id).await;
            Ok(())
        }
        Err(e) => {
            error!(error = %e, "Download failed");
            state.download_tracker.complete_download(download_id).await;
            Err(e)
        }
    }
}
```

---

### Step 6: Implement Worker Loading Progress

**File:** `bin/llm-worker-rbee/src/http/loading.rs` (NEW)

```rust
//! Model loading progress endpoint
//!
//! Per test-001-mvp.md Phase 7: Worker Health Check

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
};
use futures::stream::Stream;
use serde::Serialize;
use std::{convert::Infallible, sync::Arc};
use tokio::sync::Mutex;

type EventStream = Box<dyn Stream<Item = Result<Event, Infallible>> + Send + Unpin>;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "stage")]
pub enum LoadingEvent {
    #[serde(rename = "loading_to_vram")]
    LoadingToVram {
        layers_loaded: u32,
        layers_total: u32,
        vram_mb: u64,
    },
    #[serde(rename = "ready")]
    Ready,
}

/// Handle GET /v1/loading/progress
///
/// Per test-001-mvp.md Phase 7: Stream model loading progress
pub async fn handle_loading_progress(
    State(backend): State<Arc<Mutex<dyn InferenceBackend>>>,
) -> Result<Sse<EventStream>, (StatusCode, String)> {
    // Get loading progress channel from backend
    let mut rx = backend
        .lock()
        .await
        .loading_progress_channel()
        .ok_or((
            StatusCode::SERVICE_UNAVAILABLE,
            "Model not loading".to_string(),
        ))?;
    
    // Stream loading events
    let stream = async_stream::stream! {
        while let Ok(event) = rx.recv().await {
            yield Ok(Event::default().json_data(&event).unwrap());
            
            if matches!(event, LoadingEvent::Ready) {
                break;
            }
        }
        
        yield Ok(Event::default().data("[DONE]"));
    };
    
    Ok(Sse::new(Box::new(stream)))
}
```

---

### Step 7: Update Worker Routes

**File:** `bin/llm-worker-rbee/src/http/routes.rs`

```rust
pub fn create_router<B: InferenceBackend + 'static>(backend: Arc<Mutex<B>>) -> Router {
    Router::new()
        .route("/health", get(health::handle_health::<B>))
        .route("/v1/ready", get(ready::handle_ready::<B>))  // NEW
        .route("/v1/loading/progress", get(loading::handle_loading_progress::<B>))  // NEW
        .route("/v1/inference", post(execute::handle_execute::<B>))  // RENAMED from /execute
        .layer(middleware::from_fn(correlation_middleware))
        .with_state(backend)
}
```

---

### Step 8: Add Loading Progress to InferenceBackend Trait

**File:** `bin/llm-worker-rbee/src/http/backend.rs`

```rust
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn execute(&mut self, prompt: &str, config: &SamplingConfig) -> Result<InferenceResult, String>;
    
    fn is_healthy(&self) -> bool;
    
    /// Get loading progress channel (if model is currently loading)
    fn loading_progress_channel(&self) -> Option<broadcast::Receiver<LoadingEvent>> {
        None  // Default: no loading progress
    }
    
    /// Check if model is ready
    fn is_ready(&self) -> bool {
        true  // Default: always ready
    }
}
```

---

## Testing Plan

### Unit Tests

**File:** `bin/rbee-hive/src/http/models.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_download_progress_stream() {
        let tracker = Arc::new(DownloadTracker::new());
        let download_id = tracker.start_download().await;
        
        // Subscribe
        let mut rx = tracker.subscribe(&download_id).await.unwrap();
        
        // Send events
        tracker.send_progress(&download_id, DownloadEvent::Downloading {
            bytes_downloaded: 1024,
            bytes_total: 2048,
            speed_mbps: 10.0,
        }).await.unwrap();
        
        // Receive
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, DownloadEvent::Downloading { .. }));
    }
}
```

### Integration Tests

**File:** `bin/rbee-hive/tests/sse_streaming.rs` (NEW)

```rust
#[tokio::test]
async fn test_download_progress_sse() {
    // Start server
    let app = create_test_app().await;
    
    // Start download
    let response = app
        .post("/v1/models/download")
        .json(&json!({"model_ref": "hf:test/model"}))
        .await;
    
    let download_id = response.json::<DownloadModelResponse>().download_id;
    
    // Stream progress
    let mut stream = app
        .get(&format!("/v1/models/download/progress?id={}", download_id))
        .await
        .into_sse_stream();
    
    // Verify events
    let event1 = stream.next().await.unwrap();
    assert!(event1.data.contains("downloading"));
    
    let event2 = stream.next().await.unwrap();
    assert!(event2.data.contains("complete"));
    
    let done = stream.next().await.unwrap();
    assert_eq!(done.data, "[DONE]");
}
```

---

## Dependencies

### Add to `bin/rbee-hive/Cargo.toml`:

```toml
[dependencies]
async-stream = "0.3"
uuid = { workspace = true, features = ["v4"] }
```

### Add to `bin/llm-worker-rbee/Cargo.toml`:

```toml
[dependencies]
async-stream = "0.3"
```

---

## Timeline

### Phase 1: Download Progress (2-3 hours)
- [ ] Create `DownloadTracker`
- [ ] Implement `handle_download_progress`
- [ ] Update `handle_download_model` to use tracker
- [ ] Add unit tests

### Phase 2: Loading Progress (2-3 hours)
- [ ] Create `loading.rs` module
- [ ] Add `loading_progress_channel()` to trait
- [ ] Implement `handle_loading_progress`
- [ ] Update routes

### Phase 3: Inference Streaming (1 hour)
- [ ] Rename `/execute` to `/v1/inference`
- [ ] Verify existing SSE works
- [ ] Add integration tests

### Phase 4: Integration Testing (2 hours)
- [ ] End-to-end SSE tests
- [ ] Client reconnection tests
- [ ] Error handling tests

**Total Estimated Time:** 7-9 hours

---

## Success Criteria

✅ **Download Progress:**
- Client can stream download progress via SSE
- Progress bar updates in real-time
- `[DONE]` marker sent at completion

✅ **Loading Progress:**
- Worker streams layer-by-layer loading
- rbee-keeper displays progress bar
- Ready state detected correctly

✅ **Inference Streaming:**
- Tokens stream in real-time
- End event includes metrics
- Client can cancel mid-stream

✅ **Error Handling:**
- Connection drops handled gracefully
- Invalid download IDs return 404
- Stalled streams timeout properly

---

## Industry Standards Summary

### ✅ What We're Following

1. **[DONE] Marker (OpenAI Compatible)**
   - Source: llama.cpp line 4700, mistral.rs line 107
   - Format: `data: [DONE]\n\n`
   - **Required** for OpenAI API compatibility

2. **10-Second Keep-Alive**
   - Source: mistral.rs `DEFAULT_KEEP_ALIVE_INTERVAL_MS = 10_000`
   - Prevents proxy timeouts
   - Configurable via environment variable

3. **State Machine Pattern**
   - Source: mistral.rs `DoneState` enum
   - States: `Running` → `SendingDone` → `Done`
   - Ensures [DONE] is always sent

4. **Error Handling**
   - Source: mistral.rs lines 127-136
   - Send error as data event
   - Still send [DONE] after error
   - Never close stream abruptly

5. **Broadcast Channels**
   - Source: Both llama.cpp and mistral.rs
   - Buffer size: 100 (industry standard)
   - Fan-out to multiple subscribers

6. **Connection State Checking**
   - Source: llama.cpp `sink.is_writable()`
   - Stop generation if client disconnected
   - Prevents wasted compute

### ✅ Axum SSE Features We Use

- `Event::default().json_data()` - Structured JSON events
- `Event::default().data()` - Plain text events
- `Sse::new().keep_alive()` - Keep-alive configuration
- `KeepAlive::new().interval()` - Custom intervals

### ✅ Our Existing Pattern

- Already implemented in `bin/llm-worker-rbee/src/http/execute.rs`
- Uses `futures::stream::iter()` for simple cases
- Uses `async_stream::stream!` for complex state machines
- Type: `Sse<EventStream>` where `EventStream = Box<dyn Stream<...>>`

---

## Notes

1. **Follow Existing Pattern:** All SSE implementations use the same pattern from `llm-worker-rbee/src/http/execute.rs`

2. **Use Axum Built-in SSE:** No custom SSE implementation needed - Axum provides `axum::response::sse`

3. **Broadcast Channels:** Use `tokio::sync::broadcast` for fan-out to multiple subscribers (buffer size: 100)

4. **[DONE] Marker:** Always send `[DONE]` as final event per OpenAI spec

5. **JSON Data:** Use `Event::default().json_data(&event)` for structured events

6. **Error Events:** Send error as data event, then send `[DONE]`

7. **Keep-Alive:** 10-second interval (industry standard from mistral.rs)

8. **State Machine:** Use 3-state pattern: `Running` → `SendingDone` → `Done`

---

## References

### Our Specs
- **Spec:** `bin/.specs/.gherkin/test-001-mvp.md` (Lines 109-120, 243-254, 290-308)
- **Existing Implementation:** `bin/llm-worker-rbee/src/http/execute.rs`
- **README:** Lines 102-103, 149-150, 173-174, 281-285, 321-324, 464-509

### Industry References
- **llama.cpp:** `reference/llama.cpp/tools/server/server.cpp` (Lines 4679-4711)
- **mistral.rs:** `reference/mistral.rs/mistralrs-server-core/src/streaming.rs`
- **mistral.rs:** `reference/mistral.rs/mistralrs-server-core/src/chat_completion.rs` (Lines 88-167)

### Documentation
- **Axum SSE Docs:** https://docs.rs/axum/latest/axum/response/sse/
- **OpenAI Streaming:** https://platform.openai.com/docs/api-reference/streaming
- **SSE Spec:** https://html.spec.whatwg.org/multipage/server-sent-events.html
