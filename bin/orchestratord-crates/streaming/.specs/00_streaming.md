# Streaming SPEC — SSE Relay (STREAM-20xxx)

**Status**: Draft  
**Applies to**: `bin/orchestratord-crates/streaming/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `streaming` crate relays Server-Sent Events (SSE) from workers to clients through orchestratord. It adds orchestrator-specific metadata and handles stream lifecycle.

**Why it exists:**
- Clients connect to orchestratord, not workers directly
- Need to add orchestrator metadata (queue position, predicted start time)
- Handle stream cancellation and error propagation

**What it does:**
- Establish SSE connection to worker endpoint
- Relay worker events to client
- Add orchestrator metadata to events
- Handle stream cancellation (client disconnect → cancel worker job)
- Track active streams

**What it does NOT do:**
- ❌ Execute inference (workers do this)
- ❌ Make scheduling decisions (scheduling does this)
- ❌ Transform token data (relay as-is from worker)

---

## 1. Core Responsibilities

### [STREAM-20001] Worker Connection
The crate MUST establish SSE connection to worker.

### [STREAM-20002] Event Relay
The crate MUST relay events from worker to client.

### [STREAM-20003] Metadata Injection
The crate MUST add orchestrator metadata to events.

### [STREAM-20004] Stream Cancellation
The crate MUST handle cancellation (client disconnect, timeout).

---

## 2. Stream Lifecycle

### [STREAM-20010] Stream Establishment
```rust
pub async fn start_stream(
    &self,
    job_id: String,
    worker_uri: String,
    client_tx: Sender<SseEvent>,
) -> Result<StreamHandle> {
    // 1. Connect to worker SSE endpoint
    let worker_stream = reqwest::get(format!("{}/execute", worker_uri))
        .await?
        .bytes_stream();
    
    // 2. Relay events
    tokio::spawn(async move {
        relay_events(worker_stream, client_tx, job_id).await;
    });
    
    Ok(StreamHandle { job_id })
}
```

### [STREAM-20011] Event Relay Loop
```rust
async fn relay_events(
    mut worker_stream: BytesStream,
    client_tx: Sender<SseEvent>,
    job_id: String,
) {
    while let Some(chunk) = worker_stream.next().await {
        let event = parse_sse_event(chunk)?;
        
        // Add orchestrator metadata
        let enriched = enrich_event(event, &job_id);
        
        // Send to client
        client_tx.send(enriched).await?;
    }
}
```

---

## 3. Event Types

### [STREAM-20020] SSE Event Format
```rust
pub struct SseEvent {
    pub event_type: String,
    pub data: serde_json::Value,
}
```

### [STREAM-20021] Event Types from Worker
- `started` — Inference started
- `token` — Token generated
- `metrics` — Intermediate metrics
- `end` — Inference complete
- `error` — Error occurred

### [STREAM-20022] Orchestrator Events
Additional events from orchestrator:
- `queued` — Job in queue (before dispatch)
- `dispatched` — Job sent to worker

---

## 4. Metadata Injection

### [STREAM-20030] Enrich Events
Add orchestrator context:
```rust
fn enrich_event(event: SseEvent, job_id: &str) -> SseEvent {
    let mut data = event.data;
    
    // Add orchestrator fields
    data["job_id"] = json!(job_id);
    data["orchestrator_ts"] = json!(Utc::now().to_rfc3339());
    
    // For specific event types
    match event.event_type.as_str() {
        "queued" => {
            data["queue_position"] = json!(queue_position);
            data["predicted_start_ms"] = json!(predicted_start);
        }
        "started" => {
            data["queue_wait_ms"] = json!(wait_time);
        }
        _ => {}
    }
    
    SseEvent {
        event_type: event.event_type,
        data,
    }
}
```

---

## 5. Stream Cancellation

### [STREAM-20040] Client Disconnect
When client disconnects:
```rust
async fn handle_client_disconnect(&self, job_id: String, worker_uri: String) {
    // 1. Send cancel request to worker
    let cancel_url = format!("{}/cancel", worker_uri);
    reqwest::post(&cancel_url)
        .json(&json!({ "job_id": job_id }))
        .send()
        .await;
    
    // 2. Close worker stream
    self.close_stream(&job_id);
    
    // 3. Update job state
    self.job_tracker.mark_cancelled(&job_id);
}
```

### [STREAM-20041] Timeout
If stream exceeds timeout (default 5 minutes):
```rust
tokio::select! {
    _ = stream_relay_task => {
        // Stream completed normally
    }
    _ = tokio::time::sleep(Duration::from_secs(300)) => {
        // Timeout
        self.handle_timeout(&job_id, &worker_uri).await;
    }
}
```

### [STREAM-20042] Worker Disconnect
If worker stream closes unexpectedly:
- Emit `error` event to client with reason
- Mark job as failed
- Log error

---

## 6. Error Handling

### [STREAM-20050] Worker Unreachable
If cannot connect to worker:
```rust
SseEvent {
    event_type: "error",
    data: json!({
        "code": "WORKER_UNREACHABLE",
        "message": "Worker did not respond",
        "retriable": true
    })
}
```

### [STREAM-20051] Stream Error
If error during streaming:
- Log error with context
- Send error event to client
- Close stream gracefully

---

## 7. Stream Tracking

### [STREAM-20060] Active Streams
Track active streams:
```rust
pub struct StreamRegistry {
    streams: HashMap<String, ActiveStream>,
}

pub struct ActiveStream {
    pub job_id: String,
    pub worker_uri: String,
    pub started_at: DateTime<Utc>,
    pub client_connected: bool,
}
```

### [STREAM-20061] Stream Cleanup
Clean up completed streams:
- Remove from registry
- Log stream duration
- Emit metrics

---

## 8. Backpressure

### [STREAM-20070] Client Slow Consumer
If client cannot keep up with token stream:
- Buffer events (bounded buffer)
- If buffer full, drop oldest events
- Emit warning metric

### [STREAM-20071] Buffer Size
Default buffer: 100 events
- Enough for ~1 second of fast generation
- Prevents memory exhaustion

---

## 9. Metrics

### [STREAM-20080] Metrics
```rust
pub struct StreamMetrics {
    pub streams_started_total: Counter,
    pub streams_completed_total: Counter,
    pub streams_cancelled_total: Counter,
    pub streams_errored_total: Counter,
    pub stream_duration_ms: Histogram,
    pub active_streams: Gauge,
}
```

---

## 10. Dependencies

### [STREAM-20090] Required Crates
```toml
[dependencies]
tokio = { workspace = true, features = ["full"] }
reqwest = { workspace = true, features = ["stream"] }
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
tracing = { workspace = true }
thiserror = { workspace = true }
futures = { workspace = true }
```

---

## 11. Traceability

**Code**: `bin/orchestratord-crates/streaming/src/`  
**Tests**: `bin/orchestratord-crates/streaming/tests/`  
**Parent**: `bin/orchestratord/.specs/00_orchestratord.md`  
**Used by**: `orchestratord`, `agentic-api`, `platform-api`  
**Spec IDs**: STREAM-20001 to STREAM-20090

---

**End of Specification**
