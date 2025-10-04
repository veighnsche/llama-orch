# FT-003: SSE Streaming Implementation

**Team**: Foundation-Alpha  
**Sprint**: Sprint 1 - HTTP Foundation  
**Size**: M (2 days)  
**Days**: 3 - 4  
**Spec Ref**: M0-W-1310, M0-W-1311, M0-W-1312, WORK-3050

---

## Story Description

Implement complete Server-Sent Events (SSE) streaming with proper event types, ordering, and UTF-8 safety. This provides the real-time token streaming mechanism for inference results.

---

## Acceptance Criteria

- [ ] SSE stream emits 5 event types: `started`, `token`, `metrics`, `end`, `error`
- [ ] Event ordering enforced: `started` → `token*` → (`end` | `error`)
- [ ] Exactly one terminal event per job (either `end` or `error`, never both)
- [ ] Event payloads match spec format (JSON with correct fields)
- [ ] UTF-8 boundary safety: buffer partial multibyte sequences, never emit invalid UTF-8
- [ ] Unit tests validate event serialization and ordering
- [ ] Integration tests validate complete SSE stream lifecycle
- [ ] Error events include error code and human-readable message
- [ ] Stream closes properly after terminal event

---

## Dependencies

### Upstream (Blocks This Story)
- FT-002: Execute endpoint skeleton (Expected completion: Day 2)

### Downstream (This Story Blocks)
- FT-024: HTTP-FFI-CUDA integration test needs real SSE streaming
- FT-006: FFI integration needs to emit SSE events from CUDA callbacks

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/http/sse.rs` - SSE event types and streaming logic
- `bin/worker-orcd/src/http/execute.rs` - Wire SSE stream to execute handler
- `bin/worker-orcd/src/types/events.rs` - Event payload structures
- `bin/worker-orcd/src/util/utf8.rs` - UTF-8 boundary buffer

### Key Interfaces
```rust
use axum::response::sse::Event;
use serde::Serialize;
use std::pin::Pin;
use futures::Stream;

#[derive(Debug, Serialize)]
#[serde(tag = "event", content = "data")]
pub enum InferenceEvent {
    Started {
        job_id: String,
        model: String,
        started_at: String,  // ISO 8601
    },
    Token {
        t: String,  // token text
        i: u32,     // token index
    },
    Metrics {
        tokens_per_sec: f32,
        vram_bytes: u64,
    },
    End {
        tokens_out: u32,
        decode_time_ms: u64,
    },
    Error {
        code: String,
        message: String,
    },
}

pub struct SseStream {
    events: Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>,
}

impl SseStream {
    pub fn new() -> Self;
    pub async fn emit(&mut self, event: InferenceEvent) -> Result<(), SseError>;
    pub async fn close(self) -> Result<(), SseError>;
}

/// UTF-8 boundary-safe buffer for streaming tokens
pub struct Utf8Buffer {
    buffer: Vec<u8>,
}

impl Utf8Buffer {
    pub fn new() -> Self;
    
    /// Push bytes, return complete UTF-8 strings
    pub fn push(&mut self, bytes: &[u8]) -> Vec<String>;
    
    /// Flush remaining bytes (call at end of stream)
    pub fn flush(&mut self) -> Option<String>;
}
```

### Implementation Notes
- Event format: `event: <type>\ndata: <json>\n\n`
- `started` event includes ISO 8601 timestamp: `chrono::Utc::now().to_rfc3339()`
- `token` event uses short field names (`t`, `i`) to minimize bandwidth
- `metrics` event is optional, emitted every N tokens (configurable)
- `error` codes: `VRAM_OOM`, `CANCELLED`, `TIMEOUT`, `INVALID_REQUEST`, `INFERENCE_FAILED`
- UTF-8 buffer handles partial multibyte sequences (e.g., split emoji)
- Terminal event closes stream via `futures::stream::once()`
- Use `tokio::sync::mpsc` channel for event queueing
- Stream should handle backpressure (client slow to read)

---

## Testing Strategy

### Unit Tests
- Test InferenceEvent serializes to correct JSON
- Test Utf8Buffer handles complete UTF-8 strings
- Test Utf8Buffer buffers partial multibyte sequences (split 2-byte, 3-byte, 4-byte chars)
- Test Utf8Buffer flush returns buffered bytes
- Test event ordering validation (started before token, terminal event last)
- Test exactly one terminal event enforced

### Integration Tests
- Test SSE stream emits started event first
- Test SSE stream emits multiple token events
- Test SSE stream emits end event last
- Test SSE stream closes after terminal event
- Test error event terminates stream (no end event after error)
- Test UTF-8 safety with emoji and multibyte characters
- Test stream handles client disconnect gracefully

### Manual Verification
1. Start server with mock inference
2. Curl with SSE: `curl -N http://localhost:8080/execute -d '{"job_id":"test","prompt":"hello","max_tokens":5,"temperature":0.7,"seed":42}'`
3. Verify event sequence: started → token → token → ... → end
4. Test with emoji prompt: `{"prompt":"Hello 👋 世界"}`
5. Verify UTF-8 correctness in output

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing (7+ tests)
- [ ] Documentation updated (SSE module docs, UTF-8 buffer docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §7.2 SSE Streaming (M0-W-1310, M0-W-1311, M0-W-1312)
- Related Stories: FT-002 (execute endpoint), FT-024 (integration test)
- SSE Spec: https://html.spec.whatwg.org/multipage/server-sent-events.html
- UTF-8 Spec: https://tools.ietf.org/html/rfc3629

---

## 🎀 Narration Opportunities (v0.2.0)

**From**: Narration-Core Team  
**Updated**: 2025-10-04 (v0.2.0 - Production Ready with Builder Pattern & Axum Middleware)

### Critical Events to Narrate

#### 1. Stream Started (DEBUG level) 🔍
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD};

// NEW v0.2.0: Builder with debug level
Narration::new(ACTOR_WORKER_ORCD, "stream_start", &job_id)
    .human(format!("SSE stream started for job {}", job_id))
    .correlation_id(correlation_id)
    .job_id(&job_id)
    .emit_debug();  // ← DEBUG level
```

#### 2. Stream Completed (INFO level) ✅
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD, ACTION_INFERENCE_COMPLETE};

// NEW v0.2.0: Builder pattern
Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_COMPLETE, &job_id)
    .human(format!("Completed inference: {} tokens in {} ms", token_count, elapsed.as_millis()))
    .correlation_id(correlation_id)
    .job_id(&job_id)
    .tokens_out(token_count)
    .duration_ms(elapsed.as_millis() as u64)
    .cute(format!("Worker finished the story! 📖✨ {} tokens in {}ms!", token_count, elapsed.as_millis()))
    .emit();
```

#### 3. Stream Error (ERROR level) 🚨
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD};

// NEW v0.2.0: Builder with error level
Narration::new(ACTOR_WORKER_ORCD, "stream_error", &job_id)
    .human(format!("SSE stream error for job {}: {}", job_id, error_message))
    .correlation_id(correlation_id)
    .job_id(&job_id)
    .error_kind(&error_code)
    .emit_error();  // ← ERROR level
```

#### 4. Client Disconnect (WARN level) ⚠️
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD};

// NEW v0.2.0: Builder with warn level
Narration::new(ACTOR_WORKER_ORCD, "stream_disconnect", &job_id)
    .human(format!("Client disconnected after {} tokens", tokens_sent_so_far))
    .correlation_id(correlation_id)
    .job_id(&job_id)
    .tokens_out(tokens_sent_so_far)
    .emit_warn();  // ← WARN level
```

### Testing with CaptureAdapter

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_sse_stream_narration() {
    let adapter = CaptureAdapter::install();
    
    // Start SSE stream
    let stream = create_sse_stream(job_id, correlation_id);
    
    // Consume stream
    while let Some(event) = stream.next().await {
        // Process event
    }
    
    // Assert narration captured
    let captured = adapter.captured();
    assert!(captured.iter().any(|e| e.action == "stream_start"));
    assert!(captured.iter().any(|e| e.action == "inference_complete"));
    
    // Verify all events have same correlation ID
    let correlation_ids: Vec<_> = captured.iter()
        .filter_map(|e| e.correlation_id.as_ref())
        .collect();
    assert!(correlation_ids.iter().all(|id| id == &correlation_id));
}
```

### Why This Matters

**SSE streaming events** are critical for:
- 🔍 Tracking stream lifecycle (start → tokens → complete)
- 🐛 Diagnosing client disconnect issues
- 📈 Measuring token generation rates
- 🔗 Correlating streaming across orchestrator → worker
- 🚨 Alerting on stream error patterns

### New in v0.2.0
- ✅ **7 logging levels** (DEBUG for stream start, INFO for complete, ERROR for failures)
- ✅ **Correlation ID propagation** through entire stream lifecycle
- ✅ **Token counting** in narration fields (`tokens_out`)
- ✅ **Duration tracking** in narration fields (`duration_ms`)
- ✅ **Rich test assertions** for stream event sequences

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋  
*Narration guidance added by Narration-Core Team 🎀*

---

## 🔍 Testing Team Requirements

**From**: Testing Team (Pre-Development Audit)

### Unit Testing Requirements
- **Test InferenceEvent serializes to correct JSON** (all event types)
- **Test Utf8Buffer handles complete UTF-8 strings** (ASCII, emoji, CJK)
- **Test Utf8Buffer buffers partial 2-byte sequences** (split UTF-8)
- **Test Utf8Buffer buffers partial 3-byte sequences** (emoji split)
- **Test Utf8Buffer buffers partial 4-byte sequences** (rare chars)
- **Test Utf8Buffer flush returns buffered bytes** (end of stream)
- **Test event ordering validation** (started before token, terminal last)
- **Test exactly one terminal event enforced** (end XOR error)
- **Property test**: UTF-8 boundary safety with random byte splits

### Integration Testing Requirements
- **Test SSE stream emits started event first** (event ordering)
- **Test SSE stream emits multiple token events** (streaming)
- **Test SSE stream emits end event last** (terminal event)
- **Test SSE stream closes after terminal event** (connection cleanup)
- **Test error event terminates stream** (no end after error)
- **Test UTF-8 safety with emoji** (👋 split across chunks)
- **Test UTF-8 safety with CJK characters** (世界 split across chunks)
- **Test stream handles client disconnect gracefully** (broken pipe)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: Complete SSE stream lifecycle
  - Given an inference request
  - When the stream starts
  - Then I should receive a started event
  - And I should receive token events
  - And I should receive an end event
  - And the stream should close
- **Scenario**: UTF-8 emoji handling
  - Given a prompt with emoji "👋🌍"
  - When tokens are streamed
  - Then all emoji should be valid UTF-8
  - And no partial multibyte sequences should be emitted
- **Scenario**: Error terminates stream
  - Given an inference that fails
  - When an error occurs
  - Then I should receive an error event
  - And I should NOT receive an end event
  - And the stream should close

### Critical Paths to Test
- Event ordering enforcement (started → token* → terminal)
- UTF-8 boundary detection and buffering
- Terminal event exclusivity (end XOR error)
- Stream cleanup on client disconnect

### Edge Cases
- Empty token stream (started → end immediately)
- Single token stream
- Very long tokens (>256 bytes)
- Rapid token generation (backpressure)
- Client disconnect mid-stream
- Multiple error events (should only emit first)

---
Test opportunities identified by Testing Team 🔍
