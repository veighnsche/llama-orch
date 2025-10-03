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

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
