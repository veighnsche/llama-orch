# FT-002: POST /execute Endpoint Skeleton

**Team**: Foundation-Alpha  
**Sprint**: Sprint 1 - HTTP Foundation  
**Size**: S (1 day)  
**Days**: 2 - 2  
**Spec Ref**: M0-W-1300, M0-W-1302, WORK-3040

---

## Story Description

Implement POST /execute endpoint skeleton that accepts inference requests, validates parameters, and returns placeholder SSE stream. This establishes the request/response contract before CUDA integration.

---

## Acceptance Criteria

- [ ] POST /execute endpoint accepts JSON request body
- [ ] Request struct deserializes: `job_id`, `prompt`, `max_tokens`, `temperature`, `seed`
- [ ] Parameter validation: job_id non-empty, prompt 1-32768 chars, max_tokens 1-2048, temperature 0.0-2.0
- [ ] Returns HTTP 400 with error details for invalid parameters
- [ ] Returns HTTP 200 with `Content-Type: text/event-stream` for valid requests
- [ ] Emits placeholder SSE events: `started`, `token` (mock), `end`
- [ ] Unit tests validate request deserialization and validation logic
- [ ] Integration test validates end-to-end request/response flow
- [ ] Error responses include field name and validation constraint

---

## Dependencies

### Upstream (Blocks This Story)
- FT-001: HTTP server infrastructure (Expected completion: Day 1)

### Downstream (This Story Blocks)
- FT-003: SSE streaming implementation needs endpoint structure
- FT-006: FFI integration needs endpoint to wire inference calls

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/http/execute.rs` - Execute endpoint handler
- `bin/worker-orcd/src/http/routes.rs` - Add execute route
- `bin/worker-orcd/src/http/validation.rs` - Request validation logic
- `bin/worker-orcd/src/types/request.rs` - Request/response types
- `bin/worker-orcd/Cargo.toml` - Add dependency: validator crate

### Key Interfaces
```rust
use axum::{Json, response::sse::{Event, Sse}};
use serde::{Deserialize, Serialize};
use validator::Validate;
use futures::stream::Stream;

#[derive(Debug, Deserialize, Validate)]
pub struct ExecuteRequest {
    #[validate(length(min = 1))]
    pub job_id: String,
    
    #[validate(length(min = 1, max = 32768))]
    pub prompt: String,
    
    #[validate(range(min = 1, max = 2048))]
    pub max_tokens: u32,
    
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: f32,
    
    pub seed: u64,
}

#[derive(Debug, Serialize)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
}

pub async fn execute_handler(
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ValidationError>)>;
```

### Implementation Notes
- Use `validator` crate for declarative validation
- Validation errors return HTTP 400 with JSON: `{"field": "prompt", "message": "must be 1-32768 characters"}`
- Placeholder SSE stream emits 3 events: `started`, `token` (with mock text "test"), `end`
- SSE event format: `event: token\ndata: {"t":"test","i":0}\n\n`
- Use `futures::stream::iter()` for placeholder stream
- Log request at DEBUG level with job_id and prompt length
- Temperature validation: exactly 0.0-2.0 (inclusive)
- Seed is u64, no validation needed (all values valid)

---

## Testing Strategy

### Unit Tests
- Test ExecuteRequest deserializes valid JSON
- Test validation rejects empty job_id
- Test validation rejects prompt >32768 chars
- Test validation rejects max_tokens <1 or >2048
- Test validation rejects temperature <0.0 or >2.0
- Test ValidationError serializes correctly

### Integration Tests
- Test POST /execute with valid request returns 200
- Test POST /execute with invalid request returns 400
- Test response Content-Type is text/event-stream
- Test placeholder SSE stream emits started, token, end events
- Test validation error includes field name and message

### Manual Verification
1. Start server: `cargo run -- --port 8080`
2. Valid request: `curl -X POST http://localhost:8080/execute -H "Content-Type: application/json" -d '{"job_id":"test","prompt":"hello","max_tokens":10,"temperature":0.7,"seed":42}'`
3. Verify SSE stream output
4. Invalid request: `curl -X POST http://localhost:8080/execute -d '{"job_id":"","prompt":"x","max_tokens":10,"temperature":0.7,"seed":42}'`
5. Verify 400 error with field details

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (6+ tests)
- [ ] Integration tests passing (5+ tests)
- [ ] Documentation updated (endpoint handler docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §7.1 Inference Endpoint (M0-W-1300)
- Spec: `bin/.specs/01_M0_worker_orcd.md` §7.1 Request Validation (M0-W-1302)
- Related Stories: FT-001 (server), FT-003 (SSE streaming)
- Axum SSE: https://docs.rs/axum/latest/axum/response/sse/

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **Request received** (ACTION_INFERENCE_START)
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: ACTION_INFERENCE_START,
       target: req.job_id.clone(),
       correlation_id: Some(correlation_id),
       tokens_in: Some(req.prompt.len() as u64),
       human: format!("Starting inference for job {}", req.job_id),
       ..Default::default()
   });
   ```

2. **Validation failures** (with specific field)
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: ACTION_INFERENCE_START,
       target: req.job_id.clone(),
       correlation_id: Some(correlation_id),
       error_kind: Some("validation_failed".to_string()),
       human: format!("Validation failed for job {}: {} must be {}", req.job_id, field, constraint),
       ..Default::default()
   });
   ```

**Why this matters**: Request validation failures are common debugging scenarios. Narration helps identify which field failed and why.

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
- **Test ExecuteRequest deserializes valid JSON** (all fields present)
- **Test validation rejects empty job_id** (non-empty constraint)
- **Test validation rejects prompt >32768 chars** (length constraint)
- **Test validation rejects max_tokens <1 or >2048** (range constraint)
- **Test validation rejects temperature <0.0 or >2.0** (range constraint)
- **Test seed accepts all u64 values** (no validation needed)
- **Test ValidationError serializes correctly** (JSON format)
- **Property test**: All invalid parameter combinations rejected

### Integration Testing Requirements
- **Test POST /execute with valid request returns 200** (happy path)
- **Test POST /execute with invalid request returns 400** (validation failure)
- **Test response Content-Type is text/event-stream** (SSE header)
- **Test placeholder SSE stream emits started, token, end events** (event sequence)
- **Test validation error includes field name and message** (error details)
- **Test concurrent requests** (multiple jobs in parallel)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: Valid inference request accepted
  - Given a valid ExecuteRequest with all parameters
  - When I POST to /execute
  - Then I should receive 200 OK
  - And Content-Type should be text/event-stream
  - And SSE stream should emit started event first
- **Scenario**: Invalid job_id rejected
  - Given an ExecuteRequest with empty job_id
  - When I POST to /execute
  - Then I should receive 400 Bad Request
  - And error should indicate "job_id must not be empty"
- **Scenario**: Temperature out of range rejected
  - Given an ExecuteRequest with temperature 3.0
  - When I POST to /execute
  - Then I should receive 400 Bad Request
  - And error should indicate "temperature must be 0.0-2.0"

### Critical Paths to Test
- Request deserialization (JSON → ExecuteRequest struct)
- Validation logic for all parameters
- Error response formatting (field + message)
- SSE stream initialization

### Edge Cases
- Prompt exactly 32768 chars (boundary)
- Temperature exactly 0.0 and 2.0 (boundaries)
- max_tokens exactly 1 and 2048 (boundaries)
- Malformed JSON (invalid UTF-8, truncated)
- Missing required fields

---
Test opportunities identified by Testing Team 🔍
