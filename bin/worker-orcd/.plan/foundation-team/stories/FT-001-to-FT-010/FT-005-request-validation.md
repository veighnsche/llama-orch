# FT-005: Request Validation Framework

**Team**: Foundation-Alpha  
**Sprint**: Sprint 1 - HTTP Foundation  
**Size**: S (1 day)  
**Days**: 6 - 6  
**Spec Ref**: M0-W-1302, WORK-3120

---

## Story Description

Implement comprehensive request validation framework with detailed error reporting. This ensures all invalid requests are rejected before reaching CUDA layer, with clear error messages for debugging.

---

## Acceptance Criteria

- [ ] Validation framework validates all ExecuteRequest fields
- [ ] Validation errors return HTTP 400 with structured JSON error response
- [ ] Error response includes: field name, constraint violated, provided value
- [ ] Multiple validation errors collected and returned together (not fail-fast)
- [ ] Unit tests cover all validation rules and edge cases
- [ ] Integration tests validate error response format
- [ ] Validation logic is reusable across endpoints
- [ ] Custom validators for domain-specific constraints (e.g., temperature range)
- [ ] Validation errors logged at WARN level with correlation ID

---

## Dependencies

### Upstream (Blocks This Story)
- FT-002: Execute endpoint skeleton (Expected completion: Day 2)
- FT-004: Correlation ID middleware for error logging (Expected completion: Day 5)

### Downstream (This Story Blocks)
- FT-006: FFI integration needs validated requests before CUDA calls

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/validation/mod.rs` - Validation framework
- `bin/worker-orcd/src/validation/rules.rs` - Validation rules
- `bin/worker-orcd/src/validation/errors.rs` - Error types and formatting
- `bin/worker-orcd/src/http/execute.rs` - Wire validation to execute handler
- `bin/worker-orcd/Cargo.toml` - Ensure validator crate dependency

### Key Interfaces
```rust
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

#[derive(Debug, Serialize)]
pub struct ValidationErrorResponse {
    pub errors: Vec<FieldError>,
}

#[derive(Debug, Serialize)]
pub struct FieldError {
    pub field: String,
    pub constraint: String,
    pub message: String,
    pub value: Option<String>,
}

pub trait ValidateRequest {
    fn validate_request(&self) -> Result<(), ValidationErrorResponse>;
}

impl ValidateRequest for ExecuteRequest {
    fn validate_request(&self) -> Result<(), ValidationErrorResponse> {
        let mut errors = Vec::new();
        
        // Validate job_id
        if self.job_id.is_empty() {
            errors.push(FieldError {
                field: "job_id".to_string(),
                constraint: "non_empty".to_string(),
                message: "job_id must not be empty".to_string(),
                value: Some(self.job_id.clone()),
            });
        }
        
        // Validate prompt
        if self.prompt.is_empty() || self.prompt.len() > 32768 {
            errors.push(FieldError {
                field: "prompt".to_string(),
                constraint: "length(1..=32768)".to_string(),
                message: format!("prompt must be 1-32768 characters, got {}", self.prompt.len()),
                value: None, // Don't include full prompt in error
            });
        }
        
        // Validate max_tokens
        if self.max_tokens < 1 || self.max_tokens > 2048 {
            errors.push(FieldError {
                field: "max_tokens".to_string(),
                constraint: "range(1..=2048)".to_string(),
                message: format!("max_tokens must be 1-2048, got {}", self.max_tokens),
                value: Some(self.max_tokens.to_string()),
            });
        }
        
        // Validate temperature
        if self.temperature < 0.0 || self.temperature > 2.0 {
            errors.push(FieldError {
                field: "temperature".to_string(),
                constraint: "range(0.0..=2.0)".to_string(),
                message: format!("temperature must be 0.0-2.0, got {}", self.temperature),
                value: Some(self.temperature.to_string()),
            });
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ValidationErrorResponse { errors })
        }
    }
}

// Axum extractor that validates on extraction
pub struct ValidatedJson<T>(pub T);

#[async_trait]
impl<T, S> FromRequest<S> for ValidatedJson<T>
where
    T: DeserializeOwned + ValidateRequest,
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<ValidationErrorResponse>);
    
    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        let Json(value) = Json::<T>::from_request(req, state)
            .await
            .map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ValidationErrorResponse {
                        errors: vec![FieldError {
                            field: "body".to_string(),
                            constraint: "valid_json".to_string(),
                            message: "Request body must be valid JSON".to_string(),
                            value: None,
                        }],
                    }),
                )
            })?;
        
        value.validate_request().map_err(|e| {
            (StatusCode::BAD_REQUEST, Json(e))
        })?;
        
        Ok(ValidatedJson(value))
    }
}
```

### Implementation Notes
- Use `validator` crate for declarative validation where possible
- Custom validators for domain-specific rules (temperature, token limits)
- Collect all validation errors before returning (don't fail on first error)
- Error messages should be actionable: "field X must be Y, got Z"
- Don't include sensitive data in error responses (e.g., full prompt text)
- Log validation failures at WARN level with correlation ID
- Validation should be fast (<1ms for typical requests)
- Consider adding validation metrics (count of validation failures by field)

---

## Testing Strategy

### Unit Tests
- Test job_id validation: empty string rejected
- Test prompt validation: empty and >32768 chars rejected
- Test max_tokens validation: <1 and >2048 rejected
- Test temperature validation: <0.0 and >2.0 rejected
- Test seed validation: all u64 values accepted
- Test multiple errors collected together
- Test error response JSON format
- Test ValidatedJson extractor rejects invalid requests

### Integration Tests
- Test POST /execute with all invalid fields returns 400 with multiple errors
- Test POST /execute with valid request returns 200
- Test error response includes field names and constraints
- Test error response does not include sensitive data
- Test validation errors logged with correlation ID
- Test malformed JSON returns appropriate error

### Manual Verification
1. Start server: `cargo run -- --port 8080`
2. Invalid request: `curl -X POST http://localhost:8080/execute -H "Content-Type: application/json" -d '{"job_id":"","prompt":"","max_tokens":0,"temperature":3.0,"seed":42}'`
3. Verify 400 response with 4 errors (job_id, prompt, max_tokens, temperature)
4. Valid request: `curl -X POST http://localhost:8080/execute -d '{"job_id":"test","prompt":"hello","max_tokens":10,"temperature":0.7,"seed":42}'`
5. Verify 200 response

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (8+ tests)
- [ ] Integration tests passing (6+ tests)
- [ ] Documentation updated (validation framework docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §7.1 Request Validation (M0-W-1302)
- Related Stories: FT-002 (execute endpoint), FT-006 (FFI integration)
- Validator Crate: https://docs.rs/validator/latest/validator/

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities (v0.1.0)

**From**: Narration-Core Team  
**Updated**: 2025-10-04 (v0.1.0 - Production Ready)

### Critical Events to Narrate

#### 1. Single Field Validation Failure (WARN level) ⚠️
```rust
use observability_narration_core::narrate_warn;

narrate_warn(NarrationFields {
    actor: "worker-orcd",
    action: "validation",
    target: req.job_id.clone(),
    correlation_id: Some(correlation_id),
    job_id: Some(req.job_id.clone()),
    error_kind: Some("validation_failed".to_string()),
    human: format!("Validation failed for job {}: {} must be {}", req.job_id, field, constraint),
    ..Default::default()
});
```

#### 2. Multiple Validation Errors (WARN level) ⚠️
```rust
use observability_narration_core::narrate_warn;

narrate_warn(NarrationFields {
    actor: "worker-orcd",
    action: "validation",
    target: req.job_id.clone(),
    correlation_id: Some(correlation_id),
    job_id: Some(req.job_id.clone()),
    error_kind: Some("validation_failed".to_string()),
    human: format!("Validation failed for job {}: {} errors ({})", req.job_id, error_count, field_list),
    ..Default::default()
});
```

#### 3. Validation Success (DEBUG level) 🔍
```rust
use observability_narration_core::narrate_debug;

narrate_debug(NarrationFields {
    actor: "worker-orcd",
    action: "validation",
    target: req.job_id.clone(),
    correlation_id: Some(correlation_id),
    job_id: Some(req.job_id.clone()),
    tokens_in: Some(req.prompt.len() as u64),
    human: format!("Validated request for job {}: {} tokens", req.job_id, req.prompt.len()),
    ..Default::default()
});
```

### Testing with CaptureAdapter

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_validation_failure_narration() {
    let adapter = CaptureAdapter::install();
    
    // Invalid request
    let response = client.post("/execute")
        .json(&invalid_request)
        .send()
        .await?;
    
    // Assert validation failure narrated
    adapter.assert_includes("Validation failed");
    adapter.assert_field("action", "validation");
    adapter.assert_field("error_kind", "validation_failed");
    
    // Verify error details captured
    let captured = adapter.captured();
    assert!(captured[0].human.contains("must be"));
}
```

### Property Testing for Validation

```rust
#[test]
fn property_all_invalid_requests_rejected() {
    let invalid_cases = vec[
        ("empty_job_id", ExecuteRequest { job_id: "", .. }),
        ("empty_prompt", ExecuteRequest { prompt: "", .. }),
        ("max_tokens_zero", ExecuteRequest { max_tokens: 0, .. }),
        ("temperature_negative", ExecuteRequest { temperature: -1.0, .. }),
    ];
    
    for (case_name, request) in invalid_cases {
        let result = validate_request(&request);
        assert!(result.is_err(), "Case {} should fail validation", case_name);
    }
}
```

### Why This Matters

**Validation events** are critical for:
- 🐛 **Client debugging** (which field is wrong?)
- 📊 **API improvement** (which validations fail most?)
- 🔗 **Request tracing** (validation → execution)
- 🚨 **Anomaly detection** (unusual validation patterns)
- 📈 **SLO tracking** (validation failure rate)

### New in v0.1.0
- ✅ **7 logging levels** (WARN for failures, DEBUG for success)
- ✅ **Property tests** for validation invariants
- ✅ **Rich error context** in narration fields
- ✅ **Test assertions** for validation events
- ✅ **Secret redaction** in validation error messages

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Narration Updated**: 2025-10-04 (v0.1.0)

---
Planned by Project Management Team 📋  
*Narration guidance updated by Narration-Core Team 🎀*

---

## 🔍 Testing Team Requirements

**From**: Testing Team (Pre-Development Audit)

### Unit Testing Requirements
- **Test job_id validation: empty string rejected** (non-empty constraint)
- **Test prompt validation: empty rejected** (min length)
- **Test prompt validation: >32768 chars rejected** (max length)
- **Test max_tokens validation: <1 rejected** (min range)
- **Test max_tokens validation: >2048 rejected** (max range)
- **Test temperature validation: <0.0 rejected** (min range)
- **Test temperature validation: >2.0 rejected** (max range)
- **Test seed validation: all u64 values accepted** (no constraints)
- **Test multiple errors collected together** (not fail-fast)
- **Test error response JSON format** (field, constraint, message, value)
- **Test ValidatedJson extractor rejects invalid requests** (Axum integration)
- **Property test**: All invalid combinations rejected with correct errors

### Integration Testing Requirements
- **Test POST /execute with all invalid fields returns 400** (multiple errors)
- **Test POST /execute with valid request returns 200** (happy path)
- **Test error response includes field names and constraints** (error details)
- **Test error response does not include sensitive data** (no full prompt)
- **Test validation errors logged with correlation ID** (observability)
- **Test malformed JSON returns appropriate error** (JSON parse failure)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: All fields invalid
  - Given a request with empty job_id, empty prompt, max_tokens 0, temperature 3.0
  - When I POST to /execute
  - Then I should receive 400 Bad Request
  - And errors array should contain 4 errors
  - And each error should have field, constraint, and message
- **Scenario**: Boundary values accepted
  - Given a request with prompt exactly 32768 chars, max_tokens 2048, temperature 2.0
  - When I POST to /execute
  - Then I should receive 200 OK
- **Scenario**: Malformed JSON rejected
  - Given invalid JSON body
  - When I POST to /execute
  - Then I should receive 400 Bad Request
  - And error should indicate "Request body must be valid JSON"

### Critical Paths to Test
- Validation framework initialization
- Error collection (not fail-fast)
- Error response formatting
- Integration with Axum extractors
- Logging with correlation ID

### Edge Cases
- Prompt exactly at boundaries (1, 32768)
- max_tokens exactly at boundaries (1, 2048)
- Temperature exactly at boundaries (0.0, 2.0)
- Null bytes in strings
- Very large seed values (u64::MAX)
- Missing optional fields vs invalid values

---
Test opportunities identified by Testing Team 🔍
