# FT-009: Error Code to Result Conversion (Rust)

**Team**: Foundation-Alpha  
**Sprint**: Sprint 2 - FFI Layer  
**Size**: S (1 day)  
**Days**: 15 - 15  
**Spec Ref**: M0-W-1050, M0-W-1500

---

## Story Description

Implement Rust error type conversion from C++ error codes to typed Result enums. This provides idiomatic Rust error handling with proper error context and HTTP status code mapping.

---

## Acceptance Criteria

- [ ] CudaError enum with variants for all C++ error codes
- [ ] From<i32> implementation converts error codes to CudaError
- [ ] Display and Error trait implementations for CudaError
- [ ] HTTP status code mapping (500 for internal, 400 for invalid params, 503 for OOM)
- [ ] Error context preservation (original error code + message)
- [ ] Unit tests validate error conversion
- [ ] Integration with Axum error responses
- [ ] Error logging with structured fields

---

## Dependencies

### Upstream (Blocks This Story)
- FT-007: Rust FFI bindings (Expected completion: Day 13)
- FT-008: C++ error code system (Expected completion: Day 14)

### Downstream (This Story Blocks)
- FT-010: CUDA context init needs error handling
- FT-002: Execute endpoint needs to return errors

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/src/cuda/error.rs` - Rust error types (extend from FT-007)
- `bin/worker-orcd/src/http/error.rs` - HTTP error response mapping
- `bin/worker-orcd/src/types/error.rs` - Shared error types

### Key Interfaces
```rust
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use std::fmt;

#[derive(Debug, Clone, thiserror::Error)]
pub enum CudaError {
    #[error("Invalid CUDA device: {0}")]
    InvalidDevice(String),
    
    #[error("Out of VRAM: {0}")]
    OutOfMemory(String),
    
    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),
    
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),
    
    #[error("VRAM residency check failed: {0}")]
    VramResidencyFailed(String),
    
    #[error("Unknown CUDA error: {0}")]
    Unknown(String),
}

impl CudaError {
    /// Convert from C error code to Rust error
    pub fn from_code(code: i32) -> Self {
        use std::ffi::CStr;
        
        // Get error message from C layer
        let msg_ptr = unsafe { crate::cuda::ffi::cuda_error_message(code) };
        let msg = if msg_ptr.is_null() {
            format!("Error code {}", code)
        } else {
            unsafe { CStr::from_ptr(msg_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        
        match code {
            1 => Self::InvalidDevice(msg),
            2 => Self::OutOfMemory(msg),
            3 => Self::ModelLoadFailed(msg),
            4 => Self::InferenceFailed(msg),
            5 => Self::InvalidParameter(msg),
            6 => Self::KernelLaunchFailed(msg),
            7 => Self::VramResidencyFailed(msg),
            _ => Self::Unknown(msg),
        }
    }
    
    /// Get error code for this error
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidDevice(_) => "INVALID_DEVICE",
            Self::OutOfMemory(_) => "VRAM_OOM",
            Self::ModelLoadFailed(_) => "MODEL_LOAD_FAILED",
            Self::InferenceFailed(_) => "INFERENCE_FAILED",
            Self::InvalidParameter(_) => "INVALID_PARAMETER",
            Self::KernelLaunchFailed(_) => "KERNEL_LAUNCH_FAILED",
            Self::VramResidencyFailed(_) => "VRAM_RESIDENCY_FAILED",
            Self::Unknown(_) => "UNKNOWN",
        }
    }
    
    /// Get HTTP status code for this error
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::InvalidParameter(_) => StatusCode::BAD_REQUEST,
            Self::OutOfMemory(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::InvalidDevice(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::ModelLoadFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::InferenceFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::KernelLaunchFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::VramResidencyFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Unknown(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
    
    /// Check if error is retriable
    pub fn is_retriable(&self) -> bool {
        matches!(self, Self::OutOfMemory(_))
    }
}

impl From<i32> for CudaError {
    fn from(code: i32) -> Self {
        Self::from_code(code)
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retriable: Option<bool>,
}

impl IntoResponse for CudaError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let code = self.code().to_string();
        let message = self.to_string();
        let retriable = if self.is_retriable() { Some(true) } else { None };
        
        // Log error with structured fields
        tracing::error!(
            error_code = %code,
            error_message = %message,
            status_code = %status.as_u16(),
            retriable = retriable.unwrap_or(false),
            "CUDA error occurred"
        );
        
        let body = Json(ErrorResponse {
            code,
            message,
            retriable,
        });
        
        (status, body).into_response()
    }
}

// Wrapper type for SSE error events
#[derive(Debug, Serialize)]
pub struct SseError {
    pub code: String,
    pub message: String,
}

impl From<CudaError> for SseError {
    fn from(err: CudaError) -> Self {
        Self {
            code: err.code().to_string(),
            message: err.to_string(),
        }
    }
}
```

### Implementation Notes
- Error codes are stable strings (e.g., "VRAM_OOM", "INVALID_DEVICE")
- HTTP status codes follow REST conventions (4xx client, 5xx server)
- Only OOM errors are retriable (orchestrator can retry on different worker)
- Error logging includes structured fields for observability
- SSE error events use same error codes as HTTP responses
- Error messages preserve context from C++ layer
- thiserror crate provides Display and Error trait implementations

---

## Testing Strategy

### Unit Tests
- Test CudaError::from_code() for all error codes
- Test CudaError::code() returns correct string codes
- Test CudaError::status_code() returns correct HTTP codes
- Test CudaError::is_retriable() for OOM vs other errors
- Test ErrorResponse serialization
- Test SseError conversion from CudaError
- Test From<i32> trait implementation

### Integration Tests
- Test HTTP endpoint returns correct status code for CUDA errors
- Test SSE stream emits error event with correct format
- Test error logging includes structured fields
- Test retriable flag in error responses

### Manual Verification
1. Trigger CUDA error (e.g., invalid device)
2. Verify HTTP response: `curl http://localhost:8080/execute -d '...'`
3. Check response status code and error JSON
4. Verify logs include error_code and error_message fields

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review for agents)
- [ ] Unit tests passing (7+ tests)
- [ ] Integration tests passing (4+ tests)
- [ ] Documentation updated (error type docs, conversion docs)
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §9 Error Handling (M0-W-1500, M0-W-1510)
- Related Stories: FT-007 (FFI bindings), FT-008 (C++ errors)
- Axum Error Handling: https://docs.rs/axum/latest/axum/error_handling/

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities (v0.2.0)

**From**: Narration-Core Team  
**Updated**: 2025-10-04 (v0.2.0 - Production Ready with Builder Pattern & Axum Middleware)

### Critical Events to Narrate

#### 1. CUDA Error Converted to HTTP Response (ERROR level) 🚨
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD, ACTION_INFERENCE_ERROR};

// NEW v0.2.0: Builder with error level
Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_ERROR, &job_id)
    .human(format!("CUDA error for job {}: {} (HTTP {})", job_id, error, error.status_code().as_u16()))
    .correlation_id(correlation_id)
    .job_id(&job_id)
    .error_kind(&error.code())
    .emit_error();  // ← ERROR level
```

#### 2. Retriable Error Detected (WARN level) ⚠️
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD, ACTION_INFERENCE_ERROR};

// NEW v0.2.0: Builder with warn level
Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_ERROR, &job_id)
    .human(format!("VRAM OOM for job {} (retriable after 5s)", job_id))
    .correlation_id(correlation_id)
    .job_id(&job_id)
    .error_kind("VramOOM")
    .retry_after_ms(5000)
    .cute("Worker's memory is full! 🧠💎 Need to make room first...")
    .emit_warn();  // ← WARN level
```

#### 3. Fatal Error (ERROR level - FATAL maps to ERROR in tracing) 🔥
```rust
use observability_narration_core::{Narration, ACTOR_WORKER_ORCD, ACTION_INFERENCE_ERROR};

// NEW v0.2.0: Builder with error level (FATAL maps to ERROR)
Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_ERROR, &job_id)
    .human(format!("GPU device lost for job {} (fatal, worker restart required)", job_id))
    .correlation_id(correlation_id)
    .job_id(&job_id)
    .error_kind("DeviceLost")
    .emit_error();  // ← ERROR level (tracing doesn't have FATAL)
```

### Testing with CaptureAdapter

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_cuda_error_conversion_narration() {
    let adapter = CaptureAdapter::install();
    
    // Trigger CUDA error
    let result = cuda_context.init(-1);
    
    // Assert error narrated
    adapter.assert_includes("CUDA error");
    adapter.assert_field("action", "inference_error");
    adapter.assert_field("error_kind", "INVALID_DEVICE");
    
    // Verify HTTP status code in human message
    let captured = adapter.captured();
    assert!(captured[0].human.contains("HTTP 400"));
}
```

### Why This Matters

**Error conversion events** are critical for:
- 📈 **Error rate tracking** (CUDA → HTTP)
- 🔁 **Retry logic** (retriable vs. fatal)
- 🐛 **Client debugging** (meaningful HTTP errors)
- 🚨 **Alerting** on fatal errors
- 📊 **SLO tracking** (error rate by type)

### New in v0.2.0
- ✅ **7 logging levels** (WARN for retriable, ERROR for failures, FATAL for device loss)
- ✅ **Retry hints** in `retry_after_ms` field
- ✅ **HTTP status code** in human message
- ✅ **Test assertions** for error conversion
- ✅ **Property tests** for error mapping invariants

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Narration Updated**: 2025-10-04 (v0.2.0)

---
Planned by Project Management Team 📋  
*Narration guidance updated by Narration-Core Team 🎀*

---

## 🔍 Testing Team Requirements

**From**: Testing Team (Pre-Development Audit)

### Unit Testing Requirements
- **Test CudaError::from_code() for all error codes** (1-7, 99)
- **Test CudaError::code() returns correct string codes** ("INVALID_DEVICE", "VRAM_OOM", etc.)
- **Test CudaError::status_code() returns correct HTTP codes** (400, 500, 503)
- **Test CudaError::is_retriable() for OOM vs other errors** (only OOM is retriable)
- **Test ErrorResponse serialization** (JSON format with code, message, retriable)
- **Test SseError conversion from CudaError** (SSE event format)
- **Test From<i32> trait implementation** (automatic conversion)
- **Property test**: All error codes map to valid CudaError variants

### Integration Testing Requirements
- **Test HTTP endpoint returns correct status code for CUDA errors** (400/500/503)
- **Test SSE stream emits error event with correct format** (code + message)
- **Test error logging includes structured fields** (error_code, error_message, status_code)
- **Test retriable flag in error responses** (only OOM has retriable: true)
- **Test error context preservation** (original C++ message included)

### BDD Testing Requirements (VERY IMPORTANT)
- **Scenario**: Invalid parameter error returns 400
  - Given a CUDA invalid parameter error (code 5)
  - When converted to HTTP response
  - Then status code should be 400 Bad Request
  - And error code should be "INVALID_PARAMETER"
- **Scenario**: Out of memory error returns 503
  - Given a CUDA out of memory error (code 2)
  - When converted to HTTP response
  - Then status code should be 503 Service Unavailable
  - And error code should be "VRAM_OOM"
  - And retriable should be true
- **Scenario**: Inference failed error returns 500
  - Given a CUDA inference failed error (code 4)
  - When converted to HTTP response
  - Then status code should be 500 Internal Server Error
  - And error code should be "INFERENCE_FAILED"
  - And retriable should be absent or false

### Critical Paths to Test
- Error code to CudaError conversion
- CudaError to HTTP status code mapping
- Retriable error detection (OOM only)
- Error logging with structured fields
- SSE error event formatting

### Edge Cases
- Unknown error code (not in enum)
- Error code 0 (CUDA_SUCCESS, should not happen)
- Very long error messages from C++
- NULL error message pointer from C++
- Multiple error conversions in sequence

---
Test opportunities identified by Testing Team 🔍
