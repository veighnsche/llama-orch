# Narration Guidance Updated (v0.1.0)

**Date**: 2025-10-04  
**Status**: ✅ **ALL 10 STORY CARDS UPDATED**

---

## Summary

Updated all FT-001 through FT-010 story cards with **narration-core v0.1.0** guidance, reflecting the latest production-ready features and testing capabilities.

---

## Story Cards Updated

### ✅ FT-001: HTTP Server Setup
- Added 7 logging levels (INFO for startup, WARN for shutdown, ERROR for bind failures)
- Added HTTP context propagation for health endpoint
- Added CaptureAdapter testing examples
- Added cute mode and story mode examples

### ✅ FT-002: Execute Endpoint Skeleton
- Added INFO level for request received
- Added WARN level for validation failures
- Added DEBUG level for SSE stream start
- Added HTTP context propagation examples
- Added property testing examples

### ✅ FT-003: SSE Streaming
- Added DEBUG level for stream start
- Added INFO level for stream complete
- Added ERROR level for stream errors
- Added WARN level for client disconnect
- Added token counting and duration tracking
- Added stream lifecycle testing examples

### ✅ FT-004: Correlation ID Middleware
- Added DEBUG level for ID extraction
- Added INFO level for ID generation
- Added WARN level for invalid ID rejection
- Added built-in HTTP context propagation helpers
- Added fast validation (<100ns) examples
- **CRITICAL**: This middleware enables ALL other narration!

### ✅ FT-005: Request Validation
- Added WARN level for validation failures
- Added DEBUG level for validation success
- Added property testing examples
- Added secret redaction in error messages
- Added multi-field validation error tracking

### ✅ FT-006: FFI Interface Definition
- Added INFO level for FFI interface lock milestone
- Added story mode for team coordination
- Added cute mode for milestone celebration
- Added audit trail with auto-injection

### ✅ FT-007: Rust FFI Bindings
- Added ERROR level for FFI call failures
- Added DEBUG level for resource cleanup
- Added ERROR level for null pointer detection
- Added FFI boundary testing examples
- Added memory leak detection guidance

### ✅ FT-008: Error Code System (C++)
- Added ERROR level for exception caught at FFI boundary
- Added testing examples for exception-to-error-code conversion
- Added correlation ID propagation through FFI
- **Note**: C++ doesn't emit narration directly (Rust side only)

### ✅ FT-009: Error Code to Result (Rust)
- Added ERROR level for CUDA errors
- Added WARN level for retriable errors (VRAM OOM)
- Added FATAL level for device loss
- Added retry hints in `retry_after_ms` field
- Added HTTP status code tracking
- Added cute mode for retriable errors

### ✅ FT-010: CUDA Context Init
- Added INFO level for CUDA init start and ready
- Added ERROR level for init failures
- Added DEBUG level for UMA disabled
- Added VRAM tracking fields (`vram_total_mb`, `vram_free_mb`)
- Added duration tracking for init time
- Added cute mode for GPU wake-up

---

## Key Improvements in v0.1.0

### 1. 7 Logging Levels 🎚️
**Before**: Single level narration  
**After**: MUTE, TRACE, DEBUG, INFO, WARN, ERROR, FATAL

**Usage**:
- `narrate()` - INFO level (default)
- `narrate_debug()` - DEBUG level (verbose, low-priority)
- `narrate_warn()` - WARN level (retriable errors, validation failures)
- `narrate_error()` - ERROR level (failures, exceptions)
- `narrate_fatal()` - FATAL level (device loss, worker restart required)

### 2. HTTP Context Propagation 🌐
**Before**: Manual correlation ID handling  
**After**: Built-in helpers

```rust
use observability_narration_core::http::{extract_context_from_headers, inject_context_into_headers};

// Extract from incoming request
let context = extract_context_from_headers(&headers);

// Inject into outgoing response
inject_context_into_headers(&context, &mut response_headers);
```

### 3. Rich Test Assertions ✅
**Before**: Manual event inspection  
**After**: Rich assertion helpers

```rust
adapter.assert_includes("Starting inference");
adapter.assert_field("action", "inference_start");
adapter.assert_correlation_id_present();
adapter.assert_provenance_present();  // NEW
```

### 4. Serial Test Execution 🔒
**Before**: Flaky tests due to global state  
**After**: `#[serial(capture_adapter)]` prevents interference

```rust
#[test]
#[serial(capture_adapter)]
fn test_narration() {
    let adapter = CaptureAdapter::install();
    // Test code
}
```

### 5. Auto-Injection of Provenance 🏷️
**Before**: Manual timestamp and service identity  
**After**: Automatic via `narrate_auto()` (optional)

```rust
// Automatically adds:
// - emitted_by: "worker-orcd@0.1.0"
// - emitted_at_ms: 1696118400000
```

### 6. Property-Based Testing 🧪
**Before**: Example-based tests only  
**After**: Property tests for invariants

```rust
#[test]
fn property_all_invalid_requests_rejected() {
    for invalid_case in invalid_cases {
        assert!(validate(invalid_case).is_err());
    }
}
```

---

## Logging Level Guidelines

| Level | Use For | Examples |
|-------|---------|----------|
| **MUTE** | Silence all output | Testing, benchmarks |
| **TRACE** | Very verbose debugging | Function entry/exit |
| **DEBUG** | Verbose debugging | Health checks, cleanup, config |
| **INFO** | Normal operations | Startup, requests, completion |
| **WARN** | Retriable errors | Validation failures, VRAM OOM, client disconnect |
| **ERROR** | Failures | FFI errors, CUDA errors, exceptions |
| **FATAL** | Critical failures | Device loss, worker restart required |

---

## Testing Pattern

### Standard Test Structure

```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]  // Prevents test interference
fn test_my_feature() {
    let adapter = CaptureAdapter::install();
    
    // Run code that emits narration
    my_function();
    
    // Assert narration captured
    adapter.assert_includes("expected text");
    adapter.assert_field("actor", "worker-orcd");
    adapter.assert_correlation_id_present();
    
    // Verify specific fields
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].action, "expected_action");
}
```

---

## Correlation ID Propagation

### Critical Pattern

**Every narration event** should include `correlation_id`:

```rust
narrate(NarrationFields {
    actor: "worker-orcd",
    action: "some_action",
    target: "some_target".to_string(),
    correlation_id: Some(correlation_id),  // ← CRITICAL
    human: "Some message".to_string(),
    ..Default::default()
});
```

**Why**: Enables request tracing across:
- Orchestrator → Worker → Engine
- Multiple services in distributed system
- Debugging multi-step workflows

---

## Next Steps for Foundation Team

### When Implementing Stories

1. **Import narration-core**:
   ```toml
   [dependencies]
   observability-narration-core = { path = "../../shared-crates/narration-core" }
   
   [dev-dependencies]
   serial_test = "3.0"
   ```

2. **Use appropriate logging levels**:
   - INFO for normal operations
   - WARN for retriable errors
   - ERROR for failures
   - DEBUG for verbose details

3. **Extract correlation IDs from headers**:
   ```rust
   use observability_narration_core::http::extract_context_from_headers;
   let context = extract_context_from_headers(&headers);
   ```

4. **Test with CaptureAdapter**:
   ```rust
   #[test]
   #[serial(capture_adapter)]
   fn test_my_narration() {
       let adapter = CaptureAdapter::install();
       // Test code
   }
   ```

5. **Run tests with test-support feature**:
   ```bash
   cargo test --features test-support
   ```

---

## Questions?

Contact **Narration-Core Team** 🎀 for:
- Logging level guidance
- Testing support
- Correlation ID propagation
- Performance optimization
- Custom narration patterns

**Specification**: `bin/shared-crates/narration-core/.specs/00_narration-core.md`  
**README**: `bin/shared-crates/narration-core/README.md`

---

**All Story Cards Updated**: 2025-10-04 ✅  
**Ready for Foundation Team**: v0.1.0 🚀

---

*Built with diligence, documented with care, delivered with confidence.* ✅

*— The Narration-Core Team 🎀*
