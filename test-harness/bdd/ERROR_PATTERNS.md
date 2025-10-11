# Error Handling Patterns - Implementation Guide

**TEAM-075**  
**Date:** 2025-10-11  
**Source:** Industry research + TEAM-074 foundation

---

## Pattern 1: Retry with Exponential Backoff

**Source:** Industry standard (ollama, distributed systems)  
**Use Case:** Transient network errors, temporary resource unavailability

### Implementation

```rust
// TEAM-075: Exponential backoff with jitter
#[then(expr = "rbee-hive retries with exponential backoff")]
pub async fn then_retry_exponential_backoff(world: &mut World) {
    // Retry schedule: 1s, 2s, 4s, 8s, 16s (max 5 attempts)
    world.last_error = Some(ErrorResponse {
        code: "RETRY_BACKOFF".to_string(),
        message: "Retrying with exponential backoff".to_string(),
        details: Some(json!({
            "attempt": 1,
            "max_attempts": 5,
            "next_retry_seconds": 1,
            "backoff_strategy": "exponential_with_jitter",
            "schedule": [1, 2, 4, 8, 16]
        })),
    });
    tracing::info!("✅ Exponential backoff initiated");
}
```

### Key Points
- Start with 1 second delay
- Double delay each attempt (exponential)
- Add jitter (random 0-1000ms) to prevent thundering herd
- Max 5 attempts before giving up
- Log each retry attempt

---

## Pattern 2: Circuit Breaker

**Source:** llama.cpp server, distributed systems  
**Use Case:** Prevent cascade failures, stop trying after repeated failures

### Implementation

```rust
// TEAM-075: Circuit breaker pattern
#[then(expr = "circuit breaker opens after {int} failures")]
pub async fn then_circuit_breaker_opens(world: &mut World, failures: u8) {
    world.last_error = Some(ErrorResponse {
        code: "CIRCUIT_BREAKER_OPEN".to_string(),
        message: format!("Circuit breaker opened after {} consecutive failures", failures),
        details: Some(json!({
            "failure_count": failures,
            "failure_threshold": 5,
            "cooldown_seconds": 60,
            "state": "open",
            "suggested_action": "Wait for cooldown period before retrying"
        })),
    });
    tracing::error!("🔴 Circuit breaker OPEN after {} failures", failures);
}
```

### States
- **Closed:** Normal operation, requests pass through
- **Open:** Failing, reject immediately without trying
- **Half-Open:** Testing if recovered, allow one request

### Key Points
- Open after 5 consecutive failures
- Cooldown period: 60 seconds
- After cooldown, enter half-open state
- One success closes circuit, one failure reopens

---

## Pattern 3: FAIL FAST on GPU Errors (rbee policy)

**Source:** rbee architectural decision  
**Use Case:** GPU failures, clear error reporting

**CRITICAL POLICY:** NO FALLBACK - FAIL FAST

### Implementation

```rust
// TEAM-075: FAIL FAST on GPU errors
#[when(expr = "CUDA device {int} fails")]
pub async fn when_cuda_device_fails(world: &mut World, device: u8) {
    world.last_exit_code = Some(1);
    world.last_error = Some(ErrorResponse {
        code: "CUDA_DEVICE_FAILED".to_string(),
        message: format!("CUDA device {} initialization failed", device),
        details: Some(json!({
            "device": device,
            "suggested_action": "Check GPU drivers, verify device availability, or explicitly select CPU backend"
        })),
    });
    tracing::error!("❌ CUDA device {} FAILED - exiting immediately", device);
}

#[when(expr = "GPU VRAM is exhausted")]
pub async fn when_gpu_vram_exhausted(world: &mut World) {
    world.last_exit_code = Some(1);
    world.last_error = Some(ErrorResponse {
        code: "GPU_VRAM_EXHAUSTED".to_string(),
        message: "GPU out of memory: 8GB required, 6GB available".to_string(),
        details: Some(json!({
            "required_vram_gb": 8,
            "available_vram_gb": 6,
            "model": "llama-3.1-8b",
            "device": 0,
            "suggested_action": "Use smaller model or explicitly select CPU backend",
            "alternative_models": ["llama-3.1-3b", "phi-3-mini"]
        })),
    });
    tracing::error!("❌ GPU VRAM exhausted - FAILING FAST");
}

#[then(expr = "rbee-hive fails immediately")]
pub async fn then_gpu_fails_immediately(world: &mut World) {
    assert_eq!(world.last_exit_code, Some(1), "Expected exit code 1");
    assert!(world.last_error.is_some(), "Expected error to be set");
    tracing::info!("✅ FAIL FAST verified: exit code 1, no fallback attempted");
}
```

### FAIL FAST Policy
- **NO automatic backend fallback**
- **NO CPU fallback**
- **NO graceful degradation**
- Exit code 1 immediately on GPU error
- User must explicitly choose backend

### Key Points
- Detect GPU errors early
- Report clear, actionable error
- Exit immediately with code 1
- **NEVER attempt automatic fallback**

---

## Pattern 4: Resource Cleanup Guarantees

**Source:** Rust RAII, candle-vllm, mistral.rs  
**Use Case:** Prevent resource leaks, ensure cleanup

### Implementation

```rust
// TEAM-075: Resource cleanup verification
#[then(expr = "all resources are released")]
pub async fn then_resources_released(world: &mut World) {
    // Verify cleanup occurred
    let leaked_processes = world.rbee_hive_processes.len() + world.worker_processes.len();
    let temp_files = world.temp_dir.as_ref()
        .and_then(|d| std::fs::read_dir(d.path()).ok())
        .map(|dir| dir.count())
        .unwrap_or(0);
    
    if leaked_processes == 0 && temp_files == 0 {
        world.last_exit_code = Some(0);
        tracing::info!("✅ All resources released: 0 processes, 0 temp files");
    } else {
        world.last_exit_code = Some(1);
        world.last_error = Some(ErrorResponse {
            code: "RESOURCE_LEAK_DETECTED".to_string(),
            message: "Resource leak detected after cleanup".to_string(),
            details: Some(json!({
                "leaked_processes": leaked_processes,
                "leaked_temp_files": temp_files,
                "suggested_action": "Manual cleanup required"
            })),
        });
        tracing::error!("❌ Resource leak: {} processes, {} files", 
                       leaked_processes, temp_files);
    }
}
```

### TEAM-074 Critical Lesson

```rust
// ❌ NEVER block in Drop - causes hangs!
impl Drop for MyResource {
    fn drop(&mut self) {
        // BAD: Blocking wait
        std::thread::sleep(Duration::from_secs(5));
    }
}

// ✅ GOOD: Non-blocking cleanup
impl Drop for MyResource {
    fn drop(&mut self) {
        // Let OS handle cleanup
        // Or use explicit cleanup method before drop
    }
}
```

### Key Points
- Use RAII (Drop trait) for automatic cleanup
- Never block in Drop implementation
- Verify cleanup with leak detection
- Explicit cleanup methods for critical resources

---

## Pattern 5: Actionable Error Messages

**Source:** candle-vllm APIError, industry best practice  
**Use Case:** All error scenarios

### Implementation

```rust
// ❌ BAD - Vague error
ErrorResponse {
    code: "ERROR".to_string(),
    message: "Something went wrong".to_string(),
    details: None,
}

// ✅ GOOD - Actionable error
ErrorResponse {
    code: "GPU_VRAM_EXHAUSTED".to_string(),
    message: "GPU out of memory: 8GB required, 6GB available".to_string(),
    details: Some(json!({
        "required_vram_gb": 8,
        "available_vram_gb": 6,
        "model": "llama-3.1-8b",
        "device": 0,
        "suggested_action": "Use smaller model or CPU backend",
        "alternative_models": ["llama-3.1-3b", "phi-3-mini"]
    })),
}
```

### Required Fields
- **code:** Machine-readable error code (UPPER_SNAKE_CASE)
- **message:** Human-readable description
- **details.suggested_action:** What user should do next
- **details.retry_after_seconds:** When to retry (if applicable)
- **details.resource_info:** Current resource state (if applicable)

### Key Points
- Error codes should be specific and searchable
- Messages should explain what happened and why
- Details should include actionable next steps
- Avoid technical jargon in messages

---

## Pattern 6: HTTP Status Code Mapping

**Source:** candle-vllm ChatResponder, REST API standards  
**Use Case:** API error responses

### Implementation

```rust
// TEAM-075: HTTP status code mapping
#[then(expr = "worker returns {int} with error")]
pub async fn then_returns_status_with_error(world: &mut World, status: u16) {
    world.last_http_status = Some(status);
    
    let (code, message) = match status {
        400 => ("VALIDATION_ERROR", "Invalid request parameters"),
        401 => ("UNAUTHORIZED", "Authentication required"),
        403 => ("FORBIDDEN", "Access denied"),
        404 => ("NOT_FOUND", "Resource not found"),
        429 => ("RATE_LIMITED", "Too many requests"),
        500 => ("INTERNAL_ERROR", "Internal server error"),
        502 => ("BAD_GATEWAY", "Upstream service error"),
        503 => ("SERVICE_UNAVAILABLE", "Service temporarily unavailable"),
        504 => ("GATEWAY_TIMEOUT", "Upstream service timeout"),
        _ => ("HTTP_ERROR", "HTTP error"),
    };
    
    world.last_error = Some(ErrorResponse {
        code: code.to_string(),
        message: message.to_string(),
        details: Some(json!({
            "http_status": status,
            "retry_after_seconds": if status == 503 { Some(30) } else { None }
        })),
    });
    
    tracing::error!("HTTP {} {}: {}", status, code, message);
}
```

### Status Code Guidelines

**4xx - Client Errors (Don't Retry)**
- 400 Bad Request - Validation error
- 401 Unauthorized - Auth required
- 403 Forbidden - Auth failed
- 404 Not Found - Resource doesn't exist
- 429 Too Many Requests - Rate limited (retry after delay)

**5xx - Server Errors (Retry OK)**
- 500 Internal Server Error - System error
- 502 Bad Gateway - Upstream error
- 503 Service Unavailable - Capacity/overload (retry after delay)
- 504 Gateway Timeout - Upstream timeout

---

## Pattern 7: Timeout Cascades

**Source:** candle-vllm async patterns, distributed systems  
**Use Case:** Prevent system-wide hangs

### Implementation

```rust
// TEAM-075: Timeout cascade handling
#[given(expr = "inference timeout is {int}s")]
pub async fn given_inference_timeout(world: &mut World, timeout: u16) {
    // Store timeout expectation
    world.last_command = Some(format!("timeout:{}", timeout));
    tracing::debug!("Inference timeout set to {}s", timeout);
}

#[when(expr = "inference exceeds timeout")]
pub async fn when_inference_timeout(world: &mut World) {
    world.last_exit_code = Some(124); // Standard timeout exit code
    world.last_error = Some(ErrorResponse {
        code: "INFERENCE_TIMEOUT".to_string(),
        message: "Inference exceeded timeout".to_string(),
        details: Some(json!({
            "timeout_seconds": 30,
            "elapsed_seconds": 35,
            "suggested_action": "Increase timeout or use smaller model"
        })),
    });
    tracing::error!("⏱️  Inference timeout exceeded");
}

#[then(expr = "worker cancels inference gracefully")]
pub async fn then_cancel_gracefully(world: &mut World) {
    world.last_exit_code = Some(0); // Graceful cancellation
    tracing::info!("✅ Inference cancelled gracefully, resources released");
}
```

### Timeout Hierarchy
1. **Request timeout:** 30s (user-facing)
2. **Inference timeout:** 25s (leaves 5s for cleanup)
3. **Cleanup timeout:** 5s (force kill after this)

### Key Points
- Set timeouts at each layer
- Inner timeout < outer timeout (leave cleanup time)
- Graceful cancellation releases resources
- Force kill only as last resort

---

## Pattern 8: Concurrent Request Limits

**Source:** ollama, mistral.rs, production systems  
**Use Case:** Prevent resource exhaustion

### Implementation

```rust
// TEAM-075: Concurrent request limit enforcement
#[given(expr = "worker has max {int} concurrent requests")]
pub async fn given_max_concurrent(world: &mut World, max: u32) {
    world.last_command = Some(format!("max_concurrent:{}", max));
    tracing::debug!("Max concurrent requests: {}", max);
}

#[when(expr = "request exceeds capacity")]
pub async fn when_exceeds_capacity(world: &mut World) {
    world.last_http_status = Some(503);
    world.last_error = Some(ErrorResponse {
        code: "SERVICE_UNAVAILABLE".to_string(),
        message: "Worker at maximum capacity".to_string(),
        details: Some(json!({
            "current_requests": 10,
            "max_concurrent": 10,
            "retry_after_seconds": 30,
            "suggested_action": "Retry after 30 seconds or use different worker"
        })),
    });
    tracing::warn!("⚠️  Worker at capacity, rejecting with 503");
}

#[then(expr = "request is rejected with 503")]
pub async fn then_rejected_503(world: &mut World) {
    assert_eq!(world.last_http_status, Some(503), "Expected 503 status");
    assert!(world.last_error.is_some(), "Expected error to be set");
    tracing::info!("✅ Request rejected with 503 Service Unavailable");
}
```

### Capacity Management
- **Max concurrent:** Configurable per worker
- **Queue depth:** Limited to prevent memory exhaustion
- **Admission control:** Reject early if over capacity
- **Retry-After header:** Tell client when to retry

---

## Summary

### Pattern Checklist

- ✅ **Retry with exponential backoff** - Transient errors
- ✅ **Circuit breaker** - Repeated failures
- ✅ **FAIL FAST on GPU errors** - NO fallback (rbee policy)
- ✅ **Resource cleanup** - RAII + leak detection
- ✅ **Actionable errors** - Specific codes + suggestions
- ✅ **HTTP status mapping** - REST API standards
- ✅ **Timeout cascades** - Prevent hangs
- ✅ **Concurrent limits** - Prevent exhaustion

### Exit Code Standards

- `0` - Success or successful recovery
- `1` - Generic error
- `124` - Timeout (standard)
- `137` - SIGKILL (128 + 9)
- `255` - SSH/network failure

### Error Code Naming

- Use UPPER_SNAKE_CASE
- Be specific: `GPU_VRAM_EXHAUSTED` not `ERROR`
- Include context: `CUDA_DEVICE_FAILED` not `DEVICE_FAILED`
- Searchable: Unique codes for debugging

---

**Patterns documented:** 2025-10-11  
**Source:** Industry research + TEAM-074 lessons  
**Ready for:** TEAM-075 implementation
