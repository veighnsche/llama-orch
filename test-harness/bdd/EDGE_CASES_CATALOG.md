# Edge Cases Catalog - MVP + Critical

**TEAM-075**  
**Date:** 2025-10-11  
**Based on:** Industry research (llama.cpp, candle-vllm, mistral.rs)

---

## MVP Blockers (Must Implement) - 15 Functions

**CRITICAL POLICY:** NO FALLBACK - FAIL FAST on GPU errors

### 1. GPU/CUDA Failures - FAIL FAST (3 functions)

**Priority:** CRITICAL  
**Why:** GPU failures are common in production, must FAIL FAST with clear error

**POLICY:** NO FALLBACK - FAIL FAST on GPU errors

#### Functions to Implement:

1. `when_cuda_device_fails(device: u8)` - CUDA device initialization failure
2. `when_gpu_vram_exhausted()` - VRAM exhaustion with actionable error
3. `then_gpu_error_fails_immediately()` - Verify FAIL FAST behavior

**Error Codes:**
- `CUDA_DEVICE_FAILED` - Device initialization failed
- `GPU_VRAM_EXHAUSTED` - Insufficient VRAM
- `GPU_NOT_AVAILABLE` - GPU not available

**Exit Codes:**
- 1 = GPU failure (FAIL FAST)

**NO:**
- ❌ NO fallback to CPU
- ❌ NO fallback to shared backend
- ❌ NO graceful degradation

---

### 2. Model Corruption Detection (3 functions)

**Priority:** CRITICAL  
**Why:** Corrupted models cause silent failures, data integrity is essential

#### Functions to Implement:

1. `when_model_checksum_fails()` - SHA256 verification failure
2. `then_delete_corrupted_model()` - Cleanup corrupted file
3. `then_retry_model_download()` - Automatic re-download

**Error Codes:**
- `MODEL_CORRUPTED` - Checksum verification failed
- `MODEL_DELETED` - Corrupted file removed
- `MODEL_RETRY_DOWNLOAD` - Re-downloading model

**Exit Codes:**
- 0 = Successful re-download
- 1 = Corruption detected, retry in progress

---

### 3. Concurrent Request Limits (3 functions)

**Priority:** CRITICAL  
**Why:** Prevents resource exhaustion and system overload

#### Functions to Implement:

1. `given_worker_at_max_capacity(max: u32)` - Worker at capacity
2. `when_request_exceeds_capacity()` - Request beyond limit
3. `then_reject_with_503()` - Service Unavailable response

**Error Codes:**
- `SERVICE_UNAVAILABLE` - Worker at capacity
- `QUEUE_FULL` - Request queue overflow
- `RATE_LIMITED` - Too many requests

**HTTP Status:**
- 503 Service Unavailable
- Retry-After: 30 (seconds)

---

### 4. Timeout Cascade Handling (3 functions)

**Priority:** CRITICAL  
**Why:** Prevents system-wide hangs and resource leaks

#### Functions to Implement:

1. `given_inference_timeout(timeout: u16)` - Set timeout expectation
2. `when_inference_exceeds_timeout()` - Timeout triggered
3. `then_cancel_gracefully()` - Graceful cancellation

**Error Codes:**
- `INFERENCE_TIMEOUT` - Operation exceeded timeout
- `TIMEOUT_CANCELLED` - Gracefully cancelled
- `TIMEOUT_FORCE_KILL` - Force killed after grace period

**Exit Codes:**
- 0 = Graceful cancellation
- 124 = Timeout exit code (standard)

---

### 5. Network Partition Handling (3 functions)

**Priority:** CRITICAL  
**Why:** Distributed systems must handle network failures

#### Functions to Implement:

1. `when_network_partition_detected(target: String)` - Connection lost
2. `then_retry_with_exponential_backoff()` - Retry strategy
3. `then_circuit_breaker_opens(failures: u8)` - Stop after repeated failures

**Error Codes:**
- `NETWORK_PARTITION` - Connection lost
- `RETRY_BACKOFF` - Retrying with backoff
- `CIRCUIT_BREAKER_OPEN` - Too many failures, circuit open

**Retry Strategy:**
- Attempts: 1s, 2s, 4s, 8s, 16s (max 5)
- Circuit breaker: Open after 5 consecutive failures
- Cooldown: 60 seconds

---

## High Priority (Should Implement) - 6 Functions

### 6. Model Version Mismatches (2 functions)

**Priority:** HIGH  
**Why:** Version incompatibility causes runtime errors

1. `given_worker_expects_version(version: String)` - Expected version
2. `then_detect_version_mismatch()` - Incompatibility detected

**Error Codes:**
- `MODEL_VERSION_MISMATCH` - Incompatible version
- `MODEL_VERSION_UPGRADE_REQUIRED` - Upgrade needed

---

### 7. Partial Response Handling (2 functions)

**Priority:** HIGH  
**Why:** Preserve partial work on failures

1. `when_worker_crashes_mid_inference()` - Crash during generation
2. `then_save_partial_tokens()` - Preserve partial results

**Error Codes:**
- `PARTIAL_RESPONSE` - Incomplete inference
- `WORKER_CRASHED_MID_INFERENCE` - Crash detected

---

### 8. Resource Leak Detection (2 functions)

**Priority:** HIGH  
**Why:** Prevent memory/file descriptor leaks

1. `when_worker_crashes_without_cleanup()` - Unclean exit
2. `then_detect_leaked_resources()` - Leak detection

**Error Codes:**
- `RESOURCE_LEAK_DETECTED` - Leak found
- `CLEANUP_REQUIRED` - Manual cleanup needed

---

## Medium Priority (Nice to Have) - Future Work

### 9. Graceful Shutdown During Inference
- Complete active request before shutdown
- Reject new requests during shutdown
- Timeout for graceful period

### 10. Model Hot-Swapping
- Load new model version
- Drain active requests on old version
- Switch traffic to new version

### 11. Load Balancing Failures
- Detect unhealthy workers
- Remove from load balancer pool
- Re-add after health check passes

### 12. Authentication Token Expiry
- Detect expired tokens
- Refresh token automatically
- Retry request with new token

### 13. Rate Limiting
- Per-client rate limits
- Sliding window algorithm
- 429 Too Many Requests

---

## Implementation Guidelines

### Error Response Structure

```rust
ErrorResponse {
    code: String,           // Machine-readable error code
    message: String,        // Human-readable message
    details: Option<Value>, // Actionable context
}
```

### Required Details Fields

- `suggested_action` - What user should do
- `retry_after_seconds` - When to retry (if applicable)
- `fallback_chain` - Available fallbacks (if applicable)
- `resource_info` - Resource state (if applicable)

### Exit Code Standards

- `0` - Success or successful recovery
- `1` - Generic error
- `124` - Timeout (standard timeout exit code)
- `137` - SIGKILL (128 + 9)
- `255` - SSH/network failure

### HTTP Status Code Standards

- `400` - Bad Request (validation error)
- `401` - Unauthorized (auth required)
- `403` - Forbidden (auth failed)
- `404` - Not Found (resource doesn't exist)
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error (system error)
- `502` - Bad Gateway (upstream error)
- `503` - Service Unavailable (capacity/overload)
- `504` - Gateway Timeout (upstream timeout)

---

## Testing Strategy

### Unit Tests
- Each function sets proper error state
- Exit codes are correct
- Error details include actionable info

### Integration Tests
- Error propagation through system
- Cleanup after errors
- Recovery from errors

### Verification Commands

```bash
# Compile check
cargo check --bin bdd-runner

# Run tests
cargo test --bin bdd-runner

# Run specific feature
LLORCH_BDD_FEATURE_PATH=tests/features/error_handling.feature cargo test --bin bdd-runner
```

---

## Success Criteria

### Minimum Acceptable
- ✅ 17 MVP functions implemented
- ✅ All functions compile
- ✅ Pass rate: 42.9% → 45% (+2%)
- ✅ No test fraud (verified against Testing Team rules)

### Realistic Target
- ✅ 17+ MVP functions + 3 high priority
- ✅ Pass rate: 42.9% → 48% (+5%)
- ✅ Comprehensive error details
- ✅ Documentation complete

### Optimistic Target
- ✅ 17+ MVP + 6 high priority functions
- ✅ Pass rate: 42.9% → 50%+ (+7%)
- ✅ Industry-standard patterns adopted
- ✅ Production-ready error handling

---

## References

- **ERROR_HANDLING_RESEARCH.md** - Industry standards analysis
- **TEAM_074_HANDOFF.md** - Foundation built by TEAM-074
- **llama.cpp** - `/reference/llama.cpp/`
- **candle-vllm** - `/reference/candle-vllm/`
- **mistral.rs** - `/reference/mistral.rs/`

---

**Catalog created:** 2025-10-11  
**Total edge cases:** 23 functions (17 MVP + 6 high priority)  
**Implementation target:** 17+ MVP functions minimum
