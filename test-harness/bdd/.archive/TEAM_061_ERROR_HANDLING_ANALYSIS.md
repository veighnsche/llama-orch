# TEAM-061 ERROR HANDLING ANALYSIS

**Date:** 2025-10-10  
**Team:** TEAM-061  
**Status:** 📋 ANALYSIS COMPLETE  
**Source:** `bin/.specs/.gherkin/test-001.md` and `bin/.specs/edge-cases/test-001-professional.md`

---

## Executive Summary

Analyzed the test-001 specifications to identify robust error handling opportunities for the BDD test suite. Found **18 critical error scenarios** that should be tested, plus **7 edge cases** from the professional spec that need implementation.

**Priority:** Focus on implementing error handling for the most common failure modes first, then expand to edge cases.

---

## Critical Error Scenarios from test-001.md

### Category 1: Network & Connectivity Errors

#### EH-001: SSH Connection Failure
**Location:** Phase 2 (queen-rbee → rbee-hive SSH)

**Current Spec:**
```bash
ssh -i ~/.ssh/id_ed25519 vince@workstation.home.arpa "cd ~/rbee && ./rbee-hive daemon --port 9200"
```

**Error Scenarios:**
1. SSH host unreachable (network down)
2. SSH authentication failure (wrong key, permissions)
3. SSH timeout (slow network)
4. SSH command execution failure (binary not found)

**Required BDD Tests:**
```gherkin
Scenario: SSH connection fails due to unreachable host
  Given node "workstation" is registered in rbee-hive registry
  And the network to "workstation.home.arpa" is down
  When I run inference targeting "workstation"
  Then queen-rbee should detect connection failure within 10s
  And the error message should contain "Connection refused" or "Host unreachable"
  And the exit code should be 1

Scenario: SSH authentication fails
  Given node "workstation" is registered with invalid SSH key
  When I run inference targeting "workstation"
  Then queen-rbee should detect authentication failure within 5s
  And the error message should contain "Permission denied"
  And the exit code should be 1
```

**Implementation Needs:**
- Timeout on SSH connection attempts (10s max)
- Retry logic with exponential backoff (3 attempts)
- Clear error messages distinguishing connection vs auth failures
- Update registry status to "unreachable" on failure

---

#### EH-002: HTTP Connection Failure (queen-rbee → rbee-hive)
**Location:** Phase 3 (worker registry check)

**Current Spec:**
```
GET http://workstation.home.arpa:9200/v1/workers/list
```

**Error Scenarios:**
1. rbee-hive not responding (process crashed)
2. Port 9200 not accessible (firewall)
3. HTTP timeout (slow network)
4. Invalid response format

**Required BDD Tests:**
```gherkin
Scenario: rbee-hive HTTP server not responding
  Given queen-rbee has started rbee-hive via SSH
  And rbee-hive process has crashed
  When queen-rbee queries worker registry
  Then the HTTP request should timeout after 10s
  And queen-rbee should retry 3 times with backoff
  And the error message should contain "Connection refused"

Scenario: rbee-hive returns malformed response
  Given rbee-hive is running but buggy
  When queen-rbee queries worker registry
  And rbee-hive returns invalid JSON
  Then queen-rbee should detect parse error
  And the error message should contain "Invalid response format"
```

**Implementation Needs:**
- HTTP client with 10s timeout (✅ DONE in TEAM-061)
- Retry logic for transient failures
- JSON parse error handling with context
- Health check before worker registry query

---

#### EH-003: Worker HTTP Connection Failure
**Location:** Phase 13 (rbee-keeper → worker inference)

**Current Spec:**
```
POST http://workstation.home.arpa:8001/execute
```

**Error Scenarios:**
1. Worker crashed after registration
2. Worker port not accessible
3. Worker hung (not responding)
4. SSE stream interrupted mid-inference

**Required BDD Tests:**
```gherkin
Scenario: Worker crashes during inference
  Given a worker is registered and idle
  And inference has started
  When the worker process crashes
  Then rbee-keeper should detect connection loss within 5s
  And the error message should contain "Worker connection lost"
  And partial results should be saved (if any)

Scenario: SSE stream interrupted
  Given inference is streaming tokens
  When the network connection drops
  Then rbee-keeper should detect stream closure
  And display partial results to user
  And exit with code 1
```

**Implementation Needs:**
- SSE stream timeout detection
- Graceful handling of partial results
- Worker health check before inference
- Connection retry for transient failures

---

### Category 2: Resource Errors

#### EH-004: Insufficient RAM
**Location:** Phase 8 (worker preflight)

**Current Spec:**
```rust
let available_ram_mb = get_available_ram();  // 8000 MB
let required_ram_mb = model_size_mb * 1.2;   // 6000 MB

if available_ram_mb < required_ram_mb {
    return Err("Insufficient RAM");
}
```

**Required BDD Tests:**
```gherkin
Scenario: Insufficient RAM for model loading
  Given node "workstation" has 4GB available RAM
  And model "TinyLlama" requires 6GB RAM
  When I request inference on "workstation"
  Then rbee-hive should detect insufficient RAM
  And the error message should contain "Insufficient RAM: need 6GB, have 4GB"
  And the exit code should be 1

Scenario: RAM exhausted during model loading
  Given model loading has started
  When system RAM is exhausted by another process
  Then worker should detect OOM condition
  And worker should exit gracefully
  And error should be reported to rbee-keeper
```

**Implementation Needs:**
- RAM check before spawning worker
- OOM detection during model loading
- Clear error messages with actual vs required RAM
- Suggestion to free RAM or use smaller model

---

#### EH-005: VRAM Exhausted
**Location:** Phase 8 (CUDA device initialization)

**Current Spec:**
```rust
if !cuda_available() {
    return Err("CUDA backend not available");
}
```

**Required BDD Tests:**
```gherkin
Scenario: VRAM exhausted on CUDA device
  Given CUDA device 1 has 2GB VRAM
  And model requires 4GB VRAM
  When I request inference with CUDA backend
  Then worker should detect insufficient VRAM
  And the error message should contain "Insufficient VRAM: need 4GB, have 2GB"
  And suggest using CPU backend or smaller model

Scenario: CUDA device not available
  Given node "workstation" has no CUDA devices
  When I request inference with CUDA backend
  Then rbee-hive should detect missing CUDA
  And the error message should contain "CUDA backend not available"
  And suggest available backends (cpu, metal)
```

**Implementation Needs:**
- VRAM check before model loading
- CUDA availability detection
- Backend fallback suggestions
- Clear error messages with device info

---

#### EH-006: Disk Space Exhausted
**Location:** Phase 6 (model download)

**Error Scenarios:**
1. Insufficient disk space for model download
2. Disk full during download
3. Write permission denied

**Required BDD Tests:**
```gherkin
Scenario: Insufficient disk space for model download
  Given node "workstation" has 1GB free disk space
  And model "TinyLlama" is 5GB
  When rbee-hive attempts to download model
  Then download should fail immediately
  And the error message should contain "Insufficient disk space: need 5GB, have 1GB"

Scenario: Disk fills up during download
  Given model download has started
  When disk space is exhausted mid-download
  Then download should fail gracefully
  And partial download should be cleaned up
  And error should be reported to user
```

**Implementation Needs:**
- Disk space check before download
- Disk space monitoring during download
- Cleanup of partial downloads on failure
- Clear error messages with disk usage info

---

### Category 3: Model & Backend Errors

#### EH-007: Model Not Found
**Location:** Phase 6 (model download)

**Error Scenarios:**
1. Model doesn't exist on Hugging Face
2. Model repository is private
3. Invalid model reference format

**Required BDD Tests:**
```gherkin
Scenario: Model not found on Hugging Face
  Given model "hf:NonExistent/Model" does not exist
  When I request inference with this model
  Then rbee-hive should detect 404 from Hugging Face
  And the error message should contain "Model not found"
  And suggest checking model reference

Scenario: Model repository is private
  Given model "hf:Private/Model" requires authentication
  When I request inference without credentials
  Then download should fail with 403
  And the error message should contain "Access denied"
  And suggest providing HF token
```

**Implementation Needs:**
- HTTP status code handling (404, 403, 401)
- Model reference validation before download
- Clear error messages with suggestions
- Support for HF authentication tokens (future)

---

#### EH-008: Model Download Failure
**Location:** Phase 6 (model download)

**Error Scenarios:**
1. Network timeout during download
2. Corrupted download (checksum mismatch)
3. Download interrupted

**Required BDD Tests:**
```gherkin
Scenario: Model download times out
  Given model download has started
  When network becomes very slow
  And no progress for 60 seconds
  Then download should timeout
  And retry with exponential backoff (3 attempts)
  And error message should contain "Download timeout"

Scenario: Downloaded model is corrupted
  Given model download completes
  When checksum verification fails
  Then rbee-hive should detect corruption
  And delete corrupted file
  And retry download
  And error message should contain "Checksum mismatch"
```

**Implementation Needs:**
- Download timeout (60s stall detection)
- Retry logic with exponential backoff (3 attempts)
- Checksum verification (sha256)
- Cleanup of corrupted downloads

---

#### EH-009: Backend Not Available
**Location:** Phase 8 (backend initialization)

**Error Scenarios:**
1. Requested backend not installed
2. Backend driver missing (CUDA, Metal)
3. Backend version incompatible

**Required BDD Tests:**
```gherkin
Scenario: CUDA backend not installed
  Given node "workstation" has no CUDA installed
  When I request inference with CUDA backend
  Then rbee-hive should detect missing CUDA
  And the error message should contain "CUDA not available"
  And list available backends: ["cpu"]

Scenario: Metal backend on non-Mac
  Given node "workstation" is Linux
  When I request inference with Metal backend
  Then rbee-hive should detect incompatible backend
  And the error message should contain "Metal only available on macOS"
```

**Implementation Needs:**
- Backend detection at startup
- Backend capability registration in registry
- Clear error messages with available alternatives
- Platform-specific backend validation

---

### Category 4: Configuration & Registry Errors

#### EH-010: Node Not in Registry
**Location:** Phase 0 (registry lookup)

**Current Spec:**
```
ERROR: Node 'workstation' not found in rbee-hive registry.
Run: rbee-keeper setup add-node --name workstation ...
```

**Required BDD Tests:**
```gherkin
Scenario: Inference on unregistered node
  Given node "workstation" is NOT in registry
  When I run inference targeting "workstation"
  Then queen-rbee should detect missing node
  And the error message should contain "Node 'workstation' not found"
  And suggest running "rbee-keeper setup add-node"
  And the exit code should be 1
```

**Implementation Needs:**
- Registry lookup before SSH connection
- Clear error message with setup instructions
- List available nodes as suggestion

---

#### EH-011: Invalid Configuration
**Location:** Setup phase (add-node)

**Error Scenarios:**
1. Invalid SSH key path
2. Invalid hostname format
3. Missing required fields
4. Duplicate node name

**Required BDD Tests:**
```gherkin
Scenario: Invalid SSH key path
  When I run "rbee-keeper setup add-node" with non-existent key
  Then setup should fail immediately
  And the error message should contain "SSH key not found"

Scenario: Duplicate node name
  Given node "workstation" already exists in registry
  When I try to add another node named "workstation"
  Then setup should fail
  And the error message should contain "Node already exists"
  And suggest using "rbee-keeper setup update-node"
```

**Implementation Needs:**
- Input validation before SSH test
- File existence checks (SSH key)
- Duplicate detection in registry
- Clear error messages with suggestions

---

### Category 5: Process Lifecycle Errors

#### EH-012: Worker Startup Failure
**Location:** Phase 9 (worker spawn)

**Error Scenarios:**
1. Worker binary not found
2. Worker crashes during startup
3. Worker fails to bind port
4. Worker fails to load model

**Required BDD Tests:**
```gherkin
Scenario: Worker binary not found
  Given rbee-hive is running
  When rbee-hive attempts to spawn worker
  And worker binary does not exist
  Then spawn should fail immediately
  And the error message should contain "Worker binary not found"

Scenario: Worker port already in use
  Given port 8001 is already occupied
  When rbee-hive spawns worker on port 8001
  Then worker should fail to bind port
  And rbee-hive should detect startup failure
  And try next available port (8002)
```

**Implementation Needs:**
- Binary existence check before spawn
- Port availability check
- Automatic port selection on conflict
- Worker startup timeout (30s)
- Startup failure detection and reporting

---

#### EH-013: Worker Crash During Inference
**Location:** Phase 13 (inference execution)

**Error Scenarios:**
1. Worker segfault during inference
2. Worker OOM killed
3. Worker timeout (hung)

**Required BDD Tests:**
```gherkin
Scenario: Worker crashes mid-inference
  Given inference has started
  When worker process crashes
  Then rbee-keeper should detect process exit
  And display partial results (if any)
  And the error message should contain "Worker crashed"
  And the exit code should be 1

Scenario: Worker hangs during inference
  Given inference has started
  When worker stops responding
  And no tokens generated for 60 seconds
  Then rbee-keeper should timeout
  And cancel the request
  And the error message should contain "Worker timeout"
```

**Implementation Needs:**
- Process exit detection
- Inference timeout (configurable, default 300s)
- Token generation timeout (60s stall detection)
- Graceful handling of partial results

---

#### EH-014: Graceful Shutdown Interrupted
**Location:** Phase 14 (cascading shutdown)

**Error Scenarios:**
1. Worker doesn't respond to shutdown
2. Shutdown timeout exceeded
3. Force-kill required

**Required BDD Tests:**
```gherkin
Scenario: Worker ignores shutdown signal
  Given worker is running
  When rbee-hive sends shutdown command
  And worker doesn't respond within 30s
  Then rbee-hive should force-kill worker
  And log the force-kill event

Scenario: Ctrl+C during inference
  Given inference is in progress
  When user presses Ctrl+C
  Then rbee-keeper should cancel request immediately
  And send cancellation to worker
  And exit with code 130 (SIGINT)
```

**Implementation Needs:**
- Shutdown timeout (30s graceful, then force-kill)
- Ctrl+C handler (✅ DONE in TEAM-061)
- Cancellation propagation to worker
- Cleanup verification

---

### Category 6: Request Validation Errors

#### EH-015: Invalid Request Parameters
**Location:** Phase 1 (rbee-keeper → queen-rbee)

**Error Scenarios:**
1. Invalid model reference format
2. Invalid backend name
3. Invalid device number
4. Missing required fields

**Required BDD Tests:**
```gherkin
Scenario: Invalid model reference
  When I run inference with model "invalid-format"
  Then rbee-keeper should validate format
  And the error message should contain "Invalid model reference"
  And show expected format: "hf:org/repo" or "file:///path"

Scenario: Invalid backend
  When I request inference with backend "quantum"
  Then rbee-keeper should validate backend
  And the error message should contain "Invalid backend"
  And list valid backends: ["cpu", "cuda", "metal"]

Scenario: Device number out of range
  Given node has 2 CUDA devices (0, 1)
  When I request device 5
  Then rbee-hive should detect invalid device
  And the error message should contain "Device 5 not available"
  And list available devices: [0, 1]
```

**Implementation Needs:**
- Input validation in rbee-keeper (client-side)
- Input validation in queen-rbee (server-side)
- Input validation in rbee-hive (resource-side)
- Clear error messages with valid options

---

### Category 7: Timeout Errors

#### EH-016: Request Timeout
**Location:** Any async operation

**Error Scenarios:**
1. Overall request timeout exceeded
2. Model download timeout
3. Model loading timeout
4. Inference timeout

**Required BDD Tests:**
```gherkin
Scenario: Overall request timeout
  Given request timeout is set to 5 minutes
  When inference takes longer than 5 minutes
  Then rbee-keeper should timeout
  And cancel the request
  And the error message should contain "Request timeout after 5m"

Scenario: Model loading timeout
  Given model is very large (50GB)
  When loading takes longer than 10 minutes
  Then worker should timeout
  And the error message should contain "Model loading timeout"
```

**Implementation Needs:**
- Configurable timeouts per operation
- Timeout detection and cancellation
- Clear error messages with elapsed time
- Partial result preservation where possible

---

### Category 8: Authentication Errors

#### EH-017: API Key Authentication Failure
**Location:** Any HTTP request with auth

**Error Scenarios:**
1. Missing API key
2. Invalid API key
3. Expired API key (future)

**Required BDD Tests:**
```gherkin
Scenario: Missing API key
  Given rbee-hive requires API key
  When queen-rbee sends request without key
  Then rbee-hive should return 401 Unauthorized
  And the error message should contain "Missing API key"

Scenario: Invalid API key
  Given rbee-hive requires API key
  When queen-rbee sends request with wrong key
  Then rbee-hive should return 401 Unauthorized
  And the error message should contain "Invalid API key"
```

**Implementation Needs:**
- API key validation middleware
- 401 error handling in clients
- Clear error messages
- Secure key storage (not in logs)

---

### Category 9: Concurrent Request Errors

#### EH-018: Worker Busy
**Location:** Phase 13 (inference request)

**Error Scenarios:**
1. Worker already processing request
2. All slots busy
3. Queue full

**Required BDD Tests:**
```gherkin
Scenario: Worker busy with another request
  Given worker is processing inference
  When another inference request arrives
  Then worker should return 503 Service Unavailable
  And the error message should contain "Worker busy"
  And suggest retrying later

Scenario: All workers busy
  Given all workers for model are busy
  When new inference request arrives
  Then rbee-keeper should queue request
  Or return 503 if queue full
  And the error message should contain "All workers busy"
```

**Implementation Needs:**
- Worker slot management
- Request queuing (optional for MVP)
- 503 error handling with retry-after
- Load balancing across multiple workers

---

## Error Handling Gaps from Professional Spec

### Gap 1: Request Cancellation (G12)
**Priority:** HIGH

**Missing:**
- No explicit cancellation endpoint
- No context cancellation propagation
- No cleanup on client disconnect

**Required Implementation:**
```rust
// Worker API
DELETE /inference/{request_id}
Response: 204 No Content

// Detect client disconnect
if sse_stream.is_closed() {
    cancel_inference();
    release_slot();
}
```

**BDD Tests Needed:**
```gherkin
Scenario: Client disconnects during inference
  Given inference is streaming tokens
  When client closes connection
  Then worker should detect disconnect within 1s
  And stop token generation immediately
  And release slot
  And log cancellation event
```

---

### Gap 2: Request Queue Management (G13)
**Priority:** MEDIUM (post-MVP)

**Missing:**
- No queue size limits
- No backpressure handling
- No 503 Service Unavailable on queue full

**Required Implementation:**
```yaml
ctl:
  max_pending_requests: 100
  queue_timeout_seconds: 300
```

**BDD Tests Needed:**
```gherkin
Scenario: Queue full
  Given queue has 100 pending requests
  When new request arrives
  Then rbee-keeper should return 503
  And the error message should contain "Queue full"
```

---

### Gap 3: Progress Streaming (G14)
**Priority:** MEDIUM

**Missing:**
- No SSE endpoint for download progress
- No SSE endpoint for loading progress
- No progress updates to user

**Required Implementation:**
```
GET /models/{model_id}/download/progress
GET /loading/progress
```

**BDD Tests Needed:**
```gherkin
Scenario: Stream download progress
  Given model download has started
  When user connects to progress stream
  Then user should receive progress events
  And events should include bytes_downloaded, bytes_total, speed_mbps
```

---

### Gap 4: Health vs Readiness (G15)
**Priority:** LOW (nice to have)

**Missing:**
- No distinction between liveness and readiness
- No readiness probe during loading

**Required Implementation:**
```
GET /health  # Always 200 if process alive
GET /ready   # 200 if ready, 503 if loading
```

---

### Gap 5: Retry Logic (G18)
**Priority:** HIGH

**Missing:**
- No exponential backoff
- No jitter to avoid thundering herd
- No max retry attempts

**Required Implementation:**
```rust
fn retry_with_backoff<F, T>(
    operation: F,
    max_attempts: u32,
) -> Result<T>
where
    F: Fn() -> Result<T>,
{
    for attempt in 1..=max_attempts {
        match operation() {
            Ok(result) => return Ok(result),
            Err(e) if attempt == max_attempts => return Err(e),
            Err(_) => {
                let delay = min(100 * 2_u32.pow(attempt - 1), 5000);
                let jitter = rand::random::<f64>() * 0.5 + 0.5;
                sleep(Duration::from_millis((delay as f64 * jitter) as u64));
            }
        }
    }
}
```

---

### Gap 6: Error Taxonomy (G17)
**Priority:** HIGH

**Missing:**
- No standardized error codes
- No consistent error response format
- No HTTP status code mapping

**Required Implementation:**
```json
{
  "error": {
    "code": "WORKER_NOT_READY",
    "message": "Worker still loading model",
    "details": {
      "worker_id": "worker-abc123",
      "state": "loading",
      "progress": 0.45
    }
  }
}
```

**Error Codes:**
- `INVALID_REQUEST` (400)
- `MODEL_NOT_FOUND` (404)
- `WORKER_NOT_READY` (503)
- `QUEUE_FULL` (503)
- `REQUEST_TIMEOUT` (408)
- `REQUEST_CANCELED` (499)
- `WORKER_FAILED` (500)
- `VRAM_EXHAUSTED` (507)
- `INTERNAL_ERROR` (500)

---

### Gap 7: Graceful Shutdown (G20)
**Priority:** HIGH

**Missing:**
- No draining state
- No timeout on shutdown
- No force-kill after timeout

**Required Implementation:**
```
1. Receive SIGTERM
2. Set state to "draining"
3. Reject new requests (503)
4. Wait for active requests (max 30s)
5. Force-kill if timeout
6. Exit with code 0
```

---

## Implementation Priority

### Phase 1: Critical Error Handling (Week 1)
1. **EH-001:** SSH connection failure
2. **EH-002:** HTTP connection failure
3. **EH-010:** Node not in registry
4. **EH-015:** Invalid request parameters
5. **Gap 5:** Retry logic with exponential backoff
6. **Gap 6:** Error taxonomy and response format

### Phase 2: Resource Errors (Week 2)
7. **EH-004:** Insufficient RAM
8. **EH-005:** VRAM exhausted
9. **EH-006:** Disk space exhausted
10. **EH-009:** Backend not available

### Phase 3: Model & Download Errors (Week 3)
11. **EH-007:** Model not found
12. **EH-008:** Model download failure
13. **Gap 3:** Progress streaming

### Phase 4: Process Lifecycle (Week 4)
14. **EH-012:** Worker startup failure
15. **EH-013:** Worker crash during inference
16. **EH-014:** Graceful shutdown
17. **Gap 1:** Request cancellation
18. **Gap 7:** Graceful shutdown improvements

### Phase 5: Advanced Features (Post-MVP)
19. **EH-003:** Worker HTTP connection failure
20. **EH-016:** Request timeout
21. **EH-017:** API key authentication
22. **EH-018:** Worker busy / concurrent requests
23. **Gap 2:** Request queue management
24. **Gap 4:** Health vs readiness probes

---

## Next Steps for TEAM-062

1. **Review this analysis** with product owner
2. **Prioritize error scenarios** based on user impact
3. **Implement Phase 1 error handling** in production code
4. **Write BDD tests** for Phase 1 scenarios
5. **Update timeout implementation** to include retry logic
6. **Create error taxonomy** and standardize error responses
7. **Document error handling** in user-facing docs

---

## Testing Strategy

### Unit Tests
- Test each error condition in isolation
- Mock external dependencies (SSH, HTTP, filesystem)
- Verify error messages and codes

### Integration Tests
- Test error propagation across components
- Verify cleanup on errors
- Test retry logic end-to-end

### BDD Tests
- Test user-facing error scenarios
- Verify error messages are helpful
- Test recovery and retry flows

### Chaos Testing (Future)
- Randomly inject failures
- Verify system resilience
- Test cascading failure handling

---

**TEAM-061 signing off.**

**Status:** Error handling analysis complete  
**Next:** TEAM-062 to implement Phase 1 error handling  
**Priority:** Focus on critical errors first, then expand

🎯 **18 error scenarios identified, 7 gaps documented, ready for implementation.** 🔥
