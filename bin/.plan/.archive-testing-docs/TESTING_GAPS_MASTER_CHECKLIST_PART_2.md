# Testing Gaps Master Checklist - Part 2: Heartbeat & Timeout + Binary Components

**Date:** Oct 22, 2025  
**Source:** Phase 4 & 5 Team Investigations  
**Status:** COMPREHENSIVE TEST PLAN

---

## 5. Heartbeat & Timeout (`rbee-heartbeat` + `timeout-enforcer`)

**Source:** TEAM-234

### 5.1 Worker Heartbeat Tests

#### Background Task
- [ ] Test heartbeat task starts correctly
- [ ] Test heartbeat sent every 30s
- [ ] Test heartbeat payload correct
- [ ] Test heartbeat continues on failure (retry)
- [ ] Test heartbeat stops on task abort
- [ ] Test heartbeat with network failures
- [ ] Test heartbeat with server unavailable

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

### 5.2 Hive Heartbeat Tests

#### Background Task
- [ ] Test heartbeat task starts correctly
- [ ] Test heartbeat sent every 5s (configurable)
- [ ] Test heartbeat payload includes all workers
- [ ] Test heartbeat continues on failure (retry)
- [ ] Test heartbeat stops on task abort
- [ ] Test heartbeat with network failures
- [ ] Test heartbeat with queen unavailable

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### Worker Aggregation
- [ ] Test empty worker list (no workers)
- [ ] Test single worker
- [ ] Test multiple workers (10+)
- [ ] Test worker state updates reflected in heartbeat
- [ ] Test worker removed from heartbeat after shutdown

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 5.3 Heartbeat Retry Tests

#### Retry Logic
- [ ] Test retry on connection refused
- [ ] Test retry on timeout
- [ ] Test retry on 5xx errors
- [ ] Test no retry on 4xx errors (invalid payload)
- [ ] Test backoff between retries
- [ ] Test max retry limit (if any)

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 5.4 Staleness Detection Tests

#### Worker Staleness
- [ ] Test worker marked stale after 60s (2 missed heartbeats)
- [ ] Test worker marked active on heartbeat received
- [ ] Test stale worker not used for routing
- [ ] Test stale worker cleanup (if implemented)

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Hive Staleness
- [ ] Test hive marked stale after 30s (6 missed heartbeats)
- [ ] Test hive marked active on heartbeat received
- [ ] Test stale hive not shown in status
- [ ] Test stale hive cleanup (if implemented)

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 5.5 Heartbeat Trait Tests

#### Mock Implementations
- [ ] Test HiveCatalog trait with mock
- [ ] Test WorkerRegistry trait with mock
- [ ] Test DeviceDetector trait with mock
- [ ] Test WorkerStateProvider trait with mock
- [ ] Test trait methods called correctly

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

### 5.6 Health Status Tests

#### Status Transitions
- [ ] Test Healthy → Degraded transition
- [ ] Test Degraded → Healthy transition
- [ ] Test Healthy → Unhealthy transition
- [ ] Test Unhealthy → Healthy transition
- [ ] Test status reflected in heartbeat payload

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

### 5.7 Timeout Enforcer Tests

#### Countdown Mode
- [ ] Test countdown display (visual progress bar)
- [ ] Test countdown updates every second
- [ ] Test countdown completes on success
- [ ] Test countdown stops on timeout
- [ ] Test countdown stops on error

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### TTY Detection
- [ ] Test countdown enabled when stderr is TTY
- [ ] Test countdown disabled when stderr is not TTY
- [ ] Test countdown disabled when piped
- [ ] Test countdown disabled when redirected to file

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### job_id Propagation
- [ ] Test timeout narration includes job_id (server-side)
- [ ] Test timeout narration without job_id (client-side)
- [ ] Test timeout narration routes to SSE (with job_id)
- [ ] Test timeout narration goes to stderr only (without job_id)

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Concurrent Timeouts
- [ ] Test multiple concurrent timeout enforcers
- [ ] Test no interference between timeouts
- [ ] Test each timeout fires independently

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Edge Cases
- [ ] Test very short timeout (<1s)
- [ ] Test very long timeout (>60s)
- [ ] Test timeout with zero duration (should error)
- [ ] Test timeout with negative duration (should error)

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 0.5 days

---

## 6. Binary: rbee-keeper

**Source:** TEAM-216, TEAM-239

### 6.1 CLI Parsing Tests

#### Command Parsing
- [ ] Test all hive commands parse correctly
- [ ] Test all worker commands parse correctly
- [ ] Test all model commands parse correctly
- [ ] Test infer command parses correctly
- [ ] Test queen commands parse correctly
- [ ] Test status command parses correctly

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Argument Validation
- [ ] Test required arguments missing (should error)
- [ ] Test invalid argument types (should error)
- [ ] Test invalid argument values (should error)
- [ ] Test default values applied correctly
- [ ] Test optional arguments work

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

### 6.2 Queen Lifecycle Tests

#### Auto-Start
- [ ] Test queen not running → auto-start
- [ ] Test queen already running → no start
- [ ] Test queen start timeout (30s)
- [ ] Test queen start failure (binary not found)
- [ ] Test queen health polling (until ready)
- [ ] Test queen stays alive after task

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### Health Checks
- [ ] Test health check success (200 OK)
- [ ] Test health check failure (connection refused)
- [ ] Test health check timeout
- [ ] Test health check retry logic

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 6.3 Job Submission Tests

#### HTTP Requests
- [ ] Test POST /v1/jobs success
- [ ] Test POST /v1/jobs timeout (10s)
- [ ] Test POST /v1/jobs connection refused
- [ ] Test POST /v1/jobs 4xx errors
- [ ] Test POST /v1/jobs 5xx errors
- [ ] Test response parsing (job_id, sse_url)

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 6.4 SSE Streaming Tests

#### Stream Consumption
- [ ] Test GET /v1/jobs/{job_id}/stream success
- [ ] Test stream receives all events
- [ ] Test stream receives [DONE] marker
- [ ] Test stream closes after [DONE]
- [ ] Test stream timeout (30s)
- [ ] Test stream connection timeout (10s)
- [ ] Test stream closes early (server crash)

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### Event Parsing
- [ ] Test parse SSE data lines
- [ ] Test ignore non-data lines
- [ ] Test handle malformed events
- [ ] Test handle empty events
- [ ] Test handle very long events (>10KB)

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

### 6.5 Error Display Tests

#### Error Formatting
- [ ] Test HTTP errors displayed correctly
- [ ] Test SSE errors displayed correctly
- [ ] Test timeout errors displayed correctly
- [ ] Test job failure detection (❌ Failed)
- [ ] Test job success detection (✅ Complete)

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

### 6.6 Narration Tests

#### Client-Side Narration
- [ ] Test narration without job_id goes to stderr
- [ ] Test narration format correct
- [ ] Test narration includes operation name
- [ ] Test narration includes hive_id (if applicable)

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 0.5 days

---

## 7. Binary: queen-rbee

**Source:** TEAM-217, TEAM-239, TEAM-240

### 7.1 HTTP Server Tests

#### Startup
- [ ] Test server binds to port
- [ ] Test server listens on correct address
- [ ] Test server ready message emitted
- [ ] Test server handles SIGTERM gracefully
- [ ] Test server handles SIGKILL

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Health Endpoint
- [ ] Test GET /health returns 200 OK
- [ ] Test health response format
- [ ] Test health endpoint always available

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 0.5 days

#### Shutdown Endpoint
- [ ] Test POST /v1/shutdown stops server
- [ ] Test shutdown graceful (waits for in-flight requests)
- [ ] Test shutdown timeout (if implemented)

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 1-2 days

### 7.2 Job Creation Tests

#### POST /v1/jobs
- [ ] Test create job returns job_id
- [ ] Test create job returns sse_url
- [ ] Test job_id format (UUID)
- [ ] Test sse_url format correct
- [ ] Test job stored in registry
- [ ] Test SSE channel created
- [ ] Test payload stored correctly

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Concurrent Job Creation
- [ ] Test 10 concurrent POST /v1/jobs
- [ ] Test all jobs get unique IDs
- [ ] Test all jobs stored correctly
- [ ] Test no race conditions

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 2-3 days

### 7.3 Job Streaming Tests

#### GET /v1/jobs/{job_id}/stream
- [ ] Test stream returns SSE events
- [ ] Test stream includes narration events
- [ ] Test stream includes [DONE] marker
- [ ] Test stream closes after [DONE]
- [ ] Test stream closes on error
- [ ] Test stream closes on timeout (2s inactivity)

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### Stream Isolation
- [ ] Test multiple clients can't stream same job
- [ ] Test first client gets stream, second gets error
- [ ] Test jobs isolated (no event leakage)

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 7.4 Operation Routing Tests

#### Route Parsing
- [ ] Test all operation types parse correctly
- [ ] Test invalid operation returns error
- [ ] Test missing required fields returns error
- [ ] Test extra fields ignored

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Route Execution
- [ ] Test HiveList executes correctly
- [ ] Test HiveStart executes correctly
- [ ] Test HiveStop executes correctly
- [ ] Test HiveGet executes correctly
- [ ] Test HiveStatus executes correctly
- [ ] Test HiveRefreshCapabilities executes correctly
- [ ] Test SshTest executes correctly
- [ ] Test Status executes correctly

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 4-5 days

### 7.5 Hive Forwarding Tests (NOT YET IMPLEMENTED)

#### Worker Operations
- [ ] Test WorkerSpawn forwarded to hive
- [ ] Test WorkerList forwarded to hive
- [ ] Test WorkerGet forwarded to hive
- [ ] Test WorkerDelete forwarded to hive

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days (when implemented)

#### Model Operations
- [ ] Test ModelDownload forwarded to hive
- [ ] Test ModelList forwarded to hive
- [ ] Test ModelGet forwarded to hive
- [ ] Test ModelDelete forwarded to hive

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days (when implemented)

#### Inference Operations
- [ ] Test Infer forwarded to hive
- [ ] Test token streaming proxied correctly
- [ ] Test error propagation from hive

**Priority:** N/A (not implemented)  
**Complexity:** Very High  
**Estimated Effort:** 5-7 days (when implemented)

### 7.6 Heartbeat Receiver Tests

#### POST /v1/heartbeat
- [ ] Test receive hive heartbeat
- [ ] Test update hive registry
- [ ] Test update last_seen timestamp
- [ ] Test update worker states
- [ ] Test return acknowledgement
- [ ] Test hive not found error
- [ ] Test invalid payload error

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 7.7 Config Loading Tests

#### File-Based Config
- [ ] Test load from ~/.config/rbee/
- [ ] Test load from custom directory
- [ ] Test config.toml parsed correctly
- [ ] Test hives.conf parsed correctly
- [ ] Test capabilities.yaml parsed correctly
- [ ] Test missing files handled correctly

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 7.8 Hive Registry Tests

#### Registry Operations
- [ ] Test add hive on heartbeat
- [ ] Test update hive on heartbeat
- [ ] Test list active hives (30s threshold)
- [ ] Test get hive state
- [ ] Test hive staleness detection

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

---

## Summary - Part 2

**Total Test Categories:** 2 shared crates + 2 binaries  
**Total Test Tasks:** ~120 individual tests  
**Estimated Total Effort:** 50-70 days (with 1 developer)

**Priority Breakdown:**
- HIGH: ~80 tests (35-45 days)
- MEDIUM: ~30 tests (10-15 days)
- LOW: ~10 tests (3-5 days)
- N/A (not implemented): ~15 tests (15-20 days when implemented)

**Next:** Part 3 - Integration Flow Testing Gaps
