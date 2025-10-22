# Testing Gaps Master Checklist - Part 3: Integration Flows

**Date:** Oct 22, 2025  
**Source:** Phase 5 Team Investigations  
**Status:** COMPREHENSIVE TEST PLAN

---

## 8. Binary: rbee-hive

**Source:** TEAM-218, TEAM-240

### 8.1 HTTP Server Tests

#### Startup
- [ ] Test server binds to port
- [ ] Test server listens on correct address
- [ ] Test server ready message emitted
- [ ] Test heartbeat task starts
- [ ] Test handles SIGTERM gracefully

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Health Endpoint
- [ ] Test GET /health returns 200 OK
- [ ] Test health response format

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 0.5 days

#### Capabilities Endpoint
- [ ] Test GET /capabilities returns devices
- [ ] Test GPU detection (if available)
- [ ] Test CPU always present
- [ ] Test response format correct
- [ ] Test handles GPU detection failure

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 8.2 Heartbeat Sender Tests

#### Heartbeat Task
- [ ] Test heartbeat sent every 5s
- [ ] Test heartbeat includes hive_id
- [ ] Test heartbeat includes workers (empty for now)
- [ ] Test heartbeat continues on failure
- [ ] Test heartbeat stops on task abort

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 8.3 Worker State Provider Tests

#### State Collection
- [ ] Test get_worker_states() returns empty (no workers yet)
- [ ] Test get_worker_states() returns workers (when implemented)
- [ ] Test worker states updated correctly

**Priority:** MEDIUM (when workers implemented)  
**Complexity:** Medium  
**Estimated Effort:** 1-2 days

---

## 9. Integration: Keeper ↔ Queen

**Source:** TEAM-239

### 9.1 Happy Path Integration Tests

#### HiveList Flow
- [ ] Test full flow: CLI → POST → GET → SSE → Display
- [ ] Test narration events received
- [ ] Test [DONE] marker received
- [ ] Test ✅ Complete displayed

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### HiveStart Flow
- [ ] Test full flow: CLI → Queen start → POST → Hive spawn → Health poll → Capabilities → SSE → Display
- [ ] Test all narration events received
- [ ] Test hive actually starts
- [ ] Test capabilities cached
- [ ] Test ✅ Complete displayed

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### HiveStop Flow
- [ ] Test full flow: CLI → POST → SIGTERM → SIGKILL → SSE → Display
- [ ] Test hive actually stops
- [ ] Test graceful shutdown (5s wait)
- [ ] Test force kill if needed

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### HiveStatus Flow
- [ ] Test full flow: CLI → POST → Health check → SSE → Display
- [ ] Test hive reachable
- [ ] Test hive unreachable

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Status Flow
- [ ] Test full flow: CLI → POST → Registry query → SSE → Display
- [ ] Test with no hives (empty message)
- [ ] Test with 1 hive
- [ ] Test with multiple hives
- [ ] Test table formatting

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 9.2 Error Propagation Tests

#### Queen Unreachable
- [ ] Test queen not running → auto-start
- [ ] Test queen auto-start timeout (30s)
- [ ] Test queen auto-start failure
- [ ] Test error message displayed

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### SSE Stream Closes Early
- [ ] Test queen crashes mid-operation
- [ ] Test keeper detects stream close
- [ ] Test error message displayed
- [ ] Test no [DONE] received

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 2-3 days

#### Multiple Clients Same job_id
- [ ] Test first client gets stream
- [ ] Test second client gets error
- [ ] Test error message clear

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 1-2 days

### 9.3 Timeout Tests

#### HTTP Timeouts
- [ ] Test POST /v1/jobs timeout (10s)
- [ ] Test GET /v1/jobs/{job_id}/stream timeout (10s)
- [ ] Test error message displayed

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### SSE Streaming Timeout
- [ ] Test SSE stream timeout (30s)
- [ ] Test timeout message displayed
- [ ] Test operation continues on server (orphaned)

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 2-3 days

#### Operation Timeout
- [ ] Test hive start timeout (15s)
- [ ] Test timeout narration received via SSE
- [ ] Test ❌ Failed displayed

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Queen Startup Timeout
- [ ] Test queen startup timeout (30s)
- [ ] Test countdown displayed
- [ ] Test error message displayed

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 1-2 days

### 9.4 Network Failure Tests

#### Connection Refused
- [ ] Test POST /v1/jobs connection refused
- [ ] Test GET /v1/jobs/{job_id}/stream connection refused
- [ ] Test error messages displayed

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Network Partition
- [ ] Test network partition during POST
- [ ] Test network partition during GET
- [ ] Test timeout behavior
- [ ] Test error messages

**Priority:** MEDIUM  
**Complexity:** High  
**Estimated Effort:** 2-3 days

### 9.5 Concurrent Operation Tests

#### Multiple Operations
- [ ] Test 10 concurrent hive operations
- [ ] Test all operations complete
- [ ] Test no interference between operations
- [ ] Test all SSE streams isolated

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 3-4 days

### 9.6 Resource Cleanup Tests

#### Normal Completion
- [ ] Test job removed from registry
- [ ] Test SSE channel removed
- [ ] Test no memory leaks

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Error Completion
- [ ] Test job removed on error
- [ ] Test SSE channel removed on error
- [ ] Test no memory leaks

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Client Disconnect
- [ ] Test keeper Ctrl+C
- [ ] Test queen detects disconnect
- [ ] Test SSE channel cleanup
- [ ] Test job cleanup

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 2-3 days

#### Timeout Expiration
- [ ] Test keeper timeout fires
- [ ] Test connection closed
- [ ] Test queen cleanup
- [ ] Test no orphaned operations

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 2-3 days

---

## 10. Integration: Queen ↔ Hive

**Source:** TEAM-240

### 10.1 Hive Lifecycle Tests

#### Hive Spawn
- [ ] Test queen spawns hive daemon
- [ ] Test hive binds to port
- [ ] Test hive starts heartbeat
- [ ] Test queen polls health
- [ ] Test health check succeeds
- [ ] Test Stdio::null() prevents pipe hangs

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### Hive Stop
- [ ] Test queen sends SIGTERM
- [ ] Test hive graceful shutdown (5s)
- [ ] Test queen sends SIGKILL if needed
- [ ] Test hive actually stops

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 10.2 Heartbeat Flow Tests

#### Hive → Queen Heartbeat
- [ ] Test hive sends heartbeat every 5s
- [ ] Test queen receives heartbeat
- [ ] Test queen updates registry
- [ ] Test queen updates last_seen
- [ ] Test queen returns acknowledgement

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### Heartbeat Retry
- [ ] Test hive retries on failure
- [ ] Test hive continues after failure
- [ ] Test backoff between retries (if any)

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Staleness Detection
- [ ] Test hive marked stale after 30s (6 missed heartbeats)
- [ ] Test hive marked active on heartbeat
- [ ] Test stale hive not shown in status
- [ ] Test hive re-registration after restart

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

### 10.3 Capabilities Flow Tests

#### Capabilities Fetch
- [ ] Test queen fetches capabilities from hive
- [ ] Test GPU detection (if available)
- [ ] Test CPU always present
- [ ] Test capabilities cached
- [ ] Test cache saved to disk

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Capabilities Refresh
- [ ] Test manual refresh (HiveRefreshCapabilities)
- [ ] Test capabilities updated
- [ ] Test cache updated
- [ ] Test cache saved

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Capabilities Timeout
- [ ] Test capabilities fetch timeout (15s)
- [ ] Test timeout narration emitted
- [ ] Test error propagated to keeper

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 10.4 SSH Integration Tests

#### SSH Test
- [ ] Test SSH connection success
- [ ] Test SSH connection failure (connection refused)
- [ ] Test SSH connection failure (auth failed)
- [ ] Test SSH connection failure (timeout)
- [ ] Test error messages clear

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### Remote Hive Start (NOT YET IMPLEMENTED)
- [ ] Test SSH command execution
- [ ] Test remote hive starts
- [ ] Test health polling remote hive
- [ ] Test capabilities fetch from remote

**Priority:** N/A (not implemented)  
**Complexity:** Very High  
**Estimated Effort:** 5-7 days (when implemented)

### 10.5 Error Propagation Tests

#### Hive Unreachable
- [ ] Test hive not running
- [ ] Test connection refused
- [ ] Test error narration emitted
- [ ] Test error propagated to keeper

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Hive Crash
- [ ] Test hive crashes mid-operation
- [ ] Test queen detects crash
- [ ] Test error narration emitted
- [ ] Test error propagated to keeper

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 2-3 days

#### Heartbeat Failure
- [ ] Test hive heartbeat fails
- [ ] Test hive retries
- [ ] Test hive marked stale after threshold
- [ ] Test no error to keeper (background task)

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 10.6 Worker Status Aggregation Tests

#### Worker States
- [ ] Test hive collects worker states
- [ ] Test hive includes workers in heartbeat
- [ ] Test queen stores worker states
- [ ] Test queen displays workers in status

**Priority:** MEDIUM (when workers implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

### 10.7 Concurrent Hive Tests

#### Multiple Hives
- [ ] Test queen manages multiple hives
- [ ] Test each hive isolated
- [ ] Test heartbeats from all hives
- [ ] Test operations to different hives

**Priority:** MEDIUM  
**Complexity:** High  
**Estimated Effort:** 3-4 days

---

## 11. Integration: Hive ↔ Worker (NOT YET IMPLEMENTED)

**Source:** TEAM-241

### 11.1 Worker Lifecycle Tests

#### Worker Spawn
- [ ] Test hive spawns worker
- [ ] Test worker registers with hive
- [ ] Test worker starts heartbeat
- [ ] Test worker loads model

**Priority:** N/A (not implemented)  
**Complexity:** Very High  
**Estimated Effort:** 5-7 days

#### Worker Shutdown
- [ ] Test hive sends SIGTERM
- [ ] Test worker graceful shutdown
- [ ] Test hive sends SIGKILL if needed
- [ ] Test worker removed from registry

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

### 11.2 Worker Heartbeat Tests

#### Worker → Hive Heartbeat
- [ ] Test worker sends heartbeat every 30s
- [ ] Test hive receives heartbeat
- [ ] Test hive updates registry
- [ ] Test hive returns acknowledgement

**Priority:** N/A (not implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### Worker Failure Detection
- [ ] Test worker stops sending heartbeats
- [ ] Test hive marks worker stale after 60s
- [ ] Test hive stops routing to stale worker

**Priority:** N/A (not implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

### 11.3 Model Provisioning Tests

#### Model Discovery
- [ ] Test hive scans model directory
- [ ] Test hive lists all models
- [ ] Test model metadata parsing

**Priority:** N/A (not implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Model Download
- [ ] Test hive downloads model
- [ ] Test progress streaming
- [ ] Test checksum verification
- [ ] Test download failure handling

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### Model Loading
- [ ] Test worker loads model
- [ ] Test VRAM allocation
- [ ] Test model validation
- [ ] Test load failure handling

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

### 11.4 Inference Coordination Tests

#### Request Routing
- [ ] Test hive routes to correct worker
- [ ] Test worker selection (round-robin)
- [ ] Test worker selection (least loaded)
- [ ] Test no workers available error

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### Token Streaming
- [ ] Test worker generates tokens
- [ ] Test hive proxies tokens
- [ ] Test tokens arrive in order
- [ ] Test [DONE] marker sent

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

### 11.5 Resource Management Tests

#### GPU Assignment
- [ ] Test explicit device assignment
- [ ] Test auto device selection
- [ ] Test CPU fallback

**Priority:** N/A (not implemented)  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### VRAM Tracking
- [ ] Test VRAM usage tracked
- [ ] Test VRAM exhaustion detected
- [ ] Test VRAM cleanup on worker shutdown

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### Slot Management
- [ ] Test slot allocation
- [ ] Test slot deallocation
- [ ] Test concurrent requests (multiple slots)
- [ ] Test queue when all slots full

**Priority:** N/A (not implemented)  
**Complexity:** High  
**Estimated Effort:** 3-4 days

---

## Summary - Part 3

**Total Test Categories:** 1 binary + 3 integration flows  
**Total Test Tasks:** ~100 individual tests  
**Estimated Total Effort:** 80-110 days (with 1 developer)

**Priority Breakdown:**
- HIGH: ~60 tests (50-70 days)
- MEDIUM: ~20 tests (15-25 days)
- LOW: ~5 tests (3-5 days)
- N/A (not implemented): ~15 tests (15-20 days when implemented)

**Next:** Part 4 - E2E Flows + Test Infrastructure
