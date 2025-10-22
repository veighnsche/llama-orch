# Testing Gaps Master Checklist - Part 1: Shared Crates

**Date:** Oct 22, 2025  
**Source:** Phase 4 & 5 Team Investigations  
**Status:** COMPREHENSIVE TEST PLAN

---

## Overview

This document consolidates ALL testing gaps identified across Phases 4 and 5 investigations. These are **IMPLEMENTED features that lack tests**, not future features.

**Document Structure:**
- **Part 1:** Shared Crates (narration, daemon-lifecycle, config, job-registry, heartbeat, timeout)
- **Part 2:** Binary Components (rbee-keeper, queen-rbee, rbee-hive)
- **Part 3:** Integration Flows (keeper↔queen, queen↔hive)
- **Part 4:** Test Infrastructure & Priorities

---

## 1. Narration Core (`observability-narration-core`)

**Source:** TEAM-230

### 1.1 Core Functionality Tests

#### Task-Local Context Propagation
- [ ] Test automatic job_id injection from task-local storage
- [ ] Test automatic correlation_id injection from task-local storage
- [ ] Test context inheritance across async boundaries
- [ ] Test context isolation between concurrent tasks
- [ ] Test context cleanup on task completion

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### Format String Interpolation
- [ ] Test basic interpolation: `human("Message {0}")` with 1 context
- [ ] Test multiple contexts: `human("{0} and {1}")` with 2 contexts
- [ ] Test context reuse: `human("{0} {0}")` with 1 context
- [ ] Test missing context (should panic or error)
- [ ] Test extra contexts (should be ignored)
- [ ] Test empty context string
- [ ] Test special characters in context (quotes, newlines, unicode)

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Table Formatting Edge Cases
- [ ] Test nested objects in JSON
- [ ] Test large arrays (>100 items)
- [ ] Test deeply nested structures (>5 levels)
- [ ] Test empty objects/arrays
- [ ] Test null values
- [ ] Test mixed types in arrays
- [ ] Test very long strings (>1000 chars)
- [ ] Test unicode characters in table
- [ ] Test table width overflow

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 1.2 SSE Sink Tests

#### Concurrent Channel Operations
- [ ] Test concurrent create_job_channel() calls
- [ ] Test concurrent remove_job_channel() calls
- [ ] Test concurrent send() to same channel
- [ ] Test concurrent send() to different channels
- [ ] Test create + send race condition
- [ ] Test send + remove race condition
- [ ] Test multiple receivers attempting take (should fail)

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 3-4 days

#### Memory Leak Tests
- [ ] Test channel cleanup on job completion
- [ ] Test channel cleanup on job failure
- [ ] Test channel cleanup on timeout
- [ ] Test channel cleanup on client disconnect
- [ ] Test sender cleanup when receiver dropped
- [ ] Test receiver cleanup when sender dropped
- [ ] Test memory usage with 1000+ jobs
- [ ] Test memory usage with long-running jobs

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

#### Job Isolation Tests
- [ ] Test narration with job_id routes to correct channel
- [ ] Test narration without job_id is dropped
- [ ] Test narration with wrong job_id is dropped
- [ ] Test multiple jobs don't leak events
- [ ] Test job_id validation (format, length)

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

### 1.3 Compile-Time Validation Tests

#### NarrationFactory Actor Validation
- [ ] Test actor with 10 chars (should compile)
- [ ] Test actor with 11 chars (should NOT compile)
- [ ] Test actor with unicode (count chars, not bytes)
- [ ] Test actor with emojis (count as 1 char)
- [ ] Test empty actor (should NOT compile)

**Priority:** MEDIUM  
**Complexity:** Low (compile-time tests)  
**Estimated Effort:** 1 day

#### Action Runtime Validation
- [ ] Test action with 15 chars (should work)
- [ ] Test action with 16 chars (should panic)
- [ ] Test action with unicode (count chars, not bytes)
- [ ] Test empty action (should panic)

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 0.5 days

### 1.4 Correlation ID Tests

#### Generation Tests
- [ ] Test UUID v4 format (36 chars, correct pattern)
- [ ] Test uniqueness (generate 10000, check no duplicates)
- [ ] Test randomness (statistical distribution)

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 0.5 days

#### Validation Tests
- [ ] Test valid UUID v4 (should pass)
- [ ] Test invalid format (should fail)
- [ ] Test wrong length (should fail)
- [ ] Test wrong version (UUID v1, v3, v5 should fail)
- [ ] Test uppercase vs lowercase (should both work)
- [ ] Test with/without hyphens

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 0.5 days

#### Header Parsing Tests
- [ ] Test extract from valid header
- [ ] Test extract from missing header
- [ ] Test extract from malformed header
- [ ] Test extract from multiple headers (should use first)

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 0.5 days

---

## 2. Daemon Lifecycle (`daemon-lifecycle`)

**Source:** TEAM-231

### 2.1 Daemon Spawn Tests

#### Success Cases
- [ ] Test spawn with valid binary path
- [ ] Test spawn with command-line arguments
- [ ] Test spawn with environment variables
- [ ] Test PID capture and validation
- [ ] Test process is actually running (check /proc)
- [ ] Test Stdio::null() (stdout/stderr not inherited)

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Failure Cases
- [ ] Test binary not found (should error)
- [ ] Test binary not executable (should error)
- [ ] Test invalid arguments (should error)
- [ ] Test spawn failure (e.g., fork failure)
- [ ] Test error message clarity

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

### 2.2 Binary Resolution Tests

#### Search Order
- [ ] Test finds binary in target/debug (if exists)
- [ ] Test finds binary in target/release (if debug missing)
- [ ] Test error if not found in either
- [ ] Test priority (debug before release)

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Edge Cases
- [ ] Test with absolute path (should use as-is)
- [ ] Test with relative path
- [ ] Test with symlinks
- [ ] Test with spaces in path
- [ ] Test with unicode in path

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 0.5 days

### 2.3 SSH Agent Propagation Tests

#### Propagation
- [ ] Test SSH_AUTH_SOCK propagated if present
- [ ] Test no error if SSH_AUTH_SOCK missing
- [ ] Test daemon can use SSH agent (integration test)

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 1 day

### 2.4 Stdio::null() Behavior Tests

#### Pipe Inheritance
- [ ] Test daemon doesn't hold parent's stdout pipe
- [ ] Test daemon doesn't hold parent's stderr pipe
- [ ] Test parent can exit immediately
- [ ] Test Command::output() doesn't hang
- [ ] Test E2E test scenario (was hanging, now fixed)

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 2-3 days

### 2.5 Concurrent Spawn Tests

#### Concurrency
- [ ] Test spawn 10 daemons concurrently
- [ ] Test each daemon gets unique PID
- [ ] Test no resource conflicts
- [ ] Test all daemons running after spawn

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 1-2 days

---

## 3. Config & Operations (`rbee-config` + `rbee-operations`)

**Source:** TEAM-232

### 3.1 Config Loading Tests

#### SSH Config Parsing
- [ ] Test parse valid SSH config
- [ ] Test parse with comments
- [ ] Test parse with extra whitespace
- [ ] Test parse with tabs vs spaces
- [ ] Test parse with missing fields (should error)
- [ ] Test parse with invalid port (should error)
- [ ] Test parse with duplicate hosts (should error or use last)

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### TOML Parsing
- [ ] Test parse valid config.toml
- [ ] Test parse with comments
- [ ] Test parse with missing sections (should use defaults)
- [ ] Test parse with invalid types (should error)
- [ ] Test parse with extra fields (should ignore)

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### YAML Parsing
- [ ] Test parse valid capabilities.yaml
- [ ] Test parse with missing fields (should error)
- [ ] Test parse with invalid types (should error)
- [ ] Test parse with extra fields (should ignore)
- [ ] Test parse empty file (should use defaults)

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

### 3.2 Concurrent Access Tests

#### File Locking
- [ ] Test concurrent reads (should work)
- [ ] Test concurrent writes (should serialize or error)
- [ ] Test read during write (should see old or new, not partial)
- [ ] Test write during read (should not corrupt)

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 2-3 days

### 3.3 Config Corruption Tests

#### Corruption Handling
- [ ] Test truncated file (should error)
- [ ] Test invalid UTF-8 (should error)
- [ ] Test partial write (should error)
- [ ] Test recovery from backup (if implemented)

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 1-2 days

### 3.4 Capabilities Cache Tests

#### Staleness Detection
- [ ] Test cache age calculation
- [ ] Test stale cache warning (if >24h)
- [ ] Test cache invalidation
- [ ] Test cache refresh

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Cache Consistency
- [ ] Test cache matches actual capabilities
- [ ] Test cache update on refresh
- [ ] Test cache persistence across restarts

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 1-2 days

### 3.5 Localhost Special Case Tests

#### Localhost Handling
- [ ] Test hives.conf missing → empty config (no error)
- [ ] Test localhost operations work without hives.conf
- [ ] Test remote operations require hives.conf
- [ ] Test auto-generate template on remote request

**Priority:** HIGH  
**Complexity:** Low  
**Estimated Effort:** 1 day

### 3.6 Operation Enum Tests

#### Serialization
- [ ] Test all operation types serialize correctly
- [ ] Test all operation types deserialize correctly
- [ ] Test default values (alias → "localhost", stream → true)
- [ ] Test optional fields (top_p, top_k, device, worker_id)
- [ ] Test missing required fields (should error)
- [ ] Test extra fields (should ignore)

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

#### Forward Compatibility
- [ ] Test old client with new server (extra fields)
- [ ] Test new client with old server (missing fields)

**Priority:** LOW  
**Complexity:** Low  
**Estimated Effort:** 0.5 days

---

## 4. Job Registry (`job-registry`)

**Source:** TEAM-233

### 4.1 Concurrent Job Tests

#### Race Conditions
- [ ] Test concurrent create_job() (should generate unique IDs)
- [ ] Test concurrent set_payload() on same job
- [ ] Test concurrent take_payload() on same job (only one succeeds)
- [ ] Test concurrent set_token_receiver() on same job
- [ ] Test concurrent take_token_receiver() on same job (only one succeeds)
- [ ] Test concurrent update_state() on same job
- [ ] Test concurrent remove_job() on same job

**Priority:** HIGH  
**Complexity:** High  
**Estimated Effort:** 3-4 days

### 4.2 Memory Leak Tests

#### Job Cleanup
- [ ] Test jobs removed after completion
- [ ] Test jobs removed after error
- [ ] Test jobs removed after timeout
- [ ] Test jobs NOT removed if remove_job() not called (leak)
- [ ] Test memory usage with 1000+ jobs
- [ ] Test memory usage with long-running jobs

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2-3 days

### 4.3 execute_and_stream Tests

#### Execution
- [ ] Test with successful operation
- [ ] Test with failing operation
- [ ] Test with operation that panics
- [ ] Test with operation that times out
- [ ] Test narration emitted correctly
- [ ] Test error narration on failure

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

#### Streaming
- [ ] Test token stream consumed correctly
- [ ] Test stream closes on completion
- [ ] Test stream closes on error
- [ ] Test stream closes on timeout
- [ ] Test multiple consumers (should fail - take semantics)

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 4.4 Stream Cancellation Tests

#### Client Disconnect
- [ ] Test client disconnect mid-stream
- [ ] Test receiver dropped (sender should fail)
- [ ] Test sender dropped (receiver should see None)
- [ ] Test cleanup after disconnect

**Priority:** HIGH  
**Complexity:** Medium  
**Estimated Effort:** 2 days

### 4.5 Job State Transition Tests

#### State Machine
- [ ] Test Queued → Running transition
- [ ] Test Running → Completed transition
- [ ] Test Running → Failed transition
- [ ] Test invalid transitions (should error or be no-op)
- [ ] Test concurrent state updates

**Priority:** MEDIUM  
**Complexity:** Medium  
**Estimated Effort:** 1-2 days

### 4.6 Payload Tests

#### Serialization
- [ ] Test small payload (<1KB)
- [ ] Test large payload (>1MB)
- [ ] Test very large payload (>10MB)
- [ ] Test payload with nested structures
- [ ] Test payload with binary data
- [ ] Test payload serialization errors

**Priority:** MEDIUM  
**Complexity:** Low  
**Estimated Effort:** 1 day

---

## Summary - Part 1

**Total Test Categories:** 6 shared crates  
**Total Test Tasks:** ~150 individual tests  
**Estimated Total Effort:** 40-60 days (with 1 developer)

**Priority Breakdown:**
- HIGH: ~80 tests (25-35 days)
- MEDIUM: ~50 tests (12-18 days)
- LOW: ~20 tests (3-7 days)

**Next:** Part 2 - Binary Components Testing Gaps
