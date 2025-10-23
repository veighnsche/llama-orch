# Testing Gaps - Additional Findings (Reasonable Scope)

**Date:** Oct 22, 2025  
**Source:** Deep dive into Phase 4 behavior documents  
**Scope:** NUC-friendly tests (1-10 concurrent operations, not 100+)

---

## Overview

After reviewing all Phase 4 behavior documents, I found **additional specific testing gaps** that were missed in the initial checklist. These are **IMPLEMENTED features with NO tests**, focused on **reasonable scale** for a NUC environment.

---

## 1. SSH Client (TEAM-222) - 0% Test Coverage

**Component:** `bin/15_queen_rbee_crates/ssh-client`  
**LOC:** 263  
**Current Coverage:** 0%  
**Priority:** HIGH (used for remote hive operations)

### Missing Tests (15 tests, 5-7 days)

#### Pre-flight Checks
- [ ] Test SSH_AUTH_SOCK not set (should return helpful error)
- [ ] Test SSH_AUTH_SOCK set but empty string
- [ ] Test SSH_AUTH_SOCK points to non-existent socket

**Priority:** HIGH  
**Complexity:** Low  
**Effort:** 0.5 days

#### TCP Connection Tests
- [ ] Test connection to unreachable host (timeout behavior)
- [ ] Test connection to invalid host:port format
- [ ] Test connection timeout (5s default)
- [ ] Test connection to non-SSH server (e.g., HTTP on port 22)

**Priority:** HIGH  
**Complexity:** Medium  
**Effort:** 1-2 days

#### SSH Handshake Tests
- [ ] Test SSH protocol version mismatch
- [ ] Test handshake timeout
- [ ] Test handshake with corrupted response

**Priority:** MEDIUM  
**Complexity:** Medium  
**Effort:** 1-2 days

#### Authentication Tests
- [ ] Test with no SSH keys loaded in agent
- [ ] Test with wrong username
- [ ] Test userauth_agent success but authenticated() = false
- [ ] Test with correct keys but server rejects

**Priority:** HIGH  
**Complexity:** Medium  
**Effort:** 2 days

#### Command Execution Tests
- [ ] Test channel creation failure
- [ ] Test command exec failure
- [ ] Test command with non-zero exit status
- [ ] Test command output parsing

**Priority:** MEDIUM  
**Complexity:** Low  
**Effort:** 1 day

#### Narration Tests
- [ ] Test narration emitted on success
- [ ] Test narration emitted on failure
- [ ] Test narration includes correct target/error

**Priority:** LOW  
**Complexity:** Low  
**Effort:** 0.5 days

---

## 2. Hive Lifecycle (TEAM-220) - ~10% Test Coverage

**Component:** `bin/15_queen_rbee_crates/hive-lifecycle`  
**LOC:** ~1,629  
**Current Coverage:** ~10% (basic unit tests only)  
**Priority:** HIGH (core queen-rbee functionality)

### Missing Tests (25 tests, 10-15 days)

#### Binary Resolution Tests
- [ ] Test config.binary_path exists (use it)
- [ ] Test config.binary_path missing, debug exists (use debug)
- [ ] Test config.binary_path missing, debug missing, release exists (use release)
- [ ] Test all paths missing (error with helpful message)
- [ ] Test binary path with spaces
- [ ] Test binary path with symlinks

**Priority:** HIGH  
**Complexity:** Low  
**Effort:** 1-2 days

#### Health Polling Tests
- [ ] Test health poll success on first attempt
- [ ] Test health poll success after 5 attempts (exponential backoff)
- [ ] Test health poll timeout after 10 attempts
- [ ] Test exponential backoff timing (200ms, 400ms, 800ms...)

**Priority:** HIGH  
**Complexity:** Medium  
**Effort:** 2-3 days

#### Capabilities Cache Tests
- [ ] Test cache hit (return cached, suggest refresh)
- [ ] Test cache miss (fetch fresh, update cache)
- [ ] Test cache refresh (force fetch, update cache)
- [ ] Test cache cleanup on uninstall
- [ ] Test cache staleness detection (>24h)
- [ ] Test cache with corrupted file

**Priority:** MEDIUM  
**Complexity:** Medium  
**Effort:** 2-3 days

#### Graceful Shutdown Tests
- [ ] Test SIGTERM success (exits within 5s)
- [ ] Test SIGTERM timeout → SIGKILL fallback
- [ ] Test process already stopped (idempotent)
- [ ] Test health check polling during shutdown (1s intervals)

**Priority:** HIGH  
**Complexity:** Medium  
**Effort:** 2-3 days

#### Localhost Special Case Tests
- [ ] Test localhost operations without hives.conf
- [ ] Test remote operations require hives.conf
- [ ] Test auto-generate template on remote request
- [ ] Test localhost default configuration

**Priority:** HIGH  
**Complexity:** Low  
**Effort:** 1 day

#### Timeout Enforcement Tests
- [ ] Test capabilities fetch timeout (15s)
- [ ] Test health check timeout (2s)
- [ ] Test status check timeout (5s)
- [ ] Test timeout narration includes job_id

**Priority:** HIGH  
**Complexity:** Low  
**Effort:** 1 day

#### Error Message Quality Tests
- [ ] Test all error messages include actionable advice
- [ ] Test hive not found lists available hives
- [ ] Test binary not found suggests cargo build
- [ ] Test connection errors show exact endpoint

**Priority:** MEDIUM  
**Complexity:** Low  
**Effort:** 1 day

---

## 3. Hive Registry (TEAM-221) - ~5% Test Coverage

**Component:** `bin/15_queen_rbee_crates/hive-registry`  
**LOC:** 186  
**Current Coverage:** ~5% (basic unit tests only)  
**Priority:** HIGH (critical for queen-rbee state)

### Missing Tests (20 tests, 7-10 days)

#### Concurrent Access Tests (Reasonable Scale)
- [ ] Test 10 concurrent update_hive_state() calls (different hives)
- [ ] Test 10 concurrent update_hive_state() calls (same hive)
- [ ] Test 5 concurrent reads during 5 writes
- [ ] Test concurrent get_worker() calls (linear search)
- [ ] Test RwLock behavior (readers don't block readers)

**Priority:** HIGH  
**Complexity:** High  
**Effort:** 3-4 days

**Note:** NOT 100+ concurrent calls - that's overkill for a NUC. 10 concurrent is reasonable.

#### Staleness Detection Tests
- [ ] Test hive marked stale after 30s (6 missed heartbeats)
- [ ] Test hive marked active on heartbeat received
- [ ] Test list_active_hives() excludes stale hives
- [ ] Test staleness calculation with clock skew

**Priority:** HIGH  
**Complexity:** Medium  
**Effort:** 2 days

#### Worker Aggregation Tests
- [ ] Test hive with 0 workers
- [ ] Test hive with 1 worker
- [ ] Test hive with 5 workers
- [ ] Test worker state updates reflected in registry
- [ ] Test get_worker() with multiple hives

**Priority:** MEDIUM  
**Complexity:** Medium  
**Effort:** 2 days

#### Memory Leak Tests (Reasonable Scale)
- [ ] Test 100 hive updates (not 1000+)
- [ ] Test memory usage stays constant
- [ ] Test old states are replaced (not accumulated)

**Priority:** HIGH  
**Complexity:** Medium  
**Effort:** 1-2 days

#### Edge Cases
- [ ] Test empty registry (no hives)
- [ ] Test hive with empty hostname
- [ ] Test hive with invalid port (0, 65536)
- [ ] Test worker with empty model_id

**Priority:** LOW  
**Complexity:** Low  
**Effort:** 1 day

---

## 4. Config Loading (TEAM-232) - ~20% Test Coverage

**Component:** `bin/99_shared_crates/rbee-config`  
**LOC:** ~400  
**Current Coverage:** ~20% (basic parsing only)  
**Priority:** MEDIUM

### Missing Tests (15 tests, 5-7 days)

#### SSH Config Edge Cases
- [ ] Test config with comments (should ignore)
- [ ] Test config with extra whitespace (should trim)
- [ ] Test config with tabs vs spaces (should handle both)
- [ ] Test config with missing required fields (should error)
- [ ] Test config with invalid port (should error)
- [ ] Test config with duplicate hosts (should error or use last)

**Priority:** MEDIUM  
**Complexity:** Low  
**Effort:** 1-2 days

#### Concurrent File Access (Reasonable Scale)
- [ ] Test 5 concurrent reads (should work)
- [ ] Test 5 concurrent writes (should serialize)
- [ ] Test read during write (should see old or new, not partial)

**Priority:** HIGH  
**Complexity:** High  
**Effort:** 2-3 days

**Note:** NOT 100+ concurrent - 5 is reasonable for a NUC.

#### Config Corruption Tests
- [ ] Test truncated file (should error)
- [ ] Test invalid UTF-8 (should error)
- [ ] Test partial write (should error)
- [ ] Test empty file (should use defaults)

**Priority:** MEDIUM  
**Complexity:** Medium  
**Effort:** 1-2 days

#### YAML Capabilities Tests
- [ ] Test parse valid capabilities.yaml
- [ ] Test parse with missing GPU (CPU only)
- [ ] Test parse with multiple GPUs
- [ ] Test parse with invalid device type

**Priority:** LOW  
**Complexity:** Low  
**Effort:** 1 day

---

## 5. Narration Edge Cases (TEAM-230) - ~30% Test Coverage

**Component:** `bin/99_shared_crates/observability-narration-core`  
**LOC:** ~800  
**Current Coverage:** ~30% (basic unit tests)  
**Priority:** HIGH

### Missing Tests (20 tests, 7-10 days)

#### Format String Edge Cases
- [ ] Test context with quotes (should escape)
- [ ] Test context with newlines (should handle)
- [ ] Test context with unicode (should count chars correctly)
- [ ] Test context with emojis (should count as 1 char)
- [ ] Test very long context (>1000 chars, should truncate?)

**Priority:** MEDIUM  
**Complexity:** Low  
**Effort:** 1-2 days

#### Table Formatting Edge Cases
- [ ] Test nested objects (depth 3)
- [ ] Test large arrays (50 items, not 100+)
- [ ] Test empty objects/arrays
- [ ] Test null values
- [ ] Test mixed types in arrays
- [ ] Test very long strings (>500 chars)
- [ ] Test table width overflow

**Priority:** MEDIUM  
**Complexity:** Medium  
**Effort:** 2-3 days

#### SSE Channel Edge Cases (Reasonable Scale)
- [ ] Test 10 concurrent create_job_channel() calls
- [ ] Test create + send race condition
- [ ] Test send + remove race condition
- [ ] Test multiple receivers attempting take (should fail)

**Priority:** HIGH  
**Complexity:** High  
**Effort:** 2-3 days

**Note:** NOT 100+ concurrent - 10 is reasonable.

#### Job Isolation Edge Cases
- [ ] Test narration with malformed job_id (should drop)
- [ ] Test narration with very long job_id (should handle)
- [ ] Test job_id validation (format, length)

**Priority:** MEDIUM  
**Complexity:** Low  
**Effort:** 1 day

---

## 6. Daemon Lifecycle Edge Cases (TEAM-231) - 0% Test Coverage

**Component:** `bin/99_shared_crates/daemon-lifecycle`  
**LOC:** ~200  
**Current Coverage:** 0%  
**Priority:** HIGH

### Missing Tests (10 tests, 3-5 days)

#### Stdio::null() Critical Tests
- [ ] Test daemon doesn't hold parent's stdout pipe
- [ ] Test daemon doesn't hold parent's stderr pipe
- [ ] Test parent can exit immediately after spawn
- [ ] Test Command::output() doesn't hang (E2E scenario)

**Priority:** HIGH  
**Complexity:** High  
**Effort:** 2-3 days

**Note:** This is CRITICAL - was causing E2E test hangs (TEAM-164 fix).

#### Binary Resolution Edge Cases
- [ ] Test with absolute path (should use as-is)
- [ ] Test with relative path (should resolve)
- [ ] Test with symlinks (should follow)
- [ ] Test with spaces in path
- [ ] Test with unicode in path

**Priority:** MEDIUM  
**Complexity:** Low  
**Effort:** 1 day

#### Concurrent Spawn Tests (Reasonable Scale)
- [ ] Test spawn 5 daemons concurrently (not 10+)
- [ ] Test each daemon gets unique PID
- [ ] Test no resource conflicts

**Priority:** MEDIUM  
**Complexity:** Medium  
**Effort:** 1-2 days

**Note:** 5 concurrent is reasonable for a NUC.

---

## 7. Job Registry Edge Cases (TEAM-233) - ~15% Test Coverage

**Component:** `bin/99_shared_crates/job-registry`  
**LOC:** ~300  
**Current Coverage:** ~15% (basic unit tests)  
**Priority:** HIGH

### Missing Tests (15 tests, 5-7 days)

#### Payload Size Tests
- [ ] Test small payload (<1KB)
- [ ] Test medium payload (100KB)
- [ ] Test large payload (1MB, not 10MB+)
- [ ] Test payload with nested structures (depth 5)

**Priority:** MEDIUM  
**Complexity:** Low  
**Effort:** 1 day

**Note:** 1MB is reasonable for a NUC, 10MB+ is overkill.

#### Stream Cancellation Edge Cases
- [ ] Test client disconnect mid-stream
- [ ] Test receiver dropped before sender
- [ ] Test sender dropped before receiver
- [ ] Test cleanup after disconnect

**Priority:** HIGH  
**Complexity:** Medium  
**Effort:** 2-3 days

#### Job State Edge Cases
- [ ] Test invalid state transitions (should error or no-op)
- [ ] Test concurrent state updates (same job)
- [ ] Test state query during transition

**Priority:** MEDIUM  
**Complexity:** Medium  
**Effort:** 1-2 days

#### Memory Leak Tests (Reasonable Scale)
- [ ] Test 100 jobs (not 1000+)
- [ ] Test memory usage after cleanup
- [ ] Test jobs removed after completion

**Priority:** HIGH  
**Complexity:** Medium  
**Effort:** 1-2 days

**Note:** 100 jobs is reasonable for a NUC.

---

## 8. Heartbeat Edge Cases (TEAM-234) - 0% Test Coverage

**Component:** `bin/99_shared_crates/rbee-heartbeat`  
**LOC:** ~400  
**Current Coverage:** 0%  
**Priority:** HIGH

### Missing Tests (15 tests, 5-7 days)

#### Background Task Edge Cases
- [ ] Test task starts correctly
- [ ] Test task continues after failure (retry)
- [ ] Test task stops on abort signal
- [ ] Test task doesn't block main thread

**Priority:** HIGH  
**Complexity:** Medium  
**Effort:** 2-3 days

#### Retry Logic Edge Cases
- [ ] Test retry on connection refused
- [ ] Test retry on timeout
- [ ] Test retry on 5xx errors
- [ ] Test no retry on 4xx errors
- [ ] Test backoff timing (if any)

**Priority:** HIGH  
**Complexity:** Medium  
**Effort:** 2 days

#### Worker Aggregation Edge Cases
- [ ] Test empty worker list
- [ ] Test single worker
- [ ] Test 5 workers (not 10+)
- [ ] Test worker state updates

**Priority:** MEDIUM  
**Complexity:** Medium  
**Effort:** 1-2 days

**Note:** 5 workers is reasonable for a NUC.

#### Staleness Edge Cases
- [ ] Test staleness with clock skew
- [ ] Test staleness boundary (exactly 30s)
- [ ] Test staleness recovery (heartbeat after stale)

**Priority:** MEDIUM  
**Complexity:** Low  
**Effort:** 1 day

---

## Summary of Additional Findings

### New Test Count

**By Component:**
- SSH Client: 15 tests (5-7 days)
- Hive Lifecycle: 25 tests (10-15 days)
- Hive Registry: 20 tests (7-10 days)
- Config Loading: 15 tests (5-7 days)
- Narration Edge Cases: 20 tests (7-10 days)
- Daemon Lifecycle: 10 tests (3-5 days)
- Job Registry: 15 tests (5-7 days)
- Heartbeat: 15 tests (5-7 days)

**Total: 135 additional tests, 47-68 days effort**

### Combined with Original Checklist

**Original:** ~450 tests, 130-180 days  
**Additional:** ~135 tests, 47-68 days  
**Total:** ~585 tests, 177-248 days (1 developer)

**With Team of 3:** ~20-28 weeks (5-7 months)

### Critical Additions

1. **SSH Client** - 0% coverage, HIGH priority (15 tests)
2. **Stdio::null()** - CRITICAL for E2E tests (4 tests)
3. **Binary Resolution** - Core functionality (6 tests)
4. **Graceful Shutdown** - User-facing (4 tests)
5. **Capabilities Cache** - Performance critical (6 tests)

### Reasonable Scale Guidelines

**What's Reasonable for a NUC:**
- ✅ 5-10 concurrent operations
- ✅ 100 jobs/hives/workers
- ✅ 1MB payloads
- ✅ 5 workers per hive
- ✅ 10 concurrent SSE channels

**What's Overkill:**
- ❌ 100+ concurrent operations
- ❌ 1000+ jobs/hives/workers
- ❌ 10MB+ payloads
- ❌ 50+ workers per hive
- ❌ 100+ concurrent SSE channels

---

## Recommendations

### Immediate Actions

1. **Add SSH Client tests** - 0% coverage, HIGH priority
2. **Add Stdio::null() tests** - CRITICAL for E2E
3. **Add Binary Resolution tests** - Core functionality
4. **Add Graceful Shutdown tests** - User-facing

### Short-Term Actions

1. **Add Hive Lifecycle tests** - Most complex component
2. **Add Hive Registry tests** - Critical state management
3. **Add Config Loading tests** - Edge cases and corruption

### Medium-Term Actions

1. **Add Narration edge cases** - Format strings, tables
2. **Add Job Registry edge cases** - Payloads, cancellation
3. **Add Heartbeat edge cases** - Background tasks, retry

---

**Next:** Create testing engineer guide
