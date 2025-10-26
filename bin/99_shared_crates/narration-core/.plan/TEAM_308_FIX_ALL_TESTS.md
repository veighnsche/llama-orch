# TEAM-308: Fix All Broken Tests

**Status:** 🔧 CLEANUP REQUIRED  
**Priority:** P1 (High)  
**Estimated Duration:** 3-4 hours  
**Dependencies:** TEAM-304, TEAM-305  
**Blocks:** Production readiness

---

## Mission

Fix all broken tests after architectural changes from TEAM-304 ([DONE] signal) and TEAM-305 (circular dependency). Ensure 100% test pass rate.

---

## Problem Statement

**After TEAM-304 and TEAM-305 changes:**
- ❌ Tests expecting narration to emit [DONE] will fail
- ❌ Tests using old HashMap-based fake binaries will fail
- ❌ Integration tests may have timing issues
- ❌ Old integration.rs tests may be broken

**Impact:**
- Cannot merge changes until tests pass
- Cannot claim production readiness
- CI/CD pipeline blocked

---

## Affected Test Files

### 1. narration-core Tests

**Files:**
- `tests/e2e_job_client_integration.rs`
- `tests/job_server_basic.rs`
- `tests/job_server_concurrent.rs`
- `tests/e2e_real_processes.rs`
- `tests/integration.rs` (old, may be broken)

### 2. job-server Tests

**Files:**
- `tests/resource_cleanup_tests.rs`
- Any tests expecting [DONE] from external source

### 3. job-client Tests

**Files:**
- Any tests that verify [DONE] handling

---

## Implementation Tasks

### Task 1: Fix e2e_job_client_integration.rs (1 hour)

**File:** `narration-core/tests/e2e_job_client_integration.rs`

**Issues:**
1. Test emits `n!("done", "[DONE]")` - REMOVE
2. Test expects [DONE] from narration - UPDATE to expect from job-server

**Changes:**

```rust
// BEFORE (line 83):
n!("stream_complete", "Request complete");
tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
n!("done", "[DONE]");  // ❌ REMOVE THIS

// AFTER:
n!("stream_complete", "Request complete");
// TEAM-308: [DONE] now sent by job-server, not narration
```

**Update stream handler:**

```rust
// The stream handler needs to be updated to work with job-server's [DONE]
// Since we're using a test server, we need to manually send [DONE] via the token channel

async fn stream_job_handler(
    State(_state): State<Arc<TestServerState>>,
    Path(job_id): Path<String>,
) -> axum::response::Response {
    use axum::response::sse::{Event, Sse};
    use futures::stream;
    use axum::response::IntoResponse;
    
    let receiver = observability_narration_core::output::sse_sink::take_job_receiver(&job_id);
    
    if let Some(rx) = receiver {
        // Emit narration in background
        let job_id_clone = job_id.clone();
        tokio::spawn(async move {
            let ctx = NarrationContext::new().with_job_id(&job_id_clone);
            with_narration_context(ctx, async {
                n!("stream_start", "Stream started");
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                n!("stream_processing", "Processing request");
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                n!("stream_complete", "Request complete");
                // TEAM-308: No [DONE] here - that's job-server's job
            }).await;
        });
        
        // TEAM-308: For this test, we need to simulate job-server behavior
        // In real usage, job-server's execute_and_stream would send [DONE]
        let event_stream = stream::unfold((rx, false), |(mut rx, done_sent)| async move {
            if done_sent {
                return None;
            }
            
            match rx.recv().await {
                Some(event) => {
                    let data = event.formatted.clone();
                    Some((Ok::<_, std::io::Error>(Event::default().data(data)), (rx, false)))
                }
                None => {
                    // TEAM-308: Channel closed, send [DONE] like job-server does
                    Some((Ok::<_, std::io::Error>(Event::default().data("[DONE]")), (rx, true)))
                }
            }
        });
        
        Sse::new(event_stream).into_response()
    } else {
        let empty_stream = stream::iter(vec![
            Ok::<_, std::io::Error>(Event::default().data("error: job not found"))
        ]);
        Sse::new(empty_stream).into_response()
    }
}
```

### Task 2: Fix e2e_real_processes.rs (1 hour)

**File:** `narration-core/tests/e2e_real_processes.rs`

**Issues:**
1. Fake binaries emit `n!("done", "[DONE]")` - already fixed in TEAM-304
2. Tests may need timing adjustments

**Verify:**
```bash
cargo test -p observability-narration-core --test e2e_real_processes --features axum -- --ignored --nocapture
```

**If tests fail:**
- Check timing (increase timeouts if needed)
- Verify [DONE] is being sent by job-server logic
- Check that fake binaries are using real JobRegistry (TEAM-305)

### Task 3: Fix job_server Tests (30 min)

**File:** `job-server/tests/resource_cleanup_tests.rs`

**Check line 26:**
```rust
tx.send("[DONE]".to_string()).unwrap();  // This is OK - it's sending to token channel
```

This is actually correct - the test is sending [DONE] as a token, which job-server will then stream. No changes needed.

### Task 4: Fix or Remove integration.rs (1 hour)

**File:** `narration-core/tests/integration.rs`

**This file uses old CaptureAdapter which is deprecated.**

**Options:**

**Option A: Delete it** (RECOMMENDED)
```bash
rm narration-core/tests/integration.rs
```

These tests are superseded by:
- `job_server_basic.rs`
- `job_server_concurrent.rs`
- `e2e_job_client_integration.rs`
- `e2e_real_processes.rs`

**Option B: Fix it** (if tests are still valuable)
- Update to use new narration-core v0.5.0 API
- Remove CaptureAdapter usage
- Use SSE sink instead

### Task 5: Run Full Test Suite (30 min)

**Run all tests:**

```bash
# Build binaries first
cargo build --bin fake-queen-rbee --bin fake-rbee-hive --bin fake-worker --features axum

# Run all narration-core tests
cargo test -p observability-narration-core --features axum

# Run job-server tests
cargo test -p job-server

# Run job-client tests
cargo test -p job-client

# Run E2E tests (ignored by default)
cargo test -p observability-narration-core --test e2e_real_processes --features axum -- --ignored
```

**Expected results:**
- All tests pass
- No warnings about [DONE] in wrong place
- No circular dependency errors

---

## Verification Checklist

- [ ] `e2e_job_client_integration.rs` tests pass
- [ ] `job_server_basic.rs` tests pass
- [ ] `job_server_concurrent.rs` tests pass
- [ ] `e2e_real_processes.rs` tests pass
- [ ] `integration.rs` deleted or fixed
- [ ] job-server tests pass
- [ ] job-client tests pass
- [ ] No [DONE] emitted by narration
- [ ] All [DONE] emitted by job-server
- [ ] 100% test pass rate

---

## Success Criteria

1. **All Tests Pass**
   - narration-core: 100% pass
   - job-server: 100% pass
   - job-client: 100% pass
   - E2E tests: 100% pass

2. **Correct Architecture**
   - [DONE] only from job-server
   - No [DONE] from narration
   - Real JobRegistry in test binaries

3. **CI/CD Ready**
   - Can merge to main
   - Can deploy to production
   - No test failures

---

## Known Issues to Watch For

### Issue 1: Timing Sensitivity

**Problem:** E2E tests may be flaky due to timing

**Solution:**
- Increase timeouts if needed
- Use `--test-threads=1` for sequential execution
- Add retry logic for flaky tests

### Issue 2: Port Conflicts

**Problem:** Multiple tests using same ports

**Solution:**
- Each test uses unique port
- Or use port 0 for automatic assignment

### Issue 3: Process Cleanup

**Problem:** Orphaned processes from failed tests

**Solution:**
```bash
# Kill any stray fake binaries
pkill -f fake-queen-rbee
pkill -f fake-rbee-hive
pkill -f fake-worker
```

---

## Handoff to TEAM-309

Document in `.plan/TEAM_308_HANDOFF.md`:

1. **What Was Fixed**
   - All tests updated for new architecture
   - [DONE] signal tests corrected
   - Old integration.rs removed/fixed
   - 100% test pass rate achieved

2. **Test Results**
   - All tests passing
   - No architectural violations
   - Production ready

3. **Next Steps**
   - TEAM-309: Rename TEAM_304_PHASE_3 → TEAM_306
   - TEAM-310: Rename TEAM_305_PHASE_4 → TEAM_307
   - Ready for production deployment

---

**TEAM-308 Mission:** Fix all broken tests after architectural changes

**Priority:** P1 - HIGH (blocks production)

**Estimated Time:** 3-4 hours
