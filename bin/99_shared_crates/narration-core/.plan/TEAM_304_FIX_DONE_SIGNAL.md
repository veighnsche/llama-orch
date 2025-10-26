# TEAM-304: Fix [DONE] Signal Architecture

**Status:** 🚨 CRITICAL - MUST FIX  
**Priority:** P0 (Blocking)  
**Estimated Duration:** 4-5 hours  
**Dependencies:** None  
**Blocks:** All future work (architectural issue)

---

## Mission

Fix critical architectural violation where [DONE] signal is being emitted by narration-core instead of job-server. This violates separation of concerns and will cause production issues.

---

## Problem Statement

**Current State:**
- ❌ narration-core emits `n!("done", "[DONE]")` 
- ❌ job-server does NOT emit [DONE]
- ❌ Mixing job lifecycle with observability

**Impact:**
- Streams may never close properly in production
- Cannot track job completion
- Cannot add job cancellation/timeouts
- Architectural violation

**Root Cause:**
- job-server was created without [DONE] signal
- Tests failed because streams never closed
- Quick fix: emit [DONE] via narration (WRONG)

---

## Implementation Tasks

### Task 1: Fix job-server (2 hours)

**File:** `bin/99_shared_crates/job-server/src/lib.rs`

**Change `execute_and_stream()` to send [DONE]:**

```rust
pub async fn execute_and_stream<T, F, Exec>(
    job_id: String,
    registry: Arc<JobRegistry<T>>,
    executor: Exec,
) -> impl Stream<Item = String>
where
    T: ToString + Send + 'static,
    F: std::future::Future<Output = Result<(), anyhow::Error>> + Send + 'static,
    Exec: FnOnce(String, serde_json::Value) -> F + Send + 'static,
{
    let payload = registry.take_payload(&job_id);

    if let Some(payload) = payload {
        let job_id_clone = job_id.clone();
        let registry_clone = registry.clone();

        tokio::spawn(async move {
            NARRATE
                .action("execute")
                .job_id(&job_id_clone)
                .context(job_id_clone.clone())
                .human("Executing job {}")
                .emit();

            // Execute the job
            let result = executor(job_id_clone.clone(), payload).await;
            
            // TEAM-304: Update job state based on result
            match result {
                Ok(_) => {
                    registry_clone.update_state(&job_id_clone, JobState::Completed);
                }
                Err(e) => {
                    registry_clone.update_state(&job_id_clone, JobState::Failed(e.to_string()));
                    NARRATE
                        .action("failed")
                        .job_id(&job_id_clone)
                        .context(job_id_clone.clone())
                        .context(e.to_string())
                        .human("Job {} failed: {}")
                        .error_kind("job_execution_failed")
                        .emit_error();
                }
            }
        });
    }

    // TEAM-304: Stream results and send [DONE] or [ERROR]
    let receiver = registry.take_token_receiver(&job_id);
    let registry_clone = registry.clone();
    let job_id_clone = job_id.clone();

    stream::unfold((receiver, false, job_id_clone, registry_clone), 
        |(rx_opt, done_sent, job_id, registry)| async move {
            if done_sent {
                return None;
            }

            match rx_opt {
                Some(mut rx) => match rx.recv().await {
                    Some(token) => {
                        let data = token.to_string();
                        Some((data, (Some(rx), false, job_id, registry)))
                    }
                    None => {
                        // Channel closed - check job state and send appropriate signal
                        let state = registry.get_job_state(&job_id);
                        let signal = match state {
                            Some(JobState::Failed(err)) => format!("[ERROR] {}", err),
                            _ => "[DONE]".to_string(),
                        };
                        Some((signal, (None, true, job_id, registry)))
                    }
                },
                None => {
                    // No receiver - send [DONE] immediately
                    Some(("[DONE]".to_string(), (None, true, job_id, registry)))
                }
            }
        }
    )
}
```

### Task 2: Remove [DONE] from narration-core (30 min)

**Files to fix:**

1. `narration-core/tests/e2e_job_client_integration.rs` (line 83)
2. `narration-core/tests/bin/fake_queen.rs` (line 98)
3. `narration-core/tests/bin/fake_hive.rs` (line 103)

**Change:**
```rust
// BEFORE (WRONG):
n!("done", "[DONE]");

// AFTER (CORRECT):
// Remove this line - job-server will send [DONE]
```

### Task 3: Update job-client (30 min)

**File:** `bin/99_shared_crates/job-client/src/lib.rs`

**Add [ERROR] handling:**

```rust
// Around line 163, after [DONE] check:
if data.contains("[DONE]") {
    return Ok(job_id);
}

// TEAM-304: Add [ERROR] handling
if data.contains("[ERROR]") {
    let error_msg = data.strip_prefix("[ERROR]").unwrap_or(&data).trim();
    return Err(anyhow::anyhow!("Job failed: {}", error_msg));
}
```

### Task 4: Add Tests (1 hour)

**File:** `bin/99_shared_crates/job-server/tests/done_signal_tests.rs` (NEW)

```rust
// TEAM-304: Tests for [DONE] and [ERROR] signals

use job_server::{JobRegistry, execute_and_stream};
use std::sync::Arc;
use futures::StreamExt;

#[tokio::test]
async fn test_execute_and_stream_sends_done_on_success() {
    let registry = Arc::new(JobRegistry::new());
    let job_id = registry.create_job();
    
    // Set up payload
    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));
    
    // Set up channel
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);
    
    // Send some tokens
    tx.send("token1".to_string()).unwrap();
    tx.send("token2".to_string()).unwrap();
    drop(tx);  // Close channel to trigger [DONE]
    
    // Stream should include [DONE]
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async { Ok(()) }  // Successful execution
    ).await;
    
    let results: Vec<String> = stream.collect().await;
    
    assert_eq!(results.len(), 3);
    assert_eq!(results[0], "token1");
    assert_eq!(results[1], "token2");
    assert_eq!(results[2], "[DONE]");
}

#[tokio::test]
async fn test_execute_and_stream_sends_error_on_failure() {
    let registry = Arc::new(JobRegistry::new());
    let job_id = registry.create_job();
    
    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));
    
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);
    
    tx.send("token1".to_string()).unwrap();
    drop(tx);
    
    // Stream should include [ERROR]
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async { Err(anyhow::anyhow!("Test error")) }  // Failed execution
    ).await;
    
    let results: Vec<String> = stream.collect().await;
    
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], "token1");
    assert!(results[1].starts_with("[ERROR]"));
    assert!(results[1].contains("Test error"));
}

#[tokio::test]
async fn test_done_sent_only_once() {
    let registry = Arc::new(JobRegistry::new());
    let job_id = registry.create_job();
    
    registry.set_payload(&job_id, serde_json::json!({"test": "data"}));
    
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);
    
    drop(tx);  // Close immediately
    
    let stream = execute_and_stream(
        job_id.clone(),
        registry.clone(),
        |_, _| async { Ok(()) }
    ).await;
    
    let results: Vec<String> = stream.collect().await;
    
    // Should only have one [DONE]
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], "[DONE]");
}
```

### Task 5: Update Documentation (30 min)

**Files:**
- `job-server/README.md` - Document [DONE] and [ERROR] signals
- `job-client/README.md` - Document expected signals
- `narration-core/README.md` - Clarify narration ≠ lifecycle signals

---

## Verification Checklist

- [ ] job-server sends [DONE] when channel closes
- [ ] job-server sends [ERROR] when executor fails
- [ ] narration-core does NOT emit [DONE]
- [ ] job-client handles [DONE] correctly
- [ ] job-client handles [ERROR] correctly
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Architecture review confirms separation of concerns

---

## Success Criteria

1. **Separation of Concerns**
   - job-server manages job lifecycle
   - narration-core handles observability
   - No mixing of responsibilities

2. **Proper Signaling**
   - [DONE] sent on successful completion
   - [ERROR] sent on failure
   - Signals sent exactly once per job

3. **Tests Pass**
   - All existing tests still pass
   - New tests verify [DONE] and [ERROR]
   - No tests rely on narration for [DONE]

---

## Handoff to TEAM-305

Document in `.plan/TEAM_304_HANDOFF.md`:

1. **What Was Fixed**
   - [DONE] signal moved to job-server
   - [ERROR] signal added
   - Separation of concerns restored

2. **Test Results**
   - All tests passing
   - [DONE] and [ERROR] verified

3. **Next Steps**
   - TEAM-305: Fix circular dependency (JobRegistry)
   - TEAM-306: Context propagation tests
   - TEAM-307: Failure scenario tests

---

**TEAM-304 Mission:** Fix critical [DONE] signal architecture violation

**Priority:** P0 - MUST FIX IMMEDIATELY

**Estimated Time:** 4-5 hours
