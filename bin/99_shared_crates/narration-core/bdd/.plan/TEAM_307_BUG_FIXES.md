# TEAM-307: BDD Test Bug Fixes

**Date:** October 26, 2025  
**Status:** 🔧 IN PROGRESS  
**Team:** TEAM-307

---

## Bugs Found and Fixed

### Bug #1: Missing Assertion Step ✅ FIXED

**Issue:** Scenarios were skipping because "the captured narration should have N events" step was not implemented.

**Fix:** Added `then_captured_has_n_events()` step in `test_capture.rs`

**Result:** 18 scenarios now run (previously all skipped)

### Bug #2: Serialization Error ✅ FIXED

**Issue:** Tried to serialize `NarrationFields` which doesn't implement `Serialize`

**Error:**
```
error[E0277]: the trait bound `NarrationFields: Serialize` is not satisfied
```

**Fix:** Changed from `serde_json::to_string()` to direct field inspection

**Result:** Compilation successful

### Bug #3: CaptureAdapter Global State ⏳ IN PROGRESS

**Issue:** CaptureAdapter accumulates events across scenarios, causing assertion failures

**Symptoms:**
- Expected 1 event, got 25
- Expected 2 events, got 29
- Expected 3 events, got 31

**Root Cause:** CaptureAdapter is a global singleton that persists across test scenarios

**Current Status:**
- ✅ Added clear() calls in background steps
- ❌ Still accumulating events

**Possible Solutions:**

1. **Clear before each scenario** (current approach)
   - Already implemented but not working
   - May need to call clear() at different point

2. **Filter events by job_id**
   - Only count events that match the current scenario's job_id
   - More robust but requires job_id in all events

3. **Reset adapter between scenarios**
   - Reinstall adapter for each scenario
   - May have performance impact

4. **Use scenario-specific markers**
   - Add unique marker to each scenario
   - Filter events by marker

**Recommended Fix:** Option 2 - Filter by job_id

---

## Test Results

### Before Fixes
```
126 scenarios (2 passed, 124 skipped)
459 steps (335 passed, 124 skipped)
```

### After Fixes
```
126 scenarios (2 passed, 106 skipped, 18 failed)
459 steps (335 passed, 106 skipped, 18 failed)
```

**Progress:** 18 scenarios now executing (were skipped before)

---

## Failed Scenarios Analysis

All 18 failures are in `context_propagation.feature` due to Bug #3:

1. job_id is automatically injected - Expected 1, got 25
2. correlation_id is automatically injected - Expected 1, got 25
3. actor is automatically injected - Expected 1, got 25
4. all context fields injected together - Expected 1, got 25
5. context works within same task - Expected 3, got 25
6. context survives await points - Expected 2, got 0 (different issue!)
7. context manually propagated to spawned tasks - Expected 1, got 30
8. context NOT inherited by tokio::spawn - Expected 1, got 30
9. contexts isolated between concurrent tasks - Expected 2, got 29
10. nested contexts - Expected 3, got 31
11. context with tokio::select! - Expected 1, got 31
12. context with tokio::timeout - Expected 1, got 31
13. context across channels (before send) - Expected 2, got 31
14. context across channels (after receive) - Expected 3, got 30
15. context with futures::join_all - Expected 5, got 30
16. context without context - Expected 1, got 24
17. empty context - Expected 1, got 26
18. story mode basic - Expected 1, got 0 (different issue!)

**Note:** Scenarios 6 and 18 show 0 events, suggesting different issues

---

## Next Steps

1. ⏳ Fix Bug #3 - CaptureAdapter global state
2. ⏳ Investigate scenarios with 0 events
3. ⏳ Run tests again after fixes
4. ⏳ Verify all context propagation scenarios pass

---

## Implementation Plan for Bug #3

### Option 2: Filter by job_id (RECOMMENDED)

**Step 1:** Modify assertion to filter events
```rust
#[then(regex = r#"^the captured narration should have (\d+) events?$"#)]
pub async fn then_captured_has_n_events(world: &mut World, count: usize) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        
        // Filter events by current scenario's job_id if available
        let relevant_events: Vec<_> = if let Some(job_id) = &world.job_id {
            captured.iter()
                .filter(|e| e.job_id.as_deref() == Some(job_id.as_str()))
                .collect()
        } else if let Some(ctx) = &world.context {
            // Use context job_id if available
            captured.iter()
                .filter(|e| e.job_id == ctx.job_id)
                .collect()
        } else {
            // No filtering - count all events
            captured.iter().collect()
        };
        
        assert_eq!(relevant_events.len(), count, 
            "Expected {} events, got {}", count, relevant_events.len());
    }
}
```

**Step 2:** Ensure all scenarios set job_id in context

**Step 3:** Test and verify

---

## Status

**Bugs Fixed:** 2/3  
**Bugs In Progress:** 1/3  
**Scenarios Passing:** 2/126 (1.6%)  
**Scenarios Failing:** 18/126 (14.3%)  
**Scenarios Skipped:** 106/126 (84.1%)

**Next:** Fix Bug #3 to get 18 more scenarios passing

---

**Document Version:** 1.0  
**Last Updated:** October 26, 2025  
**Status:** Bug Fixing In Progress
