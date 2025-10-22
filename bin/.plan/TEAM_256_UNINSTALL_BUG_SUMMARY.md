# TEAM-256 Uninstall Idempotency Bug - Investigation Summary

**Status:** 🔴 BLOCKED - Narration system issue discovered

**Problem:** Second `hive uninstall` doesn't show "already uninstalled" message

**Root Cause:** Narration `.emit()` calls after `hive_debug` are silently failing or being dropped

## Investigation Timeline

1. ✅ Fixed stale config issue - reload from disk works
2. ✅ Confirmed `had_capabilities=false` is correct
3. ✅ Code structure is correct (if/else logic)
4. ✅ Binary is being rebuilt
5. 🔴 **BLOCKED:** Narration calls after line 78 never execute

## Evidence

```
[hive-life ] hive_debug     : 🐞 DEBUG: had_capabilities=false
[DONE]  ← Function exits immediately after
```

- `eprintln!()` calls don't show (daemon stderr issue)
- Narration after `hive_debug` never emits
- No panic, no error, just silent exit
- Code after line 78 never executes

## Hypothesis

The narration system or SSE stream is closing/failing after a certain point, causing subsequent narrations to be dropped and the function to exit early.

## Workaround Needed

Instead of fixing narration, modify the response message based on `had_capabilities` so the user sees the status in the final response, not in narration.

## Files

- `bin/15_queen_rbee_crates/hive-lifecycle/src/uninstall.rs` (lines 73-161)
