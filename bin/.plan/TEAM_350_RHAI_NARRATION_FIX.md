# TEAM-350: RHAI Narration SSE Routing Fix

**Status:** ✅ COMPLETE

## Problem

When pressing the "Test" button in the RHAI IDE, narration events were not reaching the GUI. The error displayed was:

```
❌ Test failed:
Unknown error
```

## Root Cause

**Missing narration context in all RHAI operations.**

The `n!()` macro gets `job_id` from thread-local context (see `narration-core/src/api/macro_impl.rs:87`):

```rust
let ctx = crate::context::get_context();
let job_id = ctx.as_ref().and_then(|c| c.job_id.clone());
```

**Without `job_id`:** Narration events are dropped by SSE sink (fail-fast security - see SYSTEM-RETRIEVED-MEMORY about job_id being REQUIRED for SSE routing).

**Result:** Events went to stdout only, never reached SSE stream → GUI never saw them.

## The Flow (Before Fix)

1. **GUI** (`RhaiIDE.tsx`) → Test button pressed
2. **React Hook** (`useRhaiScripts.ts`) → Calls `testScript(content)`
3. **SDK** (`queen-rbee-sdk`) → Submits `RhaiScriptTest` operation via `QueenClient.submitAndStream()`
4. **Backend** (`queen-rbee/src/job_router.rs:207`) → Routes to `execute_rhai_script_test()`
5. **RHAI Handler** (`rhai/test.rs`) → Calls `n!()` macro WITHOUT narration context
6. **Narration Core** → `job_id` is `None` → SSE sink drops events ❌
7. **GUI** → Never receives narration events → Shows "Unknown error"

## The Fix

Wrapped all RHAI operation handlers in `with_narration_context()` to set `job_id` for SSE routing.

### Pattern Applied

```rust
// BEFORE (BROKEN)
pub async fn execute_rhai_script_test(job_id: &str, content: String) -> Result<()> {
    n!("rhai_test_start", "🧪 Testing RHAI script");
    // ... more narration ...
    Ok(())
}

// AFTER (FIXED)
pub async fn execute_rhai_script_test(job_id: &str, content: String) -> Result<()> {
    // TEAM-350: Set narration context with job_id for SSE routing
    let ctx = NarrationContext::new().with_job_id(job_id);
    
    with_narration_context(ctx, async {
        n!("rhai_test_start", "🧪 Testing RHAI script");
        // ... more narration ...
        Ok(())
    }).await
}
```

## Files Changed

All 5 RHAI operation handlers fixed:

1. **`bin/10_queen_rbee/src/rhai/test.rs`** - Test operation (the reported bug)
2. **`bin/10_queen_rbee/src/rhai/save.rs`** - Save operation (same bug)
3. **`bin/10_queen_rbee/src/rhai/get.rs`** - Get operation (same bug)
4. **`bin/10_queen_rbee/src/rhai/list.rs`** - List operation (same bug)
5. **`bin/10_queen_rbee/src/rhai/delete.rs`** - Delete operation (same bug)

## Verification

```bash
cargo check --bin queen-rbee
# ✅ PASS (exit code 0)
```

## Expected Behavior (After Fix)

When pressing "Test" button:

1. GUI sends `RhaiScriptTest` operation
2. Backend wraps execution in narration context with `job_id`
3. All `n!()` macro calls include `job_id` automatically
4. SSE sink routes events to correct job channel
5. GUI receives narration events via SSE stream:
   - `🧪 Testing RHAI script`
   - `Script length: X bytes`
   - `✅ Script executed successfully`
   - `Output: (placeholder - implement RHAI execution)`
6. `narrationBridge.ts` parses events and sends to parent window
7. rbee-keeper displays narration in narration panel

## Key Architectural Insight

**The `n!()` macro is NOT self-contained** - it requires narration context to be set via `with_narration_context()` for SSE routing to work.

This is by design for security:
- Without `job_id`, events are dropped (fail-fast)
- Prevents narration from one job leaking to another
- Forces explicit job isolation

## Related Patterns

See how other operations do this correctly:

```rust
// bin/10_queen_rbee/src/job_router.rs:197
Operation::QueenCheck => {
    let ctx = NarrationContext::new().with_job_id(&job_id);
    with_narration_context(ctx, queen_check::handle_queen_check()).await?;
}
```

## Documentation

- **Narration Architecture:** `/home/vince/Projects/llama-orch/bin/NARRATION_AND_JOB_ID_ARCHITECTURE.md`
- **SSE Routing:** SYSTEM-RETRIEVED-MEMORY[ab27c6ba-a497-4501-ba44-880ecffead71]
- **TimeoutEnforcer Fix:** SYSTEM-RETRIEVED-MEMORY[661e7f3f-1793-4be3-9627-3e515c496dd9] (same root cause)

## Compilation

✅ All changes compile successfully with no errors.

## Team Signature

**TEAM-350** - Oct 29, 2025
