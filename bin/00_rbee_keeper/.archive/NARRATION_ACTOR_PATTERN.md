# Narration Actor Pattern for rbee-keeper

**TEAM-309** | **Date:** 2025-10-26

---

## Problem

When using the `n!()` macro without setting an actor context, all narrations show `actor="unknown"`:

```
actor="unknown" action="self_check_start" human=Starting rbee-keeper self-check
```

This makes it impossible to identify which component emitted the narration.

---

## Solution

Wrap your async handler in `NarrationContext` with `.with_actor()`:

```rust
use observability_narration_core::{NarrationContext, with_narration_context, n};

pub async fn handle_self_check() -> Result<()> {
    // Set actor context for all narrations in this handler
    let ctx = NarrationContext::new()
        .with_actor("rbee-keeper");
    
    with_narration_context(ctx, async {
        run_self_check_tests().await
    }).await
}

async fn run_self_check_tests() -> Result<()> {
    // All n!() calls inside automatically use actor="rbee-keeper"
    n!("self_check_start", "Starting rbee-keeper self-check");
    n!("version_check", "Checking version {}", env!("CARGO_PKG_VERSION"));
    Ok(())
}
```

**Result:**
```
actor="rbee-keeper" action="self_check_start" human=Starting rbee-keeper self-check
actor="rbee-keeper" action="version_check" human=Checking version 0.1.0
```

---

## Why This Works

The `n!()` macro calls `macro_emit()` which:
1. Checks for thread-local `NarrationContext`
2. Extracts `actor` from context (if present)
3. Uses `actor` in the narration fields
4. Falls back to `"unknown"` if no context

**Source:** `bin/99_shared_crates/narration-core/src/api/macro_impl.rs:33-40`

```rust
let ctx = crate::context::get_context();
let actor = ctx.as_ref().and_then(|c| c.actor).unwrap_or("unknown");
```

---

## Pattern for All Handlers

**Apply this pattern to ALL handler functions in `src/handlers/`:**

```rust
// ❌ BEFORE (actor="unknown")
pub async fn handle_queen(action: QueenAction, queen_url: &str) -> Result<()> {
    n!("queen_start", "Starting queen");
    // ...
}

// ✅ AFTER (actor="rbee-keeper")
pub async fn handle_queen(action: QueenAction, queen_url: &str) -> Result<()> {
    let ctx = NarrationContext::new().with_actor("rbee-keeper");
    
    with_narration_context(ctx, async {
        execute_queen_action(action, queen_url).await
    }).await
}

async fn execute_queen_action(action: QueenAction, queen_url: &str) -> Result<()> {
    n!("queen_start", "Starting queen");
    // All narrations automatically use actor="rbee-keeper"
}
```

---

## Additional Context Fields

You can also set `job_id` and `correlation_id` in the context:

```rust
let ctx = NarrationContext::new()
    .with_actor("rbee-keeper")
    .with_job_id("job-abc123")
    .with_correlation_id("corr-xyz789");

with_narration_context(ctx, async {
    n!("action", "message");
    // Automatically includes: actor, job_id, correlation_id
}).await
```

**When to use:**
- `actor` - Always (identifies component)
- `job_id` - When submitting jobs to queen-rbee (for SSE routing)
- `correlation_id` - When tracing requests end-to-end (optional)

---

## Files to Update

Apply this pattern to all handlers:

- ✅ `src/handlers/self_check.rs` - DONE (TEAM-309)
- ⏳ `src/handlers/queen.rs` - TODO
- ⏳ `src/handlers/hive.rs` - TODO
- ⏳ `src/handlers/worker.rs` - TODO
- ⏳ `src/handlers/model.rs` - TODO
- ⏳ `src/handlers/infer.rs` - TODO
- ⏳ `src/handlers/status.rs` - TODO

---

## References

- **Context API:** `bin/99_shared_crates/narration-core/src/context.rs`
- **Macro Implementation:** `bin/99_shared_crates/narration-core/src/api/macro_impl.rs`
- **Example:** `bin/00_rbee_keeper/src/handlers/self_check.rs`
- **Tests:** `bin/99_shared_crates/narration-core/tests/thread_local_context_tests.rs`

---

## Summary

**Problem:** `actor="unknown"` in all narrations  
**Solution:** Wrap handlers in `NarrationContext::new().with_actor("rbee-keeper")`  
**Result:** `actor="rbee-keeper"` in all narrations  
**Status:** ✅ Pattern established, ready to apply to all handlers
